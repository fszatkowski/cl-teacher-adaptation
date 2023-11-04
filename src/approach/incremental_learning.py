import time
import torch
import numpy as np
from argparse import ArgumentParser

from loggers.exp_logger import ExperimentLogger
from datasets.exemplars_dataset import ExemplarsDataset


class Inc_Learning_Appr:
    """Basic class for implementing incremental learning approaches"""

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr=1e-1, wu_fix_bn=False,
                 wu_scheduler='constant', wu_patience=None, wu_wd=0., fix_bn=False,
                 eval_on_train=False, select_best_model_by_val_loss=True, logger: ExperimentLogger = None,
                 exemplars_dataset: ExemplarsDataset = None, scheduler_milestones=None, no_learning=False):
        self.model = model
        self.device = device
        self.nepochs = nepochs
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.momentum = momentum
        self.wd = wd
        self.multi_softmax = multi_softmax
        self.logger = logger
        self.exemplars_dataset = exemplars_dataset
        self.warmup_epochs = wu_nepochs
        self.warmup_lr = wu_lr
        self.warmup_loss = torch.nn.CrossEntropyLoss()
        self.warmup_scheduler = wu_scheduler
        self.warmup_fix_bn = wu_fix_bn
        self.warmup_patience = wu_patience
        self.wu_wd = wu_wd
        self.fix_bn = fix_bn
        self.eval_on_train = eval_on_train
        self.select_best_model_by_val_loss = select_best_model_by_val_loss
        self.optimizer = None
        self.scheduler_milestones = scheduler_milestones
        self.scheduler = None
        self.debug = False
        self.no_learning = no_learning

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        return parser.parse_known_args(args)

    @staticmethod
    def exemplars_dataset_class():
        """Returns a exemplar dataset to use during the training if the approach needs it
        :return: ExemplarDataset class or None
        """
        return None

    def _get_optimizer(self):
        """Returns the optimizer"""
        return torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def _get_scheduler(self):
        if self.scheduler_milestones is not None:
            return torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer, milestones=self.scheduler_milestones,
                                                        gamma=0.1)
        else:
            return None

    def train(self, t, trn_loader, val_loader):
        """Main train structure"""
        self.pre_train_process(t, trn_loader)
        self.train_loop(t, trn_loader, val_loader)
        self.post_train_process(t, trn_loader)

    def pre_train_process(self, t, trn_loader):
        """Runs before training all epochs of the task (before the train session)"""

        # Warm-up phase
        if self.warmup_epochs and t > 0:
            prev_heads_b_norm = torch.cat([h.bias for h in self.model.heads[:-1]], dim=0).detach().norm().item()
            prev_heads_w_norm = torch.cat([h.weight for h in self.model.heads[:-1]], dim=0).detach().norm().item()
            self._log_weight_norms(t, prev_heads_w_norm, prev_heads_b_norm,
                                   self.model.heads[-1].weight.detach().norm().item(),
                                   self.model.heads[-1].bias.detach().norm().item())

            head_params = self.model.heads[-1].parameters()
            optimizer = torch.optim.SGD(head_params, lr=self.warmup_lr, weight_decay=self.wu_wd)

            if self.warmup_scheduler.lower() == 'constant':
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0)
            elif self.warmup_scheduler.lower() == 'plateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                                       min_lr=1e-4)
            elif self.warmup_scheduler.lower() == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=0)
            elif self.warmup_scheduler.lower() == 'onecycle':
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.warmup_lr,
                                                                total_steps=self.warmup_epochs,
                                                                pct_start=0.2)
            else:
                raise NotImplementedError(f"Unknown scheduler: {self.warmup_scheduler}.")

            patience = 0
            best_loss = float('inf')
            best_model = self.model.state_dict()

            # Loop epochs -- train warm-up head
            for e in range(self.warmup_epochs):
                warmupclock0 = time.time()

                if self.warmup_fix_bn:
                    self.model.eval()
                else:
                    self.model.train()
                self.model.heads[-1].train()

                for images, targets in trn_loader:
                    images = images.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)

                    outputs = self.model(images)
                    loss = self.warmup_loss(outputs[t], targets - self.model.task_offset[t])
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(head_params, self.clipgrad)
                    optimizer.step()
                warmupclock1 = time.time()

                with torch.no_grad():
                    total_loss, total_acc_taw = 0, 0
                    self.model.eval()
                    for images, targets in trn_loader:
                        images, targets = images.to(self.device), targets.to(self.device)
                        outputs = self.model(images)
                        loss = self.warmup_loss(outputs[t], targets - self.model.task_offset[t])
                        pred = torch.zeros_like(targets)
                        for m in range(len(pred)):
                            this_task = (self.model.task_cls.cumsum(0).to(self.device) <= targets[m]).sum()
                            pred[m] = outputs[this_task][m].argmax() + self.model.task_offset[this_task]
                        hits_taw = (pred == targets).float()
                        total_loss += loss.item() * len(targets)
                        total_acc_taw += hits_taw.sum().item()
                total_num = len(trn_loader.dataset.labels)
                trn_loss, trn_acc = total_loss / total_num, total_acc_taw / total_num
                warmupclock2 = time.time()
                print('| Warm-up Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
                    e + 1, warmupclock1 - warmupclock0, warmupclock2 - warmupclock1, trn_loss, 100 * trn_acc))
                self.logger.log_scalar(task=None, iter=e + 1, name="trn_loss", value=trn_loss, group=f"warmup_t{t}")
                self.logger.log_scalar(task=None, iter=e + 1, name="trn_acc", value=100 * trn_acc, group=f"warmup_t{t}")

                # Evaluate
                warmupclock3 = time.time()
                outputs = self._evaluate(t, debug=self.debug)
                for name, value in outputs.items():
                    self.logger.log_scalar(task=None, iter=e + 1, name=name, group=f"warmup_t{t}",
                                           value=value)
                if self.debug:
                    self.logger.log_scalar(task=None, iter=e + 1, name='lr', group=f"warmup_t{t}",
                                           value=optimizer.param_groups[0]["lr"])
                    self._log_weight_norms(t, prev_heads_w_norm, prev_heads_b_norm,
                                           self.model.heads[-1].weight.detach().norm().item(),
                                           self.model.heads[-1].bias.detach().norm().item())

                warmupclock4 = time.time()
                print('| Epoch {:3d}, time={:5.1f}s | Eval: loss={:.3f}, TAg loss={:.3f}, TAw acc={:5.1f}% |'.format(
                    e + 1, warmupclock4 - warmupclock3, outputs['ce_taw_current_task'],
                    outputs['ce_tag_current_task'],
                    100 * outputs['taw_acc_current_task']), end=''
                )

                if self.warmup_scheduler == 'plateau':
                    scheduler.step(outputs['tag_acc_current_task'])
                else:
                    scheduler.step()

                if self.warmup_patience is not None:
                    if outputs['ce_taw_current_task'] < best_loss:
                        best_loss = outputs['ce_taw_current_task']
                        best_model = self.model.state_dict()
                        patience = 0
                        print("*")
                    else:
                        patience += 1
                        print()
                        if patience > self.warmup_patience:
                            print(f"Stopping early at epoch {e + 1}")
                            break
                else:
                    best_model = self.model.state_dict()
                    print()

            self.model.load_state_dict(state_dict=best_model)

    def _evaluate(self, t, debug=False):
        if t == 0:
            raise ValueError()

        loaders = self.tst_loader[:t + 1]

        self.model.eval()
        per_task_taw_acc = []
        per_task_tag_acc = []
        per_task_ce_taw = []
        per_task_ce_tag = []

        with torch.no_grad():
            for i, loader in enumerate(loaders):
                if not self.debug and i != len(loaders) - 1:
                    continue

                total_acc_taw, total_acc_tag, total_ce_taw, total_ce_tag, total_num = 0, 0, 0, 0, 0
                for images, targets in loader:
                    images, targets = images.to(self.device), targets.to(self.device)
                    outputs = self.model(images)
                    hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                    ce_taw = torch.nn.functional.cross_entropy(outputs[i], targets - self.model.task_offset[i],
                                                               reduction='sum')
                    ce_tag = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets, reduction='sum')

                    # Log
                    total_acc_taw += hits_taw.sum().data.cpu().numpy().item()
                    total_acc_tag += hits_tag.sum().data.cpu().numpy().item()
                    total_ce_taw += ce_taw.cpu().item()
                    total_ce_tag += ce_tag.cpu().item()
                    total_num += len(targets)
                per_task_taw_acc.append(total_acc_taw / total_num)
                per_task_tag_acc.append(total_acc_tag / total_num)
                per_task_ce_taw.append(total_ce_taw / total_num)
                per_task_ce_tag.append(total_ce_tag / total_num)

        if debug:
            output = {
                "tag_acc_current_task": per_task_tag_acc[-1],
                "tag_acc_all_tasks": sum(per_task_tag_acc[:-1]) / len(per_task_tag_acc[:-1]),
                "taw_acc_current_task": per_task_taw_acc[-1],
                "ce_taw_current_task": per_task_ce_taw[-1],
                "ce_taw_all_tasks": sum(per_task_ce_taw) / len(per_task_ce_taw),
                "ce_tag_current_task": per_task_ce_tag[-1],
                "ce_tag_all_tasks": sum(per_task_ce_tag) / len(per_task_ce_tag),
            }
        else:
            output = {
                "tag_acc_current_task": per_task_tag_acc[-1],
                "taw_acc_current_task": per_task_taw_acc[-1],
                "ce_taw_current_task": per_task_ce_taw[-1],
                "ce_tag_current_task": per_task_ce_tag[-1],
            }
        return output

    def _log_weight_norms(self, t, prev_w, prev_b, new_w, new_b):
        self.logger.log_scalar(task=None, iter=None, name='prev_heads_w_norm', group=f"wu_w_t{t}",
                               value=prev_w)
        self.logger.log_scalar(task=None, iter=None, name='prev_heads_b_norm', group=f"wu_w_t{t}",
                               value=prev_b)
        self.logger.log_scalar(task=None, iter=None, name='new_head_w_norm', group=f"wu_w_t{t}",
                               value=new_w)
        self.logger.log_scalar(task=None, iter=None, name='new_head_b_norm', group=f"wu_w_t{t}",
                               value=new_b)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""
        lr = self.lr
        best_loss = np.inf
        patience = self.lr_patience
        best_model = self.model.get_copy()

        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()

        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0 = time.time()
            self.train_epoch(t, trn_loader)
            clock1 = time.time()
            if self.eval_on_train:
                train_loss, train_acc, _ = self.eval(t, trn_loader)
                clock2 = time.time()
                print('| Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
                    e + 1, clock1 - clock0, clock2 - clock1, train_loss, 100 * train_acc), end='')
                self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=train_loss, group="train")
                self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * train_acc, group="train")
            else:
                print('| Epoch {:3d}, time={:5.1f}s | Train: skip eval |'.format(e + 1, clock1 - clock0), end='')

            # Valid
            clock3 = time.time()
            valid_loss, valid_acc, _ = self.eval(t, val_loader)
            clock4 = time.time()
            print(' Valid: time={:5.1f}s loss={:.3f}, TAw acc={:5.1f}% |'.format(
                clock4 - clock3, valid_loss, 100 * valid_acc), end='')
            self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=valid_loss, group="valid")
            self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * valid_acc, group="valid")

            if self.scheduler is not None:
                lr = self.scheduler.get_last_lr()[0]

            if self.select_best_model_by_val_loss:
                # Adapt learning rate - patience scheme - early stopping regularization
                if valid_loss < best_loss:
                    # if the loss goes down, keep it as the best model and end line with a star ( * )
                    best_loss = valid_loss
                    best_model = self.model.get_copy()
                    patience = self.lr_patience
                    print(' *', end='')
                else:
                    patience -= 1
                    if self.scheduler is None:
                        # if the loss does not go down, decrease patience
                        if patience <= 0:
                            # if it runs out of patience, reduce the learning rate
                            lr /= self.lr_factor
                            print(' lr={:.1e}'.format(lr), end='')
                            if lr < self.lr_min:
                                # if the lr decreases below minimum, stop the training session
                                print()
                                break
                            # reset patience and recover best model so far to continue training
                            patience = self.lr_patience
                            self.optimizer.param_groups[0]['lr'] = lr
                            self.model.set_state_dict(best_model)
            else:
                best_model = self.model.get_copy()

            self.logger.log_scalar(task=t, iter=e + 1, name="patience", value=patience, group="train")
            self.logger.log_scalar(task=t, iter=e + 1, name="lr", value=lr, group="train")
            print()

        self.model.set_state_dict(best_model)

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        pass

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            # Forward current model
            outputs = self.model(images.to(self.device))
            loss = self.criterion(t, outputs, targets.to(self.device))
            # Backward
            if t == 0 or not self.no_learning:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                # Forward current model
                images, targets = images.to(self.device), targets.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(t, outputs, targets)
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def calculate_metrics(self, outputs, targets):
        """Contains the main Task-Aware and Task-Agnostic metrics"""
        pred = torch.zeros_like(targets)
        # Task-Aware Multi-Head
        for m in range(len(pred)):
            this_task = (self.model.task_cls.cumsum(0).to(self.device) <= targets[m]).sum()
            pred[m] = outputs[this_task][m].argmax() + self.model.task_offset[this_task]
        hits_taw = (pred == targets).float()
        # Task-Agnostic Multi-Head
        if self.multi_softmax:
            outputs = [torch.nn.functional.log_softmax(output, dim=1) for output in outputs]
            pred = torch.cat(outputs, dim=1).argmax(1)
        else:
            pred = torch.cat(outputs, dim=1).argmax(1)
        hits_tag = (pred == targets).float()
        return hits_taw, hits_tag

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        return torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])
