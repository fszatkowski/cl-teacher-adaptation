import torch
from copy import deepcopy
from argparse import ArgumentParser

from metrics import cka
from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Inc_Learning_Appr):
    """Class implementing A-LwF approach from 
    Achieving a Better Stability-Plasticity Trade-off via Auxiliary Networks in Continual Learning 
    (CVPR 2023)"""
    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr=1e-1, wu_fix_bn=False,
                 wu_scheduler='constant', wu_patience=None, wu_wd=0., fix_bn=False, eval_on_train=False,
                 select_best_model_by_val_loss=True, logger=None, exemplars_dataset=None, scheduler_milestones=None,
                 lamb=1, lamb_a=1, T=2, mc=False, taskwise_kd=False,
                 ta=False,
                 cka=False, debug_loss=False
                 ):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr, wu_fix_bn, wu_scheduler, wu_patience, wu_wd,
                                   fix_bn, eval_on_train, select_best_model_by_val_loss, logger, exemplars_dataset,
                                   scheduler_milestones)
        self.model_old = None
        self.model_aux = None
        self.lamb = lamb
        self.lamb_a = lamb_a
        self.T = T
        self.mc = mc
        self.taskwise_kd = taskwise_kd
        self.ta = ta
        self.cka = cka
        self.debug_loss = debug_loss

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Page 5: "lambda is a loss balance weight, set to 1 for most our experiments. Making lambda larger will favor
        # the old task performance over the new task’s, so we can obtain a old-task-new-task performance line by
        # changing lambda."
        parser.add_argument('--lamb', default=10, type=float, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')
        parser.add_argument('--lamb-a', default=1, type=float, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')
        # Page 5: "We use T=2 according to a grid search on a held out set, which aligns with the authors’
        #  recommendations." -- Using a higher value for T produces a softer probability distribution over classes.
        parser.add_argument('--T', default=2, type=int, required=False,
                            help='Temperature scaling (default=%(default)s)')
        parser.add_argument('--mc', default=False, action='store_true', required=False,
                            help='If set, will use LwF.MC variant from iCaRL. (default=%(default)s)')
        parser.add_argument('--taskwise-kd', default=False, action='store_true', required=False,
                            help='If set, will use task-wise KD loss as defined in SSIL. (default=%(default)s)')

        parser.add_argument('--ta', default=False, action='store_true', required=False,
                            help='Teacher adaptation. If set, will update old model batch norm params '
                                 'during training the new task. (default=%(default)s)')

        parser.add_argument('--cka', default=False, action='store_true', required=False,
                            help='If set, will compute CKA between current representations and representations at '
                                 'the start of the task. (default=%(default)s)')
        parser.add_argument('--debug-loss', default=False, action='store_true', required=False,
                            help='If set, will log intermediate loss values. (default=%(default)s)')
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        if len(self.exemplars_dataset) == 0 and len(self.model.heads) > 1:
            # if there are no exemplars, previous heads are not modified
            params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        else:
            params = self.model.parameters()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""
        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)
        print("Lamb: ", self.lamb)
        print("Lamb_a: ", self.lamb_a)

        self.training = True

        if t > 0:
            print('=' * 108)
            print("Training of Auxiliary Network")
            print('=' * 108)
            # Args for the new trainer
            new_trainer_args = dict(nepochs=self.nepochs, lr=self.lr, lr_min=self.lr_min, lr_factor=self.lr_factor,
                            lr_patience=self.lr_patience, clipgrad=self.clipgrad, momentum=0.9,
                            wd=5e-4, multi_softmax=self.multi_softmax, wu_nepochs=0,
                            eval_on_train=self.eval_on_train, select_best_model_by_val_loss=self.select_best_model_by_val_loss,
                            logger=self.logger, exemplars_dataset=self.exemplars_dataset, scheduler_milestones=self.scheduler_milestones)
            self.model_aux = deepcopy(self.model)
            # Train auxiliary model on current dataset
            new_trainer = NewTaskTrainer(self.model_aux, self.device, **new_trainer_args)
            new_trainer.train_loop(t, trn_loader, val_loader)
            self.model_aux.eval()
            self.model_aux.freeze_all()
        
        print('=' * 108)
        print("Training of Main Network")
        print('=' * 108)
        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        # Restore best and save model for future tasks
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.ta and self.model_old is not None:
            self.model_old.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Forward old model and auxiliary model
            targets_old = None
            targets_aux = None
            if t > 0:
                targets_old = self.model_old(images)
                targets_aux = self.model_aux(images)
            # Forward current model
            outputs = self.model(images)

            loss, loss_kd, loss_kd_a, loss_ce = self.criterion(t, outputs, targets, targets_old, targets_aux, return_partial_losses=True)
            if self.debug_loss:
                self.logger.log_scalar(task=None, iter=None, name='loss_kd', group=f"debug_t{t}",
                                       value=float(loss_kd))
                self.logger.log_scalar(task=None, iter=None, name='loss_kd_a', group=f"debug_t{t}",
                                       value=float(loss_kd_a))
                self.logger.log_scalar(task=None, iter=None, name='loss_ce', group=f"debug_t{t}",
                                       value=float(loss_ce))
                self.logger.log_scalar(task=None, iter=None, name='loss_total', group=f"debug_t{t}",
                                       value=float(loss))

            assert not torch.isnan(loss), "Loss is NaN"

            # Backward
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
            if self.model_old is not None:
                self.model_old.eval()

            for images, targets in val_loader:
                # Forward old model and auxiliary model
                targets_old = None
                targets_aux = None
                if t > 0:
                    targets_old = self.model_old(images.to(self.device))
                    targets_aux = self.model_aux(images.to(self.device))
                # Forward current model
                outputs = self.model(images.to(self.device))
                loss = self.criterion(t, outputs, targets.to(self.device), targets_old, targets_aux)
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets.to(self.device))
                # Log
                total_loss += loss.data.cpu().numpy().item() * len(targets)
                total_acc_taw += hits_taw.sum().data.cpu().numpy().item()
                total_acc_tag += hits_tag.sum().data.cpu().numpy().item()
                total_num += len(targets)
        
        if self.cka and t > 0 and self.training:
            _cka = cka(self.model, self.model_old, val_loader, self.device)
            self.logger.log_scalar(task=None, iter=None, name=f't_{t}', group=f"cka", value=_cka)


        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
        """Calculates cross-entropy with temperature scaling"""
        out = torch.nn.functional.softmax(outputs, dim=1)
        tar = torch.nn.functional.softmax(targets, dim=1)
        if exp != 1:
            out = out.pow(exp)
            out = out / out.sum(1).view(-1, 1).expand_as(out)
            tar = tar.pow(exp)
            tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
        out = out + eps / out.size(1)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        ce = -(tar * out.log()).sum(1)
        if size_average:
            ce = ce.mean()
        return ce

    def criterion(self, t, outputs, targets, targets_old=None, targets_aux=None, return_partial_losses=False):
        """Returns the loss value"""
        loss = 0
        if t > 0:
            # Knowledge distillation loss for all previous tasks on old(previous) network
            kd_outputs = torch.cat(outputs[:t], dim=1)
            kd_outputs_old = torch.cat(targets_old[:t], dim=1)

            if self.mc:
                g = torch.sigmoid(kd_outputs)
                q_i = torch.sigmoid(kd_outputs_old)
                loss_kd = sum(
                    torch.nn.functional.binary_cross_entropy(g[:, y], q_i[:, y])
                    for y in range(kd_outputs.shape[-1]))
            elif self.taskwise_kd:
                loss_kd = torch.zeros(t).to(self.device)
                for _t in range(t):
                    soft_target = torch.nn.functional.softmax(targets_old[_t] / self.T, dim=1)
                    output_log = torch.nn.functional.log_softmax(outputs[_t] / self.T, dim=1)
                    loss_kd[_t] = torch.nn.functional.kl_div(output_log, soft_target, reduction='batchmean') * (self.T ** 2)
                loss_kd = loss_kd.sum()
            else:
                loss_kd = self.cross_entropy(kd_outputs, kd_outputs_old, exp=1.0 / self.T)

            # Auxiliary KD
            # Knowledge distillation loss for current task on new network
            loss_kd_a = self.cross_entropy(outputs[t],
                                                   targets_aux[t] - self.model.task_offset[t], exp=1.0 / self.T)
        else:
            loss_kd, loss_kd_a = 0, 0

        # Current cross-entropy loss -- with exemplars use all heads
        if len(self.exemplars_dataset) > 0:
            loss_ce = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        else:
            loss_ce = torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])

        if return_partial_losses:
            return self.lamb * loss_kd + self.lamb_a * loss_kd_a + loss_ce, loss_kd, loss_kd_a, loss_ce
        else:
            return self.lamb * loss_kd + self.lamb_a * loss_kd_a + loss_ce

class NewTaskTrainer(Inc_Learning_Appr):
    def __init__(self, model, device, nepochs=160, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0.9, wd=5e-4, multi_softmax=False, 
                 wu_nepochs=0, wu_lr=1e-1, wu_fix_bn=False, wu_scheduler='constant', wu_patience=None, wu_wd=0., fix_bn=False,
                 eval_on_train=False, select_best_model_by_val_loss=True, logger=None, exemplars_dataset=None, scheduler_milestones=None):
        super(NewTaskTrainer, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr, wu_fix_bn, wu_scheduler, wu_patience, wu_wd,
                                   fix_bn, eval_on_train, select_best_model_by_val_loss, logger, exemplars_dataset,
                                   scheduler_milestones)
