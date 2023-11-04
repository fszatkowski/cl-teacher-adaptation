import contextlib
import os
import torch
from copy import deepcopy
import torch.nn.functional as F
from argparse import ArgumentParser
from torchvision.utils import save_image

from networks.network import LLL_Net
from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Inc_Learning_Appr):
    """Class implementing the Learning without Memorizing (LwM) approach
    described in http://arxiv.org/abs/1811.08051
    """

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr=1e-1, wu_fix_bn=False,
                 wu_scheduler='constant', wu_patience=None, fix_bn=False, eval_on_train=False,
                 select_best_model_by_val_loss=True, logger=None, exemplars_dataset=None, scheduler_milestones=None,
                 beta=1.0, gamma=1.0, gradcam_layer='layer3', log_gradcam_samples=0,
                 ta=False,
                 debug_loss=False
                 ):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr, wu_fix_bn, wu_scheduler, wu_patience, fix_bn,
                                   eval_on_train,
                                   select_best_model_by_val_loss, logger, exemplars_dataset, scheduler_milestones)
        self.beta = beta
        self.gamma = gamma
        self.gradcam_layer = gradcam_layer
        self.log_gradcam_samples = log_gradcam_samples
        self.model_old = None
        self._samples_to_log_X = []
        self._samples_to_log_y = []

        self.ta = ta

        self.debug_loss = debug_loss

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        parser = ArgumentParser()
        # Paper, sec 4: beta and gamma introduce in Eq.3, but no values are given in the paper
        parser.add_argument('--beta', default=1, type=float, required=False,
                            help='Trade-off for distillation loss (default=%(default)s)')
        parser.add_argument('--gamma', default=1, type=float, required=False,
                            help='Trade-off for attention loss(default=%(default)s)')
        # Paper, sec 5.4: "The feature maps of the final conv layer are used to generate attention maps using Grad-CAM"
        parser.add_argument('--gradcam-layer', default='layer3', type=str,
                            help='Which layer take for GradCAM calculations (default=%(default)s)')
        parser.add_argument('--log-gradcam-samples', default=0, type=int,
                            help='How many examples of GradCAM to log (default=%(default)s)')

        parser.add_argument('--ta', default=False, action='store_true', required=False,
                            help='Teacher adaptation. If set, will update old model batch norm params '
                                 'during training the new task. (default=%(default)s)')

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

    def _take_samples_for_current_task(self, trn_loader):
        current_task_X, current_task_y = [], []
        for X, y in trn_loader:
            n = self.log_gradcam_samples - sum(_x.size(0) for _x in current_task_X)
            if n > 0:
                current_task_X.append(X[:n])
                current_task_y.append(y[:n])
        self._samples_to_log_X.append(torch.cat(current_task_X))
        self._samples_to_log_y.append(torch.cat(current_task_y))

    def _save_gradcam_examples(self, t, trn_loader):
        self._take_samples_for_current_task(trn_loader)
        output_orig_filename = os.path.join(self.logger.exp_path, '{t}_orig_post_train.png'.format(t=t))
        output_map_filename = os.path.join(self.logger.exp_path, '{t}_gcam_post_train.png'.format(t=t))
        print('Task {} - Saving {} samples to: {}'.format(t, self.log_gradcam_samples, output_orig_filename))
        save_image(torch.cat(self._samples_to_log_X), output_orig_filename, normalize=True, nrow=(t + 1))
        print(
            'Task {} - Saving {} samples with heatmaps to: {}'.format(t, self.log_gradcam_samples, output_map_filename))
        X_with_gradcam = []
        with GradCAM(self.model_old, self.gradcam_layer) as gradcam:
            for X in self._samples_to_log_X:
                img_with_heatmaps = []
                for x in X:
                    heatmap = gradcam(x.to(self.device))
                    img = gradcam.visualize_cam(heatmap, x)
                    img = img.view([1] + list(img.size()))
                    img_with_heatmaps.append(img)
                X_with_gradcam.append(torch.cat(img_with_heatmaps))
        save_image(torch.cat(X_with_gradcam), output_map_filename, nrow=(t + 1))

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        self.model_old = deepcopy(self.model)
        if self.log_gradcam_samples > 0:
            self._save_gradcam_examples(t, trn_loader)

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        if self.model_old is None:  # First task only
            return super().train_epoch(t, trn_loader)
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        # Do training with distillation losses
        with GradCAM(self.model_old, self.gradcam_layer) as gradcam_old:
            for images, targets in trn_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                # Forward old model
                attmap_old, outputs_old = gradcam_old(images, return_outputs=True, adapt_bn=self.ta)
                with GradCAM(self.model, self.gradcam_layer, retain_graph=True) as gradcam:
                    attmap = gradcam(images)  # this use eval() pass
                self.model.zero_grad()
                self.model.train()
                outputs = self.model(images)  # this use train() pass
                loss, loss_ce, loss_kd, loss_atn = self.criterion(t, outputs, targets, outputs_old, attmap, attmap_old,
                                                                  return_partial=True)
                if self.debug_loss:
                    self.logger.log_scalar(task=None, iter=None, name='loss_kd', group=f"debug_t{t}",
                                           value=float(loss_kd))
                    self.logger.log_scalar(task=None, iter=None, name='loss_atn', group=f"debug_t{t}",
                                           value=float(loss_atn))
                    self.logger.log_scalar(task=None, iter=None, name='loss_ce', group=f"debug_t{t}",
                                           value=float(loss_ce))
                    self.logger.log_scalar(task=None, iter=None, name='loss_total', group=f"debug_t{t}",
                                           value=float(loss))

                assert not torch.isnan(loss), 'Loss is NaN, stopping training...'

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
                self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        if t == 0:
            return super().eval(t, val_loader)
        total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
        self.model.eval()
        if self.model_old is not None:
            self.model_old.eval()
        with GradCAM(self.model, self.gradcam_layer) as gradcam, \
                GradCAM(self.model_old, self.gradcam_layer) as gradcam_old:
            for images, targets in val_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                # Forward old model
                attmap_old, outputs_old = gradcam_old(images, return_outputs=True)
                # Forward current model
                attmap, outputs = gradcam(images, return_outputs=True)
                loss = self.criterion(t, outputs, targets, outputs_old, attmap, attmap_old)
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def attention_distillation_loss(self, attention_map1, attention_map2):
        """Calculates the attention distillation loss"""
        attention_map1 = torch.nn.functional.normalize(attention_map1.view(attention_map1.size(0), -1),
                                                       p=2, dim=1, eps=1e-12, out=None)
        attention_map2 = torch.nn.functional.normalize(attention_map2.view(attention_map1.size(0), -1),
                                                       p=2, dim=1, eps=1e-12, out=None)
        return torch.norm(attention_map2 - attention_map1, p=1, dim=1).mean()

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

    def criterion(self, t, outputs, targets, outputs_old=None, attmap=None, attmap_old=None, return_partial=False):
        """Returns the loss value"""
        if t > 0 and outputs_old is not None:
            # Knowledge distillation loss for all previous tasks
            loss_kd = self.beta * self.cross_entropy(torch.cat(outputs[:t], dim=1),
                                                     torch.cat(outputs_old[:t], dim=1), exp=1.0 / 2.0)
        else:
            loss_kd = 0

        # Attention Distillation loss
        if attmap is not None and attmap_old is not None:
            loss_atn = self.gamma * self.attention_distillation_loss(attmap, attmap_old)
        else:
            loss_atn = 0

        # Current cross-entropy loss -- with exemplars use all heads
        if len(self.exemplars_dataset) > 0:
            loss_ce = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        else:
            loss_ce = torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])

        loss = loss_kd + loss_ce + loss_atn
        if not return_partial:
            return loss
        else:
            return loss, loss_ce, loss_kd, loss_atn


class GradCAM:
    """
    Calculates GradCAM for the given model.

    Ref:
     - article:
        Grad-CAM: Visual Explanations from Deep Networks 
        via Gradient-based Localization, Selvaraju et al, ICCV, 2017
        https://arxiv.org/pdf/1610.02391.pdf

     - useful code repo:
        https://github.com/vickyliin/gradcam_plus_plus-pytorch
    """

    def __init__(self, model: LLL_Net, layer_name, retain_graph=False):
        self.model = model
        assert hasattr(
            self.model.model, layer_name), 'Model {} has got attribute layer: {}'.format(type(self.model), layer_name)
        self.model_layer = getattr(self.model.model, layer_name)
        # grads & activations
        self.activations = None
        self.gradients = None
        self.retain_graph = retain_graph

    def __enter__(self):
        # register hooks to collect activations and gradients
        def forward_hook(module, input, output):
            if self.retain_graph:
                self.activations = output
            else:
                self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # hook to final layer
        self.fhandle = self.model_layer.register_forward_hook(forward_hook)
        self.bhandle = self.model_layer.register_backward_hook(backward_hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.fhandle.remove()
        self.bhandle.remove()

    def __call__(self, input, class_indices=None, return_outputs=False, adapt_bn=False):
        # pass input & backpropagate for selected class
        if input.dim() == 3:
            input = input.view([1] + list(input.size()))
        if adapt_bn:
            self.model.train()
        else:
            self.model.eval()
        model_output = self.model(input)
        logits = torch.cat(model_output, dim=1)
        if class_indices is None:
            class_indices = logits.argmax(dim=1)
        score = logits[:, class_indices].squeeze()
        self.model.zero_grad()
        score.mean().backward(retain_graph=self.retain_graph)
        model_output = [o.detach() for o in model_output]

        with torch.no_grad() if not self.retain_graph else contextlib.suppress():
            weights = F.adaptive_avg_pool2d(self.gradients, 1)
            att_map = (weights * self.activations).sum(dim=1, keepdim=True)
            att_map = F.relu(att_map)
            del self.activations
            del self.gradients
            return (att_map, model_output) if return_outputs else att_map

    @staticmethod
    def visualize_cam(mask, img, alpha=1.0, cmap='jet'):
        try:
            import cv2
        except ImportError:
            print('Please install opencv library for visualization.\n')
            print('For conda environment use command:')
            print('conda install -c menpo opencv')
        with torch.no_grad():
            # upsample and normalize
            c, h, w = img.size()
            att_map = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
            map_min, map_max = att_map.min(), att_map.max()
            att_map = (att_map - map_min).div(map_max - map_min)
            # color heatmap
            heatmap = (255 * att_map.squeeze()).type(torch.uint8).cpu().numpy()
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heatmap = torch.from_numpy(heatmap).permute(
                2, 0, 1).float().div(255)
            b, g, r = heatmap.split(1)
            heatmap = torch.cat([r, g, b]) * alpha
            # combine to single image
            result = heatmap + img.cpu()
            result = result.div(result.max()).squeeze()
            return result
