import os
import time

import torch
import argparse
import importlib
import numpy as np
import torch.multiprocessing
from functools import reduce

from dotenv import load_dotenv, find_dotenv

from metrics import cm

load_dotenv(find_dotenv())

import utils
import approach
from loggers.exp_logger import MultiLogger
from datasets.dataset_config import dataset_config
from last_layer_analysis import last_layer_analysis
from networks import tvmodels, allmodels, set_tvmodel_head_var
from datasets.data_loader import get_loaders

torch.multiprocessing.set_sharing_strategy('file_system')


def main(argv=None):
    tstart = time.time()
    # Arguments
    parser = argparse.ArgumentParser(description='FACIL - Framework for Analysis of Class Incremental Learning')

    # miscellaneous args
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU (default=%(default)s)')
    parser.add_argument('--results-path', type=str, default='../results',
                        help='Results path (default=%(default)s)')
    parser.add_argument('--exp-name', default=None, type=str,
                        help='Experiment name (default=%(default)s)')
    parser.add_argument('--tags', type=str, nargs='+', default=None,
                        help='Tags for wandb run (default=%(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default=%(default)s)')
    parser.add_argument('--log', default=['disk'], type=str, choices=['disk', 'tensorboard', 'wandb'],
                        help='Loggers used (disk, tensorboard, wandb) (default=%(default)s)', nargs='*',
                        metavar="LOGGER")
    parser.add_argument('--save-models', action='store_true',
                        help='Save trained models (default=%(default)s)')
    parser.add_argument('--cache-first-task-model', action='store_true',
                        help='If set, will try to cache first task model to save time (default=%(default)s)')
    parser.add_argument('--last-layer-analysis', action='store_true',
                        help='Plot last layer analysis (default=%(default)s)')
    parser.add_argument('--no-cudnn-deterministic', action='store_true',
                        help='Disable CUDNN deterministic (default=%(default)s)')
    parser.add_argument('--cm', default=False, action='store_true', required=False,
                        help='If set, will compute and log task confusion matrix. (default=%(default)s)')

    # dataset args
    parser.add_argument('--datasets', default=['cifar100'], type=str, choices=list(dataset_config.keys()),
                        help='Dataset or datasets used (default=%(default)s)', nargs='+', metavar="DATASET")
    parser.add_argument('--num-workers', default=4, type=int, required=False,
                        help='Number of subprocesses to use for dataloader (default=%(default)s)')
    parser.add_argument('--batch-size', default=64, type=int, required=False,
                        help='Number of samples per batch to load (default=%(default)s)')
    parser.add_argument('--num-tasks', default=4, type=int, required=False,
                        help='Number of tasks per dataset (default=%(default)s)')
    parser.add_argument('--nc-first-task', default=None, type=int, required=False,
                        help='Number of classes of the first task (default=%(default)s)')
    parser.add_argument('--ne-first-task', default=None, type=int, required=False,
                        help='Number of epochs for the first task (default=%(default)s)')
    parser.add_argument('--nc-per-task', type=int, nargs='+', default=None,
                        help='Number of classes per each task, should be provided as a '
                             'space-separated values, overrides other task split settings '
                             '(default=%(default)s)')
    parser.add_argument('--use-valid-only', action='store_true',
                        help='Use validation split instead of test (default=%(default)s)')
    parser.add_argument('--use-test-as-val', action='store_true',
                        help='Use validation test split as a validation set '
                             'and do not split original train data into train and val sets (default=%(default)s)')
    parser.add_argument('--stop-at-task', default=0, type=int, required=False,
                        help='Stop training after specified task (default=%(default)s)')
    parser.add_argument('--max-classes-per-dataset', default=None, type=int, required=False,
                        help='If training on sequence of multiple datasets, number of classes sampled from the dataset. '
                             'If the dataset appears multiple times, different classes will be sampled. (default=%(default)s)')
    parser.add_argument('--max-examples-per-class-trn', default=None, type=int, required=False,
                        help='Limit for training samples per class for debugging (default=%(default)s)')
    parser.add_argument('--max-examples-per-class-val', default=None, type=int, required=False,
                        help='Limit for val samples per class for debugging (default=%(default)s)')
    parser.add_argument('--max-examples-per-class-tst', default=None, type=int, required=False,
                        help='Limit for test samples per class for debugging (default=%(default)s)')
    # model args
    parser.add_argument('--network', default='resnet32', type=str, choices=allmodels,
                        help='Network architecture used (default=%(default)s)', metavar="NETWORK")
    parser.add_argument('--keep-existing-head', action='store_true',
                        help='Disable removing classifier last layer (default=%(default)s)')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained backbone (default=%(default)s)')
    # training args
    parser.add_argument('--approach', default='finetuning', type=str, choices=approach.__all__,
                        help='Learning approach used (default=%(default)s)', metavar="APPROACH")
    parser.add_argument('--nepochs', default=200, type=int, required=False,
                        help='Number of epochs per training session (default=%(default)s)')
    parser.add_argument('--lr', default=0.1, type=float, required=False,
                        help='Starting learning rate (default=%(default)s)')
    parser.add_argument('--scheduler-milestones', default=[60, 120, 160], nargs='+', type=int, required=False,
                        help='Milestones for learning rate scheduler, overrides lr-patience scheme, '
                             'if set to None scheduler will not be used (default=%(default)s)')
    parser.add_argument('--lr-min', default=1e-4, type=float, required=False,
                        help='Minimum learning rate (default=%(default)s)')
    parser.add_argument('--lr-factor', default=3, type=float, required=False,
                        help='Learning rate decreasing factor (default=%(default)s)')
    parser.add_argument('--lr-patience', default=5, type=int, required=False,
                        help='Maximum patience to wait before decreasing learning rate (default=%(default)s)')
    parser.add_argument('--clipping', default=100., type=float, required=False,
                        help='Clip gradient norm (default=%(default)s)')
    parser.add_argument('--momentum', default=0.0, type=float, required=False,
                        help='Momentum factor (default=%(default)s)')
    parser.add_argument('--weight-decay', default=0.0, type=float, required=False,
                        help='Weight decay (L2 penalty) (default=%(default)s)')
    parser.add_argument('--wu-nepochs', default=0, type=int, required=False,
                        help='Number of warm-up epochs (default=%(default)s)')
    parser.add_argument('--wu-lr', default=0.1, type=float, required=False,
                        help='Warm-up learning rate (default=%(default)s)')
    parser.add_argument('--wu-fix-bn', action='store_true',
                        help='Fix batch norm stats during warmup. (default=%(default)s)')
    parser.add_argument('--wu-scheduler', default='constant', type=str, required=False,
                        help='Warm-up learning rate scheduler (default=%(default)s)')
    parser.add_argument('--wu-patience', default=None, type=int, required=False,
                        help='Patience for warmup, None equals to no early stopping (default=%(default)s)')
    parser.add_argument('--wu-wd', default=0.001, type=float, required=False,
                        help='Weight decay value for warmup (default=%(default)s)')
    parser.add_argument('--multi-softmax', action='store_true',
                        help='Apply separate softmax for each task (default=%(default)s)')
    parser.add_argument('--fix-bn', action='store_true',
                        help='Fix batch normalization after first task (default=%(default)s)')
    parser.add_argument('--eval-on-train', action='store_true',
                        help='Show train loss and accuracy (default=%(default)s)')
    parser.add_argument('--extra-aug', default='', type=str,
                        choices=['', 'simclr', 'simclr_cifar', 'colorjitter', 'brightness', 'fetril'],
                        help='Additional data augmentations (default=%(default)s)')

    # gridsearch args
    parser.add_argument('--gridsearch-tasks', default=0, type=int,
                        help='Number of tasks to apply GridSearch (default=%(default)s)')

    # Args -- Incremental Learning Framework
    args, extra_args = parser.parse_known_args(argv)
    args.results_path = os.path.expanduser(args.results_path)
    base_kwargs = dict(nepochs=args.nepochs, lr=args.lr, lr_min=args.lr_min, lr_factor=args.lr_factor,
                       lr_patience=args.lr_patience, clipgrad=args.clipping, momentum=args.momentum,
                       wd=args.weight_decay, multi_softmax=args.multi_softmax, wu_nepochs=args.wu_nepochs,
                       wu_lr=args.wu_lr, wu_fix_bn=args.wu_fix_bn, wu_scheduler=args.wu_scheduler,
                       wu_patience=args.wu_patience, wu_wd=args.wu_wd, fix_bn=args.fix_bn,
                       eval_on_train=args.eval_on_train, select_best_model_by_val_loss=True,
                       scheduler_milestones=args.scheduler_milestones)

    if args.no_cudnn_deterministic:
        print('WARNING: CUDNN Deterministic will be disabled.')
        utils.cudnn_deterministic = False

    utils.seed_everything(seed=args.seed)
    print('=' * 108)
    print('Arguments =')
    for arg in np.sort(list(vars(args).keys())):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 108)

    # Args -- CUDA
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = 'cuda'
    else:
        print('WARNING: [CUDA unavailable] Using CPU instead!')
        device = 'cpu'
        raise EnvironmentError('No GPU available')

    # In case the dataset is too large
    torch.multiprocessing.set_sharing_strategy('file_system')
    # Multiple gpus
    # if torch.cuda.device_count() > 1:
    #     self.C = torch.nn.DataParallel(C)
    #     self.C.to(self.device)
    ####################################################################################################################

    # Args -- Network
    from networks.network import LLL_Net
    if args.network in tvmodels:  # torchvision models
        tvnet = getattr(importlib.import_module(name='torchvision.models'), args.network)
        if args.network == 'googlenet':
            init_model = tvnet(pretrained=args.pretrained, aux_logits=False)
        else:
            init_model = tvnet(pretrained=args.pretrained)
        set_tvmodel_head_var(init_model)
    else:  # other models declared in networks package's init
        net = getattr(importlib.import_module(name='networks'), args.network)
        # WARNING: fixed to pretrained False for other model (non-torchvision)
        init_model = net(pretrained=False)

    # Args -- Continual Learning Approach
    from approach.incremental_learning import Inc_Learning_Appr
    Appr = getattr(importlib.import_module(name='approach.' + args.approach), 'Appr')
    assert issubclass(Appr, Inc_Learning_Appr)
    appr_args, extra_args = Appr.extra_parser(extra_args)
    print('Approach arguments =')
    for arg in np.sort(list(vars(appr_args).keys())):
        print('\t' + arg + ':', getattr(appr_args, arg))
    print('=' * 108)

    # Args -- Exemplars Management
    from datasets.exemplars_dataset import ExemplarsDataset
    Appr_ExemplarsDataset = Appr.exemplars_dataset_class()
    if Appr_ExemplarsDataset:
        assert issubclass(Appr_ExemplarsDataset, ExemplarsDataset)
        appr_exemplars_dataset_args, extra_args = Appr_ExemplarsDataset.extra_parser(extra_args)
        print('Exemplars dataset arguments =')
        for arg in np.sort(list(vars(appr_exemplars_dataset_args).keys())):
            print('\t' + arg + ':', getattr(appr_exemplars_dataset_args, arg))
        print('=' * 108)
    else:
        appr_exemplars_dataset_args = argparse.Namespace()

    # Args -- GridSearch
    if args.gridsearch_tasks > 0:
        from gridsearch import GridSearch
        gs_args, extra_args = GridSearch.extra_parser(extra_args)
        Appr_finetuning = getattr(importlib.import_module(name='approach.finetuning'), 'Appr')
        assert issubclass(Appr_finetuning, Inc_Learning_Appr)
        GridSearch_ExemplarsDataset = Appr.exemplars_dataset_class()
        print('GridSearch arguments =')
        for arg in np.sort(list(vars(gs_args).keys())):
            print('\t' + arg + ':', getattr(gs_args, arg))
        print('=' * 108)

    assert len(extra_args) == 0, "Unused args: {}".format(' '.join(extra_args))
    ####################################################################################################################

    # Log all arguments
    full_exp_name = reduce((lambda x, y: x[0] + y[0]), args.datasets) if len(args.datasets) > 0 else args.datasets[0]
    full_exp_name += '_' + args.approach
    if args.exp_name is not None:
        full_exp_name += '_' + args.exp_name
    logger = MultiLogger(args.results_path, full_exp_name, loggers=args.log, save_models=args.save_models,
                         tags=args.tags)
    logger.log_args(argparse.Namespace(**args.__dict__, **appr_args.__dict__, **appr_exemplars_dataset_args.__dict__))

    # Loaders
    utils.seed_everything(seed=args.seed)
    trn_loader, val_loader, tst_loader, taskcla = get_loaders(args.datasets, args.num_tasks, args.nc_first_task,
                                                              args.nc_per_task,
                                                              args.batch_size, num_workers=args.num_workers,
                                                              pin_memory=True,
                                                              max_classes_per_dataset=args.max_classes_per_dataset,
                                                              max_examples_per_class_trn=args.max_examples_per_class_trn,
                                                              max_examples_per_class_val=args.max_examples_per_class_val,
                                                              max_examples_per_class_tst=args.max_examples_per_class_tst,
                                                              extra_aug=args.extra_aug,
                                                              validation=0.0 if args.use_test_as_val else 0.1)
    # Apply arguments for loaders
    if args.use_valid_only:
        tst_loader = val_loader
    if args.use_test_as_val:
        val_loader = tst_loader
        base_kwargs["select_best_model_by_val_loss"] = False
    max_task = len(taskcla) if args.stop_at_task == 0 else args.stop_at_task

    # Network and Approach instances
    utils.seed_everything(seed=args.seed)
    net = LLL_Net(init_model, remove_existing_head=not args.keep_existing_head)
    utils.seed_everything(seed=args.seed)
    # taking transformations and class indices from first train dataset
    first_train_ds = trn_loader[0].dataset
    transform, class_indices = first_train_ds.transform, first_train_ds.class_indices
    appr_kwargs = {**base_kwargs, **dict(logger=logger, **appr_args.__dict__)}
    if Appr_ExemplarsDataset:
        appr_kwargs['exemplars_dataset'] = Appr_ExemplarsDataset(transform, class_indices,
                                                                 **appr_exemplars_dataset_args.__dict__)
    utils.seed_everything(seed=args.seed)
    appr = Appr(net, device, **appr_kwargs)

    ### Add test loader for oracle evaluation during teacher finetuning
    appr.tst_loader = tst_loader

    # GridSearch
    if args.gridsearch_tasks > 0:
        ft_kwargs = {**base_kwargs, **dict(logger=logger,
                                           exemplars_dataset=GridSearch_ExemplarsDataset(transform, class_indices))}
        appr_ft = Appr_finetuning(net, device, **ft_kwargs)
        gridsearch = GridSearch(appr_ft, args.seed, gs_args.gridsearch_config, gs_args.gridsearch_acc_drop_thr,
                                gs_args.gridsearch_hparam_decay, gs_args.gridsearch_max_num_searches)

    # Loop tasks
    print(taskcla)
    acc_taw = np.zeros((max_task, max_task))
    acc_tag = np.zeros((max_task, max_task))
    forg_taw = np.zeros((max_task, max_task))
    forg_tag = np.zeros((max_task, max_task))
    test_loss = np.zeros((max_task, max_task))

    for t, (_, ncla) in enumerate(taskcla):

        # Early stop tasks if flag
        if t >= max_task:
            continue

        print('*' * 108)
        print('Task {:2d}'.format(t))
        print('*' * 108)

        # Add head for current task
        net.add_head(taskcla[t][1])
        net.to(device)

        # GridSearch
        if t < args.gridsearch_tasks:

            # Search for best finetuning learning rate -- Maximal Plasticity Search
            print('LR GridSearch')
            best_ft_acc, best_ft_lr = gridsearch.search_lr(appr.model, t, trn_loader[t], val_loader[t])
            # Apply to approach
            appr.lr = best_ft_lr
            gen_params = gridsearch.gs_config.get_params('general')
            for k, v in gen_params.items():
                if not isinstance(v, list):
                    setattr(appr, k, v)

            # Search for best forgetting/intransigence tradeoff -- Stability Decay
            print('Trade-off GridSearch')
            best_tradeoff, tradeoff_name = gridsearch.search_tradeoff(args.approach, appr,
                                                                      t, trn_loader[t], val_loader[t], best_ft_acc)
            # Apply to approach
            if tradeoff_name is not None:
                setattr(appr, tradeoff_name, best_tradeoff)

            print('-' * 108)

        if t == 0 and args.ne_first_task is not None:
            appr.nepochs = args.ne_first_task

        # Train
        if t == 0 and args.cache_first_task_model:
            exp_tag = "_".join([d for d in args.datasets]) + "_t" + str(args.num_tasks) + "s" + str(args.nc_first_task)
            if args.use_test_as_val:
                exp_tag += "_test_as_val"
            model_tag = args.network
            if args.pretrained:
                model_tag += "_pretrained"
            model_tag += "_ep" + str(args.nepochs) + "_bs" + str(args.batch_size) + "_lr" + str(args.lr) \
                         + "_wd" + str(args.weight_decay) + "_m" + str(args.momentum) + "_clip" \
                         + str(args.clipping) + "_sched" + "_".join([str(m) for m in args.scheduler_milestones])
            model_ckpt_dir = os.path.join("checkpoints", exp_tag, model_tag)
            model_ckpt_path = os.path.join(model_ckpt_dir, "model_seed_" + str(args.seed) + ".ckpt")
            if os.path.exists(model_ckpt_path):
                print("Loading model from checkpoint: " + model_ckpt_path)
                net.load_state_dict(torch.load(model_ckpt_path))
                appr.post_train_process(t, trn_loader[t])
                appr.exemplars_dataset.collect_exemplars(appr.model, trn_loader[t], val_loader[t].dataset.transform)
            else:
                appr.train(t, trn_loader[t], val_loader[t])
                print("Saving first task checkpoint to: " + model_ckpt_path)
                os.makedirs(model_ckpt_dir, exist_ok=True)
                torch.save(net.state_dict(), model_ckpt_path)
        else:
            appr.train(t, trn_loader[t], val_loader[t])
        print('-' * 108)

        if t == 0 and args.ne_first_task is not None:
            appr.nepochs = args.nepochs

        # Test
        for u in range(t + 1):
            test_loss[t, u], acc_taw[t, u], acc_tag[t, u] = appr.eval(u, tst_loader[u])
            if u < t:
                forg_taw[t, u] = acc_taw[:t, u].max(0) - acc_taw[t, u]
                forg_tag[t, u] = acc_tag[:t, u].max(0) - acc_tag[t, u]
            print('>>> Test on task {:2d} : loss={:.3f} | TAw acc={:5.1f}%, forg={:5.1f}%'
                  '| TAg acc={:5.1f}%, forg={:5.1f}% <<<'.format(u, test_loss[t, u],
                                                                 100 * acc_taw[t, u], 100 * forg_taw[t, u],
                                                                 100 * acc_tag[t, u], 100 * forg_tag[t, u]))

        for u in range(max_task):
            logger.log_scalar(task=u, iter=t, name='loss', group='test', value=test_loss[t, u])
            logger.log_scalar(task=u, iter=t, name='acc_taw', group='test', value=100 * acc_taw[t, u])
            logger.log_scalar(task=u, iter=t, name='acc_tag', group='test', value=100 * acc_tag[t, u])
            logger.log_scalar(task=u, iter=t, name='forg_taw', group='test', value=100 * forg_taw[t, u])
            logger.log_scalar(task=u, iter=t, name='forg_tag', group='test', value=100 * forg_tag[t, u])

        # Save
        print('Save at ' + os.path.join(args.results_path, full_exp_name))
        logger.log_result(acc_taw, name="acc_taw", step=t, skip_wandb=True)
        logger.log_result(acc_tag, name="acc_tag", step=t, skip_wandb=True)
        logger.log_result(forg_taw, name="forg_taw", step=t, skip_wandb=True)
        logger.log_result(forg_tag, name="forg_tag", step=t, skip_wandb=True)
        if args.cm:
            logger.log_result(cm(appr.model, tst_loader[:t + 1], args.num_tasks, appr.device), name="cm", step=t,
                              title="Task confusion matrix", xlabel="Predicted task", ylabel="True task", annot=False,
                              cmap='Blues', cbar=True, vmin=0, vmax=1)

        logger.save_model(net.state_dict(), task=t)

        avg_accs_taw = acc_taw.sum(1) / np.tril(np.ones(acc_taw.shape[0])).sum(1)
        logger.log_result(avg_accs_taw, name="avg_accs_taw", step=t, skip_wandb=True)
        logger.log_scalar(task=None, iter=t, name='avg_acc_taw', group='test',
                          value=100 * avg_accs_taw[t])
        avg_accs_tag = acc_tag.sum(1) / np.tril(np.ones(acc_tag.shape[0])).sum(1)
        logger.log_result(avg_accs_tag, name="avg_accs_tag", step=t, skip_wandb=True)
        logger.log_scalar(task=None, iter=t, name='avg_acc_tag', group='test',
                          value=100 * avg_accs_tag[t])
        aux = np.tril(np.repeat([[tdata[1] for tdata in taskcla[:max_task]]], max_task, axis=0))
        wavg_accs_taw = (acc_taw * aux).sum(1) / aux.sum(1)
        logger.log_result(wavg_accs_taw, name="wavg_accs_taw", step=t, skip_wandb=True)
        logger.log_scalar(task=None, iter=t, name='wavg_acc_taw', group='test',
                          value=100 * wavg_accs_taw[t])
        wavg_accs_tag = (acc_tag * aux).sum(1) / aux.sum(1)
        logger.log_result(wavg_accs_tag, name="wavg_accs_tag", step=t, skip_wandb=True)
        logger.log_scalar(task=None, iter=t, name='wavg_acc_tag', group='test',
                          value=100 * wavg_accs_tag[t])

        # Last layer analysis
        if args.last_layer_analysis:
            weights, biases = last_layer_analysis(net.heads, t, taskcla, y_lim=True)
            logger.log_figure(name='weights', iter=t, figure=weights)
            logger.log_figure(name='bias', iter=t, figure=biases)

            # Output sorted weights and biases
            weights, biases = last_layer_analysis(net.heads, t, taskcla, y_lim=True, sort_weights=True)
            logger.log_figure(name='weights', iter=t, figure=weights)
            logger.log_figure(name='bias', iter=t, figure=biases)

    avg_accs_taw = acc_taw.sum(1) / np.tril(np.ones(acc_taw.shape[0])).sum(1)
    logger.log_result(avg_accs_taw, name="avg_accs_taw", step=0, skip_wandb=False)
    avg_accs_tag = acc_tag.sum(1) / np.tril(np.ones(acc_tag.shape[0])).sum(1)
    logger.log_result(avg_accs_tag, name="avg_accs_tag", step=0, skip_wandb=False)
    aux = np.tril(np.repeat([[tdata[1] for tdata in taskcla[:max_task]]], max_task, axis=0))
    wavg_accs_taw = (acc_taw * aux).sum(1) / aux.sum(1)
    logger.log_result(wavg_accs_taw, name="wavg_accs_taw", step=0, skip_wandb=False)
    wavg_accs_tag = (acc_tag * aux).sum(1) / aux.sum(1)
    logger.log_result(wavg_accs_tag, name="wavg_accs_tag", step=0, skip_wandb=False)

    # Print Summary
    utils.print_summary(acc_taw, acc_tag, forg_taw, forg_tag)
    print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))
    print('Done!')

    return acc_taw, acc_tag, forg_taw, forg_tag, logger.exp_path
    ####################################################################################################################


if __name__ == '__main__':
    main()
