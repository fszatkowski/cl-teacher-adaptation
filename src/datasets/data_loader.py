import glob
import os
from itertools import compress
from collections import defaultdict

from collections import defaultdict

import numpy as np
from torch.utils import data
import torchvision.transforms as transforms
from torchvision.datasets import MNIST as TorchVisionMNIST
from torchvision.datasets import CIFAR100 as TorchVisionCIFAR100
from torchvision.datasets import SVHN as TorchVisionSVHN

from . import base_dataset as basedat
from . import memory_dataset as memd
from .dataset_config import dataset_config
from .autoaugment import CIFAR10Policy, ImageNetPolicy
from .ops import Cutout


def get_loaders(datasets, num_tasks, nc_first_task, nc_per_task, batch_size, num_workers,
                pin_memory, validation=.1, max_classes_per_dataset=None, max_examples_per_class_trn=None,
                max_examples_per_class_val=None, max_examples_per_class_tst=None,
                extra_aug=""):
    """Apply transformations to Datasets and create the DataLoaders for each task"""

    if nc_per_task is not None:
        num_tasks = len(nc_per_task)

    trn_load, val_load, tst_load = [], [], []
    taskcla = []
    dataset_offset = 0
    dataset_to_cls_idx = defaultdict(lambda : 0)
    ds_to_cls_order = {}

    for idx_dataset, cur_dataset in enumerate(datasets, 0):
        # get configuration for current dataset
        dc = dataset_config[cur_dataset]

        # transformations
        trn_transform, tst_transform = get_transforms(resize=dc['resize'],
                                                      test_resize=dc['test_resize'],
                                                      pad=dc['pad'],
                                                      crop=dc['crop'],
                                                      flip=dc['flip'],
                                                      normalize=dc['normalize'],
                                                      extend_channel=dc['extend_channel'],
                                                      extra_aug=extra_aug,
                                                      ds_name=cur_dataset)

        if len(datasets) > 1:
            if cur_dataset in ds_to_cls_order:
                class_order = ds_to_cls_order[cur_dataset]
            else:
                class_order = None
        else:
            class_order = dc['class_order']

        # datasets
        trn_dset, val_dset, tst_dset, curtaskcla = get_datasets(cur_dataset, dc['path'],
                                                                num_tasks, nc_first_task,
                                                                nc_per_task,
                                                                validation=validation,
                                                                trn_transform=trn_transform,
                                                                tst_transform=tst_transform,
                                                                class_order=class_order,
                                                                max_examples_per_class_trn=max_examples_per_class_trn,
                                                                max_examples_per_class_val=max_examples_per_class_val,
                                                                max_examples_per_class_tst=max_examples_per_class_tst)

        if max_classes_per_dataset is not None:
            if num_tasks > 1:
                raise ValueError("'max_classes_per_dataset' only works with 'num_tasks' = 1.")

            if cur_dataset not in ds_to_cls_order:
                ds_to_cls_order[cur_dataset] = trn_dset[0].class_indices

            trn_dset = _sample_classes(trn_dset, dataset_to_cls_idx[cur_dataset], dataset_to_cls_idx[cur_dataset]+max_classes_per_dataset)
            val_dset = _sample_classes(val_dset, dataset_to_cls_idx[cur_dataset], dataset_to_cls_idx[cur_dataset]+max_classes_per_dataset)
            tst_dset = _sample_classes(tst_dset, dataset_to_cls_idx[cur_dataset], dataset_to_cls_idx[cur_dataset]+max_classes_per_dataset)

            dataset_to_cls_idx[cur_dataset] += max_classes_per_dataset
            curtaskcla = [(0, max_classes_per_dataset)]


        # apply offsets in case of multiple datasets
        if idx_dataset > 0:
            for tt in range(num_tasks):
                trn_dset[tt].labels = [
                    elem + dataset_offset for elem in trn_dset[tt].labels]
                val_dset[tt].labels = [
                    elem + dataset_offset for elem in val_dset[tt].labels]
                tst_dset[tt].labels = [
                    elem + dataset_offset for elem in tst_dset[tt].labels]
        dataset_offset = dataset_offset + sum([tc[1] for tc in curtaskcla])

        # reassign class idx for multiple dataset case
        curtaskcla = [(tc[0] + idx_dataset * num_tasks, tc[1])
                      for tc in curtaskcla]

        # extend final taskcla list
        taskcla.extend(curtaskcla)

        # loaders
        for tt in range(num_tasks):
            trn_load.append(data.DataLoader(trn_dset[tt], batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                            pin_memory=pin_memory))
            val_load.append(data.DataLoader(val_dset[tt], batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                            pin_memory=pin_memory))
            tst_load.append(data.DataLoader(tst_dset[tt], batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                            pin_memory=pin_memory))
    return trn_load, val_load, tst_load, taskcla


def get_datasets(dataset, path, num_tasks, nc_first_task, nc_per_task, validation, trn_transform,
                 tst_transform, class_order=None, max_examples_per_class_trn=None,
                 max_examples_per_class_val=None, max_examples_per_class_tst=None):
    """Extract datasets and create Dataset class"""

    trn_dset, val_dset, tst_dset = [], [], []

    if 'mnist' in dataset:
        tvmnist_trn = TorchVisionMNIST(path, train=True, download=True)
        tvmnist_tst = TorchVisionMNIST(path, train=False, download=True)
        trn_data = {'x': tvmnist_trn.data.numpy(
        ), 'y': tvmnist_trn.targets.tolist()}
        tst_data = {'x': tvmnist_tst.data.numpy(
        ), 'y': tvmnist_tst.targets.tolist()}
        # compute splits
        all_data, taskcla, class_indices = memd.get_data(trn_data, tst_data, validation=validation,
                                                         num_tasks=num_tasks, nc_first_task=nc_first_task,
                                                         nc_per_task=nc_per_task,
                                                         shuffle_classes=class_order is None, class_order=class_order)
        # set dataset type
        Dataset = memd.MemoryDataset

    elif 'cifar100' in dataset:
        tvcifar_trn = TorchVisionCIFAR100(path, train=True, download=True)
        tvcifar_tst = TorchVisionCIFAR100(path, train=False, download=True)
        trn_data = {'x': tvcifar_trn.data, 'y': tvcifar_trn.targets}
        tst_data = {'x': tvcifar_tst.data, 'y': tvcifar_tst.targets}
        # compute splits
        all_data, taskcla, class_indices = memd.get_data(trn_data, tst_data, validation=validation,
                                                         num_tasks=num_tasks, nc_first_task=nc_first_task,
                                                         nc_per_task=nc_per_task,
                                                         shuffle_classes=class_order is None, class_order=class_order)
        # set dataset type
        Dataset = memd.MemoryDataset

    elif dataset == 'svhn':
        tvsvhn_trn = TorchVisionSVHN(path, split='train', download=True)
        tvsvhn_tst = TorchVisionSVHN(path, split='test', download=True)
        trn_data = {'x': tvsvhn_trn.data.transpose(
            0, 2, 3, 1), 'y': tvsvhn_trn.labels}
        tst_data = {'x': tvsvhn_tst.data.transpose(
            0, 2, 3, 1), 'y': tvsvhn_tst.labels}
        # Notice that SVHN in Torchvision has an extra training set in case needed
        # tvsvhn_xtr = TorchVisionSVHN(path, split='extra', download=True)
        # xtr_data = {'x': tvsvhn_xtr.data.transpose(0, 2, 3, 1), 'y': tvsvhn_xtr.labels}

        # compute splits
        all_data, taskcla, class_indices = memd.get_data(trn_data, tst_data, validation=validation,
                                                         num_tasks=num_tasks, nc_first_task=nc_first_task,
                                                         nc_per_task=nc_per_task,
                                                         shuffle_classes=class_order is None, class_order=class_order)
        # set dataset type
        Dataset = memd.MemoryDataset

    elif 'imagenet_32' in dataset:
        import pickle
        # load data
        x_trn, y_trn = [], []
        for i in range(1, 11):
            with open(os.path.join(path, 'train_data_batch_{}'.format(i)), 'rb') as f:
                d = pickle.load(f)
            x_trn.append(d['data'])
            y_trn.append(np.array(d['labels']) - 1)  # labels from 0 to 999
        with open(os.path.join(path, 'val_data'), 'rb') as f:
            d = pickle.load(f)
        x_trn.append(d['data'])
        y_tst = np.array(d['labels']) - 1  # labels from 0 to 999
        # reshape data
        for i, d in enumerate(x_trn, 0):
            x_trn[i] = d.reshape(d.shape[0], 3, 32, 32).transpose(0, 2, 3, 1)
        x_tst = x_trn[-1]
        x_trn = np.vstack(x_trn[:-1])
        y_trn = np.concatenate(y_trn)
        trn_data = {'x': x_trn, 'y': y_trn}
        tst_data = {'x': x_tst, 'y': y_tst}
        # compute splits
        all_data, taskcla, class_indices = memd.get_data(trn_data, tst_data, validation=validation,
                                                         num_tasks=num_tasks, nc_first_task=nc_first_task,
                                                         nc_per_task=nc_per_task,
                                                         shuffle_classes=class_order is None, class_order=class_order)
        # set dataset type
        Dataset = memd.MemoryDataset

    else:
        if dataset == 'domainnet':
            _ensure_domainnet_prepared(path, classes_per_domain=nc_first_task, num_tasks=num_tasks)
            all_data, taskcla, class_indices = basedat.get_data(path, num_tasks=num_tasks, nc_first_task=nc_first_task,
                                                                    nc_per_task=None, validation=validation, shuffle_classes=False)
            Dataset = basedat.BaseDataset
            
        else:
            if dataset == 'imagenet_subset_kaggle':
                _ensure_imagenet_subset_prepared(path)
            elif dataset == 'tiny_scaled_imnet':
                _ensure_tinyimagenet_prepared(path)
        
            # read data paths and compute splits -- path needs to have a train.txt and a test.txt with image-label pairs
            all_data, taskcla, class_indices = basedat.get_data(path, num_tasks=num_tasks, nc_first_task=nc_first_task,
                                                                nc_per_task=nc_per_task,
                                                                validation=validation, shuffle_classes=class_order is None,
                                                                class_order=class_order)
            # set dataset type
            Dataset = basedat.BaseDataset

        

    # get datasets, apply correct label offsets for each task
    offset = 0
    for task in range(num_tasks):
        all_data[task]['trn']['y'] = [
            label + offset for label in all_data[task]['trn']['y']]
        all_data[task]['val']['y'] = [
            label + offset for label in all_data[task]['val']['y']]
        all_data[task]['tst']['y'] = [
            label + offset for label in all_data[task]['tst']['y']]
        trn_dset.append(
            Dataset(all_data[task]['trn'], trn_transform, class_indices))
        val_dset.append(
            Dataset(all_data[task]['val'], tst_transform, class_indices))
        tst_dset.append(
            Dataset(all_data[task]['tst'], tst_transform, class_indices))
        offset += taskcla[task][1]

    trn_dset = _sample_dataset(trn_dset, max_examples_per_class_trn)
    val_dset = _sample_dataset(val_dset, max_examples_per_class_val)
    tst_dset = _sample_dataset(tst_dset, max_examples_per_class_tst)

    return trn_dset, val_dset, tst_dset, taskcla


def _sample_dataset(dataset_list, max_examples_per_class):
    if max_examples_per_class is not None:
        for dataset in dataset_list:
            labels_array = np.array(dataset.labels)
            unique_labels = np.unique(labels_array)
            sampled_indices = []

            for class_label in unique_labels:
                class_indices = np.where(labels_array == class_label)[0]
                sampled_indices.extend(np.random.choice(class_indices,
                                                        size=max_examples_per_class).tolist())

            dataset.labels = [label for idx, label in enumerate(
                dataset.labels) if idx in sampled_indices]
            dataset.images = [image for idx, image in enumerate(
                dataset.images) if idx in sampled_indices]
    return dataset_list


def get_transforms(resize, test_resize, pad, crop, flip, normalize, extend_channel, extra_aug="", ds_name=""):
    """Unpack transformations and apply to train or test splits"""

    trn_transform_list = []
    tst_transform_list = []

    # resize
    if resize is not None:
        trn_transform_list.append(transforms.Resize(resize))
        tst_transform_list.append(transforms.Resize(resize))

    # padding
    if pad is not None:
        trn_transform_list.append(transforms.Pad(pad))
        tst_transform_list.append(transforms.Pad(pad))

    # test only resize
    if test_resize is not None:
        tst_transform_list.append(transforms.Resize(test_resize))

    # crop
    if crop is not None:
        trn_transform_list.append(transforms.RandomResizedCrop(crop))
        tst_transform_list.append(transforms.CenterCrop(crop))

    # flips
    if flip:
        trn_transform_list.append(transforms.RandomHorizontalFlip())

    if extra_aug == 'simclr_cifar':  # removes Gaussian Blur from Cifar augmentations as it does not work well
        simCLR_transforms = [transforms.RandomApply([transforms.ColorJitter(brightness=0.4,
                                                                            contrast=0.4,
                                                                            saturation=0.4,
                                                                            hue=0.1)
                                                     ], p=0.8),
                             transforms.RandomGrayscale(p=0.2)]
        for t in simCLR_transforms:
            trn_transform_list.append(t)
    elif extra_aug == 'simclr':
        simCLR_transforms = [transforms.RandomApply([transforms.ColorJitter(brightness=0.4,
                                                                            contrast=0.4,
                                                                            saturation=0.4,
                                                                            hue=0.1)
                                                     ], p=0.8),
                             transforms.RandomGrayscale(p=0.2),
                             transforms.GaussianBlur(kernel_size=9)]
        for t in simCLR_transforms:
            trn_transform_list.append(t)
    elif extra_aug == 'colorjitter':  # Similar as in Avalanche
        trn_transform_list.append(transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
            p=0.8))
    elif extra_aug == 'brightness':  # Similar as in PyCIL
        trn_transform_list.append(transforms.ColorJitter(brightness=63 / 255))
    if extra_aug == 'fetril':  # Similar as in PyCIL
        trn_transform_list.append(transforms.ColorJitter(brightness=63 / 255))
        if 'cifar' in ds_name.lower():
            trn_transform_list.append(CIFAR10Policy())
        elif 'imagenet' in ds_name.lower():
            trn_transform_list.append(ImageNetPolicy())
        elif 'domainnet' in ds_name.lower():
            trn_transform_list.append(ImageNetPolicy())
        else:
            raise RuntimeError(f'Please check and update the data agumentation code for your dataset: {ds_name}')

    # to tensor
    trn_transform_list.append(transforms.ToTensor())
    tst_transform_list.append(transforms.ToTensor())

    if extra_aug == 'fetril':  # Similar as in PyCIL
        trn_transform_list.append(Cutout(n_holes=1, length=16))
    # normalization
    if normalize is not None:
        trn_transform_list.append(transforms.Normalize(
            mean=normalize[0], std=normalize[1]))
        tst_transform_list.append(transforms.Normalize(
            mean=normalize[0], std=normalize[1]))

    # gray to rgb
    if extend_channel is not None:
        trn_transform_list.append(transforms.Lambda(
            lambda x: x.repeat(extend_channel, 1, 1)))
        tst_transform_list.append(transforms.Lambda(
            lambda x: x.repeat(extend_channel, 1, 1)))

    return transforms.Compose(trn_transform_list), \
        transforms.Compose(tst_transform_list)


def _sample_classes(ds, idx_start, idx_end):
    dataset = ds[0]
    cls_indices = dataset.class_indices[idx_start:idx_end]
    idx_to_label = {idx: i for i, idx in enumerate(cls_indices)}
    images = [img for img, label in zip(dataset.images, dataset.labels) if label in cls_indices]
    labels = [idx_to_label[label] for label in dataset.labels if label in cls_indices]
    dataset.class_indices = cls_indices
    dataset.images = images
    dataset.labels = labels
    return [dataset]

def _ensure_imagenet_subset_prepared(path):
    assert os.path.exists(
        path), f"Please first download and extract dataset from: https://www.kaggle.com/datasets/arjunashok33/imagenet-subset-for-inc-learn to dir: {path}"
    ds_conf = dataset_config['imagenet_subset_kaggle']
    clsss2idx = {c: i for i, c in enumerate(ds_conf['lbl_order'])}
    print(f'Generating train/test splits for ImageNet-Subset directory: {path}')

    def prepare_split(split='train', outfile='train.txt'):
        with open(f"{path}/{outfile}", 'wt') as f:
            for fn in glob.glob(f"{path}/data/{split}/*/*"):
                c = fn.split('/')[-2]
                lbl = clsss2idx[c]
                relative_path = fn.replace(f"{path}/", '')
                f.write(f"{relative_path} {lbl}\n")

    prepare_split()
    prepare_split('val', outfile='test.txt')

def _ensure_tinyimagenet_prepared(path):
    assert os.path.exists(
            path), f"Please first download and extract dataset from: https://www.kaggle.com/competitions/tiny-imagenet to dir: {path}"
    ds_conf = dataset_config['tiny_scaled_imnet']
    clsss2idx = {c: i for i, c in enumerate(ds_conf['lbl_order'])}
    print(f'Generating train/test splits for Tiny Imagenet directory: {path}')

    def prepare_split_train():
        if not os.path.exists(f"{path}/train.txt"):
            with open(f"{path}/train.txt", 'wt') as f:
                for fn in glob.glob(f"{path}/train/*/*/*.JPEG"):
                    c = fn.split('/')[-3]
                    lbl = clsss2idx[c]
                    relative_path = fn.replace(f"{path}/", '')
                    f.write(f"{relative_path} {lbl}\n")
    def prepare_split_test(split='val'):
        if not os.path.exists(f"{path}/test.txt"):
            with open(f"{path}/test.txt", 'wt') as f:
                for fn in glob.glob(f"{path}/{split}/*/*.JPEG"):
                    c = fn.split('/')[-2]
                    lbl = clsss2idx[c]
                    relative_path = fn.replace(f"{path}/", '')
                    f.write(f"{relative_path} {lbl}\n")

    prepare_split_train()
    prepare_split_test()

def _ensure_domainnet_prepared(path, classes_per_domain=50, num_tasks=6):
    assert os.path.exists(path), f"Please first download and extract dataset from: http://ai.bu.edu/M3SDA/#dataset into:{path}"
    domains = ["clipart", "infograph", "painting",  "quickdraw", "real", "sketch"] * (num_tasks // 6)
    for set_type in ["train", "test"]:
        samples = []
        for i, domain in enumerate(domains):
            with open(f"{path}/{domain}_{set_type}.txt", 'r') as f:
                lines = list(map(lambda x: x.replace("\n", "").split(" "), f.readlines()))
            paths, classes = zip(*lines)
            classes = np.array(list(map(float, classes)))
            offset = classes_per_domain * i
            for c in range(classes_per_domain):
                is_class = classes == c + ((i // 6) * classes_per_domain)
                class_samples = list(compress(paths, is_class))
                samples.extend([*[f"{row} {c + offset}" for row in class_samples]])
        with open(f"{path}/{set_type}.txt", 'wt') as f:
            for sample in samples:
                f.write(f"{sample}\n")
