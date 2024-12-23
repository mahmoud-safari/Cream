"""
This file contains the code to set up dataset before training a model
"""

import os
import argparse
import logging
import torch
from typing import Tuple, Any
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from torchvision import datasets
from torchvision.transforms import Compose
# from tasks.image_classification.utils import DATASETS, get_dataset_and_loaders
# from tasks.utils.utils import create_directory

from image_classification.mixup_cutmix import get_mixup_cutmix
from image_classification.augmentations import resnet_aug_train, resnet_aug_test
from image_classification.augmentations import wide_resnet_aug_train, wide_resnet_aug_test
from image_classification.augmentations import densenet_aug_train, densenet_aug_test
from image_classification.augmentations import vit_aug_train, vit_aug_test
from image_classification.augmentations import vit_tiny_aug_train, vit_tiny_aug_test


def create_directory(
    path: str,
    ddp: bool = False,
    master_process: bool = False
) -> None:
    """
    This function creates the required directories

    Args:
        path (str): The path to the directory to create
        ddp (bool, optional): Flag which tells if this is ddp training is done. Defaults to False.
        master_process (bool, optional): Flag which tells if this is the master process in ddp. Defaults to False.
    """
    os.makedirs(path, exist_ok=True)
    if (ddp and master_process) or not ddp:
        logging.info('Successfully created the directories!')


DATASETS = {
    'imagenet_1k': datasets.ImageNet,
    'none': None
}

def get_dataset_and_loaders(
    dataset: str,
    dataset_path: str,
    aug_train: Compose,
    aug_test: Compose,
    batch_size: int,
    num_workers: int,
    mixup_alpha: float,
    cutmix_alpha: float
) -> Tuple[DataLoader, DataLoader, int, int, int]:
    """
    This function creates the objects to hold the datasets to train on

    Args:
        dataset (str): Name of the dataset to train the model on
        dataset_path (str): Path to the dataset where it is stored
        aug_train (Compose): Augmentation to use for training set
        aug_test (Compose): Augmentation to use for test set
        batch_size (int): The batch size to train with
        num_workers (int): The number of workers used to load data
        mixup_alpha (float): The alpha value for MixUp
        cutmix_alpha (float): The alpha value for CutMix

    Returns:
        Tuple[DataLoader, DataLoader, int, int, int]: The data loaders that hold the train and test datasets, the \
            number of classes and the number of train and test images in the dataset
    """
    # Gets the dataset instance to train on ----------------------------------------------------------------------------
    use_dataset = DATASETS[dataset]
    
    if use_dataset is not None:  # Checks if the dataset exists in torchvision
        if dataset in ['cifar_10', 'cifar_100']:
            train_dataset = use_dataset(root=dataset_path, train=True, transform=aug_train, download=True)
            test_dataset = use_dataset(root=dataset_path, train=False, transform=aug_test, download=True)
        elif dataset == 'imagenet_1k':
            train_dataset = use_dataset(root=dataset_path, split='train', transform=aug_train)
            test_dataset = use_dataset(root=dataset_path, split='val', transform=aug_test)
    
    # Get meta data about the dataset ----------------------------------------------------------------------------------    
    num_classes = len(train_dataset.classes)
    num_train_images = len(train_dataset)
    num_test_images = len(test_dataset)
    
    # Setup MixUp and CutMix -------------------------------------------------------------------------------------------
    # The collate_fn only goes in the training data loader -------------------------------------------------------------
    mixup_cutmix = get_mixup_cutmix(
        mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha, num_classes=num_classes, use_v2=True
    )
    if mixup_cutmix is not None:
        logging.info('Using MixUp and CutMix!')
        def collate_fn(batch):
            return mixup_cutmix(*default_collate(batch))
    else:
        logging.info('Not using MixUp and CutMix!')
        collate_fn = default_collate

    # Instantiating the Train and Test DataLoaders ---------------------------------------------------------------------
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    logging.info('Instantiated the data loaders!')
    return train_loader, test_loader, num_classes, num_train_images, num_test_images


if __name__ == '__main__':
    
    cmdline_parser = argparse.ArgumentParser('Set up ImageNet-1K Dataset')
    
    cmdline_parser.add_argument('-dn', '--dataset_name',
                                default='cifar_10',
                                help='Name of the dataset',
                                choices=list(DATASETS.keys()),
                                type=str)
    cmdline_parser.add_argument('-dp', '--dataset_path',
                                default='./data',
                                help='Path where the dataset is stored',
                                type=str)
    
    args, unknowns = cmdline_parser.parse_known_args()
    logging.basicConfig(level=logging.INFO)
    if unknowns:
        logging.warning('Found unknown arguments!')
        logging.warning(str(unknowns))
        logging.warning('These will be ignored')
    
    # The following function will try to create the dataset instances and loaders
    # The ImageNet-1K files downloaded from the web will be extracted and the required folders will be created
    dataset_name = args.dataset_name
    # dataset_path = f'{args.dataset_path}/{dataset_name}'
    dataset_path = f'{args.dataset_path}'
    create_directory(
        path=dataset_path
    )
    # dataset_path = "/hkfs/work/workspace/scratch/fr_ms2108-data/data/imagenet"
    logging.info(dataset_path)
    train_loader, test_loader, num_classes, num_train_images, num_test_images = get_dataset_and_loaders(
        dataset=dataset_name,
        dataset_path=dataset_path,
        aug_train=transforms.Compose([]),
        aug_test=transforms.Compose([]),
        batch_size=256,
        num_workers=12,
        mixup_alpha=0.0,
        cutmix_alpha=0.0
    )
    logging.info('Dataset Extracted!')
