import multiprocessing

# Pytorch Framework
from torchvision import datasets
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, random_split
import torch
from torch import nn


def cifar10_dataset(root='./data', train=True, test_batch=28, train_batch=128, num_workers=multiprocessing.cpu_count()):
    """
    Returns CIFAR10 training and testing dataset in a Pytorch DataLoader as a dictionary.
    :param bool train: Whether to return a training dataset generator
    :param int num_workers: Default is most available.
    :return dict:
    """
    dataset = dict()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
    testset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    dataset['testing'] = torch.utils.data.DataLoader(testset, batch_size=test_batch, shuffle=False, num_workers=num_workers)
    dataset['testing_length'] = len(dataset['testing'])

    if train:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize])
        trainset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
        dataset['training'] = torch.utils.data.DataLoader(trainset, batch_size=train_batch, shuffle=True, num_workers=num_workers)
        dataset['training_length'] = len(dataset['training'])
    return dataset