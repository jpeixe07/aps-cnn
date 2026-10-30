from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def make_loaders(name: str, root: Path, batch_size: int, seed: int, workers: int,
                 pretrained: bool = False, limit: int | None = None):
    name = name.lower()
    if name not in {"mnist", "cifar10"}:
        raise ValueError("dataset deve ser 'mnist' ou 'cifar10'")

    if pretrained:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        # 64 px preserves useful context while making full CPU feature extraction practical.
        common = [transforms.Resize((64, 64)), transforms.ToTensor(),
                  transforms.Normalize(mean, std)]
        train_ops = ([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
                     if name == "cifar10" else [])
        train_transform = transforms.Compose(train_ops + common)
        eval_transform = transforms.Compose(common)
    elif name == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), transforms.Normalize(mean, std),
        ])
        eval_transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(mean, std),
        ])
    else:
        mean, std = (0.1307,), (0.3081,)
        train_transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(mean, std),
        ])
        eval_transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(mean, std),
        ])

    if name == "mnist":
        train_aug = datasets.MNIST(root, train=True, download=True, transform=train_transform)
        train_eval = datasets.MNIST(root, train=True, download=True, transform=eval_transform)
        test_data = datasets.MNIST(root, train=False, download=True, transform=eval_transform)
    else:
        train_aug = datasets.CIFAR10(root, train=True, download=True, transform=train_transform)
        train_eval = datasets.CIFAR10(root, train=True, download=True, transform=eval_transform)
        test_data = datasets.CIFAR10(root, train=False, download=True, transform=eval_transform)

    if pretrained and name == "mnist":
        # A ResNet ImageNet recebe três canais; replicar tons de cinza não altera os rótulos.
        rgb = transforms.Lambda(lambda image: image.convert("RGB"))
        train_aug.transform = transforms.Compose([rgb, *train_aug.transform.transforms])
        train_eval.transform = transforms.Compose([rgb, *train_eval.transform.transforms])
        test_data.transform = transforms.Compose([rgb, *test_data.transform.transforms])

    size = len(train_aug) if limit is None else min(limit, len(train_aug))
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(train_aug), generator=generator).tolist()[:size]
    val_size = max(1, int(size * 0.1))
    train_indices, val_indices = order[val_size:], order[:val_size]
    train_set = Subset(train_aug, train_indices)
    val_set = Subset(train_eval, val_indices)
    if limit is not None:
        test_data = Subset(test_data, list(range(min(limit, len(test_data)))))

    kwargs = {"num_workers": workers, "pin_memory": False}
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              generator=generator, **kwargs)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader
