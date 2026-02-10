from dataclasses import dataclass
from typing import Tuple, Dict
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

@dataclass
class DataConfig:
    name: str = "cifar10"
    data_dir: str = "./data"
    batch_size: int = 128
    num_workers: int = 2
    img_size: int = 224
    val_split: float = 0.1

# simple, stable normalization (ImageNet-style) to work well with pretrained backbones
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def get_transforms(img_size: int, train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def _build_dataset(name: str, root: str, train: bool, transform):
    name = name.lower()
    if name == "cifar10":
        ds = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
        num_classes = 10
    elif name == "cifar100":
        ds = datasets.CIFAR100(root=root, train=train, download=True, transform=transform)
        num_classes = 100
    elif name == "stl10":
        split = "train" if train else "test"
        ds = datasets.STL10(root=root, split=split, download=True, transform=transform)
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {name}. Use cifar10|cifar100|stl10")
    return ds, num_classes

def make_dataloaders(cfg: DataConfig) -> Tuple[DataLoader, DataLoader, int, Dict]:
    train_tf = get_transforms(cfg.img_size, train=True)
    val_tf   = get_transforms(cfg.img_size, train=False)

    full_train, num_classes = _build_dataset(cfg.name, cfg.data_dir, train=True, transform=train_tf)

    val_len = int(len(full_train) * cfg.val_split)
    train_len = len(full_train) - val_len
    train_ds, val_ds = random_split(full_train, [train_len, val_len])

    # IMPORTANT: apply val transforms to val subset
    val_ds.dataset.transform = val_tf

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)

    preprocess_info = {
        "img_size": cfg.img_size,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "dataset": cfg.name,
    }
    return train_loader, val_loader, num_classes, preprocess_info
