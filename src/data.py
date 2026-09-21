from pathlib import Path
from collections import Counter
from typing import Optional

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _split_path(root: Path, split: str) -> Path:
    candidates = [root / split]
    if split == "val":
        candidates.append(root / "validation")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Missing {split} folder in {root}")


def build_transforms(train: bool) -> transforms.Compose:
    steps = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if train:
        steps.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(7),
            transforms.ColorJitter(brightness=0.12, contrast=0.12),
        ])
    steps.extend([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    return transforms.Compose(steps)


def make_datasets(data_dir: str):
    root = Path(data_dir)
    train = datasets.ImageFolder(_split_path(root, "train"), transform=build_transforms(True))
    val = datasets.ImageFolder(_split_path(root, "val"), transform=build_transforms(False))
    test = datasets.ImageFolder(_split_path(root, "test"), transform=build_transforms(False))
    if train.class_to_idx != val.class_to_idx or train.class_to_idx != test.class_to_idx:
        raise ValueError("Train, validation and test class folders must match")
    return train, val, test


def make_loaders(data_dir: str, batch_size: int = 32, workers: int = 0):
    train, val, test = make_datasets(data_dir)
    counts = Counter(train.targets)
    weights = torch.tensor([1.0 / counts[label] for label in train.targets], dtype=torch.double)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    kwargs = {"batch_size": batch_size, "num_workers": workers, "pin_memory": torch.cuda.is_available()}
    return (
        DataLoader(train, sampler=sampler, **kwargs),
        DataLoader(val, shuffle=False, **kwargs),
        DataLoader(test, shuffle=False, **kwargs),
        train.classes,
        counts,
    )
