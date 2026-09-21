from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

NUM_CLASSES = 5


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def save_checkpoint(model: nn.Module, path: str, class_names: list[str]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "class_names": class_names}, destination)


def load_checkpoint(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    class_names = checkpoint.get("class_names", [str(index) for index in range(NUM_CLASSES)])
    model = build_model(len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, class_names
