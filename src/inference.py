from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from .data import build_transforms, MEAN, STD, IMAGE_SIZE
from .gradcam import GradCAM

SEVERITY = {0: "No radiographic OA", 1: "Doubtful", 2: "Mild", 3: "Moderate", 4: "Severe"}


def predict(image: Image.Image, model, class_names: list[str], device: torch.device):
    tensor = build_transforms(False)(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    index = int(probabilities.argmax())
    grade = int(class_names[index].split("_")[-1]) if class_names[index].split("_")[-1].isdigit() else index
    return {"grade": grade, "severity": SEVERITY.get(grade, "Unknown"), "confidence": float(probabilities[index]), "tensor": tensor}


def overlay_cam(image: Image.Image, cam: np.ndarray):
    import matplotlib.cm as cm
    base = np.asarray(image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))).astype(np.float32) / 255
    heat = cm.get_cmap("jet")(cam)[..., :3]
    return np.clip(0.55 * base + 0.45 * heat, 0, 1)
