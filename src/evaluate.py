import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch

from .data import make_datasets, build_transforms
from .metrics import classification_metrics
from .model import load_checkpoint
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained KL grade checkpoint")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test = make_datasets(args.data_dir)
    test.transform = build_transforms(False)
    loader = DataLoader(test, batch_size=32, shuffle=False)
    model, class_names = load_checkpoint(args.checkpoint, device)
    targets, predictions = [], []
    with torch.no_grad():
        for images, labels in loader:
            predictions.extend(model(images.to(device)).argmax(1).cpu().tolist())
            targets.extend(labels.tolist())
    metrics = classification_metrics(targets, predictions, labels=range(len(class_names)))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plt.figure(figsize=(6, 5))
    sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted grade")
    plt.ylabel("True grade")
    plt.tight_layout()
    plt.savefig(output / "confusion_matrix.png", dpi=160)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
