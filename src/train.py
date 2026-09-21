import argparse
from pathlib import Path

import torch
from torch import nn

from .data import make_loaders
from .metrics import classification_metrics
from .model import build_model, save_checkpoint


def run_epoch(model, loader, criterion, optimizer, device, training):
    model.train(training)
    total_loss, targets, predictions = 0.0, [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * images.size(0)
        targets.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
    metrics = classification_metrics(targets, predictions)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 for KL grade classification")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", default="checkpoints")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _, class_names, counts = make_loaders(args.data_dir, args.batch_size)
    model = build_model(len(class_names), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_f1 = -1.0
    print(f"device={device}; train_counts={dict(counts)}")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, False)
        print(f"epoch={epoch} train_loss={train_metrics['loss']:.4f} val_f1={val_metrics['f1_macro']:.4f} val_kappa={val_metrics['quadratic_weighted_kappa']:.4f}")
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            save_checkpoint(model, str(Path(args.output_dir) / "best.pt"), class_names)
    print(f"saved best checkpoint to {Path(args.output_dir) / 'best.pt'}")


if __name__ == "__main__":
    main()
