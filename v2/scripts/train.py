#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.brats import Brats2020Dataset
from src.data.synthetic import SyntheticSegmentationDataset
from src.losses.segmentation import CompositeSegmentationLoss
from src.metrics.segmentation import dice_score
from src.models.unet import UNet


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    dice_scores = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * x.size(0)
            probabilities = torch.sigmoid(logits)
            predictions = probabilities > 0.5

            for pred, target in zip(predictions, y):
                dice_scores.append(dice_score(pred.cpu().numpy(), target.cpu().numpy()))

    return total_loss / len(loader.dataset), sum(dice_scores) / len(dice_scores)


def main():
    parser = argparse.ArgumentParser(description="Train NeuroSeg-U v2 on BraTS 2020 data.")
    parser.add_argument("--config", required=True, help="YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["seed"])
    device_name = cfg["training"]["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    print(f"Using device: {device}")

    data_cfg = cfg["data"]
    if data_cfg.get("synthetic", False):
        train_dataset = SyntheticSegmentationDataset(
            size=data_cfg["image_size"],
            channels=cfg["model"]["in_channels"],
            length=data_cfg["train_samples"],
            seed=cfg["seed"],
        )
        val_dataset = SyntheticSegmentationDataset(
            size=data_cfg["image_size"],
            channels=cfg["model"]["in_channels"],
            length=data_cfg["val_samples"],
            seed=cfg["seed"] + 10000,
        )
    else:
        train_dataset = Brats2020Dataset(
            data_root=data_cfg["root"],
            split="train",
            modalities=data_cfg.get("modalities", ["t1", "t1ce", "t2", "flair"]),
            context=data_cfg.get("num_slices_context", 3),
            target_size=tuple(data_cfg.get("target_size", [240, 240])),
            seed=cfg["seed"],
        )
        val_dataset = Brats2020Dataset(
            data_root=data_cfg["root"],
            split="val",
            modalities=data_cfg.get("modalities", ["t1", "t1ce", "t2", "flair"]),
            context=data_cfg.get("num_slices_context", 3),
            target_size=tuple(data_cfg.get("target_size", [240, 240])),
            seed=cfg["seed"] + 1,
        )

    train_loader = DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["training"]["batch_size"], shuffle=False)

    model = UNet(**cfg["model"]).to(device)
    loss_fn = CompositeSegmentationLoss(**cfg["loss"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])

    best_dice = -1.0
    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_dice = evaluate(model, val_loader, loss_fn, device)

        print(f"Epoch {epoch + 1}/{cfg['training']['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch + 1, "val_dice": val_dice}, "best.pt")
            print(f"  ✓ New best checkpoint (Dice={val_dice:.4f})")

    print(f"\nBest validation Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
