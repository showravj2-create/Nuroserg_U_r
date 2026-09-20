import argparse

import torch
import yaml

from torch.utils.data import DataLoader

from src.data.synthetic import (
    SyntheticSegmentationDataset
)

from src.models.unet import UNet

from src.losses.segmentation import (
    CompositeSegmentationLoss
)

from src.metrics.segmentation import (
    dice_score
)


def evaluate(
    model,
    loader,
    loss_fn,
    device
):
    model.eval()

    total_loss = 0.0
    dice_scores = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            loss = loss_fn(
                logits,
                y
            )

            total_loss += (
                loss.item()
                * x.size(0)
            )

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities > 0.5
            )

            for pred, target in zip(
                predictions,
                y
            ):

                score = dice_score(
                    pred.cpu().numpy(),
                    target.cpu().numpy()
                )

                dice_scores.append(score)

    average_loss = (
        total_loss / len(loader.dataset)
    )

    average_dice = (
        sum(dice_scores)
        / len(dice_scores)
    )

    return average_loss, average_dice


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True
    )

    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(
        cfg["seed"]
    )

    device_name = cfg["training"]["device"]

    if (
        device_name == "cuda"
        and not torch.cuda.is_available()
    ):
        device_name = "cpu"

    device = torch.device(
        device_name
    )

    print(f"Using device: {device}")

    # --------------------------------------------------
    # DATA
    # --------------------------------------------------

    train_dataset = (
        SyntheticSegmentationDataset(
            size=cfg["data"]["image_size"],
            channels=cfg["model"]["in_channels"],
            length=cfg["data"]["train_samples"],
            seed=cfg["seed"]
        )
    )

    val_dataset = (
        SyntheticSegmentationDataset(
            size=cfg["data"]["image_size"],
            channels=cfg["model"]["in_channels"],
            length=cfg["data"]["val_samples"],
            seed=cfg["seed"] + 10000
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False
    )

    # --------------------------------------------------
    # MODEL
    # --------------------------------------------------

    model = UNet(
        **cfg["model"]
    ).to(device)

    # --------------------------------------------------
    # LOSS
    # --------------------------------------------------

    loss_fn = CompositeSegmentationLoss(
        **cfg["loss"]
    )

    # --------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"]
    )

    # --------------------------------------------------
    # TRAINING
    # --------------------------------------------------

    best_dice = -1.0

    for epoch in range(
        cfg["training"]["epochs"]
    ):

        model.train()

        running_loss = 0.0

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            loss = loss_fn(
                logits,
                y
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * x.size(0)
            )

        train_loss = (
            running_loss
            / len(train_loader.dataset)
        )

        val_loss, val_dice = evaluate(
            model,
            val_loader,
            loss_fn,
            device
        )

        print(
            f"Epoch "
            f"{epoch + 1}/"
            f"{cfg['training']['epochs']} | "
            f"Train Loss: "
            f"{train_loss:.4f} | "
            f"Val Loss: "
            f"{val_loss:.4f} | "
            f"Val Dice: "
            f"{val_dice:.4f}"
        )

        if val_dice > best_dice:

            best_dice = val_dice

            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg,
                    "epoch": epoch + 1,
                    "val_dice": val_dice
                },
                "best.pt"
            )

            print(
                f"  ✓ New best checkpoint "
                f"(Dice={val_dice:.4f})"
            )

    print()
    print(
        f"Best validation Dice: "
        f"{best_dice:.4f}"
    )


if __name__ == "__main__":
    main()