import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
import numpy as np

from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.brats import Brats2020Dataset
from src.data.synthetic import SyntheticSegmentationDataset
from src.models.unet import UNet

from src.metrics.segmentation import (
    dice_score,
    iou_score,
    confusion_metrics,
    hd95,
    bootstrap_ci,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True
    )

    parser.add_argument(
        "--checkpoint",
        default="best.pt"
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # CONFIG
    # --------------------------------------------------

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # --------------------------------------------------
    # DEVICE
    # --------------------------------------------------

    device_name = cfg["training"]["device"]

    if (
        device_name == "cuda"
        and not torch.cuda.is_available()
    ):
        device_name = "cpu"

    device = torch.device(device_name)

    print(f"Using device: {device}")

    # --------------------------------------------------
    # DATA
    # --------------------------------------------------

    data_cfg = cfg["data"]
    if data_cfg.get("synthetic", False):
        dataset = SyntheticSegmentationDataset(
            size=data_cfg["image_size"],
            channels=cfg["model"]["in_channels"],
            length=data_cfg["val_samples"],
            seed=cfg["seed"] + 10000,
        )
    else:
        dataset = Brats2020Dataset(
            data_root=data_cfg["root"],
            split="val",
            modalities=data_cfg.get("modalities", ["t1", "t1ce", "t2", "flair"]),
            context=data_cfg.get("num_slices_context", 3),
            target_size=tuple(data_cfg.get("target_size", [240, 240])),
            seed=cfg["seed"] + 10,
        )

    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
    )

    # --------------------------------------------------
    # MODEL
    # --------------------------------------------------

    model = UNet(
        **cfg["model"]
    ).to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only= False
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval()

    print(
        f"Loaded checkpoint from epoch "
        f"{checkpoint['epoch']}"
    )

    # --------------------------------------------------
    # EVALUATION
    # --------------------------------------------------

    dice_values = []
    iou_values = []
    sensitivity_values = []
    specificity_values = []
    precision_values = []
    hd95_values = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)

            logits = model(x)

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

                pred_np = (
                    pred.cpu()
                    .numpy()
                    .astype(bool)
                )

                target_np = (
                    target.numpy()
                    .astype(bool)
                )

                dice_values.append(
                    dice_score(
                        pred_np,
                        target_np
                    )
                )

                iou_values.append(
                    iou_score(
                        pred_np,
                        target_np
                    )
                )

                confusion = confusion_metrics(
                    pred_np,
                    target_np
                )

                sensitivity_values.append(
                    confusion["sensitivity"]
                )

                specificity_values.append(
                    confusion["specificity"]
                )

                precision_values.append(
                    confusion["precision"]
                )

                distance = hd95(
                    pred_np,
                    target_np
                )

                if not np.isnan(distance):
                    hd95_values.append(
                        distance
                    )

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    dice_mean, dice_ci = bootstrap_ci(
        dice_values
    )

    iou_mean, iou_ci = bootstrap_ci(
        iou_values
    )

    sensitivity_mean = np.mean(
        sensitivity_values
    )

    specificity_mean = np.mean(
        specificity_values
    )

    precision_mean = np.mean(
        precision_values
    )

    hd95_mean = (
        np.mean(hd95_values)
        if hd95_values
        else float("nan")
    )

    results = {

        "checkpoint_epoch":
            checkpoint["epoch"],

        "checkpoint_val_dice":
            checkpoint["val_dice"],

        "dice": {
            "mean": float(dice_mean),
            "ci95": [
                float(dice_ci[0]),
                float(dice_ci[1])
            ]
        },

        "iou": {
            "mean": float(iou_mean),
            "ci95": [
                float(iou_ci[0]),
                float(iou_ci[1])
            ]
        },

        "sensitivity":
            float(sensitivity_mean),

        "specificity":
            float(specificity_mean),

        "precision":
            float(precision_mean),

        "hd95":
            float(hd95_mean)
    }

    # --------------------------------------------------
    # PRINT
    # --------------------------------------------------

    print()
    print("Evaluation Results")
    print("------------------")

    print(
        f"Dice:        {dice_mean:.4f}"
    )

    print(
        f"Dice 95% CI: "
        f"[{dice_ci[0]:.4f}, "
        f"{dice_ci[1]:.4f}]"
    )

    print(
        f"IoU:         {iou_mean:.4f}"
    )

    print(
        f"IoU 95% CI:  "
        f"[{iou_ci[0]:.4f}, "
        f"{iou_ci[1]:.4f}]"
    )

    print(
        f"Sensitivity:  "
        f"{sensitivity_mean:.4f}"
    )

    print(
        f"Specificity:  "
        f"{specificity_mean:.4f}"
    )

    print(
        f"Precision:    "
        f"{precision_mean:.4f}"
    )

    print(
        f"HD95:         "
        f"{hd95_mean:.4f}"
    )

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    with open(
        "evaluation.json",
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    print()
    print(
        "Saved results to evaluation.json"
    )


if __name__ == "__main__":
    main()