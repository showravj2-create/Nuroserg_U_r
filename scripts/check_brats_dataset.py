#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.splits import discover_patients


def main():
    parser = argparse.ArgumentParser(description="Validate a BraTS 2020 dataset before training.")
    parser.add_argument("--data-root", required=True, help="Root directory containing the BraTS training set")
    args = parser.parse_args()

    root = Path(args.data_root)
    try:
        patients = discover_patients(root)
    except Exception as exc:
        print(f"Dataset check failed: {exc}")
        raise SystemExit(1)

    print(f"Dataset root: {root}")
    print(f"Patients found: {len(patients)}")

    required_modalities = ["t1", "t1ce", "t2", "flair", "seg"]
    missing = []
    for patient in patients[:5]:
        files = {p.name for p in patient.iterdir() if p.is_file()}
        missing_modalities = [
            modality
            for modality in required_modalities
            if not any(
                name.endswith((f"_{modality}.nii", f"_{modality}.nii.gz"))
                for name in files
            )
        ]
        print(f"{patient.name}: missing={missing_modalities}")
        if missing_modalities:
            missing.append((patient.name, missing_modalities))

    if missing:
        print("\nSome patients are missing expected modalities.")
        raise SystemExit(1)

    print("\nBraTS dataset layout looks valid for training.")


if __name__ == "__main__":
    main()
