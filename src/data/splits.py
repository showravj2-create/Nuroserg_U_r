from pathlib import Path
import random


def discover_patients(data_root):
    """
    Discover patient directories containing MRI data.

    Supports both the legacy lab layout and the BraTS 2020 training layout:

    data_root/
        BraTS-001/
            BraTS-001_t1n.nii.gz
            BraTS-001_t1c.nii.gz
            BraTS-001_t2w.nii.gz
            BraTS-001_t2f.nii.gz
            BraTS-001_seg.nii.gz

    or

    data_root/
        Training/
            BraTS20_Training_001/
                BraTS20_Training_001_t1.nii.gz
                BraTS20_Training_001_t1ce.nii.gz
                BraTS20_Training_001_t2.nii.gz
                BraTS20_Training_001_flair.nii.gz
                BraTS20_Training_001_seg.nii.gz
    """

    data_root = Path(data_root)

    if not data_root.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {data_root}"
        )

    nifti_files = sorted(data_root.rglob("*.nii.gz"))
    patients = sorted({path.parent for path in nifti_files if path.is_file()})

    if not patients:
        raise ValueError(
            f"No patient directories containing .nii.gz files "
            f"were found in {data_root}"
        )

    return patients


def split_patients(
    patients,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
):
    """
    Split patients into train/validation/test sets.

    Splitting is performed at patient level.
    """

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            "train_ratio + val_ratio + test_ratio must equal 1.0"
        )

    patients = list(patients)

    if len(patients) < 3:
        raise ValueError(
            "At least 3 patients are required for train/val/test splitting."
        )

    rng = random.Random(seed)
    rng.shuffle(patients)

    n = len(patients)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    # Guarantee that every split gets at least one patient.
    n_train = max(1, n_train)
    n_val = max(1, n_val)

    if n_train + n_val >= n:
        n_val = 1
        n_train = n - 2

    train = patients[:n_train]
    val = patients[n_train:n_train + n_val]
    test = patients[n_train + n_val:]

    return {
        "train": train,
        "val": val,
        "test": test,
    }