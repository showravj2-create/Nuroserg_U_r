from pathlib import Path

from src.data.splits import discover_patients, split_patients


def test_discover_patients(tmp_path):
    data_root = tmp_path / "dataset"
    data_root.mkdir()

    for index, patient_id in enumerate(["BraTS-001", "BraTS-002", "BraTS-003"]):
        patient_dir = data_root / patient_id
        patient_dir.mkdir()

        extension = ".nii" if index == 0 else ".nii.gz"
        (patient_dir / f"{patient_id}_t1n{extension}").touch()

    patients = discover_patients(data_root)

    assert len(patients) == 3
    assert all(isinstance(p, Path) for p in patients)


def test_patient_split():
    patients = [
        Path(f"BraTS-{i:03d}")
        for i in range(20)
    ]

    splits = split_patients(
        patients,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    assert len(splits["train"]) == 14
    assert len(splits["val"]) == 3
    assert len(splits["test"]) == 3

    train = set(splits["train"])
    val = set(splits["val"])
    test = set(splits["test"])

    # Critical leakage check.
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)


def test_split_is_reproducible():
    patients = [
        Path(f"BraTS-{i:03d}")
        for i in range(20)
    ]

    split_a = split_patients(patients, seed=42)
    split_b = split_patients(patients, seed=42)

    assert split_a == split_b