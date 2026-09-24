# NeuroSeg-U v2

This is the BraTS 2020-ready version of the repository. It is designed to run on a real multi-modal MRI dataset and keeps the synthetic smoke test available as a fallback.

## What is different in v2?

- real BraTS 2020 naming support (`t1`, `t1ce`, `t2`, `flair`, `seg`)
- patient-level discovery for nested BraTS dataset layouts
- a dedicated dataset sanity check before training
- config tuned for real MRI training on CPU/GPU
- a clean launch path for cloud GPU environments

## Directory layout

```text
v2/
  README.md
  requirements.txt
  configs/
    brats_2p5d.yaml
  scripts/
    check_brats_dataset.py
    train.py
```

## Quick start

```bash
cd /workspaces/Nuroserg_U_r/v2
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_brats_dataset.py --data-root /mnt/data/BraTS2020
python scripts/train.py --config configs/brats_2p5d.yaml
```

## Expected dataset layout

```text
/mnt/data/BraTS2020/
  Training/
    BraTS20_Training_001/
      BraTS20_Training_001_flair.nii.gz
      BraTS20_Training_001_t1.nii.gz
      BraTS20_Training_001_t1ce.nii.gz
      BraTS20_Training_001_t2.nii.gz
      BraTS20_Training_001_seg.nii.gz
```

## Notes

- This v2 keeps the root project as the canonical implementation.
- Training and evaluation use the real BraTS 2020 dataset unless you explicitly set the synthetic flag.
- For cloud GPU runs, mount your dataset at a stable path like `/mnt/data/BraTS2020` before launching the train command.
