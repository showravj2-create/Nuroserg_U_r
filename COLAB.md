# Running NeuroSeg-U on Google Colab

This is the easiest path for anyone to run the repo on Colab with a real BraTS 2020 dataset.

## 1) Open Colab

Create a new notebook in Google Colab and run:

```python
!git clone https://github.com/showravj2-create/Nuroserg_U_r.git
%cd Nuroserg_U_r
!pip install -r requirements.txt
```

## 2) Download BraTS 2020

Use the official BraTS challenge page to download the dataset, then unzip it into a folder such as:

```python
from google.colab import drive

drive.mount('/content/drive')
# then copy or move the dataset into /content/BraTS2020
```

Expected layout:

```text
/content/BraTS2020/
  Training/
    BraTS20_Training_001/
      BraTS20_Training_001_flair.nii.gz
      BraTS20_Training_001_t1.nii.gz
      BraTS20_Training_001_t1ce.nii.gz
      BraTS20_Training_001_t2.nii.gz
      BraTS20_Training_001_seg.nii.gz
```

## 3) Validate the dataset

```python
!python scripts/check_brats_dataset.py --data-root /content/BraTS2020
```

## 4) Run training

```python
!python scripts/train.py --config configs/brats_2p5d.yaml
```

If you want to use the helper script instead:

```python
!python colab_run.py --dataset-root /content/BraTS2020 --config configs/brats_2p5d.yaml
```

## 5) Recommended config change for Colab

In [configs/brats_2p5d.yaml](configs/brats_2p5d.yaml), ensure the GPU config is set for Colab:

```yaml
training:
  device: cuda
```

If Colab runs out of memory, reduce:

```yaml
training:
  batch_size: 1
```

## Notes

- This project expects a real BraTS dataset, not a synthetic sample.
- The repo supports both the legacy naming and BraTS 2020 naming conventions.
- For best results, use the official BraTS challenge data and avoid changing patient splits during evaluation.
