from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset

from src.data.nifti import NiftiPatient
from src.data.preprocessing import normalize_modalities
from src.data.splits import discover_patients, split_patients


class Brats2020Dataset(Dataset):
    """Simple 2.5D BraTS 2020 dataset that emits axial slices with per-modality context."""

    def __init__(
        self,
        data_root,
        split="train",
        modalities=None,
        context=3,
        target_size=(240, 240),
        seed=42,
    ):
        self.data_root = Path(data_root)
        self.modality_names = modalities or ["t1", "t1ce", "t2", "flair"]
        self.context = int(context)
        self.target_size = tuple(target_size)
        self.seed = seed

        patients = discover_patients(self.data_root)
        if len(patients) < 3:
            raise ValueError(
                "BraTS training requires at least 3 patient folders with MRI volumes. "
                f"Found {len(patients)} under {self.data_root}."
            )

        self.patients = split_patients(patients, seed=self.seed)[split]
        self.samples = []
        for patient_dir in self.patients:
            patient = NiftiPatient(patient_dir)
            try:
                images = patient.load_modalities(self.modality_names)
                label = patient.load_segmentation()
            except FileNotFoundError:
                continue

            if not images:
                continue

            images = normalize_modalities(images, mask=label > 0)
            for z in range(label.shape[2]):
                x, y = self._slice_from_volume(images, label, z)
                if x is not None:
                    self.samples.append((x, y))

        if not self.samples:
            raise ValueError(
                f"No usable slices were found in {self.data_root} for the {split} split. "
                "Check the BraTS folder layout or modality names."
            )

    def _slice_from_volume(self, images, label, z):
        slices = []
        for modality in self.modality_names:
            volume = images[modality]
            if volume.shape != label.shape:
                raise ValueError(
                    f"Modality {modality} shape {volume.shape} does not match label {label.shape} "
                    f"for patient {self.data_root.name}."
                )

            indices = self._slice_indices(volume.shape[2], z)
            stack = []
            for idx in indices:
                slice_2d = volume[:, :, idx]
                stack.append(slice_2d)
            stack = np.stack(stack, axis=0)
            slices.append(stack)

        x = np.concatenate(slices, axis=0).astype(np.float32)
        y = label[:, :, z].astype(np.float32)

        x = self._resize_volume(x)
        y = self._resize_volume(y[None, :, :], is_label=True)[0]
        return x, y

    def _slice_indices(self, depth, center):
        if self.context <= 1:
            return [min(max(center, 0), depth - 1)]

        indices = []
        for offset in range(self.context):
            idx = int(round(center - (self.context - 1) / 2 + offset))
            idx = min(max(idx, 0), depth - 1)
            indices.append(idx)
        return indices

    def _resize_volume(self, array, is_label=False):
        if array.ndim == 2:
            array = array[None, :, :]

        if array.shape[-2:] == self.target_size:
            return array.astype(np.float32)

        target_h, target_w = self.target_size
        zoom_y = target_h / array.shape[-2]
        zoom_x = target_w / array.shape[-1]
        out = ndimage.zoom(array, (1.0, zoom_y, zoom_x), order=0 if is_label else 1)
        if out.shape[-2:] != self.target_size:
            out = np.pad(out, ((0, 0), (0, max(0, target_h - out.shape[-2])), (0, max(0, target_w - out.shape[-1]))), mode="edge")
            out = out[:, :target_h, :target_w]
        return out.astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x, y = self.samples[index]
        return (
            torch.from_numpy(x),
            torch.from_numpy(y[None, :, :]),
        )
