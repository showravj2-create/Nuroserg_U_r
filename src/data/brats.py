from __future__ import annotations

from pathlib import Path
from collections import OrderedDict

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
        max_cached_patients=2,
    ):
        self.data_root = Path(data_root)
        self.modality_names = modalities or ["t1", "t1ce", "t2", "flair"]
        self.context = int(context)
        self.target_size = tuple(target_size)
        self.seed = seed
        self.max_cached_patients = max(1, int(max_cached_patients))

        patients = discover_patients(self.data_root)
        if len(patients) < 3:
            raise ValueError(
                "BraTS training requires at least 3 patient folders with MRI volumes. "
                f"Found {len(patients)} under {self.data_root}."
            )

        self.patients = split_patients(patients, seed=self.seed)[split]
        self.samples = []
        self._patient_cache = OrderedDict()
        for patient_index, patient_dir in enumerate(self.patients):
            patient = NiftiPatient(patient_dir)
            try:
                label_shape = patient.load_segmentation().shape
                for z in range(label_shape[2]):
                    self.samples.append((patient_index, z))
            except FileNotFoundError:
                continue

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
        patient_index, z = self.samples[index]
        images, label = self._load_patient(patient_index)
        x, y = self._slice_from_volume(images, label, z)
        return (
            torch.from_numpy(x),
            torch.from_numpy(y[None, :, :]),
        )

    def _load_patient(self, patient_index):
        if patient_index in self._patient_cache:
            images, label = self._patient_cache.pop(patient_index)
            self._patient_cache[patient_index] = (images, label)
            return images, label

        patient = NiftiPatient(self.patients[patient_index])
        images = patient.load_modalities(self.modality_names)
        label = patient.load_segmentation()
        images = normalize_modalities(images, mask=label > 0)
        self._patient_cache[patient_index] = (images, label)
        while len(self._patient_cache) > self.max_cached_patients:
            self._patient_cache.popitem(last=False)
        return images, label
