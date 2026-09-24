from pathlib import Path

import nibabel as nib
import numpy as np


class NiftiPatient:

    MODALITIES = {
        "t1n": "t1n",
        "t1c": "t1c",
        "t2w": "t2w",
        "t2f": "t2f",
        "t1": "t1",
        "t1ce": "t1ce",
        "t2": "t2",
        "flair": "flair",
    }

    MODALITY_ALIASES = {
        "t1n": ["t1n", "t1"],
        "t1c": ["t1c", "t1ce"],
        "t2w": ["t2w", "t2"],
        "t2f": ["t2f", "flair"],
        "t1": ["t1", "t1n"],
        "t1ce": ["t1ce", "t1c"],
        "t2": ["t2", "t2w"],
        "flair": ["flair", "t2f"],
    }

    MODALITY_NAMES = ["t1n", "t1c", "t2w", "t2f", "t1", "t1ce", "t2", "flair"]

    def __init__(self, patient_dir):
        self.patient_dir = Path(patient_dir)

    def _find_file(self, modality):
        modality = modality.lower()
        suffixes = self.MODALITY_ALIASES.get(modality, [self.MODALITIES.get(modality, modality)])

        for suffix in suffixes:
            matches = list(self.patient_dir.glob(f"*_{suffix}.nii.gz"))
            if matches:
                return matches[0]

        raise FileNotFoundError(
            f"Could not find {modality} for patient {self.patient_dir.name}"
        )

    def _detect_modality_name(self, file_name):
        basename = file_name.replace(".nii.gz", "")
        for suffix in self.MODALITY_NAMES:
            if basename.endswith(f"_{suffix}"):
                return suffix
        return None

    def _discover_modalities(self):
        found = []
        seen = set()

        for path in sorted(self.patient_dir.glob("*.nii.gz")):
            modality = self._detect_modality_name(path.name)
            if modality is None or modality in seen:
                continue
            if all(not path.name.endswith(f"_{candidate}.nii.gz") for candidate in ["seg", "label", "labels"]):
                found.append(modality)
                seen.add(modality)

        if found:
            return found

        for modality_order in (["t1n", "t1c", "t2w", "t2f"], ["t1", "t1ce", "t2", "flair"]):
            for modality in modality_order:
                try:
                    self._find_file(modality)
                except FileNotFoundError:
                    continue
                if modality not in seen:
                    found.append(modality)
                    seen.add(modality)

        return found

    def load_modality(self, modality):
        path = self._find_file(modality)
        image = nib.load(str(path))
        return image.get_fdata(dtype=np.float32)

    def load_modalities(self, modalities=None):
        if modalities is None:
            modalities = self._discover_modalities()
            if not modalities:
                modalities = ["t1n", "t1c", "t2w", "t2f"]

        images = {}
        for modality in modalities:
            images[modality] = self.load_modality(modality)
        return images

    def load_segmentation(self):
        for suffix in ["seg", "label", "labels"]:
            matches = list(self.patient_dir.glob(f"*_{suffix}.nii.gz"))
            if not matches:
                continue
            image = nib.load(str(matches[0]))
            return image.get_fdata(dtype=np.float32)

        raise FileNotFoundError(
            f"Could not find segmentation for patient {self.patient_dir.name}"
        )