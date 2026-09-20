from pathlib import Path

import nibabel as nib
import numpy as np


class NiftiPatient:

    MODALITIES = {
        "t1n": "t1n",
        "t1c": "t1c",
        "t2w": "t2w",
        "t2f": "t2f",
    }

    def __init__(
        self,
        patient_dir
    ):
        self.patient_dir = Path(patient_dir)

    def _find_file(self, modality):

        suffix = self.MODALITIES[modality]

        matches = list(
            self.patient_dir.glob(
                f"*_{suffix}.nii.gz"
            )
        )

        if not matches:

            raise FileNotFoundError(
                f"Could not find {modality} "
                f"for patient "
                f"{self.patient_dir.name}"
            )

        return matches[0]

    def load_modality(self, modality):

        path = self._find_file(
            modality
        )

        image = nib.load(
            str(path)
        )

        data = image.get_fdata(
            dtype=np.float32
        )

        return data

    def load_modalities(
        self,
        modalities=None
    ):

        if modalities is None:

            modalities = [
                "t1n",
                "t1c",
                "t2w",
                "t2f",
            ]

        images = {}

        for modality in modalities:

            images[modality] = (
                self.load_modality(
                    modality
                )
            )

        return images

    def load_segmentation(self):

        matches = list(
            self.patient_dir.glob(
                "*_seg.nii.gz"
            )
        )

        if not matches:

            raise FileNotFoundError(
                f"Could not find segmentation "
                f"for patient "
                f"{self.patient_dir.name}"
            )

        image = nib.load(
            str(matches[0])
        )

        return image.get_fdata(
            dtype=np.float32
        )