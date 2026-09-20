import numpy as np
import pytest

from src.data.preprocessing import (
    validate_modalities,
    normalize_volume,
    normalize_modalities,
)


def test_validate_modalities():

    images = {
        "t1n": np.zeros((16, 16, 8)),
        "t1c": np.zeros((16, 16, 8)),
        "t2w": np.zeros((16, 16, 8)),
        "t2f": np.zeros((16, 16, 8)),
    }

    assert validate_modalities(
        images
    )


def test_validate_modalities_rejects_mismatch():

    images = {
        "t1n": np.zeros((16, 16, 8)),
        "t1c": np.zeros((16, 16, 8)),
        "t2w": np.zeros((16, 16, 8)),
        "t2f": np.zeros((16, 15, 8)),
    }

    with pytest.raises(ValueError):

        validate_modalities(
            images
        )


def test_normalize_volume():

    volume = np.zeros(
        (10, 10),
        dtype=np.float32
    )

    volume[2:8, 2:8] = np.arange(
        36,
        dtype=np.float32
    ).reshape(6,6)

    normalized = normalize_volume(
        volume
    )

    valid = normalized[
        volume != 0
    ]

    assert abs(
        valid.mean()
    ) < 1e-5

    assert abs(
        valid.std() - 1.0
    ) < 1e-5

    assert np.all(
        normalized[
            volume == 0
        ] == 0
    )


def test_normalize_modalities():

    images = {
        "t1n": np.random.randn(
            8, 8, 4
        ).astype(np.float32),

        "t1c": np.random.randn(
            8, 8, 4
        ).astype(np.float32),

        "t2w": np.random.randn(
            8, 8, 4
        ).astype(np.float32),

        "t2f": np.random.randn(
            8, 8, 4
        ).astype(np.float32),
    }

    normalized = normalize_modalities(
        images
    )

    assert set(
        normalized.keys()
    ) == set(
        images.keys()
    )

    for name in images:

        assert (
            normalized[name].shape
            == images[name].shape
        )