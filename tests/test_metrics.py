import numpy as np

from src.metrics.segmentation import (
    dice_score,
    iou_score,
    hd95,
)


def test_perfect():

    x = np.array([
        [1, 0],
        [0, 1]
    ])

    assert dice_score(x, x) == 1.0
    assert iou_score(x, x) == 1.0


def test_disjoint():

    x = np.array([
        [1, 0],
        [0, 0]
    ])

    y = np.array([
        [0, 1],
        [0, 0]
    ])

    assert dice_score(x, y) < 1e-5


def test_hd95_identical():

    x = np.zeros(
        (32, 32),
        dtype=np.uint8
    )

    x[10:20, 10:20] = 1

    assert hd95(x, x) == 0.0


def test_hd95_shifted():

    x = np.zeros(
        (32, 32),
        dtype=np.uint8
    )

    y = np.zeros(
        (32, 32),
        dtype=np.uint8
    )

    x[10:20, 10:20] = 1
    y[12:22, 10:20] = 1

    distance = hd95(x, y)

    assert distance > 0.0