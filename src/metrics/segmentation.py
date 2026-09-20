import numpy as np
from scipy import ndimage


def dice_score(pred, target, eps=1e-7):

    p = np.asarray(pred).astype(bool)
    t = np.asarray(target).astype(bool)

    return (
        2 * (p & t).sum() + eps
    ) / (
        p.sum() + t.sum() + eps
    )


def iou_score(pred, target, eps=1e-7):

    p = np.asarray(pred).astype(bool)
    t = np.asarray(target).astype(bool)

    return (
        (p & t).sum() + eps
    ) / (
        (p | t).sum() + eps
    )


def confusion_metrics(
    pred,
    target,
    eps=1e-7
):

    p = np.asarray(pred).astype(bool)
    t = np.asarray(target).astype(bool)

    tp = (p & t).sum()
    tn = (~p & ~t).sum()
    fp = (p & ~t).sum()
    fn = (~p & t).sum()

    return {
        "sensitivity":
            (tp + eps) / (tp + fn + eps),

        "specificity":
            (tn + eps) / (tn + fp + eps),

        "precision":
            (tp + eps) / (tp + fp + eps)
    }


def _surface(mask):
    """
    Extract the boundary of a binary mask.

    Supports masks with singleton dimensions such as
    (1, H, W), which are common for binary segmentation.
    """

    mask = np.asarray(mask).astype(bool)

    # Remove singleton dimensions.
    mask = np.squeeze(mask)

    if mask.ndim != 2:
        raise ValueError(
            f"HD95 currently expects a 2D mask. "
            f"Received shape: {mask.shape}"
        )

    if not mask.any():
        return np.zeros_like(
            mask,
            dtype=bool
        )

    structure = np.ones(
        (3, 3),
        dtype=bool
    )

    eroded = ndimage.binary_erosion(
        mask,
        structure=structure,
        border_value=0
    )

    return mask ^ eroded


def hd95(pred, target):

    p = np.squeeze(
        np.asarray(pred).astype(bool)
    )

    t = np.squeeze(
        np.asarray(target).astype(bool)
    )

    if p.ndim != 2 or t.ndim != 2:
        raise ValueError(
            "HD95 expects 2D segmentation masks. "
            f"Got {p.shape} and {t.shape}"
        )

    if not p.any() or not t.any():
        return float("nan")

    p_surface = _surface(p)
    t_surface = _surface(t)

    distance_to_t = ndimage.distance_transform_edt(
        ~t_surface
    )

    distance_to_p = ndimage.distance_transform_edt(
        ~p_surface
    )

    distances_p_to_t = distance_to_t[
        p_surface
    ]

    distances_t_to_p = distance_to_p[
        t_surface
    ]

    distances = np.concatenate([
        distances_p_to_t,
        distances_t_to_p
    ])

    return float(
        np.percentile(
            distances,
            95
        )
    )


def bootstrap_ci(
    values,
    statistic=np.mean,
    n_boot=2000,
    seed=42
):

    rng = np.random.default_rng(seed)

    x = np.asarray(
        values,
        dtype=float
    )

    samples = rng.choice(
        x,
        size=(n_boot, len(x)),
        replace=True
    )

    statistics = np.apply_along_axis(
        statistic,
        1,
        samples
    )

    return (
        float(statistic(x)),
        (
            float(np.percentile(
                statistics,
                2.5
            )),
            float(np.percentile(
                statistics,
                97.5
            ))
        )
    )