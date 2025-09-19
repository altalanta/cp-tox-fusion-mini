import numpy as np

from src.qc_metrics import compute_focus_metric, fit_illumination_gradient, compute_debris_ratio


def test_focus_metric_detects_blur():
    sharp = np.zeros((32, 32))
    sharp[8:24, 8:24] = 1.0
    blurred = np.ones((32, 32)) * 0.5
    sharp_metric = compute_focus_metric(sharp)
    blurred_metric = compute_focus_metric(blurred)
    assert sharp_metric > blurred_metric


def test_illumination_gradient_nonzero_for_tilt():
    yy, xx = np.mgrid[:16, :16]
    gradient_image = xx.astype(float)
    gradient = fit_illumination_gradient(gradient_image)
    assert gradient > 0


def test_debris_ratio_counts_small_objects():
    mask = np.zeros((10, 10), dtype=int)
    mask[1:3, 1:3] = 1
    mask[5:9, 5:9] = 2
    ratio = compute_debris_ratio(mask, min_area=6)
    assert 0 < ratio < 1
