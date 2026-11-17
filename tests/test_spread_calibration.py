import numpy as np
import torch

from argo_guided_cycle import bounded_zero_mean_anomalies, load_spread_factors
from fit_spread_calibration import isotonic_increasing


def test_inflation_preserves_mean_and_bounds():
    generator = torch.Generator().manual_seed(7)
    anomalies = torch.randn((8, 4, 5, 6), generator=generator)
    anomalies -= anomalies.mean(dim=0, keepdim=True)
    center = torch.zeros((1, 4, 5, 6))
    center[:, :2] = 10.0
    center[:, 2:] = 35.0
    mask = torch.ones_like(center)
    factor = torch.full((1, 4, 1, 1), 50.0)
    bounded, ratio = bounded_zero_mean_anomalies(
        anomalies, center, factor, levels=2,
        temperature_bounds=(-3.0, 40.0), salinity_bounds=(0.0, 42.0),
        ocean_mask=mask,
    )
    ensemble = center + bounded
    assert torch.max(torch.abs(bounded.mean(dim=0))) < 1e-5
    assert ensemble[:, :2].min() >= -3.00001
    assert ensemble[:, :2].max() <= 40.00001
    assert ensemble[:, 2:].min() >= -0.00001
    assert ensemble[:, 2:].max() <= 42.00001
    assert ratio.min() >= 0.0 and ratio.max() <= 1.0


def test_lead_dependent_factor_loading(tmp_path):
    factors = np.linspace(1.0, 4.0, 12, dtype=np.float32).reshape(3, 4)
    path = tmp_path / "lead_factors.npz"
    np.savez_compressed(path, inflation_factor=factors)
    loaded = load_spread_factors(path, channels=4)
    np.testing.assert_allclose(loaded, factors)


def test_weighted_isotonic_fit_is_non_decreasing():
    values = np.asarray([1.0, 3.0, 2.0, 5.0, 4.0])
    weights = np.asarray([1.0, 2.0, 1.0, 1.0, 3.0])
    fitted = isotonic_increasing(values, weights)
    assert np.all(np.diff(fitted) >= -1e-12)
    assert fitted.shape == values.shape
