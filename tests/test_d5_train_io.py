"""The data-loading and metric-flag paths added for the D5.2 server runs.

``scene_batch`` / ``scene_item`` exist because a prepared cube may be a read-only
memmap; ``include_ssim`` exists because SSIM is by far the most expensive metric
term. Neither may change the numbers when the default path is used.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments" / "d5"))
sys.path.insert(0, str(REPO / "experiments" / "d5_3"))

import metrics as d5_metrics  # noqa: E402
import summarize_seeds  # noqa: E402
import train as d5_train  # noqa: E402


def cube(seed=0, size=32):
    rng = np.random.default_rng(seed)
    return (rng.random((21, size, size)) * 0.6 + 0.2).astype(np.float32)


def test_include_ssim_false_is_not_a_zero():
    truth = cube(1)
    prediction = truth + 0.01
    measurement = truth.mean(0)
    full = d5_metrics.reconstruction_metrics(prediction, truth, measurement, measurement)
    skipped = d5_metrics.reconstruction_metrics(
        prediction, truth, measurement, measurement, include_ssim=False)
    assert isinstance(full["ssim_band_mean"], float)
    assert skipped["ssim_band_mean"] is None
    for key, value in full.items():
        if key == "ssim_band_mean":
            continue
        assert skipped[key] == pytest.approx(value), key
    assert len(skipped["band_relative_l2"]) == 21


def test_include_ssim_true_is_the_default():
    truth = cube(2)
    measurement = truth.mean(0)
    default = d5_metrics.reconstruction_metrics(truth, truth, measurement, measurement)
    explicit = d5_metrics.reconstruction_metrics(
        truth, truth, measurement, measurement, include_ssim=True)
    assert default == explicit
    assert default["ssim_band_mean"] == pytest.approx(1.0, abs=1e-6)


class StubReconstructor(torch.nn.Module):
    """Minimal physics/forward pair: measurement is the band mean, cube is repeated."""

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def physics(self, x):
        return x.mean(1, keepdim=True)

    def forward(self, y):
        return dict(spectral_cube_hat=y.repeat(1, 21, 1, 1), reprojected_measurement=y)


def test_scene_batch_copies_out_of_a_read_only_mapping(tmp_path):
    array = np.load(_save(tmp_path / "cube.npy"), mmap_mode="r")
    batch = d5_train.scene_batch(array, 0)
    item = d5_train.scene_item(array, 0)
    assert tuple(batch.shape) == (1, 21, 32, 32)
    assert tuple(item.shape) == (21, 32, 32)
    batch.fill_(7.0)
    item.fill_(9.0)
    assert float(np.asarray(array[0]).max()) < 1.0        # mapping untouched
    assert float(np.asarray(array[1]).max()) < 1.0        # neighbour untouched


def _save(path):
    np.save(path, np.stack([cube(3), cube(4)]))
    return path


def test_evaluate_metrics_tolerates_a_skipped_ssim():
    data = np.stack([cube(5)])
    records = [dict(array_index=0, dataset_id="CAVE", scene_id="s0")]
    model = StubReconstructor()
    with_ssim = d5_train.evaluate_metrics(model, data, records)
    without = d5_train.evaluate_metrics(model, data, records, include_ssim=False)
    assert isinstance(with_ssim["ALL"]["ssim_band_mean"], float)
    assert without["ALL"]["ssim_band_mean"] is None
    assert without["CAVE"]["ssim_band_mean"] is None
    assert isinstance(without["ALL"]["psnr_unit_range_db"], float)
    assert isinstance(without["ALL"]["measurement_relative_l2"], float)
    assert len(without["ALL"]["band_relative_l2_mean"]) == 21
    assert without["ALL"]["psnr_unit_range_db"] == pytest.approx(
        with_ssim["ALL"]["psnr_unit_range_db"])


def test_plateau_std_skips_unmeasured_metrics():
    def row(value):
        metrics = {key: 1.0 for key in summarize_seeds.KEYS}
        metrics["ssim_band_mean"] = value
        return dict(metrics={"ALL": metrics})

    rows = [row(None), row(None), row(None), row(None)]
    spread = summarize_seeds.plateau_std(rows)
    assert spread["ssim_band_mean"] is None
    assert spread["psnr_unit_range_db"] == 0.0
    mixed = [row(0.90), row(0.92), row(None), row(0.94)]
    assert summarize_seeds.plateau_std(mixed)["ssim_band_mean"] == pytest.approx(0.02)
    assert summarize_seeds.plateau_std(rows[:2]) == {}
