import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments" / "d5_3"))

import summarize_seeds  # noqa: E402

KEYS = summarize_seeds.KEYS


def metrics(value):
    return {key: float(value) for key in KEYS}


def make_run(root, name, seed, psnr, best_epoch, epochs=400, patience=60, stale=20,
             scene_psnr=None):
    directory = root / name
    directory.mkdir()
    history = [dict(epoch=e, global_step=e * 100, validation_loss=0.02 - e * 1e-5,
                    stale=stale if e == epochs else 0, best_epoch=best_epoch,
                    metrics={"ALL": metrics(30.0)}) for e in range(max(1, epochs - 3), epochs + 1)]
    scenes = [dict(scene_id=f"s{i}", dataset_id="CAVE" if i < 5 else "KAIST",
                   metrics=dict(metrics(scene_psnr if scene_psnr is not None else psnr),
                                band_relative_l2=[0.2] * 21, band_rmse=[0.02] * 21))
              for i in range(9)]
    validation = dict(best_epoch=best_epoch, best_validation_loss=0.012,
                      results=scenes,
                      selection=dict(rule="min_validation_loss", best_epoch=best_epoch,
                                     metrics_in_checkpoint={"ALL": metrics(psnr),
                                                            "CAVE": metrics(psnr),
                                                            "KAIST": metrics(psnr)}),
                      psf_provenance=dict(kernel_file="kernels.npz"),
                      psf=dict(policy="matched"))
    execution = dict(config=dict(seed=seed, patience=patience, epochs=epochs),
                     kernel_sha256_array="a" * 64, train_noise_std_absolute=0.01)
    (directory / "history.json").write_text(json.dumps(history), encoding="utf-8")
    (directory / "validation.json").write_text(json.dumps(validation), encoding="utf-8")
    (directory / "execution.json").write_text(json.dumps(execution), encoding="utf-8")
    return directory


def test_summary_uses_the_recorded_best_metrics_and_never_reselects(tmp_path):
    # metrics_in_checkpoint says 30.0 vs 29.0; the per-scene rows say 99.0, which would
    # look far better if this script re-selected or recomputed anything.
    make_run(tmp_path, "s1c_s1", 1, 30.0, 340, scene_psnr=99.0)
    make_run(tmp_path, "s3a_s1", 1, 29.0, 340, scene_psnr=99.0)
    document = summarize_seeds.analyse(tmp_path)
    assert document["pairs"][0]["deltas"]["psnr_unit_range_db"] == 1.0
    assert document["runs"]["s1c_s1"]["metrics"]["psnr_unit_range_db"] == 30.0
    assert document["runs"]["s1c_s1"]["metrics_recomputed"] is False


def test_budget_censoring_marks_a_best_epoch_inside_the_patience_window(tmp_path):
    make_run(tmp_path, "s1c_s1", 1, 30.0, 380, epochs=400, patience=60, stale=20)
    make_run(tmp_path, "s3a_s1", 1, 29.0, 380, epochs=400, patience=60, stale=20)
    document = summarize_seeds.analyse(tmp_path)
    assert document["runs"]["s1c_s1"]["budget_censored"] is True
    assert document["any_censored"] is True


def test_patience_stop_is_not_budget_censoring(tmp_path):
    make_run(tmp_path, "s1c_s1", 1, 30.0, 340, epochs=400, patience=60, stale=60)
    make_run(tmp_path, "s3a_s1", 1, 29.0, 340, epochs=400, patience=60, stale=60)
    document = summarize_seeds.analyse(tmp_path)
    assert document["runs"]["s1c_s1"]["stopped_by_patience"] is True
    assert document["runs"]["s1c_s1"]["budget_censored"] is False


def test_seed_level_statistics_and_scene_direction(tmp_path):
    for seed, delta in ((1, 0.5), (2, 0.4), (3, 0.45)):
        make_run(tmp_path, f"s1c_s{seed}", seed, 30.0 + delta, 200)
        make_run(tmp_path, f"s3a_s{seed}", seed, 30.0, 200)
    document = summarize_seeds.analyse(tmp_path)
    entry = document["aggregate"]["psnr_unit_range_db"]
    assert entry["per_seed"] == pytest.approx([0.5, 0.4, 0.45])
    assert entry["across_seeds"]["n"] == 3
    assert entry["across_seeds"]["mean"] == pytest.approx(0.45)
    assert entry["sign_consistent_across_seeds"] is True
    assert entry["scenes_favouring_chromatic"] == [9, 9, 9]
    assert entry["stable"] is True


def test_missing_arm_is_refused(tmp_path):
    make_run(tmp_path, "s1c_s1", 1, 30.0, 200)
    try:
        summarize_seeds.analyse(tmp_path)
    except ValueError as error:
        assert "s3a" in str(error)
    else:
        raise AssertionError("a missing achromatic arm must not be summarised")


def test_reference_arm_can_be_the_550_control(tmp_path):
    # D5.3 S3 replaces the mean kernel with the 550 nm kernel; the reducer must take
    # that arm as the reference instead of silently comparing against s3a.
    for seed in (1, 2, 3):
        make_run(tmp_path, f"s1c_s{seed}", seed, 30.4, 300)
        make_run(tmp_path, f"s3a_s{seed}", seed, 30.0, 300)
        make_run(tmp_path, f"s3a550_s{seed}", seed, 30.1, 300)
    document = summarize_seeds.analyse(tmp_path, reference_arm="s3a550")
    assert document["reference_arm"] == "s3a550"
    assert sorted(document["runs"]) == [f"s1c_s{s}" for s in (1, 2, 3)] + \
        [f"s3a550_s{s}" for s in (1, 2, 3)] + [f"s3a_s{s}" for s in (1, 2, 3)]
    assert document["pairs"][0]["deltas"]["psnr_unit_range_db"] == pytest.approx(0.3)
    assert document["pairs"][0]["censored"] == [False, False]


def test_unknown_reference_arm_is_refused(tmp_path):
    make_run(tmp_path, "s1c_s1", 1, 30.0, 200)
    make_run(tmp_path, "s3a_s1", 1, 29.0, 200)
    with pytest.raises(ValueError, match="s3a999"):
        summarize_seeds.analyse(tmp_path, reference_arm="s3a999")


def test_primary_arm_can_be_the_550_control(tmp_path):
    # The secondary S3 comparison is s3a550 vs s3a: the 550 arm is the primary side.
    for seed in (1, 2, 3):
        make_run(tmp_path, f"s1c_s{seed}", seed, 30.4, 300)
        make_run(tmp_path, f"s3a_s{seed}", seed, 30.0, 300)
        make_run(tmp_path, f"s3a550_s{seed}", seed, 30.2, 300)
    document = summarize_seeds.analyse(tmp_path, reference_arm="s3a", primary_arm="s3a550")
    assert document["primary_arm"] == "s3a550"
    assert document["reference_arm"] == "s3a"
    assert document["pairs"][0]["deltas"]["psnr_unit_range_db"] == pytest.approx(0.2)
    with pytest.raises(ValueError, match="s3a777"):
        summarize_seeds.analyse(tmp_path, primary_arm="s3a777")
