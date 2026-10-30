import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from lcd_reconstruction.operators.psf_provenance import array_digest  # noqa: E402


def make_prepared(root, seed=5):
    rng = np.random.default_rng(seed)
    prepared = root / "prepared"
    prepared.mkdir()
    np.savez_compressed(prepared / "train.npz",
                        cube=rng.random((2, 21, 16, 16), dtype=np.float32) * .4 + .1)
    np.savez_compressed(prepared / "val.npz",
                        cube=rng.random((1, 21, 16, 16), dtype=np.float32) * .4 + .1)
    kernels = rng.random((21, 4, 4), dtype=np.float32) * .01
    np.savez_compressed(prepared / "kernels.npz", kernels=kernels)
    achromatic = np.repeat(kernels.mean(0, keepdims=True), 21, axis=0)
    np.savez_compressed(prepared / "kernels_achromatic.npz", kernels=achromatic)
    (prepared / "manifest.json").write_text(json.dumps([
        dict(scene_id="train_a", dataset_id="CAVE", split="train", array_index=0),
        dict(scene_id="train_b", dataset_id="KAIST", split="train", array_index=1),
        dict(scene_id="val_a", dataset_id="CAVE", split="val", array_index=0),
    ]), encoding="utf-8")
    config = dict(seed=123, model="A", stages=2, width=24, depth=3, batch=1,
                  steps_per_epoch=2, epochs=4, patience=10, eval_every=5,
                  learning_rate=2e-4, weight_decay=1e-5, grad_clip=1.0,
                  train_noise_rms_fraction=0.005, brightness_range=[1.0, 1.0],
                  lambda_grad=0.2, lambda_spectral=0.05, lambda_data=0.02)
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return prepared, kernels


def run_train(prepared, root, output, *extra):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src")
    return subprocess.run(
        [sys.executable, "-X", "utf8", str(REPO / "experiments/d5/train.py"),
         "--prepared", str(prepared), "--config", str(root / "config.json"),
         "--device", "cpu", "--output", str(output), *extra],
        cwd=str(REPO), env=env, capture_output=True, text=True)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("best_metrics")
    prepared, kernels = make_prepared(root)
    output = root / "run"
    result = run_train(prepared, root, output)
    assert result.returncode == 0, result.stderr[-3000:]
    return dict(root=root, prepared=prepared, kernels=kernels, output=output, log=result.stdout)


def test_best_epoch_records_complete_metrics(trained):
    history = json.loads((trained["output"] / "history.json").read_text(encoding="utf-8"))
    assert len(history) == 4
    best_row = min(history, key=lambda row: row["validation_loss"])
    assert "metrics" in best_row
    assert "best" in best_row["metrics_roles"]
    assert "ALL" in best_row["metrics"]
    assert history[-1]["metrics_roles"] == ["final"] or "final" in history[-1]["metrics_roles"]
    assert "metrics" in history[-1]


def test_checkpoint_carries_rule_and_provenance(trained):
    payload = torch.load(trained["output"] / "best.pt", map_location="cpu", weights_only=True)
    assert payload["selection"]["rule"] == "min_validation_loss"
    assert payload["selection"]["is_best"] is True
    assert payload["selection"]["metrics"] is not None
    assert payload["selection"]["epoch"] == payload["epoch"] == payload["best_epoch"]
    record = payload["psf_provenance"]
    assert record["protocol"] == "d5.2-psf-v1"
    assert record["kernel_sha256_array"] == array_digest(trained["kernels"])
    assert record["physics_buffers"]


def test_validation_json_records_selection_and_consistency(trained):
    summary = json.loads((trained["output"] / "validation.json").read_text(encoding="utf-8"))
    assert summary["selection"]["rule"] == "min_validation_loss"
    assert summary["selection"]["best_epoch"] == summary["best_epoch"]
    assert summary["consistency"]["consistent"] is True
    assert summary["psf_provenance"]["kernel_sha256_array"] == array_digest(trained["kernels"])


def test_resume_with_the_other_arm_kernel_is_refused(trained):
    damaged = trained["root"] / "resume_wrong_kernel"
    damaged.mkdir()
    shutil.copyfile(trained["output"] / "last.pt", damaged / "last.pt")
    shutil.copyfile(trained["output"] / "history.json", damaged / "history.json")
    result = run_train(trained["prepared"], trained["root"], damaged,
                       "--resume", str(damaged / "last.pt"),
                       "--kernel-file", str(trained["prepared"] / "kernels_achromatic.npz"))
    assert result.returncode != 0
    assert "does not match the checkpoint operator" in result.stderr
