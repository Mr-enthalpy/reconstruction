"""Strict resume-equivalence check on a tiny synthetic prepared set."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]


def compare_state(a, b, path="root"):
    diffs = []
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape:
            diffs.append((path, "shape", list(a.shape), list(b.shape)))
        elif not torch.equal(a.cpu(), b.cpu()):
            diffs.append((path, "max_abs", float((a.cpu() - b.cpu()).abs().max())))
    elif isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            diffs.append((path, "keys", sorted(a), sorted(b)))
        for key in set(a) & set(b):
            diffs += compare_state(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            diffs.append((path, "length", len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            diffs += compare_state(x, y, f"{path}[{i}]")
    elif a != b:
        diffs.append((path, "value", a, b))
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rng = np.random.default_rng(17)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        prepared = tmp / "prepared"
        prepared.mkdir()
        train = rng.random((2, 21, 16, 16), dtype=np.float32) * .5 + .1
        val = rng.random((1, 21, 16, 16), dtype=np.float32) * .5 + .1
        kernels = rng.random((21, 4, 4), dtype=np.float32) * .01
        np.savez_compressed(prepared / "train.npz", cube=train)
        np.savez_compressed(prepared / "val.npz", cube=val)
        np.savez_compressed(prepared / "kernels.npz", kernels=kernels)
        records = [
            dict(scene_id="train_a", dataset_id="CAVE", split="train", array_index=0),
            dict(scene_id="train_b", dataset_id="KAIST", split="train", array_index=1),
            dict(scene_id="val_a", dataset_id="CAVE", split="val", array_index=0),
        ]
        (prepared / "manifest.json").write_text(json.dumps(records), encoding="utf-8")
        config = dict(
            seed=123, model="A", stages=2, width=24, depth=3, batch=1,
            steps_per_epoch=2, epochs=4, patience=4, eval_every=1,
            learning_rate=2e-4, weight_decay=1e-5, grad_clip=1.0,
            train_noise_rms_fraction=0.005, brightness_range=[1.0, 1.0],
            lambda_grad=0.2, lambda_spectral=0.05, lambda_data=0.02,
        )
        (tmp / "config.json").write_text(json.dumps(config), encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO / "src")
        base = [sys.executable, "-X", "utf8", str(REPO / "experiments/d5/train.py"),
                "--prepared", str(prepared), "--config", str(tmp / "config.json"),
                "--device", "cpu"]

        def run(*extra):
            subprocess.run(base + list(extra), cwd=REPO, env=env, check=True,
                           stdout=subprocess.DEVNULL)

        full = tmp / "full"
        interrupted = tmp / "interrupted"
        run("--output", str(full))
        run("--output", str(interrupted), "--stop-after", "2")
        run("--output", str(interrupted), "--resume", str(interrupted / "last.pt"))

        left = torch.load(full / "last.pt", map_location="cpu", weights_only=True)
        right = torch.load(interrupted / "last.pt", map_location="cpu", weights_only=True)
        fields = ["state", "optimizer", "scheduler", "epoch", "next_epoch", "global_step",
                  "best", "best_epoch", "stale", "torch_rng", "numpy_rng"]
        diffs = []
        for field in fields:
            diffs += compare_state(left[field], right[field], field)
        full_best = torch.load(full / "best.pt", map_location="cpu", weights_only=True)
        int_best = torch.load(interrupted / "best.pt", map_location="cpu", weights_only=True)
        diffs += compare_state(full_best["state"], int_best["state"], "best.state")
        result = dict(protocol="D5.2-resume-equivalence", passed=not diffs,
                      compared_fields=fields, differences=diffs[:30],
                      full_epoch=left["epoch"], resumed_epoch=right["epoch"],
                      full_global_step=left["global_step"],
                      resumed_global_step=right["global_step"],
                      full_best_epoch=left["best_epoch"],
                      resumed_best_epoch=right["best_epoch"])
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result))


if __name__ == "__main__":
    main()

