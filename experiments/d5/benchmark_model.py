"""Measure D5 model step time and peak CUDA memory at declared resolution."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from train import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batches", default="1,2")
    ap.add_argument("--steps", type=int, default=5)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for model benchmark")
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    kernels = np.load(args.prepared / "kernels.npz")["kernels"]
    results = []
    for batch in [int(v) for v in args.batches.split(",")]:
        torch.manual_seed(cfg["seed"])
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = build_model(cfg, kernels, (args.size, args.size)).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
        x = torch.rand(batch, 21, args.size, args.size, device="cuda")
        with torch.no_grad():
            y = model.physics(x)
        for _ in range(2):
            out = model(y)
            loss = out["spectral_cube_hat"].square().mean()
            loss.backward()
            opt.step()
            opt.zero_grad()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.steps):
            out = model(y)
            loss = out["spectral_cube_hat"].square().mean()
            loss.backward()
            opt.step()
            opt.zero_grad()
        end.record()
        torch.cuda.synchronize()
        results.append(dict(
            model=cfg["model"], stages=cfg["stages"], size=args.size, batch=batch,
            parameters=sum(p.numel() for p in model.parameters()),
            step_ms=start.elapsed_time(end) / args.steps,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            status="ok"))
        del model, opt, x, y, out, loss
        torch.cuda.empty_cache()
    args.output.write_text(json.dumps(dict(
        protocol="D5-G2-model-benchmark",
        config=cfg, results=results,
        note="Synthetic input, forward+backward+optimizer step; not data-loader throughput."),
        indent=2), encoding="utf-8")
    print(json.dumps(results))


if __name__ == "__main__":
    main()
