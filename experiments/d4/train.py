"""Label-free general N1. CUDA training by default; explicit CPU smoke is not acceptance."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from lcd_reconstruction.unrolling.general_pgd import GeneralSpectralPGD, reconstruction_loss


def augment(x, seed, brightness):
    """Whole-scene transforms precede simulation and share all spectral channels."""
    rng = np.random.default_rng(seed)
    x = torch.rot90(x, int(rng.integers(4)), (-2, -1))
    if rng.random() < .5:
        x = x.flip(-1)
    if rng.random() < .5:
        x = x.flip(-2)
    return x * float(rng.uniform(*brightness))


def metrics(x, truth, prediction, measurement):
    """No clipping; PSNR/SSIM use fixed unit range, not scene/test extrema."""
    a, b = x.cpu().numpy()[0], truth.cpu().numpy()[0]
    e = a-b
    relative = lambda u, v: float(np.linalg.norm(u)/max(np.linalg.norm(v), 1e-12))
    sam_valid = np.linalg.norm(b, axis=0) > 1e-3
    cosine = (a*b).sum(0)/np.maximum(np.linalg.norm(a, axis=0)*np.linalg.norm(b, axis=0), 1e-12)
    ssim = []
    for ai, bi in zip(a, b):
        blur = lambda z: cv2.GaussianBlur(z, (11, 11), 1.5)
        ma, mb = blur(ai), blur(bi)
        va, vb, cov = blur(ai*ai)-ma*ma, blur(bi*bi)-mb*mb, blur(ai*bi)-ma*mb
        ssim.append(float((((2*ma*mb+.01**2)*(2*cov+.03**2)) /
                           ((ma*ma+mb*mb+.01**2)*(va+vb+.03**2)))[5:-5, 5:-5].mean()))
    gy, gx = np.meshgrid(np.fft.fftfreq(a.shape[-2]), np.fft.fftfreq(a.shape[-1]), indexing="ij")
    high = np.hypot(gy, gx) >= .25
    return dict(psnr_unit_range_db=float(-10*np.log10(max(float((e*e).mean()), 1e-12))),
                ssim_band_mean=float(np.mean(ssim)),
                sam_deg=float(np.degrees(np.arccos(np.clip(cosine[sam_valid], -1, 1))).mean()),
                valid_spectral_fraction=float(sam_valid.mean()),
                map_relative_l2=relative(e, b),
                global_spectrum_relative_l2=relative(e.mean((1, 2)), b.mean((1, 2))),
                band_rmse=np.sqrt((e*e).mean((1, 2))).tolist(),
                band_relative_l2=[relative(ei, bi) for ei, bi in zip(e, b)],
                high_frequency_relative_l2=relative(np.fft.fft2(e)[..., high], np.fft.fft2(b)[..., high]),
                spatial_gradient_relative_l2=relative(
                    np.concatenate([np.diff(e, axis=-1).ravel(), np.diff(e, axis=-2).ravel()]),
                    np.concatenate([np.diff(b, axis=-1).ravel(), np.diff(b, axis=-2).ravel()])),
                measurement_relative_l2=relative((prediction-measurement).cpu().numpy(), measurement.cpu().numpy()),
                negative_fraction=float((a < 0).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    ap.add_argument("--cpu-smoke", action="store_true")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if not args.cpu_smoke and not torch.cuda.is_available():
        raise RuntimeError("CUDA required for declared D4 training; --cpu-smoke is only an execution test")
    args.output.mkdir(parents=True, exist_ok=False)
    device = "cpu" if args.cpu_smoke else "cuda"
    torch.set_num_threads(4)
    torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    records = json.loads((args.prepared/"manifest.json").read_text(encoding="utf-8"))
    data = torch.from_numpy(np.load(args.prepared/"scenes.npz")["cube"])
    if any(r["array_index"] is not None for r in records if r["split"] == "reserved_final"):
        raise ValueError("final holdout must not be present in prepared training arrays")
    execution = dict(config=cfg, cpu_smoke=args.cpu_smoke,
                     train_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     model_script_sha256=hashlib.sha256(
                         (Path(__file__).parents[2]/"src/lcd_reconstruction/unrolling/general_pgd.py").read_bytes()).hexdigest(),
                     torch_version=torch.__version__, seed=cfg["seed"],
                     tf32=False, cudnn_deterministic=True)
    (args.output/"execution.json").write_text(json.dumps(execution, indent=2), encoding="utf-8")
    # Only train and validation arrays enter fitting. Test observations are generated afterwards.
    ids = lambda split: [r["array_index"] for r in records if r["split"] == split]
    train_ids, val_ids = ids("train"), ids("val")
    if not train_ids or not val_ids or set(train_ids) & set(val_ids):
        raise ValueError("disjoint nonempty train/val required")
    kernels = np.load(args.prepared/"kernels.npz")["kernels"]
    model = GeneralSpectralPGD(kernels, tuple(data.shape[-2:]),
                              cfg["stages"], cfg["hidden"], cfg["depth"]).to(device)
    physics = model.physics
    with torch.no_grad():
        scale = torch.stack([physics(data[i:i+1].to(device)).square().mean()
                             for i in train_ids]).mean().sqrt().clamp_min(1e-8)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=1e-5)
    epochs = 2 if args.cpu_smoke else cfg["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=cfg["learning_rate"]/20)
    history, best, stale = [], float("inf"), 0
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        order = np.random.default_rng(cfg["seed"]+epoch).permutation(train_ids)
        losses = []
        for offset in range(0, len(order), cfg["batch"]):
            batch = order[offset:offset+cfg["batch"]]
            x = torch.stack([augment(data[i], cfg["seed"]+epoch*1000+int(i),
                                     cfg["brightness_range"]) for i in batch]).to(device)
            with torch.no_grad():
                y = physics(x)
                y = y + torch.randn_like(y)*scale*cfg["train_noise_rms_fraction"]
            out = model(y)
            loss = reconstruction_loss(out, x, scale)
            opt.zero_grad()
            loss.backward()
            if not torch.isfinite(loss) or not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError("nonfinite objective or gradients")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            val_loss = float(np.mean([float(reconstruction_loss(model(physics(data[i:i+1].to(device))),
                            data[i:i+1].to(device), scale)) for i in val_ids]))
        row = dict(epoch=epoch+1, train_loss=float(np.mean(losses)), validation_loss=val_loss,
                   learning_rate=opt.param_groups[0]["lr"])
        history.append(row)
        if val_loss < best:
            best, stale = val_loss, 0
            torch.save(dict(state=model.state_dict(), config=cfg, epoch=epoch+1,
                            hw=list(data.shape[-2:]), validation_loss=best), args.output/"best.pt")
        else:
            stale += 1
        scheduler.step()
        (args.output/"history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)
        if stale >= cfg["patience"]:
            break
    checkpoint = torch.load(args.output/"best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state"])
    model.eval().requires_grad_(False)
    results = []
    timings = []
    with torch.no_grad():
        for r in records:
            if r["split"] == "reserved_final":
                continue
            x = data[r["array_index"]:r["array_index"]+1].to(device)
            y = physics(x)
            if device == "cuda":
                torch.cuda.synchronize()
            tick = time.perf_counter()
            out = model(y)
            if device == "cuda":
                torch.cuda.synchronize()
            timings.append(time.perf_counter()-tick)
            row = dict(scene_id=r["scene_id"], split=r["split"],
                       n1=metrics(out["spectral_cube_hat"], x, out["reprojected_measurement"], y))
            # Fixed Landweber baseline, same measurement and full-channel state; no test tuning.
            land = physics.adjoint(y)/model.L
            for _ in range(100):
                land -= physics.adjoint(physics(land)-y)/model.L
            row["landweber100"] = metrics(land, x, physics(land), y)
            if r["split"] == "development_test":
                row["noise_sensitivity"] = {}
                for fraction in [.01, .05]:
                    noisy = y+torch.randn_like(y)*scale*fraction
                    pred = model(noisy)
                    row["noise_sensitivity"][str(fraction)] = metrics(
                        pred["spectral_cube_hat"], x, pred["reprojected_measurement"], noisy)
                np.savez_compressed(args.output/f"private_{r['scene_id']}.npz",
                    truth=x.cpu().numpy(), observation=y.cpu().numpy(),
                    reconstruction=out["spectral_cube_hat"].cpu().numpy(),
                    global_spectrum=out["global_spectrum"].cpu().numpy())
            results.append(row)
    result = dict(status="CPU execution smoke only" if args.cpu_smoke else "GPU experiment completed; quality requires review",
                  config=cfg, best_epoch=checkpoint["epoch"], results=results,
                  torch_version=torch.__version__, device=device,
                  device_name=torch.cuda.get_device_name() if device=="cuda" else "CPU",
                  parameters=sum(p.numel() for p in model.parameters()),
                  seconds_total=time.perf_counter()-started,
                  inference_seconds_median=float(np.median(timings)),
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device=="cuda" else None,
                  manifest_sha256=hashlib.sha256((args.prepared/"manifest.json").read_bytes()).hexdigest(),
                  physics_sha256=hashlib.sha256((args.prepared/"physics.json").read_bytes()).hexdigest(),
                  final_holdout_loaded=False, measurement_train_rms=float(scale))
    (args.output/"results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
