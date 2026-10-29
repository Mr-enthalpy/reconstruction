"""D5 G2/G3 training entry with validation-only selection and resumable state."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from lcd_reconstruction.unrolling.general_pgd import GeneralSpectralPGD
from lcd_reconstruction.unrolling.hsfaut import HSFAUReconstructor, reconstruction_loss
from lcd_reconstruction.operators.psf_provenance import (
    PSFProvenanceError, array_digest, build_psf_record, physics_buffers_from_model,
    physics_buffers_from_state, resolve_psf_policy)
from metrics import reconstruction_metrics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_model(cfg, kernels, hw):
    if cfg["model"] == "A":
        return GeneralSpectralPGD(kernels, hw, cfg["stages"], cfg["width"], cfg["depth"])
    if cfg["model"] in {"B", "C"}:
        return HSFAUReconstructor(
            kernels, hw, stages=cfg["stages"], width=cfg["width"], depth=cfg["depth"],
            prior="cnn" if cfg["model"] == "B" else "sfat",
            sfat_width=cfg["sfat_width"], sfat_levels=cfg["sfat_levels"])
    raise ValueError("model must be A, B or C")


def augment(x, seed, brightness):
    rng = np.random.default_rng(seed)
    x = torch.rot90(x, int(rng.integers(4)), (-2, -1))
    if rng.random() < .5:
        x = x.flip(-1)
    if rng.random() < .5:
        x = x.flip(-2)
    return x * float(rng.uniform(*brightness))


def atomic_save(payload, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


class Sampler:
    def __init__(self, records, seed):
        self.by_source = {}
        for index, record in enumerate(records):
            self.by_source.setdefault(record["dataset_id"], []).append(index)
        self.sources = sorted(self.by_source)
        self.rng = np.random.default_rng(seed)

    def batch(self, size):
        if len(self.sources) == 1:
            return list(self.rng.choice(self.by_source[self.sources[0]], size, replace=True))
        picks = []
        for source in self.sources:
            count = max(1, size // len(self.sources))
            picks += list(self.rng.choice(self.by_source[source], count, replace=True))
        while len(picks) < size:
            picks.append(int(self.rng.choice(self.by_source[self.sources[len(picks) % len(self.sources)]])))
        self.rng.shuffle(picks)
        return picks[:size]


def evaluate_loss(model, data, records, scale):
    model.eval()
    values = []
    with torch.no_grad():
        for index, record in enumerate(records):
            x = torch.from_numpy(data[record["array_index"]:record["array_index"] + 1]).to(
                next(model.parameters()).device)
            y = model.physics(x)
            out = model(y)
            values.append(reconstruction_loss(out, x, scale)[0])
    return float(torch.stack(values).mean())


def evaluate_metrics(model, data, records):
    model.eval()
    values = {}
    with torch.no_grad():
        for record in records:
            x = torch.from_numpy(data[record["array_index"]:record["array_index"] + 1]).to(
                next(model.parameters()).device)
            y = model.physics(x)
            out = model(y)
            values.setdefault(record["dataset_id"], []).append(reconstruction_metrics(
                out["spectral_cube_hat"][0].cpu().numpy(), x[0].cpu().numpy(),
                y[0].cpu().numpy(), out["reprojected_measurement"][0].cpu().numpy()))
    keys = ["psnr_unit_range_db", "ssim_band_mean", "sam_deg", "cube_relative_l2",
            "global_spectrum_relative_l2", "high_frequency_relative_l2",
            "spatial_gradient_relative_l2", "measurement_relative_l2"]
    grouped = {}
    for source, rows in values.items():
        grouped[source] = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    grouped["ALL"] = {key: float(np.mean([row[key] for rows in values.values() for row in rows]))
                      for key in keys}
    return grouped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--kernel-file", type=Path)
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--stop-after", type=int)
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--overfit-steps", type=int, default=100)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    seed_source = "config"
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
        seed_source = "cli"
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if not args.resume and args.output.exists():
        raise FileExistsError(args.output)
    if args.resume and not args.output.exists():
        args.output.mkdir(parents=True)
    torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records = json.loads((args.prepared / "manifest.json").read_text(encoding="utf-8"))
    train = np.load(args.prepared / "train.npz")["cube"]
    val = np.load(args.prepared / "val.npz")["cube"]
    train_records = [r for r in records if r["split"] == "train"]
    val_records = [r for r in records if r["split"] == "val"]
    if any(r["split"] == "test" for r in train_records + val_records):
        raise ValueError("test records must not enter training")
    kernel_file = args.kernel_file or (args.prepared / "kernels.npz")
    kernels = np.load(kernel_file)["kernels"]
    model = build_model(cfg, kernels, tuple(train.shape[-2:])).to(args.device)
    psf_record = build_psf_record(model, kernels, kernel_file, tuple(train.shape[-2:]))
    execution = dict(config=cfg, model=cfg["model"], prepared=str(args.prepared),
                     train_manifest_sha256=sha(args.prepared / "manifest.json"),
                     train_sha256=sha(args.prepared / "train.npz"),
                     val_sha256=sha(args.prepared / "val.npz"),
                     kernel_sha256=sha(kernel_file),
                     kernel_sha256_array=psf_record["kernel_sha256_array"],
                     kernel_shape=psf_record["kernel_shape"],
                     psf_buffers=psf_record["physics_buffers"],
                     torch_version=torch.__version__, seed=cfg["seed"],
                     seed_source=seed_source,
                     device=torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
                     parameters=sum(p.numel() for p in model.parameters()),
                     tf32=False, cudnn_deterministic=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "execution.json").write_text(json.dumps(execution, indent=2), encoding="utf-8")

    if args.overfit:
        picks = [i for i, r in enumerate(train_records) if r["dataset_id"] == "CAVE"][:1]
        if not picks:
            picks = [0]
        record = train_records[picks[0]]
        x = torch.from_numpy(train[record["array_index"]:record["array_index"] + 1]).to(args.device)
        y = model.physics(x)
        scale = y.square().mean().sqrt().clamp_min(1e-8)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=1e-5)
        history = []
        start = time.perf_counter()
        for step in range(args.overfit_steps):
            model.train()
            out = model(y)
            loss, parts = reconstruction_loss(out, x, scale)
            opt.zero_grad()
            loss.backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError("nonfinite gradients")
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            if step == 0 or (step + 1) % 10 == 0:
                history.append(dict(step=step + 1, **parts))
        payload = dict(state=model.state_dict(), config=cfg, model=cfg["model"],
                       hw=list(train.shape[-2:]), device=args.device,
                       source_scene=record["scene_id"])
        atomic_save(payload, args.output / "overfit.pt")
        restored = build_model(cfg, kernels, tuple(train.shape[-2:])).to(args.device)
        restored.load_state_dict(torch.load(args.output / "overfit.pt", map_location=args.device,
                                            weights_only=True)["state"])
        restored.eval(); model.eval()
        with torch.no_grad():
            before = model(y)["spectral_cube_hat"]
            after = restored(y)["spectral_cube_hat"]
        result = dict(mode="single-sample overfit", scene=record["scene_id"],
                      steps=args.overfit_steps, history=history,
                      initial_loss=history[0]["total"], final_loss=history[-1]["total"],
                      loss_decreased=history[-1]["total"] < history[0]["total"],
                      checkpoint_reload_max_abs=float((before - after).abs().max()),
                      seconds=time.perf_counter() - start,
                      parameters=execution["parameters"])
        (args.output / "overfit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result))
        return

    sampler = Sampler(train_records, cfg["seed"])
    with torch.no_grad():
        probe = torch.cat([torch.from_numpy(train[r["array_index"]:r["array_index"] + 1])
                           for r in train_records[:min(8, len(train_records))]]).to(args.device)
        scale = model.physics(probe).square().mean().sqrt().clamp_min(1e-8)
    execution.update(
        measured_forward_rms=float(scale),
        noise_probe_scenes=min(8, len(train_records)),
        train_noise_rms_fraction=float(cfg["train_noise_rms_fraction"]),
        train_noise_std_absolute=float(scale) * float(cfg["train_noise_rms_fraction"]),
        cosine_epochs=int(cfg["epochs"]),
        steps_per_epoch=int(cfg["steps_per_epoch"]),
        noise_note="noise scale is this arm's own forward RMS; the same scalar also "
                   "normalises the data loss, so validation loss is not comparable across arms")
    (args.output / "execution.json").write_text(json.dumps(execution, indent=2), encoding="utf-8")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                            weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, cfg["epochs"], eta_min=cfg["learning_rate"] / 20)
    start_epoch, best, stale, best_epoch, global_step = 0, float("inf"), 0, 0, 0
    eval_every = int(cfg.get("eval_every", 5))
    history = []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device, weights_only=True)
        psf_policy = resolve_psf_policy(
            checkpoint.get("psf_provenance"), physics_buffers_from_model(model),
            physics_buffers_from_state(checkpoint["state"]),
            allow_mismatch=False, requested_array=array_digest(kernels))
        model.load_state_dict(checkpoint["state"])
        if physics_buffers_from_model(model) != psf_policy["stored"]:
            raise PSFProvenanceError("operator changed after loading the checkpoint state")
        print("D5.2 stage: psf", psf_policy["policy"], flush=True)
        opt.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint.get("next_epoch", checkpoint["epoch"])
        best = checkpoint["best"]
        stale = checkpoint["stale"]
        best_epoch = checkpoint.get("best_epoch", 0)
        global_step = checkpoint.get("global_step", start_epoch * cfg["steps_per_epoch"])
        history = json.loads((args.output / "history.json").read_text(encoding="utf-8"))
        torch.set_rng_state(checkpoint["torch_rng"].to(torch.uint8).cpu())
        if args.device == "cuda" and "cuda_rng" in checkpoint:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
        sampler.rng.bit_generator.state = checkpoint["numpy_rng"]
    started = time.perf_counter()
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        losses, parts = [], []
        for step in range(cfg["steps_per_epoch"]):
            indices = sampler.batch(cfg["batch"])
            x = torch.stack([augment(torch.from_numpy(train[train_records[i]["array_index"]]),
                                     cfg["seed"] + epoch * 100000 + step * 100 + j,
                                     cfg["brightness_range"])
                             for j, i in enumerate(indices)]).to(args.device)
            with torch.no_grad():
                y = model.physics(x)
                y = y + torch.randn_like(y) * scale * cfg["train_noise_rms_fraction"]
            out = model(y)
            loss, part = reconstruction_loss(
                out, x, scale, cfg["lambda_grad"], cfg["lambda_spectral"], cfg["lambda_data"])
            opt.zero_grad()
            loss.backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError("nonfinite gradients")
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            global_step += 1
            losses.append(float(loss.detach()))
            parts.append(part)
        val_loss = evaluate_loss(model, val, val_records, scale)
        improved = val_loss < best
        if improved:
            best, stale, best_epoch = val_loss, 0, epoch + 1
        else:
            stale += 1
        scheduler.step()
        row = dict(epoch=epoch + 1, global_step=global_step, train_loss=float(np.mean(losses)),
                   validation_loss=val_loss, best=best, best_epoch=best_epoch, stale=stale,
                   learning_rate=opt.param_groups[0]["lr"],
                   **{key: float(np.mean([p[key] for p in parts])) for key in parts[0]})
        metric_roles = []
        if (epoch + 1) % eval_every == 0:
            metric_roles.append("cadence")
        if improved:
            metric_roles.append("best")
        if (epoch + 1 == cfg["epochs"] or stale >= cfg["patience"]
                or (args.stop_after and epoch + 1 >= args.stop_after)):
            metric_roles.append("final")
        if metric_roles:
            row["metrics"] = evaluate_metrics(model, val, val_records)
            row["metrics_roles"] = metric_roles
        history.append(row)
        payload = dict(state=model.state_dict(), optimizer=opt.state_dict(),
                       scheduler=scheduler.state_dict(), config=cfg, model=cfg["model"],
                       epoch=epoch + 1, next_epoch=epoch + 1, global_step=global_step,
                       hw=list(train.shape[-2:]), best=best, best_epoch=best_epoch,
                       stale=stale, validation_loss=val_loss, scale=float(scale),
                       psf_provenance=psf_record,
                       selection=dict(rule="min_validation_loss", epoch=epoch + 1,
                                      is_best=bool(improved), validation_loss=val_loss,
                                      best_epoch=best_epoch, best_validation_loss=best,
                                      metrics=row.get("metrics")),
                       torch_rng=torch.get_rng_state(),
                       cuda_rng=torch.cuda.get_rng_state_all() if args.device == "cuda" else None,
                       numpy_rng=sampler.rng.bit_generator.state)
        atomic_save(payload, args.output / "last.pt")
        if improved:
            atomic_save(payload, args.output / "best.pt")
        (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)
        if args.stop_after and epoch + 1 >= args.stop_after:
            break
        if stale >= cfg["patience"]:
            break
    checkpoint = torch.load(args.output / "best.pt", map_location=args.device, weights_only=True)
    if physics_buffers_from_state(checkpoint["state"]) != physics_buffers_from_model(model):
        raise PSFProvenanceError("best.pt operator differs from the training operator")
    model.load_state_dict(checkpoint["state"])
    model.eval().requires_grad_(False)
    results = []
    with torch.no_grad():
        for record in val_records:
            x = torch.from_numpy(val[record["array_index"]:record["array_index"] + 1]).to(args.device)
            y = model.physics(x)
            out = model(y)
            results.append(dict(scene_id=record["scene_id"], dataset_id=record["dataset_id"],
                                metrics=reconstruction_metrics(
                                    out["spectral_cube_hat"][0].cpu().numpy(),
                                    x[0].cpu().numpy(), y[0].cpu().numpy(),
                                    out["reprojected_measurement"][0].cpu().numpy())))
    recorded_metrics = (checkpoint.get("selection") or {}).get("metrics")
    consistency = None
    if recorded_metrics:
        recomputed = {key: float(np.mean([r["metrics"][key] for r in results]))
                      for key in recorded_metrics.get("ALL", {})}
        differences = {key: abs(recomputed[key] - float(recorded_metrics["ALL"][key]))
                       for key in recomputed if recorded_metrics["ALL"].get(key) is not None}
        consistency = dict(
            recorded_epoch=checkpoint.get("selection", {}).get("epoch"),
            rule=checkpoint.get("selection", {}).get("rule"),
            max_abs_difference=max(differences.values()) if differences else None,
            consistent=all(value <= max(1e-9, 1e-6 * abs(float(recorded_metrics["ALL"][key])))
                           for key, value in differences.items()))
    summary = dict(status="training completed; validation only",
                   best_epoch=checkpoint["epoch"], best_validation_loss=checkpoint["validation_loss"],
                   results=results, seconds=time.perf_counter() - started,
                   selection=dict(rule="min_validation_loss", best_epoch=checkpoint["epoch"],
                                  best_validation_loss=checkpoint["validation_loss"],
                                  checkpoint=str(args.output / "best.pt"),
                                  checkpoint_sha256=sha(args.output / "best.pt"),
                                  metrics_in_checkpoint=recorded_metrics),
                   consistency=consistency,
                   psf_provenance=checkpoint.get("psf_provenance", psf_record),
                   execution=execution)
    (args.output / "validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(dict(best_epoch=checkpoint["epoch"], best=checkpoint["validation_loss"],
                          seconds=summary["seconds"])))


if __name__ == "__main__":
    main()







