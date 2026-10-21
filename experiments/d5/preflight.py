"""D5 G1 forward/adjoint contract and operator benchmark."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import torch

from lcd_reconstruction.operators.measured_psf import MeasuredPSFOperator
from lcd_reconstruction.unrolling.full_spectral import SinglePlanePhysics
from lcd_reconstruction.unrolling.spatial_pgd import FFTMeasuredOperator


def rel(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-15))


def check(name, value, threshold, detail=None):
    return dict(name=name, value=value, threshold=threshold,
                passed=bool(value <= threshold), detail=detail or {})


def numeric_checks():
    checks = []
    rng = np.random.default_rng(20261021)
    k = rng.normal(size=(21, 4, 6)) * 0.01
    x = rng.normal(size=(21, 7, 9))
    y = rng.normal(size=(7, 9))
    ref = MeasuredPSFOperator(k, np.ones(21))
    op = FFTMeasuredOperator(torch.tensor(k), (7, 9))
    fwd = op(torch.tensor(x)).numpy()
    adj = op.adjoint(torch.tensor(y)).numpy()
    impulse = np.zeros((21, 7, 9))
    impulse[7, 3, 4] = 1
    checks.append(check("single_band_impulse_vs_reference",
                        rel(op(torch.tensor(impulse)).numpy(), ref.forward(impulse)), 1e-11))
    checks.append(check("adjoint_vs_reference", rel(adj, ref.adjoint(y)), 1e-11))

    parts = sum((op(torch.tensor(x[b:b + 1] * np.eye(21)[b][:, None, None])).numpy()
                 for b in range(21)), np.zeros_like(fwd))
    checks.append(check("21_band_linear_superposition", rel(parts, fwd), 1e-11))

    lhs = float((op(torch.tensor(x)) * torch.tensor(y)).sum())
    rhs = float((torch.tensor(x) * op.adjoint(torch.tensor(y))).sum())
    checks.append(check("forward_adjoint_inner_product", abs(lhs - rhs) /
                        max(abs(lhs), abs(rhs), 1e-15), 1e-11))

    gpu_available = torch.cuda.is_available()
    if gpu_available:
        cpu = op(torch.tensor(x)).numpy()
        op_gpu = FFTMeasuredOperator(torch.tensor(k), (7, 9)).cuda()
        gpu = op_gpu(torch.tensor(x, device="cuda")).cpu().numpy()
        checks.append(check("cpu_gpu_forward", rel(gpu, cpu), 1e-6))
        cpu_a = op.adjoint(torch.tensor(y)).numpy()
        gpu_a = op_gpu.adjoint(torch.tensor(y, device="cuda")).cpu().numpy()
        checks.append(check("cpu_gpu_adjoint", rel(gpu_a, cpu_a), 1e-6))

    k64 = torch.tensor(k, dtype=torch.float64)
    op64 = SinglePlanePhysics(k64, (7, 9))
    x64 = torch.tensor(x[None], dtype=torch.float64, requires_grad=True)
    y64 = torch.tensor(y[None, None], dtype=torch.float64)
    d = torch.tensor(rng.normal(size=x64.shape), dtype=torch.float64)
    f = lambda z: 0.5 * (op64(z) - y64).square().sum()
    grad, = torch.autograd.grad(f(x64), x64)
    numerical = (f(x64 + 1e-6 * d) - f(x64 - 1e-6 * d)) / 2e-6
    checks.append(check("directional_derivative_fp64", abs(float(numerical - (grad * d).sum())) /
                        max(abs(float((grad * d).sum())), 1e-15), 1e-7))

    even = torch.tensor(k, dtype=torch.float64)
    even_op = FFTMeasuredOperator(even, (7, 9))
    checks.append(check("asymmetric_even_kernel_reference",
                        rel(even_op(torch.tensor(x)).numpy(), ref.forward(x)), 1e-11))

    boundary = np.zeros((21, 7, 9))
    for y0, x0 in ((0, 0), (0, 8), (3, 4), (6, 8)):
        boundary[:] = 0
        boundary[10, y0, x0] = 1
        checks.append(check(f"boundary_impulse_{y0}_{x0}",
                            rel(FFTMeasuredOperator(even, (7, 9))(torch.tensor(boundary)).numpy(),
                                ref.forward(boundary)), 1e-11))

    batch = torch.stack([torch.tensor(x), torch.tensor(x) * 0.5, torch.tensor(x) * 2.0])
    out = op(batch)
    checks.append(check("batch_consistency", max(
        rel(out[i].numpy(), op(torch.tensor(x) * s).numpy())
        for i, s in enumerate((1.0, 0.5, 2.0))), 1e-11))
    return checks


def measured_checks(prepared):
    checks = []
    kernels = np.load(Path(prepared) / "kernels.npz")["kernels"]
    k = torch.tensor(kernels)
    hw = (128, 128)
    op = SinglePlanePhysics(k, hw)
    x = torch.rand(1, 21, *hw, generator=torch.Generator().manual_seed(7))
    y = op(x)
    test = torch.randn(y.shape, generator=torch.Generator().manual_seed(8))
    lhs = float((y * test).sum())
    rhs = float((x * op.adjoint(test)).sum())
    checks.append(check("measured_kernel_adjoint", abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-15), 1e-4))
    ref = MeasuredPSFOperator(kernels, np.ones(21))
    checks.append(check("measured_kernel_fft_vs_scipy",
                        rel(y[0, 0].numpy(), ref.forward(x[0].numpy())), 1e-4))
    if torch.cuda.is_available():
        gpu = SinglePlanePhysics(k.cuda(), hw)
        checks.append(check("measured_kernel_cpu_gpu",
                            rel(gpu(x.cuda()).cpu()[0, 0].numpy(), y[0, 0].numpy()), 1e-4))
    return checks


def benchmark(prepared, sizes, batches):
    kernels = np.load(Path(prepared) / "kernels.npz")["kernels"]
    kernels1024 = np.load(Path(prepared) / "kernels1024.npz")["kernels"]
    rows = []
    if not torch.cuda.is_available():
        return rows
    for support, kernel in ((512, kernels), (1024, kernels1024)):
        for size in sizes:
            for batch in batches:
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    op = FFTMeasuredOperator(torch.tensor(kernel), (size, size)).cuda()
                    x = torch.rand(batch, 21, size, size, device="cuda")
                    y = torch.rand(batch, size, size, device="cuda")
                    for _ in range(2):
                        op(x); op.adjoint(y)
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(5):
                        op(x); op.adjoint(y)
                    end.record()
                    torch.cuda.synchronize()
                    rows.append(dict(support_sensor_pixels=support, size=size, batch=batch,
                                     forward_adjoint_ms=start.elapsed_time(end) / 5,
                                     peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                                     status="ok"))
                except RuntimeError as exc:
                    rows.append(dict(support_sensor_pixels=support, size=size, batch=batch,
                                     status="error", error=str(exc)))
                    torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--sizes", default="128,256,512")
    ap.add_argument("--batches", default="1,2")
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    checks = numeric_checks() + measured_checks(args.prepared)
    sizes = [int(v) for v in args.sizes.split(",")]
    batches = [int(v) for v in args.batches.split(",")]
    script = Path(__file__).resolve()
    prepared_checks = Path(args.prepared) / "checks.json"
    result = dict(
        protocol="D5-G1-forward-validation",
        preflight_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
        prepared_checks_sha256=hashlib.sha256(prepared_checks.read_bytes()).hexdigest(),
        git_commit=subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
        cuda_available=torch.cuda.is_available(),
        device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        torch_version=torch.__version__,
        checks=checks,
        benchmark=benchmark(args.prepared, sizes, batches),
        note="Numerical identity is not optical validation; zero-boundary scalar shape-only model.",
    )
    result["all_passed"] = all(c["passed"] for c in checks)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(dict(all_passed=result["all_passed"],
                          failed=[c["name"] for c in checks if not c["passed"]],
                          benchmark=len(result["benchmark"]))))


if __name__ == "__main__":
    main()




