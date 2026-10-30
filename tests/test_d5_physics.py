import importlib.util
from pathlib import Path
import pytest

torch = pytest.importorskip("torch")


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "experiments" / "d5" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"d5_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_forward_adjoint_contract_bundle():
    checks = load_script("preflight").numeric_checks()
    failed = [c for c in checks if not c["passed"]]
    assert not failed, failed
    assert {c["name"] for c in checks} >= {
        "single_band_impulse_vs_reference", "adjoint_vs_reference",
        "21_band_linear_superposition", "forward_adjoint_inner_product",
        "directional_derivative_fp64", "asymmetric_even_kernel_reference",
        "batch_consistency",
    }


def test_integration_reduces_band_separability():
    import numpy as np
    from lcd_reconstruction.operators.measured_psf_source import load_measured_kernel
    from pathlib import Path
    root = Path(__file__).resolve().parents[2] / "optic_system" / "outputs" / "phase_c" / "spectral21_20261018_reset01"
    if not root.exists():
        pytest.skip("measured PSF source is not available")
    signed = []
    integrated = []
    for nm in range(450, 651, 10):
        k, ki, _ = load_measured_kernel(root, nm, 512, 8)
        signed.append(k)
        integrated.append(ki)
    signed = np.stack(signed)
    integrated = np.stack(integrated)

    def mean_offdiag(a):
        c = np.corrcoef(a.reshape(21, -1))
        return float(c[~np.eye(21, dtype=bool)].mean())

    assert mean_offdiag(integrated) > mean_offdiag(signed)
