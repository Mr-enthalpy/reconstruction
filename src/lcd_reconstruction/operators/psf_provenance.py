"""PSF provenance: canonical kernel identity and physics-buffer digests.

A checkpoint stores the physics operator as buffers (``kernel_fft``, ``weights``,
``L``). Loading such a state into a model that was built from a different kernel
file silently replaces those buffers, so a report can name one kernel file while
the run used another. This module provides the identity records and the single
policy decision used by training, evaluation and mismatch diagnostics.

Kernel identity is the canonical array digest (float64, C order) rather than the
``.npz`` file hash, because two containers of the same array hash differently
(zip timestamps are embedded in the file).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

PROTOCOL = "d5.2-psf-v1"
PHYSICS_BUFFER_NAMES = ("L",)
PHYSICS_BUFFER_SUFFIXES = ("kernel_fft", "weights")


class PSFProvenanceError(RuntimeError):
    """Raised when the checkpoint operator and the requested operator disagree."""


def array_digest(array) -> str:
    a = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    h = hashlib.sha256()
    h.update(repr(tuple(a.shape)).encode("ascii"))
    h.update(a.tobytes())
    return h.hexdigest()


def file_digest(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_digest(tensor) -> str:
    t = torch.as_tensor(tensor).detach().to("cpu").contiguous()
    if t.is_complex():
        t = torch.view_as_real(t)
    h = hashlib.sha256()
    h.update(repr(tuple(t.shape)).encode("ascii"))
    h.update(str(t.dtype).encode("ascii"))
    h.update(t.numpy().tobytes())
    return h.hexdigest()


def is_physics_key(name: str) -> bool:
    leaf = name.rsplit(".", 1)[-1]
    return leaf in PHYSICS_BUFFER_NAMES or leaf.endswith(PHYSICS_BUFFER_SUFFIXES)


def physics_buffers_from_state(state) -> dict:
    return {k: tensor_digest(v) for k, v in state.items()
            if is_physics_key(k) and torch.is_tensor(v)}


def physics_buffers_from_model(model) -> dict:
    return physics_buffers_from_state(model.state_dict())


def build_psf_record(model, kernels, kernel_path, hw=None) -> dict:
    return dict(
        protocol=PROTOCOL,
        kernel_file=Path(kernel_path).name,
        kernel_sha256_file=file_digest(kernel_path),
        kernel_sha256_array=array_digest(kernels),
        kernel_shape=list(np.shape(kernels)),
        hw=list(hw) if hw is not None else None,
        physics_buffers=physics_buffers_from_model(model),
    )


def _summary(digests) -> str:
    return "{" + ", ".join(f"{k}={v[:12]}" for k, v in sorted(digests.items())) + "}"


def resolve_psf_policy(recorded, requested: dict, stored: dict, allow_mismatch=False,
                       split=None, requested_array=None) -> dict:
    """Decide how a checkpoint may be used with the requested kernel file.

    ``requested`` is the digest of the operator built from the requested file and
    ``stored`` is the digest of the operator inside the checkpoint state. The
    returned ``use_checkpoint_operator`` tells the caller which buffers must be
    active afterwards, so nothing is ever substituted silently.
    """
    if recorded is not None:
        claimed = dict(recorded.get("physics_buffers") or {})
        if claimed != stored:
            raise PSFProvenanceError(
                "checkpoint is internally inconsistent: its stored operator does not "
                "match the operator recorded in psf_provenance "
                f"(recorded {_summary(claimed)} vs stored {_summary(stored)})")

    matched = requested == stored
    if matched:
        if recorded is not None and requested_array is not None and \
                recorded.get("kernel_sha256_array") != requested_array:
            raise PSFProvenanceError(
                "checkpoint psf_provenance does not describe the requested kernel "
                f"(recorded array {str(recorded.get('kernel_sha256_array'))[:12]} vs "
                f"requested {requested_array[:12]})")
        policy = "matched" if recorded is not None else "matched-legacy-no-record"
        return dict(policy=policy, formal_result=recorded is not None,
                    use_checkpoint_operator=True,
                    reason="requested operator equals the checkpoint operator"
                           + ("" if recorded is not None else
                              "; checkpoint predates psf provenance records"),
                    requested=requested, stored=stored,
                    recorded_kernel_file=(recorded or {}).get("kernel_file"),
                    recorded_array=(recorded or {}).get("kernel_sha256_array"))

    if not allow_mismatch:
        raise PSFProvenanceError(
            "requested kernel file does not match the checkpoint operator; refusing to "
            "substitute silently. Requested "
            f"{_summary(requested)} vs checkpoint {_summary(stored)}. "
            "Use the matching kernel file, or pass the explicit mismatch interface "
            "(--allow-psf-mismatch) and treat the result as a diagnostic."
            + ("" if recorded is None else
               f" Recorded training kernel: {recorded.get('kernel_file')} "
               f"(array {str(recorded.get('kernel_sha256_array'))[:12]})."))

    if split == "test":
        raise PSFProvenanceError("declared PSF mismatch is not allowed on the frozen test split")

    return dict(policy="declared-mismatch", formal_result=False,
                use_checkpoint_operator=False,
                reason="operator from the requested kernel file is used; checkpoint physics "
                       "buffers are not loaded",
                requested=requested, stored=stored,
                recorded_kernel_file=(recorded or {}).get("kernel_file"),
                recorded_array=(recorded or {}).get("kernel_sha256_array"))
