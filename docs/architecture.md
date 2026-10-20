# Current D2 architecture

Authorized hyperspectral ROI -> shared spatial augmentation -> explicit dimensionless
scene convention -> measured shape-only PSF -> one scalar sensor plane -> low-rank
spatial-spectral PGD unfolding -> inspectable map/global spectrum/reprojection -> C1.

The spatial state supports consistency and task representation. Full-resolution
spectral reconstruction is not a success criterion. Forward and adjoint operators
have explicit weights, boundary and scale conventions; no CASSI substitution.

UWA groups are frozen by subject/acquisition; CAVE is held out by scene. Protocols
A/B/C separate same-domain, expanded-negative and source-confounded tests. Common
normalization is a numerical assumption, not radiometric calibration.

Initial empty subpackages and handoff notes are historical scaffolding, not
implemented services or requirements. Current scope is this document and AGENTS.md.
