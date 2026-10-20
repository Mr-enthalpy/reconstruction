# D2 method principles

Use an explicit forward/adjoint pair and a small trainable proximal correction.
The internal low-rank spatial coefficient maps produce an inspectable 21-band map,
a spatial-mean spectrum and a measurement reprojection. Spectral state is an
intermediate representation for task validity, not a high-resolution PSNR target.

Keep source/acquisition/subject/scene separation. Fit the basis only on training
originals. Match source samples and observation budget across PSF controls; an
oracle sees privileged information and is not an equal-budget sensor baseline.

Current measurements are a dimensionless, noiseless single-plane approximation.
Do not interpret shape-normalized kernels or processed dataset intensities as
absolute throughput. No arbitrary-band, multiple-exposure or future-classifier
framework is needed for this explicitly scoped 21-band pilot.
