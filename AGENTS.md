# Agent Instructions

This repository is currently in architecture-initialization mode.

Do not implement reconstruction algorithms unless explicitly scoped.

Do not invent final handoff schemas.

Do not import `optic_system`, `LCD_forward`, `lcd_mask_families`, or external repository internals during initialization.

Do not use dense PSF tensors as the default reconstruction contract.

Do not hard-code `L=3`.

Do not treat `T` as video time.

Do not add video reconstruction assumptions, motion modelling, optical flow, temporal dynamics, temporal consistency losses, or video restoration priors unless explicitly scoped later.

Do not add black-box frames-to-spectrum networks as the mainline.

Do keep placeholders empty and well documented.

Do keep the package import test passing.

Use the active package name `lcd_reconstruction`.
