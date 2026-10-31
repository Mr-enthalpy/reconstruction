# Scientific decision — main sweeps complete, baseline sensitivity pending

The decision is conditional on noise, prior information and the tested perturbation
family. A single unconditional A/B/C/D/E label would hide the central result.
All evidence remains validation-only measured-PSF simulation, not a new hardware
mechanism experiment. The 14-scene frozen test was not loaded.

## Q1, Q2 and Q3

| Condition | Known-template detection | True-baseline, known-direction map inverse | Frozen A6 | Interpretation |
|---|---|---|---|---|
| No noise | Chromatic templates identifiable; achromatic blind | 27/27 groups pass, each in 9/9 scenes | No group passes in any seed | C, limited to this privileged noiseless problem |
| 0.1% noise | 25/27 groups have ideal scalar SNR >=3 in at least 8/9 scenes | 0/27 groups pass | No group passes | B for detectable groups under this fixed estimator/prior |
| 0.5% noise | 16/27 groups meet that scalar SNR description | 0/27 groups pass | No group passes; response largely noise-driven | B for detectable groups; no target-noise neural-capacity gap established |

Groups below the scalar SNR description have weaker ideal detectability; this
does not prove all their possible observations are indistinguishable. A failed
regularized inverse is not an information-theoretic impossibility theorem.
Condition D is unsupported: the new construction does not yield successful A6
recovery. E was not tested because no enhanced network was trained.

The restricted inverse's scene-balanced relative L2 is 0.074716 / 0.951518 /
1.445389 across the three noise levels. Every Chromatic solve converged, so the
noisy failure is not explained by the final iteration cap. However, regularization
was selected from a finite grid on only three CAVE training scenes, and noisy
optima lie at the largest tested gradient weight. Those limitations prevent
claiming the best possible inverse has failed. The grid was not expanded using
validation outcomes.

The Oracle knows X0 and the spectral direction; A6 does not. A6 also uses float32,
and its 0/0.1% noise inputs shift its 0.5%-noise training distribution. Consequently
the noiseless difference supports further diagnosis, but does not identify whether
capacity, training distribution, objective or optimization is causal.

## Six scientific propositions kept separate

| Proposition | Evidence admitted now | Boundary / negative evidence |
|---|---|---|
| Spatial structure recovery | Qualified D5.3 SSIM/gradient/residual stability result remains | PSNR did not pass every D5.3 criterion; frequency-matched control still absent |
| Per-band energy recovery | D5.3 per-band results retain their original qualifications | D5.4 does not turn those into local spectral-shape recovery |
| Pixelwise spectral shape | No new positive general-network claim | D5.3 SAM worsened; D5.4 directional response is poor |
| Fixed-wideband difference detection | Chromatic responds; ideal known-template detection passes some conditions | True template/baseline privilege; one deterministic noise realization; not empirical blind detection |
| Fixed-wideband difference inversion | Successful noiseless known-direction map inverse | No noisy group passes; A6 fails all noise levels; no general 21-band recovery theorem |
| Capacity and training objective | Existing update path and actual loss audited | No architecture or loss ablation; no causal capacity conclusion |

The natural-perturbation labels are training-statistics diagnostics, not proof of
strict distribution membership. The three PCA directions and three masks do not
cover all natural spectral changes. No material, face, liveness or classification
experiment was performed. No conclusion extends to a commercial phone.

## Next research action and S4

Retain A6 as the existing full-21-band reconstruction reference. There is a sound
reason to investigate N1's use of differential information, but insufficient
evidence to spend this round's budget on A-large/C6 formal training. At the target
noise, prioritize the detection-to-local-inversion gap, coding/noise conditions
and constrained-prior diagnostics. For the noiseless gap, first separate noise
distribution and baseline/direction privilege from model capacity. Any matched
noise retraining would be a separately frozen variable with matched controls.

The predeclared baseline-substitution test is being run on every noiseless group
that passed, using the fixed first Chromatic seed. It grants an extra clean
reference exposure. Direct measurement subtraction with that reference would
already reproduce the ideal residual, so this test diagnoses avoidable A6
baseline-estimation bias rather than the fundamental limit of a less privileged
single-shot inverse. Its results must retain that narrower interpretation.

No dynamic display, classification, real-phone transfer, new model output
representation or Meta pin update is proposed by these results.

## G4 recommendation

- Keep A6 as the formal general-reconstruction baseline candidate; no D5.4 model
  replacement has earned selection.
- Spatial reconstruction evidence and local spectral-difference recovery evidence
  are now explicitly separated. Only the qualified validation findings above may
  enter a paper, with measured-PSF simulation and Oracle privilege stated.
- Do not claim practical local spectral gain, classification gain, hardware noise
  calibration, complete frequency-response exclusion or a capacity diagnosis.
- Do not open G4 during D5.4. An independent G4 protocol can reasonably test the
  already frozen general-reconstruction models even if local spectral recovery
  remains limited. First freeze its exact models, controls, noise, metrics and
  restricted claims in a separate research decision. The current controls are
  insufficient for a G4 claim attributing all spatial benefit to spectral encoding.

No test result or test-level generalization claim is made. Further capacity or
test work is not automatically authorized by finishing these diagnostics.

Evidence: `diagnostic_gates.json`, `restricted_inverse_summary.json`,
`frozen_a6_strata.json`, `observability_strata.json`, `inverse_conditioning.md`,
`spectral_response_analysis.md`, and the D5.3 evidence amendment. These retain
source hashes, original-scene strata and model seeds separately.
