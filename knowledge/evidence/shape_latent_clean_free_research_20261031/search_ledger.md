# Targeted literature search — 2026-10-31

Purpose: distinguish clean-global headroom from usable clean-data-free
proposal direction, after bounded-prior null and positive clean-z oracle.
Primary sources only support technical claims. This is not an exhaustive
forward-citation census or a novelty search sufficient for a priority claim.

Queries included:
- diffusion inverse problems latent manifold projection PSLD gluing latent encoder2023
- diffusion test time adaptation3D point cloud global latent LION shape denoising2025 2026
- diffusion posterior sampling out of distribution initialization latent prior score DAPS ReSample
- site:arxiv.org RED-Diff variational perspective solving inverse problems diffusion models2023
- site:arxiv.org diffusion inverse problems latent reencoding denoised encoder cycle consistency2024 2025
- site:arxiv.org diffusion encoder self-consistency inverse problems2025 2026
- site:arxiv.org2026 inverse problems latent diffusion consistency
- site:arxiv.org3DD-TTA2026 diffusion
- site:arxiv.org DMPlug plugin method diffusion inverse problems2024
- site:arxiv.org Measurement-Consistent Langevin Corrector
- site:arxiv.org REED-VAE

Actually inspected primary sections:

| Version | Sections | Inclusion decision |
|---|---|---|
|2305.04391v2 RED-Diff|Sec.3, Algorithm1|Global prior as optimization regularizer, defer implementation.|
|2307.00619v1 PSLD|Sec.2 Eq.7, Sec.3|Cycle/measurement connection; linear known-operator theory not transplanted.|
|2405.16749v2 DMPlug|Sec.3.1–3.3|Optimize initial seed through fixed reverse map; expensive third candidate.|
|2501.11746v1 SILO|Sec.4.1–4.4|Corrupted encoder issue relevant; learned operator violates present no-training route.|
|2504.18989v1 REED-VAE|Sec.3–4|Reencoding drift motivates AE-only control, no decoder training proposed.|
|2601.04791v1 MCLC|Sec.3|Measurement-orthogonal corrector, missing calibrated measurement in our setting.|
|2604.11375v1 DiLO|Sec.3.2 Algorithm1|Clean-output objective/input-noise optimization; PDE result not TTA evidence.|

Revisited LION2210.06978v1 Sec.3.2 and existing pinned code audit;
reused prior LION/DPS/ReSample/DAPS record rather than calling them new findings.
Other2026 search hits (e.g. FunBCS2610.04624 and FB-GDM2609.29216) were
screened only; full papers not verified here and no technical recommendation
is based on them. Failed guessed HTML versions were replaced by actual
version metadata; accepted version URLs/bytes/SHA are in the receipts.

No identical, verified LION global-z frozen TTA method was established by
this targeted search. That statement does not assert none exists.
