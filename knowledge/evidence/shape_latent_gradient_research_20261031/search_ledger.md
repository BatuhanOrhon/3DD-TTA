# Search and source ledger — 2026-10-31

[User report] Find an inference-clean-data-free gradient that guides global
LION diffusion toward useful clean z, with classifier accuracy as primary.
[Inference] Targeted narrative review, not exhaustive/systematic coverage.
Primary papers and official author code only support technical claims.

## Search families

- Continuous condition/prompt optimization through frozen diffusion.
- Conditional diffusion likelihood and noise-prediction energy as a discriminator.
- Latent diffusion inverse-problem guidance; noisy versus clean derivatives.
- 2025–2026 prompt-gradient flows and anisotropic native-latent guidance.
- Point-cloud TTA/LION global-latent guidance and unknown corruption limits.

Recorded exact late-stage queries (earlier exploratory searches are represented
by the families above, not reconstructed as an exact transcript):

1. `"conditional diffusion" "condition" denoising loss optimization likelihood z test time latent 2025 2026`
2. `"3DD-TTA" "latent" guidance 2026`
3. `"LION" "global" "latent" "guidance" diffusion test-time adaptation`

Followed primary references from P2L/LATINO-PRO/CWGF; searched Diffusion-TTA,
Diffusion Classifier and ANaLOG directly. Existing local clean-free report's
conditional-energy and DMPlug proposals were read before reprioritization.

## Selected primary sections

| Receipt ID | Examined source / sections | Claim boundary |
|---|---|---|
| diffusion_tta | arXiv2311.16102v2 §3.2/Eq.2/Alg.1, implementation details, §4.4/Tab.5 | Condition gradient/variance; paper updates model weights, our z-only scheme is a transfer. |
| diffusion_classifier | arXiv2303.16203v2 §3.2/Eq.8, §3.3/Eq.9, §4.1 | Paired (t,epsilon) and energy ranking, not calibrated LION likelihood. |
| cwgf | arXiv2605.07907v1 PDFpp.3–7 §3/Eq.8–25, especially §3.2/Eq.20–21 | Conditional score-matching gradient is directly relevant; full LCM/linear-measurement algorithm and16NFE claims are not transferred. |
| p2l / p2l_icml_metadata | arXiv2310.01110v1 §3/Alg.1–4 plus official ICML2024 PMLR record | Optimize condition C; don't confuse it with the paper's latent z. Readable full text is v1, not asserted final proceedings version. |
| latino_pro | arXiv2503.12615v1 §5/Eq.7–9, App.C; §4.3 for LCM | Prompt MMLE/Fisher identity;8NFE fixed prompt differs from68NFE PRO. |
| dps | arXiv2209.14687v4 §3/Eq.4–7 | Differentiate clean-estimate likelihood to noisy state; data energy still needed. |
| dmplug | arXiv2405.16749v2 §3.1/3.3 | Reverse-chain seed optimization and noise overfitting; no new LION result. |
| cloudfixer | arXiv2407.16193v1 §4.2/Eq.1 | Stop-gradient target for geometric transformation, not condition-gradient replacement. |
| analog | arXiv2609.31933v1 §3/Eq.4–7, App.F.4/G | Learned mean/covariance and limitations; not training-free plug-in. |
| conditional_likelihood_caution | Cross/Ragni PMLR255 PDF §4.2/§5/Limitations, pp.7–9 | Exact conditional likelihood behavior in two image/audio models; not proof about LION's MSE surrogate. |
| difftta_code | official mihirp1998/Diffusion-TTA pinned eedeb6e, diff_tta/models/tta.py | context→noise prediction→MSE graph; inspected, not executed. Git blob byte identity checked. |

## Retrieval and exclusions

[Run] Primary HTML/PDFs were downloaded locally and SHA256 recorded. CWGF
versioned PDF was retrieved locally successfully even when web parsing timed
out; original equations were inspected from PDF/extracted text. Some arXiv
HTML revisions and CVF pages returned404/403; working pinned arXiv full texts
are used, not inferred unavailable revisions. P2L's conference status is
verified separately from its v1 text.

[Paper/Inference] LATINO-PRO/CWGF whole pipelines require LCM/known measurement
assumptions; their **condition-gradient concept** remains relevant. ANaLOG's
extra training conflicts with this task's protocol. Other review hits on
time-series/tabular/training changes were not promoted. ICLR2026 Condition
Errors Refinement was screened at abstract level only: autoregressive image
generation/OT distribution correction is outside the current frozen LION
instance-level gradient question; no full-text conclusion is attributed.
Previous PSLD/RED-Diff/SILO/REED-VAE/MCLC/DiLO notes remain context and are
not represented as newly revalidated sources in this receipt set.

[Open] No primary paper in this selected set establishes clean-global-z
recovery or improved Point-MAE accuracy for frozen LION under unknown
ModelNet40-C corruptions. No novelty or convergence guarantee is claimed.
