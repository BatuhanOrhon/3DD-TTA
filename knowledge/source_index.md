# Source Index and Citation Map

## Primary papers

### 3DD-TTA

- Local PDF: `Test-Time_Adaptation_of_3D_Point_Clouds_via_Denoising_Diffusion_Models kopyası.pdf`
- Official paper page: <https://openaccess.thecvf.com/content/WACV2025/html/Dastmalchi_Test-Time_Adaptation_of_3D_Point_Clouds_via_Denoising_Diffusion_Models_WACV_2025_paper.html>
- Citation: A. Dastmalchi et al., “Test-Time Adaptation of 3D Point Clouds via Denoising Diffusion Models,” WACV 2025.
- Local note: [papers/3dd_tta.md](papers/3dd_tta.md)

### GSDTTA

- Local PDF: `Wei_3D_Test-time_Adaptation_via_Graph_Spectral_Driven_Point_Shift_ICCV_2025_paper.pdf`
- Official paper page: <https://openaccess.thecvf.com/content/ICCV2025/html/Wei_3D_Test-time_Adaptation_via_Graph_Spectral_Driven_Point_Shift_ICCV_2025_paper.html>
- Citation: Y. Wei et al., “3D Test-time Adaptation via Graph Spectral Driven Point Shift,” ICCV 2025.
- Local note: [papers/gsdtta.md](papers/gsdtta.md)

### PixelAsParam

- Local PDFs: `PixelAsParam_ A Gradient View on Diffusion Sampling with Guidance.pdf` and repository copy `dinh23a.pdf`
- Official paper page: <https://proceedings.mlr.press/v202/dinh23a.html>
- Citation: T. M. Dinh et al., “Pixel-as-Param: A Gradient View on Diffusion Sampling with Guidance,” ICML 2023 / PMLR 202.
- Local note: [papers/pixelasparam.md](papers/pixelasparam.md)

### LION dependency

- Local PDF: `lion.pdf`
- Role: hierarchical global/local latent diffusion model used by 3DD-TTA and this fork. It is a dependency, not one of the three primary synthesis papers.

## Primary repository evidence

**Branch context, 2026-09-12:** active clean branch is `baseline-repro-clean` from main `107305f`. The GSD/dynamic/PxP paths listed below describe the preserved legacy branch at `53ba252`, not code imported into the clean branch. Four root source PDFs were preserved locally as untracked files; official source links remain available when cloning the documentation-only branch without those local files.

- `README.md`: upstream usage and prepublication result table.
- `tta.py`, `main_3dd_tta.py`, `utilities_3dd_tta.py`: original 3DD-TTA path.
- `graph_spectral.py`, `tta_gsd.py`, `eval_gsd_tta.py`: current static/dynamic latent spectral path.
- `tta_gsd.py`, `eval_gsd_tta.py`: dynamic mode is integrated in the same files.
- `tta_gsd_physical.py`, `eval_gsd_tta_physical.py`: physical-basis path.
- `tta_gsd_dual_seq.py`, `tta_gsd_dual_sync.py`: global/local diffusion experiments.
- `tta_pxp.py`, `tta_pxp_sym.py`: one-way and symmetric conflict projection.
- `grid_search_tta.py` and PxP grid/evaluation scripts: pilot search infrastructure.
- `models/lion.py`: LION encode/decode/sample contracts.
- `requirements.txt`, `env.yaml`, the exact Colab setup is not committed: environment definitions; the exact Colab setup must be archived per run.

## Provenance rules

- Paper claims cite paper section/table/page.
- Implementation claims cite file and function/class, plus commit if historical.
- Numerical experiment claims cite a directory under `result/`.
- GitHub README values are repository claims, not automatically equivalent to the published table.
- When PDF text extraction makes an equation ambiguous, inspect the rendered equation or official source before coding from it.
