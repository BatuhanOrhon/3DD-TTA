# Multibackbone GSD native binding — 2026-10-18

[User decision] The latest confirmed selection parks DGCNN and continues with
CurveNet and PointNeXt for the guidance phase. This native readiness note does
not authorize another GPU run or expand the task into a new 3DD-TTA method.

[Run] CPU audit v1 completed on r7 at 20:36:40–47 UTC, identifying the three
project JIT backends. Source inspection then exposed a fourth unconditional
JIT call in installed `knn_cuda.__init__`: `load_cpp_ext('knn')`. Audit v2 ran
at 20:39:19–25 UTC and found all four binaries uniquely. No extension was imported
or built and no CUDA forward ran during either audit. The runtime remains the
existing A100/High-RAM assignment; the operator retains sole CLI ownership.

Audit v2's local raw archive is
`result/preparation/multibackbone_20261018/gsd_native_audit_v2.zip`, 801,121 bytes,
SHA256 `c77d71e934e7b52f65493e49957f2ff3f45986babf4cd4b6f3d862b10932fc2e`.
Drive copy:
`/content/drive/MyDrive/thesis/result/preparation/gsd_multibackbone_study/gsd_native/gsd_native_20261018T203919266876Z/gsd_native_20261018T203919266876Z.zip`.
The operator verified local/Drive size and SHA; this implementation batch also
independently checked the local archive SHA, CRC and all 99 recoverable member
identities. Original audit v1 remains preserved separately, SHA256
`8d2afbca46c7ee480922e7bde1cf76b458f49a32e7806ac004e92bac365ff2f7`.

| Backend | Actual unchanged r7 binary | SHA256 |
|---|---|---|
| `_pvcnn_backend` | `/root/.cache/torch_extensions/py38_cu121/_pvcnn_backend/_pvcnn_backend.so` | `d9c2313be5c2b75088a4d5fe54408b000136dc5a34d3e2fe4a123041b0e481b0` |
| `chamfer_3D` | `/content/3DD-TTA/third_party/ChamferDistancePytorch/tmp/chamfer_3D.so` | `08040c76ff4e98b157c88a43b9cf0e178e6713b98be2abeaa0efd2b11d42c5e0` |
| `emd_ext` | `/root/.cache/torch_extensions/py38_cu121/emd_ext/emd_ext.so` | `67b729df603ae255c404d5db2792304de52cfb706d77ddfd619107dc9cdaf154` |
| `knn` | `/usr/local/envs/3dd_tta_env/lib/python3.8/site-packages/knn_cuda/csrc/_ext/knn/knn.so` | `d37e5e4ec02a44553c6329c8918cd5b6b7e121d4055c79b82968eb4177d57fbb` |

The audit records exact normal `PyInit_<name>` symbols, dynamic dependencies,
build.ninja/ninja metadata and source bytes. Runtime metadata identifies
Python 3.8.20, PyTorch 2.1.2+cu121, CUDA 12.1.105 and GCC/G++ 11.4.0. Original
checkout identity is dirty `dev@a458cd46bba1288a3f41302548bc777700cbc7d4`.
The repaired `extensions/emd` kernel and PointNet2 fallback architecture changes
are retained as provenance; they are not sources consumed by the four JIT
calls. Existing installed `chamfer` and PointNet2 `_ext` remain distinct native
modules and must retain their independently recorded identities.

LION assets verified in audit v2:
- Checkpoint `/content/3DD-TTA/lion_ckpts/epoch_10999_iters_2100999.pt`,
  1,508,051,672 bytes, SHA256
  `807f6732ad087a1ffdaeeaa456e32b4130c9a8a1708446cabc0059654a0a86c2`.
- Config `/content/3DD-TTA/lion_ckpts/unconditional_all55_cfg.yml`, 7,860 bytes,
  SHA256 `9d4a826f35fe9b3053210105eb4857811c8bfd53cd8c6a998f8ea7953e975816`.
  Config bytes are archived; the large checkpoint is identified without copying.

[Code] `scripts/multibackbone_native_bootstrap.py` exposes
`bootstrap(asset_root)`. The caller first verifies its externally pinned asset
manifest. `guidance.native_audit` names the declared audit JSON inside the
asset root; `guidance.native_bootstrap` names this declared bootstrap file in
`source_snapshot/scripts/`. The hook verifies its own bytes, audit bytes and
all 43 consumed project wrapper/source/header files against both audit and new
source-snapshot manifest, plus all 3 installed KNN source files against the
audit. It verifies all four binaries before any native consumer import,
including exact initialization symbols and the recorded Python/PyTorch ABI.

During imports of `knn_cuda`, `models.lion`, `tta_gsd_v2`, `graph_spectral_v2`
and `utilities_3dd_tta`, a temporary `torch.utils.cpp_extension.load` guard
accepts only the four exact names and their complete source lists. It rechecks
source/binary bytes and directly loads the existing binary. Unknown calls
fail; compilation is never delegated. The original loader is restored in a
`finally` block on success or failure. Fresh worker imports are required.
No original repository directory is added to `sys.path`, no source/native
bytes are rewritten, and no spectral/reconstruction math is changed.
Successful return includes `no_build=True`, audit SHA, binding identities,
verified-source count, observed JIT requests and imported module names.

[Run] Five focused CPU tests passed after observing missing-implementation
RED failures. They cover exact dispatch without invoking the original JIT
loader; binary/header/audit tampering before native import; wrong Python ABI;
unknown names/source lists; and loader restoration on failed imports.
The binary loader is replaced at the external ELF boundary for these tests.
They do not prove the Linux binaries load or execute on the live GPU. Real
CPU import preflight on r7, followed by the selected approved guidance smoke,
remains the next gate; no Colab, GPU or commit occurred in this code batch.
