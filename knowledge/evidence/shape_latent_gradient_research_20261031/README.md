# Shape-latent conditional-gradient literature audit — 2026-10-31

Evidence labels: [Paper], [Code], [Run], [User report], [Inference], [Open].
See [the research report](../../shape_latent_gradient_research_20261031.md).

- `primary_source_receipts.json`:10 primary-paper full texts, ICML metadata
  and official Diffusion-TTA pinned source; physical URL/byte/SHA receipts.
- `code_audit.json`: actual dirty checkout identities and relevant paths;
  code observations do not establish a native gradient or an accuracy gain.
- `search_ledger.md`: bounded search, selected sections, transfer limits.
- `runtime_inventory.json`: CLI server inventory; no active sessions,
  no teardown/allocation. Prior results are archived independently.
- `validation.json`: receipt/link/code identity checks for this docs-only work.

Full third-party sources and extracted text remain outside Git under
`result/preparation/shape_latent_gradient_research_20261031/primary_sources/`.
PDF text extraction used temporary pypdf6.21.4/fonttools4.68.0; no repository
dependency changed. The first extraction warned about font decoding; it was
repeated with fonttools and exited successfully without those warnings.
Raw HTML/PDF bytes were not rewritten. Math claims refer to original paper
equations, not to font-sensitive text extraction alone.

No GPU run, external reproduction, new method implementation or numeric
gradient experiment is included. Existing results are linked, not rerun.
