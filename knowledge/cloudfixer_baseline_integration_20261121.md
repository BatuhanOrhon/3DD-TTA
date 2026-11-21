# Selective XYZ A/B baseline integration — 2026-11-21

[User request/Code/Run/Decision] Publish the completed CloudFixer-denoiser code
and accepted A/B research record to `origin/thesis-integration`. Source:
`research/cloudfixer-transfer-20261114@92b456c49a1d2c5bd03013fd2e0bec4d5a4486e6`; target base `be2ad78c7de559c081ee9d67e584553bd2efeee8`.
Work prepared in the dedicated `integration/cloudfixer-ab-baseline-20261121`
checkout; main and peer checkouts/runtimes untouched. This is a selective
publication, not a merge of the entire research branch.

[Included] Tested XYZ sampler/proximal helper/store; Point2Vec and preserved
Point-MAE runners/setup/assets; all related analyzers and146 focused CPU tests;
complete A/B/previous finite readout result families including negatives;
linked scientific history, source/archive identities and baseline decision.
[Exact selected files, hashes and verification](evidence/cloudfixer_baseline_20261121/integration_scope.json).
Shared knowledge was extended by meaning: existing LION full-data and citation
work is retained. LION entrypoints/worker/sampler and citation review are
byte-identical to the target base. The older shared-helper refactor and older
whole shared notes were deliberately excluded. Raw archives, weights, dataset,
auth/recovery wrappers and redundant operator dumps remain outside Git.

[Decision] Main new-method reference is N8/K80/tau.1/retained.95 equal late
PM[20,10,0], existing tested `proximal_pm` arm. Plain CF remains the stronger
matched benchmark. [Detailed rationale, rotation/background arithmetic and
next-stage falsifiers](cloudfixer_baseline_rotation_background_20261121.md).
No new model/default dispatcher or GPU experiment was added by publication.

[Verification]146 focused tests pass; actual CPU official-source packaging
for Point2Vec and Point-MAE hypotheses passes. All selected source bytes match,
all479 B paired rows pass arithmetic, accepted B table hashes and original
ZIP CRC/member checks pass. Shared accepted text/order and selected local
Markdown links verified (large original `result/` artifacts intentionally
remain on Mac/Drive). Original CSV CRLF is preserved for byte provenance;
`git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol diff --cached --check`
is the appropriate whitespace check. No CUDA replay, external replication,
new seed inference or claimed rotation/background repair result.
