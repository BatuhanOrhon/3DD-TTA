# Selected Scan DGCNN / session-ownership integration — 2026-10-30

[User report/Code/Run/Decision] Publication target is **origin/thesis-integration**,
superseding the earlier main suggestion. No main/upstream write, force-push,
peer-runtime intervention or new GPU experiment is authorized by this integration.
Worktree:`../3DD-TTA-worktrees/scan-dgcnn-integration`;
branch:`integrate/scan-dgcnn-audit-20261030`.

## Sources and selection

- Source branch`gsd-multibackbone-study`, initialHEAD
  `f57b170460360b029a5b0f2a0ebc130a41083206`; dirty state and hashes recorded
  before fetch in local`result/preparation/scan_dgcnn_checkpoint_audit_20261030/integration_initial_state.json`.
- Source commit`817128a63f9a82701f05d9644908942f2825d6b8`:24-line Colab skill
  addition requiring task names and protecting peer/unknown sessions. Staged
  selectively; existing user monitor hunks were preserved and excluded.
- Source commit`5d0b89bafc3018ab7effe599f32113e2184e387b`: completed
  [CPU/static checkpoint audit](scan_dgcnn_checkpoint_audit_20261030.md), four
  compact evidence files and additive README/findings/open-question records.
- Clean integration base`2778a663cb8cfcce96d8837490e6a9386fb0665e`; selected
  commits became`cb3623b` and`82bd6c1`. The source baseline and previous
  literature/numerical matrices already existed on target; no duplicate import.

[Decision] Preserve negative/unresolved findings: correct public checkpoint
bytes and compatible head do not reproduce SVWA37.9% or GSDTTA55.72%; tail
omission is insufficient; zero stored metrics follow disabled validation.
OBJ_ONLY training-family evidence is a lead, not proven causal attribution.
Alternative checkpoint inspection supplies no new accuracy claim.

[Code] Exclude existing user monitor skill/script/tests, local research-skill
policy edits and setup-note edits; those remain in the original worktree.
Downloaded weights, full public HTML/issues dumps, raw archives, cache and
temporary analysis/setup scripts remain local. No production inference code,
configuration or dependency change was needed for the completed audit.

## Combined-tree verification

[Run] Clean target baseline:345CPUtests passed in21.152s. After both selected
commits:345CPUtests passed in20.584s, exit0. Command:
`uv run --offline --no-project --with torch --with numpy python -m unittest discover -s tests -v`.
Pre-existing archived-source SyntaxWarnings remain; no failure is hidden.
Evidence hashes/counts/derived means, new local note links, whitespace and
ordered preservation of all prior README/findings/open-question lines pass.
README conflict was resolved by adding our8-line headline to the full target
version; others' accepted positive/null/failed research records were retained.

[Artifact/Open] Local32-member evidence ZIP93963bytes, SHA256
`e5cb6e1185cf695d96af7e2e10c3c5907eee3f8769ac3f682c2836b0daf381d3`:
`result/preparation/scan_dgcnn_checkpoint_audit_20261030.zip`.
CRC and member hashes pass. No result Drive backup was performed because no
runtime was owned; public checkpoint download is distinct from result backup.
[Machine-readable selection, checks and limits](evidence/scan_dgcnn_checkpoint_audit_20261030/integration_receipt.json).
The final remote SHA is reported after normal push; this note's source/content
commit identities do not claim a self-referential final commit hash.
