# Proximal global shape pilot evidence — 2026-10-31

[Run/Code] Complete32exposedobjects × fourcorruptions × fivearms/seed0/Point-MAE.
Raw archives and source/operator identities are in the receipts and result note.
`outcomes.npz` has five Boolean(4,32)matrices. `paired_latents.npz` has physical
finalglobalvectors and pairedcleantargets(4,32,128), in summary.json's ID/order.
The clean targets are posthoc except explicitly diagnosticoracle3.

Recompute after retrieving the immutable rawZIP and safe fresh extraction:

```bash
PYTHONPATH=. python analyze_phase.py PHASE PHASE/paired FRESH_ACCEPTANCE
# First compare acceptance.json and all outcomes.npz arrays exactly to PHASE.
python summarize_phase.py PHASE FRESH_ACCEPTANCE FRESH_SUMMARY
```

NumPy1 required for physical RNGpickle checks; recordedlocalNumPy1.21.6.
Analyzer expects repo scripts onPYTHONPATH. Never write derived output intoPHASE.
Full manifest/CRC/source/state/proof validation precedes summary; summary trusts
that acceptance/outcomes pairing. Local exact gates are recorded separately.
`compare_repeats.py` ran in archivedoperatorroot containing smoke_extracted and
pilot_extracted; it reads existing artifacts, never starts inference.
Technical/pilot numerical variation and conditional/multiple-comparison limits
are explicit in the result note. No newdefault or unseenbenchmark claim.

[Open] Final extraDrivecapsule/publication receipt is added after archival;
this initial evidence text is not a claim those later steps already passed.

## Final archival acceptance — 2026-10-31

[Run] Source/assets/operator/derived copies were verified on Drive at
`MyDrive/thesis/result/shape_latent_proximal_pilot_20261031/final-archive-20261031T190545Z/`.
The 2,045,023-byte derived/source/operator capsule has SHA256
`322f3ff3ba16ed2f412b4b1e425be1d62cdd0d9fef62c00acceeec9dcc099094`;
Drive download roundtrip, all68 member identities and raw extraction recheck pass.
Both raw phases remain unchanged after analysis. The sealed capsule intentionally
retains its then-pending publication notes; these final receipts supersede them.
See compact `final_archive_receipt.json` and `derived_roundtrip_receipt.json`.
No GPU work remains and no runtime teardown was issued.

The compact Git CSV uses LF line endings; immutable raw and sealed derived bytes are preserved.
