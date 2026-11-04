# Zero-guidance repeat diagnosis — 2026-11-04

[Code/Open] Frozen before GPU execution. User authorizes continuation. Question: does the same observer reproduce on a failing Background case, independently of switching decoder grad mode?

Reuse owned A100 session `tta-gsd-physical-spectral-a100-20261103-r1`, monitor97932. Unchanged pinned GPU sourcecdbdfb1 and assetmanifest05b6fcde7c6982954594bec34fd6bd7d62b8b020f86129d2d8d16fafd8c0363c. Branch `research/gsd-zero-repeat-20261104` from integration6e7c60a, existing isolated worktree. No source/method patch.

Cases: Background IDs495/512 (prior failure), Gaussian44/91 (prior passing technical case); reuse exact recorded caches. Control versus live_control, weight0, controlled backend, seed0/B2/N8-B35/PM, three fresh processes each;12 new repeats, alternating order. Estimate8–12min, whole GPU hard15min. Fresh remote `/content/studies/gsd-zero-repeat-20261104`, Drive `thesis/result/gsd_zero_repeat_20261104`, local `result/preparation/gsd_zero_repeat_20261104`.

Checks: same source/native/model/cache/reference/RNG; within-arm3pair comparisons and cross-arm9pairs per case. Frozen prior gates maxlogit≤.05, per-object/checkpointRMS≤.01, firstdecodedRMS≤1e-4. Read first-clean latent/style hashes and decoded/terminal geometry differences. Bit hashes locate differences but do not quantify cause. Same-arm failure falsifies an explanation requiring observer switching; passing limited repeats cannot certify all inputs. Preserve all accepted/failed attempts; no threshold change or label-based case/coefficient search. Archive raw/source/native/cache, verify Drive/local, then interpret. Anchor experiment remains gated on this diagnosis. Runtime teardown not authorized.

Operator scriptSHA3d0aaaf3c31522b901d3b81d73ad7e29a24bef7908b46de486a6732fd68085af; wrapperSHA0ab1bd3db9c63c36b8526a2cbb4d94ca5cf2f6f085b583d2c7193ee781909543. Status complete; raw/Drive/local integrity accepted.

## Accepted results

[Run] All12 fresh processes complete,408.1089s(6min48s) under15mincap. Raw134240759bytes/SHA`6f03abf056d98bdcde817e7a800f2b150a1104ee2e5ac2bf602a71035a6c4e01`; local `result/preparation/gsd_zero_repeat_20261104/zero_repeat_raw.zip`; Drive `thesis/result/gsd_zero_repeat_20261104/zero_repeat_raw.zip`. All package member sizes/SHA/CRC,12sealedbundles, frozenmodel/source/native/cache/reference/RNG identities accepted; nativeELFs and two caches are archived.12new canonicalZIPs retained.

| Case | Control/control pass | Live/live pass | Control/live pass | Worst logit difference |
|---|---:|---:|---:|---:|
| gaussian | 3/3 | 3/3 | 9/9 | 0.0000648 |
| background | 0/3 | 0/3 | 0/9 | 2.5460446 |

[Inference] Observer switching is not necessary for the selected Background instability. The operational control has no physical graph/loss and still fails all3same-observer pair comparisons. Live0alsofails3/3; all9crosscomparisonsfail. Gaussian passesall15 comparisons. Therefore a problem requiring the GSD matrix, its inverse, or decoder gradient-mode switching cannot be the sole explanation. This does not validate the physical objective or locate a specific kernel.

[Code/Run] Within and cross comparisons reuse exact same encoded state/noise, model weights, native binaries and recorded RNG. First clean-latent/style hash witnesses and decoded differences are retained in [full pair evidence](evidence/gsd_zero_repeat_20261104/analysis.json) and [summary](evidence/gsd_zero_repeat_20261104/summary.json). A hash mismatch detects bits, not magnitude. The comparison pairs share runs and are not independent trials; this selected four-object/two-corruption study cannot certify all-input stability or separate corruption from object/horizon effects.

[Decision/Open] Keep the previous no-promotion decision. Next test should capture the earliest divergence within the original prior/SCD path on the Background35 trajectory: prior clean estimate, Chamfer distances/selected indices, SCD VJP, and updated local/style tensors at each step, followed by fixed-input native forward/backward repeats. Native atomic reductions and discrete-neighborhood amplification remain hypotheses; no speculative kernel fix, coefficient search or fresh accuracy run was performed. Anchor/correspondence accuracy tests remain gated on numerical diagnosis.

Analysis source uses assert-based acceptance against actual immutable raw files and existing tested `analyze_accuracy` helpers; no production code changed. Its archived `analyze.py` expects the original preparation directory layout. Runtime remains protected and monitor97932 stays open.

[Run] All selected repeat PM class predictions agree despite the largeBackgroundlogit/geometry differences. This phase identifies numerical nonrepeatability, not a measured new accuracy loss. Derivedarchive12801bytes/SHA`be6029108aabb5604fdc3e9b411fff29d200871983342009268c74a9ca7363fd` retains exact analysis/wrapper bytes.

[Run/Code] Spectral physical weight is0 in all repeats; original SCD remains active(.03/.003/.95). Raw and derived Drive/local copies verified; [receipt](evidence/gsd_zero_repeat_20261104/final_receipt.json). Result integrated as`07b5ae43f2a22831c3348f536edda94e0026e27c` into`origin/thesis-integration`; [scope](evidence/gsd_zero_repeat_20261104/publication.json). A100 runtime responsive09:19:48UTC, monitor97932 retained. No further GPU run started.

[User report/Run] 2026-11-04 13:09UTC: user explicitly requested Colab termination. Named session stopped with `colab stop -s tta-gsd-physical-spectral-a100-20261103-r1`; CLI confirmed termination and server inventory confirmed no active sessions. Monitor34871 stopped after its stalled liveness-only CLI75359 was terminated; both PIDs absent. No research job was active; verified local/Drive raw and derived archives retained. Previous runtime-retained statements are superseded by this closure.
