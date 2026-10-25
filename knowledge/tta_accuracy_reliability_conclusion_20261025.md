# Interpretation of reported accuracy differences — 2026-10-25

[Inference/Decision] **The audited evidence weakens a blanket explanation that our evaluation implementation is the sole cause of the paper gaps. It does not prove every local condition correct or establish corruption regeneration as the cause of every discrepancy.** The defensible conclusion is that identically named datasets, backbones and methods do not consistently identify the same experiment across the reviewed papers.

This is an interpretation of the existing [16-publication audit](pointcloud_tta_reporting_audit_20261025.md), accepted local controls and sealed evidence, not a new literature search or model experiment. Evidence base: study commit`794ddb84bcbfe0dcf21ebb8eb0ce957cceb31f97`; literature archive`result/preparation/tta_literature_expansion_20261025T144349Z.zip`, SHA256`e5c57dfb54ad3e6448006821d937658b408d0e4255d819b2c65b9fb88348f578`. The823 table observations include repeated tables/variants and are not823 independent replications.

## What is supported

[Paper/Run] Our ModelNet source-only observations are numerically close to appropriate published source families: Point-MAE53.6899 versus CloudFixer53.70; DGCNN58.7979 versus SVWA58.8; CurveNet64.0113 versus SVWA supplementary64.0. DGCNN/CurveNet weights were selected from SVWA's released assets before evaluation, so agreement is a meaningful source-evaluation consistency check. It is not independent retraining or proof that all paper assets match. Existing strict loads, frozen-state, indexed-label/logit, Point-MAE boundary parity and artifact checks strengthen internal validity within their tested conditions.

[Paper/Code] Several discrepancies have explicitly documented causes of **non-comparability**, even without regenerated data: classification-only versus joint source training; batch size and standard/online/reset protocol; six versus15 corruptions; input-only CloudFixer versus CF-O; one severity versus severity averages; and seven-corruption averages versus clean+seven. Point-Cache/BayesMM's48 paired rows retain identical displayed cells while changing the averaging denominator. Some papers explicitly reuse earlier baseline numbers; others report rerunning under changed conditions. Neither repeated means nor citation counts establish independent reproduction.

[Paper/Code/Open] Some ShapeNet-C/ScanObjectNN-C protocols describe generating corruptions from clean data. Released3DD code has an inactive seed statement and generator/evaluator severity mismatch. These are real reproducibility risks. However, inspected papers do not supply enough author-array hashes, seeds and execution histories to establish that **every** group regenerated data, used a distinct random draw, or obtained its discrepancy for that reason. Identical displayed cells may reflect inherited reporting; they do not reveal actual file identity.

[Run] ModelNet40-C has a fixed canonical corrupted archive; our downloaded archive and evaluated arrays already passed byte-parity checks against that archive. Local random regeneration is therefore not a sufficient explanation for our current ModelNet gap. Existing Scan severity and split controls also did not close the broader paper-source profile. Random-draw variance was not measured in this literature task.

## What remains open

[Open] ScanDGCNN32.7826 remains5.1174pp below SVWA37.9; the original3DD Point-MAE source/terminal and exactGSDTTA checkpoint/protocol gaps remain unresolved. The reviewed literature does not quantitatively allocate those differences among checkpoints, source training, arrays, preprocessing, randomness or local implementation. A possible classification-only versus joint-trained Point-MAE release mismatch is a hypothesis, not an established cause. Unreported author reductions/raw logs also prevent resolving every flagged printed mean.

## Reporting decision

[Decision] Describe the field-level issue as **benchmark/protocol and result-provenance heterogeneity in the reviewed literature**, rather than claiming universal random dataset generation or declaring all local runs error-free. Preserve paper-reproduction gaps as open. Report our evaluated checkpoint/data/preprocessing/severity/aggregation and compare methods against their own matched source baseline. Original3DD versus ourGSD remains a whole-policy comparison where settings differ; isolated spectral benefit still requires matched component controls and repeated seeds. The literature audit does not supply that causal evidence or change peer agents' experiment authorizations.

A thesis-safe statement is: *Cross-paper accuracy differences are already present in source-only controls and are not sufficient evidence of a local evaluation defect. Several differences are associated with documented protocol, source-model and reporting variations; exact causal attribution requires matching experimental assets and reductions.*

Falsifiers: a demonstrated local preprocessing/label/state defect would reopen affected matched comparisons; author-provided matching arrays/checkpoints/configurations and auditable reruns would strengthen or refute asset-parity explanations; controlled fixed-protocol repeated corruption generation would quantify random-draw effects. No such new experiment, GPU/Colab operation, author contact or Drive transfer was performed here.
