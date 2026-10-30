# Scan CurveNet entropy-guidance follow-up — 2026-10-30

[User report/Decision] Prepare a stronger alternate semantic guidance model for
ScanObjectNN-C after the entire current campaign, timing and raw-data collection
finish. The user authorizes this follow-up without another GPU approval question.
This preparation performs no upload/GPU action and does not modify the original
campaign/controller, frozen GPU source or its continuous five-hour clock.

[Run/Inference] CurveNet is the single preselected alternative: historical all15
source52.0711% versus PointNeXt49.0878%. The actual archive audit also reproduces
CurveNet dev70=49.3333%, held140=53.2857%; these are historical source observations,
not current follow-up results or coefficient-selection data. Exact source-model,
checkpoint, dataset and preprocessing evidence is in
[the independent source-choice audit](evidence/chamfer_author_code_audit_20261030/scan_alternate_teacher_source_evidence_20261030.json).
Historical source used B32/benchmarkTrue and a different native FPS binary;
current source is not asserted byte/parity-equivalent to that historical row.
The earlier Point-MAE H development gain on ModelNet does not prove a Scan gain.

[Code] `--semantic-guidance-head curvenet` is opt-in and restricted to Scan,
marginal entropy and a selected CurveNet head. The old Point-MAE default is
unchanged; a test compares its preview config to the original frozen source.
The frozen Scan `study_3deformrs.CurveNet` consumes BCN coordinates through the
existing wrapper. Raw-X and decoded inputs both retain the existing centroid
subtraction, per-object L-infinity normalization and native FPS1024 contract.
Only CurveNet's reused baseline logits provide the H gradient; the other three
heads remain readouts. There is one cached detached X forward per batch,
three reused clean decodes and the same three-probe-plus-terminal probability
mean. The reference never becomes an extra vote. C1/S0/invariant0, graphNone,
B70, rates.01/.01, p.95,10/30 steps and checkpoints5/8 or15/23 remain fixed.

[Code/Open] Normalization and selected-point gathers support input gradients.
CurveNet additionally uses discrete neighborhood/sampling choices, detached
crossover weights and its author's straight-through curve selection. These
algorithms, BN/dropout flags and checkpoint values remain untouched. Toy CPU
tests prove the normalized-FPS/wrapper and latent/style boundary; actual CUDA
CurveNet/decoder gradient validity is a mandatory subsequent native gate.

[Code] `scripts/calibrate_scan_curvenet_entropy.py run` collects only C and H
gradients on a C trajectory (H weight0), Scan dev70/seed0/all15/CurveNet-only.
At each of two clean checkpoints it records per-object gradient norms/cosines,
the H gradient to decoded XYZ and exact-zero gradients of0*H to decoded XYZ,
local and style (same gradient call, four calls per probe in total).
Every positive path must be finite/nonzero; zero gradients must be finite and
exactly zero; `.grad` fields and RNG must remain unchanged. Ordinary runner
frozen-state and artifact checks remain active. No KL, weighted-Chamfer, graph
or spectral calculations are executed by this collector.

[Code] A separate `scan_curvenet_entropy_report.py` purpose
`scan_curvenet_entropy_calibration` requires exactly2100 rows (70×15×2),30
complete native zero/nonzero probe proofs, one Scan dataset, the exact CurveNet
checkpoint/BCN/preprocessing contract, source/native/runtime/model provenance,
sum loss reduction and fixed alpha=0.1/median(H/C combined update-norm ratio).
The combined norm uses the actual sampler mapping: local eta, style gamma.
Missing/nonfinite/near-zero evidence rejects calibration. The old two-dataset,
three-candidate4200-row V1 validator is unchanged and rejects the new purpose.
The new report also rejects old Point-MAE calibration. Positive CurveNet runs
require this report/SHA; exact bytes are sealed as `calibration_report.json`.
Analyzer arm `semantic_h_curvenet` stays distinct from Point-MAE `semantic_h`.

[Decision] Freeze this one coefficient before any new accuracy evaluation;
no dev-accuracy run or HPO is planned. After native/calibration acceptance run
matched C/H on the same reused Scan held140 IDs, all15 corruptions and seeds0/1/2,
with four readouts and three fresh Gaussian/background timing repeats per arm.
Primary endpoint: CurveNet's own probability-mean C→H accuracy/delta and
object-cluster CI. Four-head aggregate and the other three heads' transfer are
secondary. Reusing held140 after prior analyses makes this an exploratory
post-hoc source-model check, not a new independent holdout confirmation.
Native C/C trajectory variability is already observed; do not reinstate a false
bitwise trajectory-parity claim or invent a tolerance that hides an error.

[Inference/Open] Current Scan worker measurements are C184.46s/H190.70s for
dev70 and C302.94s/H317.33s for held140; the older all-candidate Scan calibration
took179.84s. New H-only calibration plus six heldout workers and six timing
workers should take roughly40–45min including setup/I/O, with a conservative
55–60min planning allowance. CurveNet backward is unmeasured: update the forecast
from its actual first native/calibration measurements and stop on nonfinite/OOM;
do not silently change B70 or introduce another teacher/weight grid.

[Test/Open] Local tests cover config/default preservation, normalized-FPS XYZ
and latent/style gradients, head/scope restrictions, report/weight/checkpoint
binding, native proof rejection,2100-row coverage, no weighted work, gradient/RNG
purity, and sealed summaries without reading label/prediction files. Full
`pytest -q tests` passes464 tests and348 subtests with6 existing warnings;
scoped diff whitespace checks, Python3.8 AST and a dependency-free `python -S`
report validation pass. Bare `pytest -q` additionally discovers ignored source
archives/vendor tests and fails collection on unavailable `hydra`/`emd_cuda`,
obsolete `importlib.find_loader` and duplicate archived test modules; these
external collection errors are not hidden or repaired by this batch. Real native
CurveNet calibration and follow-up accuracy/runtime are still pending.

[Code/Test] Root verification on the final reviewed local source:464 tests and348
subtests pass in26.07s (six existing warnings); Python3.8 AST and whitespace
checks pass. Independent95-test/74-subtest review and exact original40-bundle
analyzer/default-config regressions pass. See the [CPU receipt](evidence/chamfer_author_code_audit_20261030/scan_curvenet_entropy_cpu_validation_20261030.json)
and [review](evidence/chamfer_author_code_audit_20261030/scan_curvenet_entropy_independent_review_20261030.json). Native GPU checks remain pending.

[Code/Test/Open] Separate post-campaign controls pass14 CPU tests and an
independent review, including fresh/replay preparation, exact C/H initial
tensors/RNG/IDs/labels, root isolation, parent454-file collection proof and
continuous70min cap. This is13workers/49raw ZIPs:1calibration,6matched C/H
and6fresh timing. Nominal2640s;10% plus600s reserve3504s;4200s hard cap.
Actual CurveNet backward will update the forecast before the six accuracy jobs.
Source95a4997 has298runtime files; new658-entry asset manifest differs from
the old656-entry manifest only in6source/ZIP entries. Source seal and controls
remain local; no new GPU phase or environment deployment has occurred.
[Controls CPU receipt](evidence/chamfer_author_code_audit_20261030/scan_curvenet_h_controls_cpu_20261030.json),
[independent review](evidence/chamfer_author_code_audit_20261030/scan_curvenet_h_controls_review_20261030.json),
[source comparison](evidence/chamfer_author_code_audit_20261030/scan_curvenet_h_source_comparison_20261030.json).

## 2026-10-30 — native calibration accepted; matched comparison running

[Run/Code] Same owned A100/session; separate newroot
`/content/studies/scan-curvenet-h-postcampaign-20261030`, Drive
`MyDrive/thesis/result/scan_curvenet_h_postcampaign_20261030`. Real plan
SHA297badb31d81667e86467d84d0d2f35912b3e3546d234f7a75bdf5435b0f9d46
binds all454parent physical identities after complete original collection.
CPU deploy/strict CurveNet load,298source/658asset checks and existing native
setup pass without CUDA initialization or forward; archive5694059bytes
SHAfdd668b3f1467c0077e9754ed7ad3a4b84899062e9d827e15adb45f53168c789
is verified on Drive and locally. Teacher state30c4d9194aca84009cca843d13de3e5e1334b9eb7bcaf49d8ad5a906a75b19b6.

[Run] H-only native calibration completes in164.0003s. All2100rows/30probes
pass finite/nonzero H and finite exact-zero0H in decoded XYZ/local/style,
RNG/.grad/frozen state and zero guidance graph/spectral counters. Source
classifier intrinsic KNN/Curve computations remain its existing architecture;
these counters refer to guidance GraphSpectralDNA/eigenspectrum work.
Independent recomputation exactly matches frozen report after only proof-path
relocation. FrozenlambdaH=3.153885581354952=.1/.03170692069210661. Scaled
median update ratio.1, observed calibration maximum.5579733; no per-object
cap is introduced. Report52546bytes SHA8e8dd33e15edc17487bbcc1636064bf3de1b4a1fdad5714d6dd777f62e2e5d07.
[Report](evidence/chamfer_author_code_audit_20261030/scan_curvenet_frozen_calibration_20261030.json),
[independent acceptance](evidence/chamfer_author_code_audit_20261030/scan_curvenet_calibration_acceptance_20261030.json).

[Run/Open] Separate continuous research clock18:23:23.664750UTC, hard
deadline19:33:23.664750UTC. One localCLI prelude failed before user-code
execution:0matching history entries, no heldout directory and byte-identical
calibration ledger. Only ownedlocalCLIchild227 was stopped; same6-job phase
relaunch starts18:33:23UTC, preserving VM/monitor/config/source/cap. New C/H
accuracy remains pending; no labels were used to select the fixed coefficient.
Comparison is the CurveNet guidance package with its own Scan calibration,
not an isolated causal attribution to teacher quality versus the old pooled
Point-MAE coefficient. All models remain frozen; no method combinations.

## 2026-10-30 — CurveNet C/H accuracy accepted

[Run]24head ZIP/720batch ve90yeni fiziksel cache bağımsız kabul edildi;
95a4997 kaynağı,658asset,8e8d...frozenH raporu, model/native state ve gerçek
C/H initial h0/noise/RNG, nesne kimlikleri/etiketler tam eşleşir. Hiçbir
eski0fd C arşivi veya cache yeni kontrolle karıştırılmadı. Ham arşivler ve
yeni Drive kökü `scan_curvenet_h_postcampaign_20261030` içinde korunur.

[Run] Birincil sonuç CurveNet’in kendi temporal probability mean’idir;
all15/severity5,140nesne, seed0/1/2. Her head6300karar. CI140nesne
cluster bootstrap2000/seed0; seçili seed/head koşulludur, multiplicity
düzeltmesi yoktur. Aynı140IDs’nin tekrar kullanımı posthoc/keşifseldir.

| Head | C accuracy% | C+CurveNet-H accuracy% | Fark pp [95%CI] |
|---|---:|---:|---:|
| CurveNet (birincil) | 55.6667 | 55.9048 | +.2381 [−.1587,+.6349] |
| DGCNN (transfer) | 38.3810 | 38.5714 | +.1905 [−.3810,+.7460] |
| PointNeXt (transfer) | 46.6984 | 47.1111 | +.4127 [+.0159,+.8258] |
| Point-MAE (transfer) | 39.0000 | 38.9365 | −.0635 [−.3175,+.1746] |
| Dört-head accuracy ortalaması (ikincil) | 44.9365 | 45.1310 | +.1944 [+.0040,+.4049] |

[Run/Inference] CurveNet seed0/1/2 C→H doğru sayıları1165→1166,1168→1176,
1174→1180 /2100; toplam+15doğru/6300. Yön üç seed’de pozitiftir fakat
birincil nesne-cluster aralık sıfırı kapsar; güçlü/genel bir katkı kanıtı
yoktur. PointNeXt ikincil olumlu gözlemi, birincil endpointin yerine
geçmez. Source-model gücü ve yeni Scan kalibrasyonunun saf etkileri izole
edilmedi; kendi matched C’sine karşı bir paket karşılaştırmasıdır.
Dört-head ortalama bir model ensemble oyu değildir: her classifier’ın
kendi dört-zamanlı probability-mean accuracy’sinin ortalamasıdır.

[Run/Code] Root ve bağımsız tam JSON raporları eşleşir; ham/source/ID/RNG
ve modellerin donmuş durumları kabul edildi.
[Kompakt proof](evidence/chamfer_author_code_audit_20261030/scan_curvenet_heldout_acceptance_20261030.json).
Bu tarihli accuracy kabulünde süre henüz bekleniyordu; aşağıdaki nihai
fresh timing kabulü bu durumu günceller. Varsayılan çalışma akışı
experimental flags dışında otomatik değişmedi; birleşik yöntem yoktur.

## 2026-10-30 — tüm deneyler ve fresh süre ölçümleri tamamlandı

[Run] CurveNet takip aşamasının 13 worker/49 head ZIP ve90 fiziksel cache'i
kabul edildi. Üç döndürülmüş fresh tekrar: dev70/B70/seed0,
Gaussian+Background, ortak rekonstrüksiyon bir kez ve dört terminal classifier.
C medyanı20.955446s, H medyanı22.759866s; birincil eşlenmiş H/C oranları
1.08557173,1.08981485,1.08468403; medyan ek maliyet **%8.5572**.
Gaussian ek maliyet medyanı%13.0753, Background%6.0390.
Model/native kurulum, toplu veri yükleme, paketleme ve Drive aktarımı bu
batch ölçümüne dahil değildir; tek classifier veya all15 deploy süresi değildir.

[Run/Code]24 timing head/48 batch bağımsız kabul edildi. Input/noise/scheduler/
RNG, kaynak95a4997,658asset, donmuş checkpoint/state ve katsayı tam eşleşir.
40 head-batch encoded-state farkı açıkça kayıtlıdır; fresh encode bitwise
paritesi iddia edilmez ve bu koşulardan accuracy çıkarılmaz. Root ve bağımsız
tam JSON, yalnız encoded-difference satırlarının sırası normalize edilince
birebir aynıdır. Guidance spectral/graph hesap sayıları sıfırdır; classifier
mimarisinin kendi KNN/Curve işlemleri aynen korunur.
[Süre kabulü](evidence/chamfer_author_code_audit_20261030/scan_curvenet_timing_acceptance_20261030.json).
Timing phase ZIP183436bytes SHA1e106a454c0df905ba444355f431ecd8ac9b0164bb2b14f9c6a368e0fb2e2cd9.

[Run] Yeni sürekli GPU saati18:23:23.664750→19:14:23.569UTC,
yaklaşık3059.90s/51dakika;4200s sınırının altında. Ledger active_seconds
2448.191s bekleme/aktarım aralarını içermez; bütçede sürekli süre kullanılır.
Orijinal4saat42dakika/5saat bütçesi değiştirilmedi.

[Inference/Decision] CurveNet kendi accuracy'sindeki +.2381pp farkın95%CI'si
[−.1587,+.6349] olduğundan, %8.56 batch maliyetine karşı güçlü bir fayda
kanıtlanmadı. PointNeXt/ortalama ikincil olumlu işaretleri birincil belirsizliği
kaldırmaz. Şimdilik hiçbir varyant varsayılan yapılmadı veya birleştirilmedi.
Tüm sonuçlar Chamfer-only ve mevcut dört-zamanlı probability mean üzerindedir.
Kullanıcının talimatıyla kalan iş yalnız nihai Drive özeti, dosya doğrulaması
ve sadece sahip olduğumuz runtime'ın kapatılmasıdır; yeni GPU işi yoktur.

## 2026-10-30 — nihai Drive kabulü ve yalnız sahipli runtime kapanışı

[Run/Artifact]407ham head ZIP+304NPZ'nin711yerel dosyası SHA/CRC/güvenli yol
kontrolünden geçti; eksik/çakışma yok.304physicalcache ayrıca Drive'da
`scan_curvenet_h_postcampaign_20261030/phase-evidence/paired_physical_caches_final_remote_20261030T1926Z.zip`:
1290931819bytes SHAea98e68d50330bd4b988594089564c35010a4e86e674c4920c65e510e85bbebd.
Geri indirilen ZIP'nin tam SHA/CRC'si ve304member SHA'sı ayrı doğrulandı.
Nihai260member bilimsel özet aynı Drive klasöründe
`final_research_summary_20261030T1930Z_verified.zip`:78981869bytes,
SHA19e7c6735f0b60438c3347d68d87183b1ee6d886aaca084cb1605746742077f5.
Geri indirme ve259member SHA/güvenli yol/CRC kabul edildi. Dört kurulum
history üyesi paylaşılan özet kopyasından çıkarıldı; orijinal kaynak ZIP'leri
ve ham araştırma kanıtları değiştirilmedi. Özet, kapanış öncesi bilimsel
snapshot'tır; gerçek kapanış makbuzu aşağıdaki yerel/Git kaydıdır.
Son roundtrip kabul makbuzları da Drive'a ayrıca başarıyla yüklendi.

[Run] CLI'nin büyük doğrudan upload'ları400/brokenpipe verdi; başarı olarak
raporlanmadı.304cache VM üzerinde CPU ile paketlenip SHA/CRC doğrulandı;
özet5×16MiBparça ile yüklenip aynı byte'lara birleştirildi. Bir CPU archive
CLI prelude bağlantı kaybı kullanıcı kodu çalışmadan önce kanıtlandı;
yalnız sahipli lokal CLI5577 sonlandırılıp aynı script yeniden çalıştı.
Bu arşiv işleri yeni GPU deneyi veya araştırma bütçesi uzatımı değildir.
Private URL/token içeren ham CLI history paylaşılmadı.

[User report/Run/Decision] Kullanıcının tamamlanınca yalnız kendi session'ımızı
onaysız kapatma talimatı uygulandı. `tta-chamfer-guidance-variants-a100-20261030-r1`,
endpoint`gpu-a100-hm-ass1c0-3nh8smksm50de`; izleyici36160/PID59645
19:34:38.530UTC'de stop protokolüyle exit0 kapandı. Ardından yalnız tam bu
adla `colab stop -s tta-chamfer-guidance-variants-a100-20261030-r1` exit0;
19:34:46UTC itibarıyla sunucu envanteri aktif assignment göstermiyor.
Peer zaten önceki envanterde yoktu; diğer agent'ın runtime/izleyicisine veya
bilinmeyen süreçlere stop/exec/reconnect işlemi yapılmadı. Yeni GPU işi yoktur.
[Cache yedeği](evidence/chamfer_author_code_audit_20261030/final_cache_backup_acceptance_20261030.json) ·
[Özet yedeği](evidence/chamfer_author_code_audit_20261030/final_summary_backup_acceptance_20261030.json) ·
[Kapanış kanıtı](evidence/chamfer_author_code_audit_20261030/owned_runtime_closure_20261030.json).
