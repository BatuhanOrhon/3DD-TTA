# Yeni session handoff — Legacy üzerinde diffusion çıktı seçimi

Tarih: **2026-10-11**. Etiketler: [User report], [Run], [Code], [Inference],
[Open]. Bu dosya kullanıcı tarafından istenen geliştirme handoff'udur.
Bu session'da yalnızca knowledge güncellendi; yeni inference geliştirmesi,
model testi veya GPU koşusu yapılmadı.

## Başlangıç ve korunacak koşullar

- Repo: `D:\Akademik\Okul\Thesis\code\3DD-TTA`
- Branch: `gsd-smooth-spectrum`
- İncelenen HEAD: `f670c07440533f01e10fd149fad2480d36f9194c`.
  Yeni session gerçek HEAD/status'u yeniden okumalı; bu commit'e reset atmamalı.
- [User report/Run] Kullanıcının seçtiği başlangıç: son başarılı **Legacy**
  B70/M800 koşusu, macro15 **64.570502%**, seed0, 23.904/37.020 doğru.
  Tarihsel64.5327% pratik olarak yeniden üretilmiş kabul ediliyor.
  Başka araştırma kolunu veya offdiag'ı sessizce baseline yapma.
- Temel ayarlar: batch70, spectral1.17, Chamfer1.0, M_max800, beta2,
  gamma0.01, eta0.01, lambdaa0.95, normal10/Background30, isolate_policy=legacy,
  statik graph, mevcut güncellenen style/final decode, mevcut scheduler ve
  normalize/rotation/FPS yolu. ModelNet40-C severity5, tüm15 corruption.
- Point-MAE ağırlıkları, BN istatistikleri, token akışı ve inference modu sabit.
  Entropiyi sınıflandırıcıya geri yayma; PG/BN adaptasyonu bu geliştirmeye dahil değil.

Ham kanıt (değiştirme):
`result/modelnet40_c/legacy_isolate_control/attempt-0002_all15_seed0/20261010T114156946419Z_handoff_20261010T141536060023Z.zip`

Boyut12.277.729 bayt; SHA-256:
`534c8634ed16aa0a0787e3be2f92294268453029a0d0a175ebcd7b6618726968`.
İçindeki `legacy_all15_seed0.zip` baseline'dır; `offdiag_all15_seed0.zip`
farklı kontroldür. Denetim: aynı dizinde `analysis_ingested/validation.md`.
`analysis_20261011_selection/` içindeki +0.2782pp olasılık ortalaması farklı
iki tam koşunun analizidir; ara diffusion adımları için kanıt değildir.
Mevcut arşivde ara clean noktalar yok; yeni gözlemci gerekli.

## Online kullanım düzeltmesi — son kullanıcı yönlendirmesi

[User report] Hedef online TTA'da tek kayıt veya küçük batch'lerle de çalışmak.
Önceki büyük-batch hız uyarısı bu senaryoya genellenmemeli.

[Inference] **B1'de erken durdurma kalan diffusion adımlarını gerçekten atlar.**
Başka örneklerin bitmesini bekleme veya aktif batch'i sıkıştırma gereği yoktur.
Dolayısıyla burada doğrudan bir hesaplama tasarrufu mekanizması vardır.
Net uçtan uca hızlanmanın miktarı, atlanan hesap ile ek VAE decode/FPS/Point-MAE
kontrollerinin maliyetine bağlıdır; her örnekte kesin hızlanma henüz ölçülmedi.
Küçük batch>1'de farklı bitiş zamanları için aktif örnek yönetimi yine geçerlidir.

İki ayrı değerlendirme hattı tutulacak:

1. **B70 benchmark:** mevcut Legacy accuracy'sini ve eşleştirilmiş aday farkını
   koruyan all15 değerlendirme. Throughput ve toplam süre ölçülür.
2. **Online B1:** her örnek için sabit-horizon Legacy ile aynı örneğin erken
   durdurmalı varyantı; gecikme p50/p95/ortalama, accuracy ve atlanan gerçek
   adımlar ölçülür. Kuyruk/toplu bekleme süresi ayrı belirtilir. Batch boyutunu
   değiştirince RNG/kernel etkileri değişebilir; B70'in64.5705% sonucunu B1'in
   ölçülmüş baseline'ı gibi kullanma. Aynı hazırlanmış state/noise ile karşılaştır.

Online kural tüm test dosyasının gelecekteki örneklerine, gerçek etiketlere,
gelecekteki diffusion tahminlerine veya etikete göre ayarlanan eşiklere erişemez.
İlk offline gözlem çalışmasında saklı logits ile kural geliştirmek mümkündür;
değerlendirmede kural kilitli ve nedensel olmalıdır. Mevcut10/30 ufku corruption
adını kullanıyor: bunu benchmark legacy protokolü olarak açıkça koru. Gerçek
corruption kimliği bilinmeyen online kullanım için ortak sabit veya yalnızca
gözlenen örnekten seçilen horizon ayrıca tanımlanmalı; corruption-oracle erişimi
varsayma. Bu soru ilk gözlemciye eşzamanlı scheduler değişikliği eklememeli.

## Yeni session'da ilk geliştirme kapsamı

1. Mevcut legacy davranışı varsayılan olarak aynen koruyan **opt-in trajectory
   gözlemcisi** ve ayrı analiz/artifact akışı hazırla. Önce kısa uygulama planı
   çıkar; teknik belirsizlikleri koddan çöz, rutin konularda tekrar izin isteme.
2. İlk checkpoint önerisi: reverse-loop tamamlanma sayacıyla10 adımda3/5/8,
   30 adımda8/15/23; ayrıca **değiştirilmemiş orijinal terminal decode**.
   Bunlar gözlem noktalarıdır; başlangıç gürültü derinliğini değiştirme.
   Checkpoint listesi/config ve indeks manifesti koşudan önce kaydedilmeli.
3. `scheduler_output.pred_original_sample` temiz local latent'tir. Aynı prior
   forward'ında kullanılan **pre-update style** ile decode et. Fiziksel noktaları
   baseline ön işlemesi ve sabit Point-MAE ile değerlendir. Post-update style
   ile karıştırma; son clean tahmini original final decode'a eşit varsayma.
4. Probe RNG'yi, trajectory'yi, model buffer'larını ve guidance gradientlerini
   etkilememeli. Classifier'ın kullanılmayan mask üretimi de RNG tüketebilir.
   Probe graph'larını tutma; detachable/no_grad gözlem kullan. Karşılaştırılabilir
   hazırlanmış girdiler/state/noise hash'lerini ve örnek indekslerini kaydet.
5. Tek trajectory kaydı üzerinden terminal, min-entropy ve eşit **sınıf olasılığı**
   ortalamasını hesaplayan GPU'suz analyzer hazırla. Koordinat/label ID ortalaması
   yapma. Sabit checkpoint kontrolü ve yalnızca tanısal, etikete bakan oracle
   sonucu açıkça ayrılmalı. Düşük entropi != doğruluk != temporal kararlılık.
6. Artifact'ler: her corruption/seed/original index/checkpoint için logits,
   prediction, target (yalnızca analiz), entropy, loop index/timestep, candidate
   tipi/style zamanlaması; component süreleri, runtime/memory/source/assets/native
   kimlikleri ve completion durumu. Ara geometri saklama maliyetini sınırlı,
   açık bir seçenek yap; temporal logits analizi için tekrar GPU koşusu gerekmesin.
7. Gözlemci kapalı/açık terminal eşdeğerliği, RNG/buffer korunması, snapshot
   style/zamanlama ve analyzer sayım/policy doğruluğu için yerel CPU kontrollerini
   hazırla. CUDA gerektiren kontrolleri Colab smoke'a ayır; yerelde model inference yok.
8. Önceki notebook'un ilk6 hazırlık hücresinden sonra kullanılacak ayrı, açık
   Colab hücreleri hazırla. Python/GPU işlerinin tamamı
   `conda run --no-capture-output -n 3dd_tta_env python ...` ile çalışmalı.
   Mevcut build/lib generated Python farklarını tüm repo için temiz-kod guard'ına
   takma; gerçek inference kaynaklarını manifest/hash ile doğrula. Kullanıcı
   dosyalarını resetleme. Smoke/sonuç indirme/Drive yedekleme ve durak noktalarını belirt.

İlk araştırma kapsamı: all15, önerilen280 sabit örnek/corruption, tohumlar0/1/2;
280 = dört batch70. İndeksler henüz seçilmedi, pilot henüz koşulmadı. İlk aşamada
erken durdurma **uygulama zorunluluğu yok**: tam trace üzerinden seçim/ortalama
mekanizmasını ölç, sonra kilitli nedensel stop kuralını geliştir. Sayısal entropi
eşiği veya terfi sınırı icat etme. Sonraki geliştirme için onlineB1 arayüzünü ve
süre kayıtlarını düşün, ancak bunu yeni baseline/scheduler aramasına dönüştürme.

## Sonraki koşullu aşamalar

- Yarar varsa gözlenen prefix üzerinde güven + temporal tutarlılıkla erken
  durdurma; fallback ve minimum gözlem sayısı geliştirme verisinde belirlenip
  değerlendirmeden önce sabitlenir. Tüm trajectory'nin minimum entropisini seçmek
  erken durdurma değildir. Offline atlanan adım hesabı canlı hız ölçümü değildir.
- OnlineB1 erken çıkışını birinci hız hedefi olarak ölç; küçük/büyük batch için
  aktif örnek yönetimini ayrı değerlendir. Net kazanç = atlanan gelecek iş
  eksi ek probe/yönetim işi. Baştaki graph maliyeti erken çıkışla geri alınmaz.
- Başarı halinde matched GSD+SCD/SCD-only/GSD-only/unguided2x2 accuracy/süre
  matrisi. Sıfır katsayı gerçek compute skip değildir: mevcut kodda graph yine
  kuruluyor, losses ortak backward kullanıyor. Birini silme kararı henüz yok.
- Daha sonra kilitli adayla full2468x15x3. ModelNet40-C artık development
  kapsamındadır; farklı seed bağımsız benchmark doğrulaması sayılmaz. Timesteps
  bağımsız örnek değil; örnek/seed eşleşmeleri korunur.
- Purge-Gate hocayla konuşulacak konu ve olası karşılaştırıcı olarak kalır.
  Bu kapsamda Point-MAE token/BN değişikliği yok. M1800 ve başarısız
  Background100/1000 kollarını değiştirme veya yeniden çalıştırma.

## Okuma sırası ve kaynak kod haritası

1. `.agents/skills/iterative-thesis-researcher/SKILL.md`, `knowledge/README.md`,
   `knowledge/thesis_scope.md`, `knowledge/experiment_protocol.md`.
2. Bu handoff; `knowledge/gsd_trajectory_selection_20261011.md`;
   `knowledge/advisor_meeting_log.md`.
3. `knowledge/gsd_legacy_smooth_restore_20261008.md`,
   `knowledge/gsd_spectral_math_audit_20261008.md`,
   `knowledge/gsd_isolate_control_colab_20261010.md` ve baseline validation.md.
4. `knowledge/gsd_accuracy_research_plan_20261011.md` (son ekler eski PG-first
   sıralamasını güncelliyor), `knowledge/method_synthesis.md`,
   `knowledge/open_questions.md`, `knowledge/findings_log.md` tarihli son ekleri.

Kod:

- `tta_gsd_v2.py`: `tta_gsd_reconstruct`, DDIM temiz tahmini, guidance ve final decode.
- `eval_gsd_tta_v2.py`: CLI/model modları, `process_batches`, dönüşüm/FPS/classifier.
- `legacy_gsd_artifacts.py`: provenance, seeding, gözlemci, NPZ ve paketleme.
- `graph_spectral_v2.py`: seçilen Legacy graph; `graph_spectral_v2_offdiag.py`
  ayrı kontrol, varsayılan baseline yerine geçirilmemeli.
- `models_mate/Point_MAE.py`: `classification_only`, MaskTransformer RNG ve BN.
- `cfgs/tta_modelnet.yaml`: classifier kontratı.
- `gsd_paired_inputs.py`: hazırlanmış state/noise eşleme örüntülerine referans;
  farklı runner'ın matematiğini veya RNG protokolünü körlemesine aktarma.
- `tests/test_legacy_gsd_restore.py`, `tests/test_legacy_gsd_numerics.py`,
  `tests/test_shared_trajectory_decoder_control.py`, `tests/test_gsd_trajectory.py`:
  mevcut kontrol örüntüleri; sonuncular aynı legacy host varsayılmamalı.
- `scripts/gsd_guidance_ablations_colab.ipynb`: kullanıcının ilk6 hazırlık hücresi.
- `scripts/gsd_legacy_isolate_git_colab.ipynb`: conda/Git/handoff örüntüsü;
  kullanıcıya ait yerel notebook değişikliklerini koru.

## Çalışma ağacı ve sınırlar

Tracked ve untracked kullanıcı değişiklikleri mevcut; yeni knowledge dosyalarının
bir kısmı henüz untracked. Yeni session bunları GitHub'da veya HEAD içinde var
saymamalı, çalışma ağacından okumalı. `git status` al; reset/clean/silme yok.
Ham ZIP'i değiştirme. Yeni türev çıktılar ayrı dizinlerde. Yerel GPU/inference
yok; Colab'ı kullanıcı çalıştıracak. Stage/commit/push yalnızca yeni açık talep
olursa. Önceki push iznini bu yeni değişikliklere otomatik uygulama.

Teslim: değişen dosyalar, uygulanan kapsam, CPU kontrolleri ve sınırları, conda
Colab hücreleri, smoke/pilot durakları, tam ZIP'in nereye kaydedileceği/indirileceği,
güncel knowledge ve henüz ölçülmeyen accuracy/runtime soruları.

## Yeni session için kopyalanabilir prompt

```text
D:\Akademik\Okul\Thesis\code\3DD-TTA reposundaki tez çalışmasına devam edeceğiz.
Branch gsd-smooth-spectrum. Önce git status ve mevcut HEAD'i incele; tracked ve
untracked kullanıcı değişikliklerini koru. Reset/clean/stage/commit/push yapma.

iterative-thesis-researcher becerisini kullan. Önce knowledge/README.md ve
knowledge/gsd_trajectory_session_handoff_20261011.md dosyalarını oku; handoff'un
okuma sırasındaki ilgili knowledge ve kaynak kod referanslarını takip et.
Başlangıç: son doğrulanmış Legacy B70/M800/10-30, seed0 macro15 %64.570502.
Tarihsel %64.5327 pratik olarak yeniden üretilmiş kabul ediliyor. Offdiag veya
Phase6 C_SCD'yi baseline yerine geçirme. Ham ZIP ve SHA handoff'ta; değiştirme.

Handoff'un ilk geliştirme kapsamını uygula: mevcut Legacy davranışını koruyan
opt-in sparse trajectory gözlemcisi; ara clean latent'i aynı pre-update style
ile decode edip sabit Point-MAE'den logits alma; ayrı GPU'suz analyzer ile
terminal/minimum entropy/eşit olasılık ortalaması karşılaştırması; provenance,
RNG korunması, CPU testleri ve conda içinde çalışan Colab smoke/pilot hücreleri.
Önce kısa uygulama planını belirt, sonra bu kapsamı geliştir. Yerelde GPU/model
inference başlatma; Colab'ı ben çalıştıracağım. İlk6 ortam hücresi
scripts/gsd_guidance_ablations_colab.ipynb içinde; sonrasında kullanılacak
hücreleri, durak noktalarını ve sonuç ZIP indirme/Drive yedeğini açıkça ver.
Python işleri conda run --no-capture-output -n 3dd_tta_env python ... ile çalışsın.

Point-MAE ağırlıkları/BN/token akışı sabit kalsın. Probe'lar trajectory/RNG'yi
değiştirmesin; clean latent ile fiziksel point cloud'u ve original terminal
decode'u ayır. Düşük entropi doğruluk/temporal kararlılık garantisi değildir.
İlk aşamada erken durdurma veya GSD/SCD kaldırma zorunlu değil; önce aynı tam
trajectory üzerinde seçim/ortalama faydasını ölçebileceğimiz altyapıyı tamamla.

Online kullanım hedefini koru: B1'de erken çıkış kalan diffusion hesaplarını
doğrudan atlar; büyük-batch bekleme uyarısını B1'e genelleme. Daha sonraki
nedensel stop aşamasında net B1 gecikmesini ek probe maliyetiyle birlikte aynı
B1 full-horizon baseline'a karşı ölçeceğiz. B70 accuracy/throughput ayrı hat.
Mevcut10/30 corruption-name horizon'unu ilk gözlemcide değiştirme; bilinmeyen
corruption'lı online kullanımda horizon seçimi açık soru olarak kalsın.

Purge-Gate yalnızca hoca gündemi/olası karşılaştırıcı; bu geliştirmeye ekleme.
M1800 ve başarısız Background100/1000 kollarını koru; tamamlanan deneyleri
gereksiz tekrarlama. Eşik/accuracy artışı/hızlanma uydurma. Değişiklikleri ve
CPU doğrulamasını raporla, ilgili knowledge kayıtlarını güncelle.
```


## Later 2026-10-11 — İlk geliştirme kapsamı uygulandı

[Code/CPU/Open] Bu dosyadaki ilk geliştirme kapsamı working tree'de uygulandı.
Önceki “yeni inference geliştirmesi yok” ifadeleri önceki handoff session'ının
tarihsel durumudur. Güncel giriş: [uygulama ve Colab teslimi](gsd_trajectory_implementation_20261011.md).
Opt-in off/audit/observe, doğru pre-update clean decode, ayrı terminal, CPU
analyzer, provenance ve sentetik CPU testleri eklendi. Mevcut notebook'lar
korundu; yeni devam notebook'u ve SHA sabitlenmiş kaynak ZIP'i hazırlandı.
GPU smoke/pilot kullanıcı tarafından henüz çalıştırılmadı. Baseline, M1800,
Background100/1000, onlineB1 açıklaması ve mevcut10/30 horizon aynen korunur.


## Later 2026-10-11 — Seed0-first launch correction

[User decision/Code/Open] User postpones seed1/2 to save time. Current smoke and
all15/280 pilot launch only seed0; pilot count4,200. Prior three-seed preparation
text is historical and superseded operationally. Keep the fixed IDs and10/30
horizon. Conditional seed1/2 replication follows seed0 review and policy lock;
no seed-stability/significance claim or numerical promotion gate is established.
Use the revised source package and [current Colab handoff](gsd_trajectory_implementation_20261011.md).


## 2026-10-11 - Latest pilot handoff state

[Run/CPU/Open] Seed0 all15x280 pilot completed and was independently reanalyzed from the user's handoff ZIP. Policy gains are modest on this selected subset and Background regresses; see [full pilot evidence and paired corruption results](gsd_trajectory_implementation_20261011.md). Next, analyze entropy-weighted probability averaging offline from saved logits as an exploratory policy; then decide whether to freeze a policy for seeds1/2. No early stopping, latency gain, full-data accuracy claim or baseline change is supported.


## 2026-10-11 - Colab preflight rerun behavior and current handoff

[User report/Open] In the full-data all15 section of `scripts/gsd_guidance_ablations_colab.ipynb`, the first execution of its preflight code cell reportedly completed without error; rerunning that same cell produced a notebook `CalledProcessError`. The shell deliberately refuses a second initialization when `/content/legacy_trajectory_full_session.txt` already exists, to protect the session and avoid overwriting outputs. This is the expected behavior if the initial pass created the session pointer. The reported traceback showed only the IPython wrapper, not the underlying Bash stderr, so the specific guard is a strong inference rather than directly confirmed from the printed error line. If the first pass printed `Preflight passed`, keep the pointer/session and proceed to the next cell, ?Start the full all-15 run?; do not rerun preflight or delete its pointer. The inference cell separately refuses any existing run output path.

[User report/Run/Open] As of this handoff, the user has not reported starting the inference cell or receiving inference output. Treat the full 37,020-example seed0 run as pending. Current inspected local branch is `gsd-smooth-spectrum`, HEAD `a351e66ec58f08f3a2104e46cab2cb8b20cdf961`. The workspace has many pre-existing tracked and untracked user/research changes, including the Colab notebooks, PDFs, results and knowledge records. Preserve them; no reset, clean, deletion, stage, commit or push.

## 2026-10-11 - User confirms preflight success and active full-data inference

[User report] In the continuation session, the user explicitly confirmed:
`Preflight passed` was printed and the `Start the full all-15 run` inference
cell is running. This supersedes the preceding launch-pending status. No
completion log, runtime session path, or full-data result ZIP has been supplied;
the run is in progress by user report, not independently verified complete.

[Code] Local Git was rechecked: branch `gsd-smooth-spectrum`, HEAD
`a351e66ec58f08f3a2104e46cab2cb8b20cdf961`, with existing tracked/untracked
user changes. The four full-data notebook cells were read without editing.
Preserve the active Colab process, session pointer and output directory; do not
repeat preflight/inference, old pilot cells, setup or Run all. After inference
ends, run `Validate, package, back up to Drive, and prepare download`, then
the download cell. On an ordinary error with the runtime alive, package the
partial attempt before any retry; a hard reset can lose local partial files.
The full-data inference cell invokes eval directly: its Drive backup occurs
in the separate packaging cell, unlike the older pilot launcher's backup.

[Open] Obtain the complete `full_all15_seed0_*_handoff_*.zip`, including logs,
run plan, source/config/state evidence and nested run ZIP. Check the printed
`drive_backup_status: verified` and SHA-256. Review all15 x 2468 coverage,
provenance and frozen model state before reporting same-trace terminal,
minimum entropy and equal probability mean; inspect corrected/broken counts,
macro15/macro14 and Background separately. No full-data accuracy, elapsed
runtime, gain, seed stability or speedup is established. No tau is selected.

## 2026-10-11 - Full-data seed0 screen completed and analyzed

[Run/CPU] Complete user ZIP is preserved at
`result/modelnet40_c/legacy_trajectory/full_all15_seed0_20261011T154536Z_handoff_20261011T171147Z.zip`,
SHA-256 `44f5366448263dc02d7932111685948ed0a5ead39d4a8f53c3bda4ec2969d07a`.
Outer/nested CRC, all15x2468 coverage, seed0, source commit/hash manifest and
frozen model state validate. Independent CPU analysis is at the matching
`..._ingested/independent_analysis/` directory. No raw result was changed.

[Run/Inference] Mean65.156672% (+0.564560pp) and minimum entropy65.051324%
(+0.459211pp) versus same-trace terminal64.592112%. Mean correct/broken
transitions953/744; Background mean is-2.6742pp, minimum entropy-3.4846pp.
One-seed conditional index intervals do not measure seed stability. The mean
is descriptively+0.5862pp over the separate accepted Legacy64.570502% run,
not paired. Thus evidence is encouraging but below+1pp and not ready for broad
promotion. Instrumented time80.4min, wall span86.1min; previous Legacy68.98min.
Approximate separate-run comparison only; no matched speedup. No early stopping.

Next: inspect the full per-corruption table and same-cost terminal voting;
consider offline entropy-weighted means on saved logits without choosing tau
confirmatorily on this set. If signal is retained, conditionally repeat seeds1/2,
then seek a separate evaluation. Keep the accepted Legacy baseline and do not
claim generalized performance or reduced runtime.


## 2026-10-11 - Common-horizon development focus (user decision)

[User decision/Run/Open] Prioritize the14 non-Background corruptions sharing
10 reverse steps. The user points out that selecting30 steps specifically for
Background assumes corruption identity at inference; Background repair is no
longer the next development objective or candidate-selection requirement.
Preserve and report the existing Background/all15 legacy results separately.
On the existing full-data seed0 trace, macro14 terminal/minimum entropy/equal
mean are64.960060/65.700972/65.755962% (+0.740912/+0.795902pp vs terminal).
Use one shared label-free policy across14; per-corruption reporting must not
become corruption-specific inference routing. These are macro14 development
results, not all15 improvements or demonstrated unknown-corruption deployment.
No horizon/model/code/raw-artifact change or new inference. Entropy-weighted
averaging remains a proposed offline candidate, with no tau/threshold selected.
See [updated decision and limits](gsd_trajectory_selection_20261011.md).

### 2026-10-11 - Macro14 offline policy analyzer complete

[Code/CPU] New analyzer and tests are `scripts/analyze_legacy_trajectory_macro14.py` and `tests/test_analyze_legacy_trajectory_macro14.py`. It verifies complete seed0/all15 provenance and frozen state, then evaluates shared policies only on the14 common10-step corruption types; no inference rerun or tau fitting. Results: `..._ingested/macro14_policy_analysis_v2/`. Entropy-weighted mean65.7270% (+0.7670pp vs terminal), slightly below equal probability mean65.7560% (+0.7959pp). Weighted vs equal mean:26 corrected,36 broken. Keep equal mean as the stronger development candidate, not yet promoted; seed stability and external validation remain open. The3 focused tests and45 existing regression tests passed. Raw archive and user notebooks remain unchanged.

## 2026-10-11 - Next-session priority: sampling smoke, not old full-data rerun

[User decision/Code/CPU/Open] User authorized the explicitly referenced two-view sampling-consistency experiment. Read `gsd_sampling_consistency_20261011.md` for the plan, controls and exact paper titles. The complete implementation is in the working tree; base HEAD remains `a351e66ec58f08f3a2104e46cab2cb8b20cdf961`. Delivery is a pinned source snapshot plus declared/hash-audited overrides, not a Git-only source revision.66 CPU tests and extracted-source preview pass; CUDA not run.

Use `result/preparation/trajectory_sampling_source_20261011_v1.zip` (SHA-256 `6e42659c5cb3684a03bd4b741e098385877dadaaa1c45d971b8a44a10b005686`) with the appended notebook section **Sampling consistency: macro14 seed-0 smoke and pilot**. Existing21 cells were retained and9 appended. The prepared `/content/3DD-TTA` and `3dd_tta_env` are reused for assets/environment, while source is unpacked into a new directory. Pointer is `/content/legacy_sampling_session.json`; preserve it. Run only new preflight, Gaussian140 smoke and download, then review full ZIP before the separate3920-example pilot. Normal failures are packaged and copied to Drive with SHA verification; optional recovery packages surviving partial files. No Run all, old pilot/full-data repeat, GPU launch locally, stage/commit/push, raw ZIP change, or success claim before artifacts arrive.


## 2026-10-12 - Sampling source delivery v2 (current)

[Code/CPU/Open] Base remains `gsd-smooth-spectrum@a351e66ec58f08f3a2104e46cab2cb8b20cdf961` plus the declared hashed working-tree overrides. Use `result/preparation/trajectory_sampling_source_20261011_v2.zip` (642,170 bytes; 318 source files), SHA-256 `d0c48e7f5a90f8fe19bfc615961e0ed1db3451b19cf5c98494cb613181b57295`. This supersedes v1 for new launches; v1 is preserved.

The standalone CPU analyzer now independently compares recorded checkpoint/config/label/selected-data SHA-256 and sizes with the accepted baseline specification and verifies that specification against source delivery. Changed or absent identities are rejected before scoring. The launcher still hashes actual assets before inference.67 targeted CPU tests pass; extracted v2 CRC, all source SHA checks, CPU macro14 preview (34,552 full-scope examples; the staged runs remain140/3920), and Python3.8 syntax checks pass. Tests use synthetic CPU fixtures, not native CUDA.

The notebook now requests v2 and its exact SHA. Only the filename/hash in the newly added section were revised; original21 cells are unchanged. Amendment receipt: `result/preparation/trajectory_sampling_notebook_receipt_20261012_v2.json`. The plan and its explicit paper attribution remain in `knowledge/gsd_sampling_consistency_20261011.md` (sampling variation/weight averaging paper, MEMO, AugMix; our H+JS selector is an unvalidated synthesis).

Next: user-operated Colab preparation, new sampling preflight once, Gaussian140 seed0 smoke, then download/provide its complete handoff ZIP for review. Pilot macro14x280 remains separate and conditional on smoke review. No Run all, old-run repetition, local GPU, stage/commit/push, or raw baseline changes. Native CUDA behavior, accuracy, cost and repeatability remain open; failed invariants or no paired gain over the controls would reject promotion. Acquisition14 forwards is not deployment latency for the8-forward method.
