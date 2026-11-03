# Encoder öncesi fiziksel graf / decoder spectral guidance — 2026-11-03

[User report/Decision] Kullanıcı önceki altı hipoteze ek olarak encoder öncesi
nokta bulutunun graph spectrum'u ile PM'de zaten decode edilen çıktının
karşılaştırılmasını ve Colab incelemelerinin başlamasını istedi.
[Code] Yeni çalışma mevcut `research/gsd-guidance-math-audit-20261103` worktree'sinde;
latest integration2177493 merge'i9729513. Önceki rapor ve kaynaklar korunur.

## Matematik ve test edilecek ek hipotez

[Inference] Aynı hazırlanmış X (B×2048×3, LION scale/rotation, encoder'ın gerçek
girdisi) üzerinde detached graph L_X kurulur. Y_t=D(hat_h0(h_t,s_t),z0).
L_phys=trace((Y_t−X)^T F_X(Y_t−X))/(3N), F_X=exp(−beta Lbar_X).
Aynı referans baz/filter iki sinyale uygulanır. Bağımsız U_X/U_Y katsayılarını
hizalama olmadan karşılaştırmak geçerli değildir. Raw değişken-N/FPS edilmiş
classifier input'u doğrudan2048-slot decoder çıkışıyla çıkarılmaz.
Slot correspondence residual LION decoder sayesinde makul, fakat doğrulanmalıdır;
NN/set-based hata ve slot hatası birlikte ölçülür. Bu loss hâlâ bozuk X'e
sadakattir; önceki corrupted-anchor bias hipotezini otomatik çözmez.

[Code] SharedTrajectoryObserver.clean decoder'a detached latent/style verir ve
isolated_probe no_grad kullanır. Bu yüzden PM decode'u gradient açısından
halihazırda ücretsiz hazır değildir. Yeni opt-in observer seçili PM adımlarında
tek **live** decoder forward'unu paylaşacak, PM classifier'a detached çıktı
verecek, spectral gradienti decoder/denoiser üzerinden taşıyacak. Classifier
backward gerekmeyecek. Native PVCNN grad-enabled/no-grad branch farkları nedeniyle
aynı RNG tek başına exact output parity garantisi değildir; doğrudan ölçülecek.
Final decode original z0; style guidance yalnız sonraki prior adımlarını etkiler.

## İlk bilimsel faz: bounded technical/diagnostic, benchmark değil

[Proposal] ModelNet40-C/s5, fixed Point-MAE ve raw/eval/EMAoff LION,
normal8/Background35, local.03/style.003/p.95/original-z/finalalpha1.
Gaussian/Background/shear/LiDAR; başlangıçta iki nesne ve sonra en fazla8nesne,
seed0. Primary amaç actual connected gradients, correspondence/decoder domain,
spectral/SCD direction ve fresh compute/memory. Önce non-applying probes;
teknik gate geçerse yalnız sabit küçük weight-zero/control ve label-free
scale'li opt-in smoke; geniş weight/beta/M araması veya full benchmark yok.
Exact IDs, source/data/checkpoint hashes ve native identities çalıştırmadan
önce manifestte dondurulacak. Mevcut kabul edilmiş accuracy koşulları tekrar
edilmeyecek; bu yeni diagnostics veya yeni objective koşuludur.

Bilimsel GPU fazı tahmini20–45dk, bütün science/technical retries dahil55dk
(3300s) cumulative watchdog. Setup ayrı. Bu ilk faz≤60dk beklendiği için ek
süre onayı gerekmiyor. Devamın60dk aşması beklenirse, genişletmeden önce proje
kuralı uygulanacak. Önceki agent çalışmasına ait süre muafiyeti taşınmaz.
Eksik setup/native/data/checkpoint veya nonfinite gradientte science durur.
Loss düşüşü accuracy artışı sayılmaz; small prefix yalnız smoke/diagnostic.

## Runtime sahipliği ve hazırlık

[Run/Open] CLI0.7.4 ile server inventory boş doğrulandı. Yeni own session:
`tta-gsd-physical-spectral-a100-20261103-r1`, endpoint
`gpu-a100-hm-ass1c0-i933qghuewn8`, A100-SXM4-80GB/High-RAM.
Owner/root task: bu decoded physical-spectral araştırması; ayrı source worktree
`/Users/batuhan/Desktop/projects/3DD-TTA-worktrees/gsd-guidance-math-audit-20261103`.
Cell1 ayrı çalıştı, aynı VM'ye reconnect/Conda/GPU doğrulaması geçti.
Cell2 canonical env kurulumu yalnız kullanıcıca ertelenmiş fastrlock0.8.4 wheel hatasında durdu;
Python3.8.20 environment ayrı subprocess ile doğrulandı. Cell3–10 henüz yürütülmedi.
No research inference yet. Teardown izni verilmedi; runtime korunacak.

Yerel wrapper/log/manifest kökü:
`result/preparation/gsd_physical_spectral_20261103/setup/`.
Drive hedefi: `/content/drive/MyDrive/thesis/result/gsd_physical_spectral_20261103/`.
Science hamları source/env/RNG/state/IDs, standard7files+probe/prediction NPZ,
failure/complete status ve SHA/CRC ile saklanacak. Tarihsel native build'lerle
aynı byte iddiası yapılmayacak; bütün paired kollar yeni ortak build'i kullanacak.
Cell2 primary exec84331 tamamlandı; environment verify37904 tamamlandı.
Drive mount TTY CLI70086 kullanıcı consent/Enter bekliyor; başka cell çalıştırılmıyor.
Ayrı monitor henüz yok. OAuth URL/state arşive veya knowledge içine kopyalanmaz.

[Open] H7: physical-domain fidelity aynı update scale'de latent fidelity'ye
göre daha yararlı yön üretiyor mu; PM forward reuse sonrası **net** hız/bellek
maliyeti nedir? Temiz/geometrik yön ve küçük applied-step etkisi ters düşerse
hipotez zayıflar. Sonuç önceden seçilmez; default baseline korunur.

## Primary-literature bağlantısı ve uygulanmış prototip

[Paper/Inference] [Functional Maps, §3–4](https://geometry.stanford.edu/paper/fmfrmbs-obsbg-12/fmfrmbs-obsbg-12.pdf)
ayrı şekil fonksiyon bazlarını bir correspondence operatorüyle ilişkilendirir.
Burada şimdilik daha küçük varsayım kullanılır: aynı vertex-order ve aynı F_X.
Slot/set hata farkı büyükse, doğrudan coefficient residual'ı güvenilir guidance
kabul etmeyiz; spectral map/matching ayrı yeni hipotez olur.
[PSLD, §3/Algorithm2](https://papers.nips.cc/paper_files/paper/2023/file/9c70cfa2e7d9328c649c94d50cbf8faf-Paper-Conference.pdf)
latent diffusion'da decoder üzerinden measurement guidance için ilgili bir
öncüldür. Buradaki bozuk-X graph loss'u için recovery/accuracy garantisi vermez.

[Code/CPU] `decoded_spectral.DecodedSpectralObserver`, `tta.py` default-off
`decoded_spectral_weight`, `scripts/diagnose_decoded_spectral.py` ve iki dar test
hazırlandı. İlk scalar reference prepared input ile tam eşitlik kontrolü;
bir live decoder forward/checkpoint; classifier detached; sadece explicit
pozitif ağırlıkta SCD+physical scalar joint backward. Local ve conditioning-style
etkisi toy sampler'da gösterildi; original z0 decode korunur. Weight0 state/RNG
parity ve decoder directional FD CPU kontrolleri geçer. Yeni testin doğrudan
module çağrısındaki import namespace hatası düzeltilip7targeted test geçti.

[Code/Proposal] Koşucu önce control/diagnostic kollarını ayırır: no-grad PM'den
live PM'ye geçişin tek başına sayısal etkisi ölçülür. Physical weight karşılaştırması
aynı live observer'ın zero-weight kolunu da kullanmalıdır. Native branch farkı
ve farklı VJP maliyeti ortak draw ile otomatik giderilmez.
B2/seed0 ID prefix[44,91,103,137,158,223,236,265]; current normal2/4/6+8,
BG9/18/27+35. Diagnostic observer'da ek spectral VJP'ler, ayrıca bir SCD forward
ve VJP vardır; bu maliyet optimized deployment latency diye yorumlanamaz.
Unit-weight SCD/physical update norm ratio/cosine'ları kaydedilir; gelecekte
label-free calibration mümkün, bu faz henüz herhangi katsayıyı dondurmadı.

[Open] Drive onayı bitmeden Cell3 wheel/data/weights ve sonraki native setup
başlatılamıyor. Runtime canlı korunur; setup bitmiş veya scientific GPU işi
çalışmış sayılmaz. Prototip gerçek native decoder backward desteği ve nodal
correspondence bakımından henüz doğrulanmadı.

## Bağımsız code review — CPU hazırlığı

[Code/CPU] Reviewer ilk worker denemesinde `capture['initial_h0_xyz']` eksikliği
buldu (P1). Nested recorder dosyaya anchor yazıyor, diagnostic SCD hesabının
kullandığı in-memory key'i doldurmuyordu. `record_reference` şimdi **aynı**
cloned/detached h0XYZ'yi capture ve reference.npz'ye yazar. Yeni regression
B2'nin ilkXYZ slotlarını ve dosya/capture eşitliğini kontrol eder; önce fail,
sonra8targeted test geçti. Diğer material review finding yok.
Native çalışmadan önce bulundu/düzeltildi; henüz başarısız GPU job yok.

Devam için tek zorunlu dış adım: açık Google Drive consent ekranında kullanıcı
onayı. Kaynak kural: `.codex/skills/colab-research-operator/SKILL.md`,
“interactive account consent remains user-operated”. Onaydan sonra CLI70086'ya
Enter ver, gerçek mount ve KNN wheel'i doğrula; canonicalCell3, ardından
`setup/run_remaining.py` ile Cell4–10'un her marker'ını ayrı kabul et.
Science için setup/deployment+strictCPU gate'lerini atlama.

[Run: CPU] İnceleme sonrası son kaynak ağacında canonical first-party unittest:
473test çalıştı,472pass/1historical-source skip,26.555s;8targeted test geçti.
[CPU/source/setup receipt](evidence/decoded_physical_spectral_20261103/cpu_verification.json)
ve `result/preparation/gsd_physical_spectral_20261103/cpu_suite_reviewed.log`.
Bu code-review ve CPU hazırlığıdır; native/Colab inference sonucu değildir.

## Kaynak dondurma / devam noktası — 2026-11-03

[Code/Run/Open] CPU-verified scientific source commit
`c415e874ce06f1020f4caf80d26ed64525857f8b`;247-file recoverable overlay
676893bytes/SHA256`ecebdcd6dea47fce073a724da01f2f38e6df9307c6813e234e6b460ddc30b98d`.
Operator package670897bytes/SHA256`4f40f5f59222f9e30cad38ee3d97206f6e042f207e8757ee6ac06778d46c0d45`
CLI upload ile own VM `/content/gsd_physical_operator.zip` yoluna aktarıldı;
remote byte verification/deployment henüz çalışmadı. Notebook wrapperCell3'e
canonical source değiştirilmeden gerçek mount/KNN wheel SHA assert'i eklendi.

Yerel partial CPU/setup evidence `result/preparation/gsd_physical_spectral_20261103/cpu_setup_preparation_partial.zip`:
1395637bytes/SHA256`a17e0238787c291d8ca8381c01e8f884e4badaceef133857e1846b6f6469f310`,42üye/CRC geçti.
Drive backup mount bekliyor; OAuth consent URL/state arşive alınmadı.

Sonraki sıra: kullanıcı consent → CLI70086 Enter → gerçek mount →
canonicalCell3 (`setup/cell_03.py`) → ayrı marker-gatedCell4–10
(`setup/run_remaining.py`) → operator `install_phase.py` (exact package SHA,
pinned asset base/native preservation+restore, strictCPU) → first Gaussian/B2
control/diagnostic. Science start/deadline kaydı ilk actual technical GPU job'dan
önce yapılmalı;55dk cumulative cap yeniden başlatılmaz.
Bilinen dört native binary'yi eski doğrulanmış paketten restore etmek için
hazırlık var; fresh bytes önce korunur ve restore açıkça kaydedilir.
Bunun yapılmış veya CPU deployment'ın geçmiş olduğu henüz iddia edilmez.

Task hâlâ aktif; izin bekleyen runtime açık. Yeni prototype integration'a
yayınlanmadı; source branch'te commitli, payload yerelde/ownVM'de korunuyor.

## 2026-11-03 — Drive consent received; persistent owner monitor started

[User report/Run/Open] User states Drive authorization granted and directs
continuation. Existing CLI70086 propagated credentials but returned a reply
timeout (the original mount request had waited). Same r1 assignment remains;
read-only file API cannot yet find MyDrive/thesis. No inference or duplicate
setup job launched. Read-only same-kernel Drive/process audit submitted before
any bounded mount retry.

[User report/Run] User asks whether the persistent background monitor is active.
It was not previously running; root starts the project monitor at19:23:42UTC,
local exec97932, expected endpointgpu-a100-hm-ass1c0-i933qghuewn8/A100,60s cadence,
540s bounded idle liveness. Ownership/source hash in setup/monitor_ownership.json.
It is now sole CLI cell owner; all phases use stdin run controls. No fake
keep-alive/peer session action. This supervision does not guarantee Colab lifetime.
Runtime stays open; no teardown permission. Exact audit/mount outcome is pending.

## 2026-11-03 19:36UTC — Drive mounted; canonical preparation continues

[Run] The original consent request timed out after credentials propagated.
A same-kernel read-only audit confirmed no mounted MyDrive, then the bounded
Cell3 retry passed with real mount and pinned KNN wheel verification at19:28:34UTC.
Cell4 started19:29:02UTC through sole-owner monitor97932. PyTorchcu121/KNN
installation passed; requirements preparation still running at19:36UTC.
The server still lists the recorded A100 endpoint. No research inference has
started. CanonicalCell5–10, deployment and actual native decoder acceptance
remain gated on their own exact completion markers.

[Code/CPU/Open] Added opt-in `--native-decoder-audit` to the diagnostic arm.
At the first selected checkpoint per B2 case, before any observer backward,
six gradient-enabled decodes check the local frozen-graph derivative using
two central-difference scales(.001,.01 times first-sample latent norm).
Tolerance1e-5+.1|AD| and repeat-noise slope checks are frozen before CUDA;
unresolved derivatives fail. RNG, buffers and live inputs must remain unchanged.
JSON/finite NPZ evidence is saved before failing the technical gate. This
certifies neither whole-denoiser derivatives nor decoded/input point matching.
Independent review found/fixed the premature observer backward, and an
executable regression verifies failure artifacts precede that call.
Eleven focused CPU tests pass. Source sealing/full-suite verification and
remote native results remain pending; no accuracy or speed claim.

Continuation: poll monitor97932, submit Cell5–10 individually only after the
prior exact completion marker; never use the old direct-exec supervisor while
the monitor owns the runtime. Seal a new operator_v2 with the audit helper and
updated driver; preserve V1 payloads. Then deployment/strictCPU gate followed
by Gaussian/B2 control and diagnostic. Start the immutable55min cumulative
science deadline immediately before the first scientific/native-GPU job.

### Predeclared small applied-step scale (before native inference)

[Inference/Plan] If Gaussian/B2 technical FD and preservation gates pass, set
one candidate `weight = .1 * median(first-checkpoint per-object
unit_weight_scd_over_physical_update_ratio_per_object)` over the two fixedIDs.
The ratio concatenates local/style update vectors after their fixed.03/.003
step sizes. Require both ratios finite and positive; otherwise no applied arm.
Freeze formula/value before consulting selected-ID accuracy. No grid, fallback
rho or accuracy-selected coefficient. This is a small update-direction smoke,
not an accuracy estimate. Current diagnostic extra VJPs/FD forwards must be
excluded from any optimized-deployment latency claim; report measured boundaries.

## H8 — Decoder residual freeze: cheap physical surrogate, not an exact gradient

[Code] `models/latent_points_ada.py:255–272` returns
`D(h,z)=Πh+s R(h,z)`, whereΠ selects each latent point's first3channels ands
is the checkpoint's configured `latent_pts.skip_weight` (its actual value
must be read from the resolved config). The current proposal does not require
assuming that the full decoder is linear or identity.

[Inference] At checkpointt compute the ordinary PM decodeYt and freeze
`bt=stopgrad(Yt−Πht)`. Then use the local surrogate

\[
\widetilde L_t(h)=\frac1{3N}\langle\Pi h+b_t-X,F_X(\Pi h+b_t-X)\rangle.
\]

At the same point its value equals the physical loss, but gradients differ:

\[
g_X=\frac2{3N}F_X(Y_t-X),\quad
g_{sur}=\Pi^\top g_X,\quad
g_{true}=\Pi^\top g_X+sJ_R^\top g_X.
\]

There is no decoder backward in the surrogate. Its remaining denoiser VJP
can join the existing SCD backward; graph/filter work and decoder forward
still cost time. Error is `s J_RᵀgX`, bounded by
`|s| ||J_R|| ||gX||`. If its norm is strictly smaller than `||g_sur||`, then
`<g_true,g_sur>>0` (sufficient, not necessary). After the denoiser Jacobian and
local/style step sizes this condition must be checked in actual update space;
local decoder alignment alone cannot certify the sampler's direction.

[Open] First use saved native-audit gradient/decoded/filter arrays to compare
the cheap local gradient with the exact local gradient, including cosine,
norm ratio and feature-channel residual. Then a separate declared paired
directional test must check the actual physical loss under small updates,
including frozen decoder branch/RNG and upstream denoiser. Compare total cost
and correspondence bias before any accuracy claim. This is an explicitly
different iterative surrogate; never label its omitted residual Jacobian an
exact or bug-fixed physical gradient. Current technical/applied full-gradient
gates are unchanged; native failure does not silently authorize a surrogate arm.

### Additional primary reading for the gradient/cost distinction

[Paper] [Chung et al., Diffusion Posterior Sampling, §3.1–3.2/Eqs10–16](https://arxiv.org/html/2209.14687v2)
approximates the measurement likelihood through the predicted clean sample;
the noisy-variable gradient is still a chain-rule gradient. Its observation
operator/noise assumptions do not establish a posterior interpretation for
our heuristic corrupted-point-cloud heat fidelity.

[Paper] [Mohseni-Sehdeh et al., Piecewise Guidance,2025 arXivv1, §III/Eqs7,11/Algorithm1](https://arxiv.org/html/2507.18654v1)
switches between a cheap low-noise approximation and denoiser-Jacobian guidance
for a linear noisy measurement model. It explicitly treats VJP computation
as a cost. This is related motivation for checking approximation error against
speed, not the same algorithm as freezing a LION decoder residual. Its image
restoration results do not predict point-cloud classification accuracy. No
novelty claim or extra GPU arm follows from these two additional readings.
