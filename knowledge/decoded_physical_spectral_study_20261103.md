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
