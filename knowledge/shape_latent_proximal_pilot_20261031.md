# Global DDIM sırasında temiz-latent önerisine yakınlık guidance'ı

2026-10-31. [User report/Decision] Kullanıcı ek shape-latent hipotezlerini,
global diffusion dahil, aynı Colab oturumunu kapatmadan test etmeyi yetkilendirdi.
Önkoşulu önceki çalışmanın kaydı/yayınıydı: tamamlandı ve uzak hedef
`1bacabfe20a70bf94e9a3c8a5988084d0ae48d3f` ile doğrulandı.
[Önceki kabul edilmiş sonuç](shape_latent_refresh_results_20261031.md).
Yeni çalışma henüz GPU'ya gönderilmedi; bu prelaunch tasarım/planıdır.

Etiketler: [Paper], [Code], [Run], [User report], [Inference], [Open].
Worktree/branch korunur:7bf0/3DD-TTA,
integration/chamfer-guidance-variants-20261030,a822dirty. Kendi mevcut
`tta-shape-refresh-a100-20261031-r1` endpoint
`gpu-a100-hm-euw4a0-3e5wlo5wf3or6` yeni bu çalışmaya atanır; yeniden setup veya
runtime kapatma yok. Setup1–10/strictCPU/native/data receipt'leri geçerli,
yeni bilimsel source/CPU gate tekrar doğrulanır.

## Hipotez ve sınır

[Inference] Global prior'ın yalnız başına çekişi nesne kimliğini korumayabilir.
Önceki bozuk-z anchor/%5 sınırı yerine, local denoising'in sağladığı
`z_ref=mu_new+sigma0*eps_z` önerisini **her global temiz tahmininde** kullanmak,
global prior ile bu veri bilgisini birleştirebilir. Mean-refresh tek başına
yalnız+.8333puan ve CI0içeren sonuç verdi; target güvenilir sayılmaz.

[Inference/Proposal] İlk deneme gradient ölçeği aramak yerine kapalı-form
quadratic proximal guidance uygular:
`c_guided=argmin_c .5||c-c_prior||²+.5||c-z_ref||²=(c_prior+z_ref)/2`.
Bu, temiz tahminde bir data-consistency yaklaşımıdır; extra network backward
gerektirmez. Global **denoising ağı açık**, ağırlıklar frozen. DDNM/DiffPIR'ın
genel clean-estimate/data-consistency fikriyle bağlantılıdır; bu yöntemlerin
bilinen fiziksel ölçüm operatörünü/likelihood'unu veya tüm algoritmasını
uygulamıyoruz. Proxy latent'i true measurement veya calibrated posterior diye
sunmuyoruz. Kareli hedef enerjisini o prior tahminine göre1/4'e düşürmek
cebirsel kontrattır; temiz oracle'a/accuracy'ye iyileşme garantisi değildir.

## Beş sabit kol, küçük kapsam

32önceden exposed development nesnesi,seed0,B32,Point-MAE;
Gaussian/Background/Rotation/LiDAR. Teknik gate yalnız G+B, aynı32IDs.
N8/B35,gamma.03,eta.003,p.95,total100; four-candidate PM, raw/eval/EMAoff.

1. original: ilk z0, mevcut operasyonel referans.
2. mean: önceki z_ref önerisi, yeni phase içinde fresh matched readout.
3. prior3: z0'dan globalDDIM3, guidance yok.
4. proxy3: aynı global chain, her temiz tahmin z_ref ile yarı yarıya birleşir.
5. oracle3: yalnız mekanizma teşhisi, gerçek clean-encoder z'ye aynı guidance.

Global schedule20/10/0(modeltime21/11/1), start noise std~.079,DDIMη0,
final_alpha1. Üç prior çağrısı; ekstra global backward0. Beşinci kol temiz veri
kullanır ve deployable TTA sonucu sayılmaz. İlk dört kolun proposal API/CLI'si
clean target kabul etmez. λ1, tek schedule; grid/blend/radius taraması yok.
Önceki %5 küre uygulanmaz; global/local güncelleme adedi değiştirilmez.

Her adımda DDIM'in `pred_original_sample` tahmini c_prior alınır;
`prev_guided=prev_DDIM+sqrt(alpha_prev)*(c_guided-c_prior)`.
Bu, deterministic DDIM temiz bileşeninin değiştirilmesine cebirsel eşdeğerdir.
Prior3/proxy3/oracle3 aynı private-generator global noise'u kullanır;
local/classifier RNG akışı ilerletilmez. Tüm ara/global vektörler fiziksel
saklanır; sadece hash kaydedilmez. İkinci local zincirde h0/noise/anchor aynı,
clean/final decoder her kolun kendi ilk seçilmiş z'sini kullanır.

## Veri/state tekrar kullanımı

Önceki yayımlanmış refresh pilotunun ilk32IDs için dört bozulmadaki fiziksel
preparedX/z0/h0/noise/postprepare RNG ve refresh proof'u **salt-okunur study
input'u** olarak alınır; file/tensor/source/checkpoint/ID kimlikleri mühürlenir.
Yeni dört-bozulma sırası eski all15 öncesi RNG durumunu yeniden üretmez.
Bu yüzden rawX'i yeniden interpolate etmek yerine frozen preparedX ve RNG
boundary açıkça replay edilir; yeni selected cache'ler kendi fresh adlarında
oluşur. Yeni yöntemin computed proposal'ı current executed-source'a bağlıdır;
eski source-bound proof current code üretmiş gibi etiketlenmez.
Clean oracle hedefleri ayrı manifest/CLI alanında, yalnız oracle3 kolunda
model hedefi olur. Deployed kollarda oldclean posteriors load edilmez.

## Ölçüm ve kabul

Primary mekanizma: final global z'nin paired clean-encoder z'ye karesel
uzaklığının original/mean/prior3/proxy3/oracle3 karşılaştırması. Temiz hedefe
yaklaşmak sınıflandırma garantisi değildir; aynı readout ile actual accuracy
ayrıca raporlanır. Oracle3 gerçek yöntem/üst sınır değildir.
32nesne×4koşul=128olay/kol,32bağımsız nesne;10k object-cluster bootstrap,
seed20261031. Çok küçük/exposed/seed0, unseen claim veya yöntem seçimi yok.

Hipotez proxy3'ün prior3'e göre clean distance ve accuracy bakımından faydalı
olmasıdır. Oracle3 bile mesafeyi iyileştirmiyorsa implementation/scheduler
kontratı tekrar denetlenir; mesafe düşüp accuracy düşerse ayrı negatif sonuç.
Oracle3 çalışıp proxy3 çalışmazsa target kalitesi sınırlaması desteklenir,
global diffusion'un genel yararsızlığı kanıtlanmaz. Önceki sonuçlarla accuracy
transplant edilmez; yeni original/mean kontrolü içinde karşılaştırılır.

## Bütçe ve işlem

Whole planned GPU fazı teknik+küçük pilot+arşiv için15–25dk tahmin,
tek2400s üst sınır; önceki631s tamamlanmış/yayımlanmış ayrı çalışmadır.
Toplam bu iki bilimsel üst sınır bile~50.5dk; >60dk çalışma planlanmıyor.
Hata/nonfinite/source/state/RNG/proof mismatch'da dur, failure hamını koru.
Yerel root `result/preparation/shape_latent_proximal_pilot_20261031/`,
remote `/content/studies/shape-proximal/`, Drive aynı görev adı altında.
Model/tta/default refactor yok. Önce CPU RED→GREEN, fresh review,
exactsource seal/strictCPU; sonra G+B32 ve dört-bozulma pilotu.

## Native execution plan

Spec: this approved-user-delegated research design. Existing isolated worktree.
Tech: existing Torch/NumPy/diffusers/CLI. Review focus: correct DDIM alpha_prev,
half-target algebra, clean-target isolation, immutable reference lineage,
actual RNG/source/frozen-model/readout and incomplete-stage rejection.

- [ ] Tests first: zero-prior analytic3half-steps, deterministic noise/RNG,
  λfixed, no %5 projection, clean target rejected outside oracle, invalid state.
- [ ] Add opt-in `shape_latent_proximal.py` and fixed `scripts/eval_shape_proximal.py`;
  tests for imported state/ID/proof/source/scope and own-z routing.
- [ ] Seal read-only imported32-ID references/newsource; strictCPU deployment,
  fresh reviewer and operator contract checks.
- [ ] Same liveVM: G+B32 technical gate then fixed four-corruption pilot.
- [ ] SHA/CRC/raw/paired independent recount, own clean-state distance diagnosis;
  preserve negative/oracle limits; update knowledge and selectively publish.

Ruling: user delegated additional hypothesis testing and explicitly kept this
runtime; no new permission loop for this sub60min phase. Proximal closed-form
guidance chosen over simultaneous conditional-denoiser energy: fewer new
assumptions and no extra local Jacobian/metric-calibration step. It tests
global prior plus improved target directly; conditional energy remains unrun.

## Unexpected owned-runtime loss before new GPU phase

[Run/Failure/Open]17:50UTC initial payload upload returned sessionnotfound
for tta-shape-refresh-a100-20261031-r1; independent serverinventory0active.
No stop/release/delete or close instruction was issued. Last verified live
inventory17:36UTC; exact loss time/cause unknown. No new proximal GPU job
was started. Previous refresh fullraw/source/derived Drive+local and Git
1bacabf remain safe. New346file source and19test/operator preparation local.
Same-VM preference cannot be fulfilled after external assignment loss.
Within authorized short study/setup/retry scope, replacement requested as
tta-shape-proximal-a100-20261031-r1(A100/HighRAM); fullcanonical10setup
will be replayed, then new CPU/GPU gates. No other/unknownruntime touched.
Prior session/operator-v1 identities preserved; replacement registration
will be versioned. Research2400s ledger has not started.

## Yeni Codex oturumuna devir: doğrulanmış güncel durum

2026-10-31, son kontrol18:06UTC. [User report/Decision] Kullanıcı yeni bir
Codex session'ında sıfırdan başlayabilmek için kaynakları ve adım adım testleri
içeren prompt istedi; ardından mevcut durumun knowledge'a kaydını açıkça istedi.
[Tam bağımsız başlangıç promptu](shape_latent_proximal_handoff_20261031.md)
bu dosyanın başındaki eski same-VM/no-setup beklentisini günceller.

[Run] `colab sessions` yeni kendi `tta-shape-proximal-a100-20261031-r1`
oturumunu A100/High-RAM, endpoint`gpu-a100-hm-ass1c0-zwl345mh6780` ile gösterir.
Cell1 Conda restart/reconnect tamamlandı; Cell2 localCLI16663 exit0 ve exact
`SETUP_WRAPPER_CELL_2_COMPLETED` ile bitti. Log yalnız bilinen fastrlock wheel
pip hatasını gösterir; bunu erteleme yetkisi vardır. Python3.8 doğrulaması,
Drive mount, Cell3–10 ve strictCPU gate **yapılmadı**. Upload19745/50463
exit0 ve başarılı dosya mesajıyla tamamlandı. Yeni scientific phase başlamadı.

[Code/CPU]19yeni/69ilgili test+16subtest ve4operator check önceden geçti;
fresh review substantive RNG/style/source kontrolleri sonrası blocker bırakmadı.
346file source manifest`690caa62d2a2603ebf9480cc86e3a66f5d13fca77ec1dbe34a810cd19fde15f5`;
assetsZIP29970678byte SHA`347ef9c3c5304f2a168046cced7298b828fc387e93093b8de29d076c7276f6fa`;
operatorv2ZIP18507byte SHA`1f8a7da3fce912f694fe912233503e92217e2d09b7de4c54f7d60eaf09203220`.
Uzak fiziksel SHA/strictCPU henüz doğrulanmadı; CPUtesti deployment kabulü değildir.

[Code/Open] Devir kontrolünde dış `operator/install_and_deploy_wrapper.py`
içindeki registration session assertion'ının eski refresh adında kaldığı
bulundu. Sealedv2registration/run_phase yeni proximal adıyla doğru.
Yeni oturumda çalıştırmadan önce dış wrapper assertion'ı düzeltilip gerçek
ZIP registration'ıyla kontrol edilmeli; bu turda düzeltme veya GPU işi yapılmadı.
Yeni4×32/fivearm distance/cluster-bootstrap özeti de henüz yazılmadı.

[Run] Önceki refresh yayını uzak `origin/thesis-integration` üzerinde
`1bacabfe20a70bf94e9a3c8a5988084d0ae48d3f` ile tekrar doğrulandı.
Worktree/branch/HEAD a822dirty korunur. Eski sonuca yeni gain eklenmez;
operasyonel N8/globaloff default korunur. Kullanıcı runtime kapatma yetkisi
vermedi; bu devirde stop/release/delete çağrısı yapılmadı.
