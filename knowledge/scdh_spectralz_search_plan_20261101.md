# Local h için SCD, z/style için smooth spectral loss: yeni agent test planı

2026-11-01. [User report/Proposal/Code/Open] Kullanıcı yeni Colab oturumunda
local-SCD/style-spectral routing'in güncel smoothing ile yeniden denenmesini;
ayrı update parametreleri ve farklı M değerlerinin taranmasını istiyor.
Bu agent yalnız plan ve başlangıç promptu hazırlar. **Kod implementasyonu,
Colab allocation/setup, GPU çalışması, runtime sorgusu veya kapatma yapılmadı.**
Kapsam ve sayılar aşağıda bir yürütme önerisidir; sonuç/optimum iddiası değildir.

## 1. Başlangıç ve okunacak kayıtlar

- Güncel `origin/thesis-integration` fetch edilerek ayrı çalışma checkout'ı
  oluşturulmalı. Planın incelendiği ortak taban:
  `7cec870e09ed464b9b0dbeabbb7f834a224e5791`; ilerlemişse yeni kayıtlarla uzlaştır.
- Kaynak kullanıcı worktree'si `/Users/batuhan/.codex/worktrees/7bf0/3DD-TTA`,
  branch `integration/chamfer-guidance-variants-20261030`, HEAD
  `a82200adbe117ce56a134dc3c22601cb73dc2654` + dirty/untracked işler korunur.
  Reset/stash/clean/branch switch uygulama; bu eski checkout'ta güncel
  `knowledge/current_baseline.md` ve baseline entrypoint'i bulunmayabilir.
- `AGENTS.md`, RTK, research/development/Colab skill'leri, knowledge README,
  thesis_scope ve experiment_protocol önce okunur. `.codex/knowledge/INDEX.md`
  bu incelenen checkout ve ortak tabanda yok; bu görev için knowledge README
  ve bu dosya araştırma index/artifact işlevini görür.
- Yeni checkout'ta `knowledge/current_baseline.md`,
  `shape_latent_spectral_guidance_audit_20261101.md`,
  `graph_spectral_retrospective_audit_20261101.md`,
  `gsd_spectral_math_audit_20261008.md`,
  `gsd_pilot_review_action_plan_20261005.md`,
  `guidance_component_sensitivity_results_20261028.md` ve
  `chamfer_depth_tuning_results_20261030.md` okunur.

[Run/Limit] Eski hard-v1 R_SG65.234375% vs style-off64.0625%,64örnek/dört
bozulma/seed0 idi; standart SCD üstünlüğü değildi. Style projection ayrı
çalışmadır. Sonraki all15 C+S Point-MAE sonuçları küçük/karışık; bir önceki
global conditional-energy pilotu spectral loss kullanmadı. Tarihsel64.53,
68.74,66.00 veya42.97 değerleri yeni koşuya referans olarak taşınmaz.
Smoothing'in varlığı kanıttır; accuracy'yi güçlendirdiği bu yeni routing için
henüz kanıtlanmış değildir. Eski GPU işleri bu protokolü kurmak için tekrarlanmaz.

## 2. Sorulacak hipotez ve kilitli host

[Proposal] Aynı normal8/Background35 local diffusion altında, h'yi SCD ile,
z/style'ı smooth spectral fidelity ile güncellemek operasyonel SCD referansını
iyileştiriyor mu? Spectral step ölçeği ve M için ayrı bir optimum var mı?

Point-MAE, ModelNet40-C/severity5, B32,2048 LION noktası, DDIMtotal100,
final-alpha1/clipoff, raw/eval LION EMAoff, frozen classifier/weights/buffers,
p=.95 ve aynı normalization/FPS sözleşmesi korunur. Spectral kayıp XYZ'de;
4D, invariant/power/Huber, entropy/KL ve dinamik graf bu taramaya eklenmez.
Global diffusion prior'ı **kapalı**. İkinci encoder/local geçiş veya refresh yok.

Sembolik update'ler:

```
h_prev = DDIM(h_t; z_t) - alpha_h * grad_h L_SCD
z_next = z_t - kappa_z * grad_z L_spectral(h_hat_0(h_t, z_t), h0)
```

İki gradient aynı live local-clean tahminde hesaplanır; h'ye spectral,
z'ye SCD katkısı aday routing'de uygulanmaz. Tanısal gradientler ayrı
hesaplanabilir ama uygulanan update'e sızamaz. Model ağırlıkları donuk olsa
da local-prior koşullama yolu z gradientini korur; global denoiser açılmaz.
SCD ve graph referansı ilk h0XYZ'dir; graph/basis/target detach edilir.

Final decode ve temiz PM probe'ları bütün kollarda başlangıç encoder z0'ı
kullanır. Güncellenen style sonraki local prior çağrılarını etkiler. Bu
historical/original-style sözleşmesini tarama sırasında updated decode'a
çevirme. Son style update'inin final decode'da tüketilmemesi kaydedilir.
PM normal clean2/4/6+terminal8, Background9/18/27+terminal35; temperature1,
eşit olasılık ortalaması; terminal ayrı tanıdır. Etiketli selector yok.

## 3. Eta/gamma ve spectral implementasyonu için zorunlu uzlaştırma

[Code] `tta.py` ve `tta_gsd.py` composition yolunda gamma=local,
eta=style; `tta_gsd_v2.py` yolunda eta=local, gamma=style'dır. Parametreleri
yalnız CLI isimleriyle aktarma. Config ve tabloda **alpha_h_local** ve
**kappa_z_spectral_effective** semantik isimleri ile gerçek eşlemeyi kaydet.
Graph gamma=.6 bunlardan üçüncü, farklı bir parametredir.

[Code] İki smooth aile farklıdır:

1. `spectral_guidance_losses.py:legacy_l2`: raw eigenvalue için
   exp(-beta*lambda), M_max sonrası0, loss sum/(3*N), batch sum.
2. `graph_spectral.py` modern smooth: mean-degree-normalized eigenvalue,
   bütün aktif modlar. Burada `modes` smooth heat filtresinin cutoff'u değildir.

**Primary öneri:** Kullanıcının M=800 karşılaştırması için1'in mevcut
truncated smooth denklemini kullan; beta=2 sabit. Graph builder olarak
`graph_spectral_v2_offdiag.py` minimal isolate-corrected varyantını dondur:
k10/delta.1/graph-gamma.6/static XYZ, full eigensystem. Bu seçim legacy
graph_v2 ile byte-identik değildir; iki graph varyantını grid içinde karıştırma.
Eski isolate düzeltmesi tek başına accuracy kazancı göstermemiştir; burada
bilinen kusuru sessizce taşımamak için açık implementation tercihi yapılır.

\[
L_S=\frac{1}{3N}\sum_{b}\sum_{i<M}
 e^{-2\lambda_{b,i}}\|[U_b^T(\hat h_{XYZ}-h_{0,XYZ})]_i\|^2.
\]

M_max gerçekten loss'a gider; yalnız graph constructor.M alanını değiştirip
tarama yaptığını sanma. Eigenbasis sırası, raw eigenvalues, graph pruning/
isolate sayısı, effective spectral mass, cutoff gap/tie, nonzero weight sayısı
saklanır. Tied-space cutoff ve raw-lambda ölçek duyarlılığı bilinen sınırlardır.
M'ler aynı fiziksel eigensystem'i kullanabilir; gradient/model/RNG ortaklığı
korunur. Etikete bakmadan tamamen aynı weight vector üreten M adayları varsa
eşdeğerlik olarak bildir; sahte bağımsız M etkisi üretme.

Bu dar primary profile makul bir plan tercihidir, smoothing'in tek doğru
tanımı değildir. Native/CPU audit bu profile uymazsa araştırma başlamadan
gerekçeyi kaydet ve planı uzlaştır. Graph/self-loop/degree-normalization veya
beta ailesini sonuçlara göre gizlice değiştirme. Modern all-active heat
profile'ı ve beta taraması ayrı gelecekteki deneydir.

## 4. Sonlu tarama: stil ölçeği/M, ardından local step

[Proposal] Aşağıdaki grid bounded öneridir. Uygulayıcı agent gerçek süre ve
kaynak/indeks manifestiyle yürütme kaydını dondurur; accuracy'ye bakıp grid'i
genişletmez. Spectral coefficient ile style step yalnız çarpımları üzerinden
update'e girer. İkisini bağımsız iki grid ekseni yapma; lambda_spectral=1
tut, tek effective kappa_z tara.

### Label-free ölçek kalibrasyonu

Sadece development grubundan32nesne; Gaussian/Background/Rotation/LiDAR,
seed0, C_base trajectory'sinde üç nonterminal quarter/half/three-quarter state.
Her M için nesne/state bazlı raw style gradient normları hesaplanır:

```
b_M = median(0.003 * ||grad_z L_SCD|| / ||grad_z L_spectral_M||)
kappa_z(M, rho) = rho * b_M
```

Bu sabit b_M, label kullanmadan bütün örnek/state oranlarının pooled medyanıdır;
accuracy ve clean z kalibrasyona girmez. Sonlu/nonzero payda gate'i ve kullanılan
gözlem/atlanma sayıları kaydedilir; sıfır gradient'i epsilon ile büyük update'e
çevirmek yok. Bir M için kullanılabilir gradient yoksa onu unavailable yap.
b_M/rho kalıcı JSON'a gerçek sayılarla yazılıp ilk development skorundan önce
dondurulur. Bu, örnek başına adaptif normalization veya zorunlu eşit norm
algoritması değildir; farklı M'leri anlamlı sabit step ölçeklerinde tarama
yöntemidir. Step başına gerçekleşen normlar yine farklı olabilir.

### A — M ve spectral-z step screen

- Development:64önceden sabitlenmiş orijinal ID, all15, seed0.
- alpha_h=.03; M_max={200,400,800,1200,1800}.
- rho_z={.1,.3,1,3}; beta=2;20routing aday.
- Aynı hazırlıktan C_base(alpha_h=.03, style-SCD step=.003) ve
  Z0_base(alpha_h=.03, style-off):2kontrol. En çok22all15koşul.
- Rank primary all15 PM; macro14/Background/per-corruption ikincil.
- En iyi2(M,rho) çifti sonraki aşama için dondurulur. Finite adaylar arasında
  eşit doğru sayısında önce küçük M, sonra küçük rho, sonra sabit lexical ID.
  Smoothing katsayılarını test skoruyla yeniden kalibre etme.

### B — Local SCD step'in routing altında yeniden ayarlanması

- Aynı development ID/seed; A'nın en iyi2(M,rho) çifti.
- alpha_h={.01,.03,.05}; toplam6routing konfigürasyonu. A'daki iki .03
  sonucu aynen kullanılır; tamamlanan işleri yeniden çalıştırma:4yeni koşul.
- Yeni alpha_h=.01 ve.05 için C_match(style-SCD=.003) ve Z0_match(style-off)
  ekle:4yeni kontrol. B en çok8yeni all15koşul, A+B toplam30.
- b_M/kappa_z A'da dondurulduğu gibi kalır. Bu iki-stage search global
  optimum değildir; yalnız bu sonlu grid'de seçilen operating point'tir.
- Winner all15 PM; tie: küçük M, küçük rho, alpha_h'nin.03'e yakınlığı,
  ardından küçük alpha_h/lexical ID. Başarısız adayların sonucu null kalır.

### C — Kilitli bağımsız-stage doğrulama

- Development'tan ayrık128orijinal ID, all15, seeds0/1/2.
- Winner'a ek C_base, C_match(winner alpha_h, style-SCD=.003) ve
  Z0_match(winner alpha_h) aynı fiziksel hazırlıktan çalışır.
- Winner alpha_h=.03 ise C_base=C_match tek koşul olur.3veya4kol ×3seed:
  en çok12all15koşul. Winner/coeff/M/horizon confirmation'da değiştirilmez.
- A+B sonunda winner C_base'den pozitif development kazancı vermiyorsa
  confirmation otomatik açılmaz; tamamlanmış null tarama ve kaynaklar sunulur.
  Pozitif pilot kazanç significance şartı değildir; selection bias açık kalır.
- Primary paired fark winner−C_base; secondary winner−C_match ve−Z0_match.
  Baseline'a policy kazancı ile spectral-z bileşen katkısını ayrı yorumla.
- Her nesnenin15bozulması ve üç seed'i cluster'da beraber tutularak10k
  bootstrap; seed başına değerler, düzeltti/bozdu, koşullu95%CI ve maliyet.
  Unadjusted secondary aralıkları bağımsız significance iddiasına çevrilmez.

En çok30×64×15 +12×128×15 = **51,840 prediction event**; calibration,
technical gate ve aşağıdaki384eventlik maliyet ölçümü buna dahil değildir.
Bunlar51,840bağımsız nesne değildir.
64+128ID seçimi label-free hash ranking ile önceden sabitlenir. Mümkünse
önceki kayıtlı development/confirmation ID'lerini dışla; exclusion union ve
seçim SHA'sı saklanır. Yeterli temiz ayrım gösterilemiyorsa historical exposure
bildirilir; "unseen benchmark" denmez. Confirmation ID'lerini selection'da
kullanma. Full2468/all15, Scan, başka head ve ekstra seed bu görevde yok.

## 5. Bütçe, yeni Colab ve izin sınırı

[Proposal/Open] Ön tahmin bütün GPU fazı için **2–4saat**: fresh setup,
technical/calibration, en çok30development ve12confirmation, recovery,
raw/Drive persistence dahil. Bu kaba forecast'tur; graph eigensystem maliyeti
ölçülmedi. Tarihsel64/all15 local/probe süreleri yaklaşık87–105s ve ek graph/
gradient maliyeti nedeniyle bunu <60dk diye varsayma. Fiyat/compute-unit maliyeti
bilinmiyor; quote gibi sunma. Hazırlanmış teknik/batch komutları, source/asset
SHA'ları ve bu bütün planla **>60dk GPU fazı için tek açık kullanıcı onayı** al.
Bu handoff metni o uzun-GPU onayı değildir. Bütçeyi kısa CLI çağrılarına bölme.
CPU implementasyon/inceleme ve yetkili canonical setup ayrı ilerleyebilir;
bilimsel calibration/technical GPU başlamadan duration/approval kaydı net olsun.

Önerilen session `tta-scdh-spectralz-a100-20261101-r1`; tarih/suffix gerçek
allocation'a göre ve server collision kontrolüyle seçilir. Başka runtime'ı
görünür diye kullanma/kapatma. Kendi endpoint/owner/worktree/primary CLI ve
monitor PID kayıtlı olsun. A100 tahsisi başarısızsa peer session'ı kapatma.

On kod hücreli canonical setup: cell1 ayrı, aynıVM'ye reconnect, sonra2–10
sırayla; /content/3DD-TTA cwd, yalnız izinli cell2 fastrlock deferral;
native/data/checkpoint gate'leri stop'tur. Güncel kurulum notunu oku. Full
research notebook Run all yasak. Drive consent kullanıcı tarafından yapılır.
Anlamlı phase heartbeat + read-only inventory monitor kullan; fake keep-alive
hücresi/busy loop yok. Çalışma bitmesi runtime kapatma yetkisi değildir.

Approved hard deadline bütün fazı/recovery'yi kapsar; receipt'te başlangıç,
onay ve cap immutable olsun. Teknik ölçüm tahmini kötüleştirirse kalan bütün
işi yeniden değerlendir, gerekirse daha uzun faz için onay iste. Timeout/
disconnect'te aynıVM process/marker/artifact'ı denetle; duplicate job açma.

## 6. Uygulama batch'leri ve kabul kapıları

### Batch1 — CPU sözleşmesi ve opt-in sampler/runner

- Goal: yalnız local-SCD/style-spectral yolunu yeni profile ile kurmak.
- Scope: semantic step eşlemesi, loss/graph profile, immutable paired-state,
  PM observer, bounded controller; mevcut default/archived source'a dokunma.
- Touched areas: önce `tta.py`, `tta_gsd.py`, `gsd_composition.py`,
  `spectral_guidance_losses.py`, `graph_spectral_v2_offdiag.py`, güncel
  `scripts/eval_chamfer_baseline.py` ve shared-trajectory/depth worker'ı oku.
  En küçük opt-in adapter/entrypoint'i seç; hard-v1'e smooth etiketi yapıştırma.
- Stack context: local CPU contracts; araştırma forward/backward Colab CUDA.
- Dependencies: current baseline ve graph/math audit; gerçek source/asset receipt.
- Implementation notes for $development-agent: baseline paylaşımını/decoder/
  scheduler/RNG'yi koru. eta/gamma mapping CPU assertion ile doğrulansın.
  Spectral weight ile style step çarpımı dışında redundant grid ekleme.
- Verification: synthetic batch-scale/finite-difference spectral-z path,
  h'ye spectral ve z'ye SCD uygulanmadığı update denklemleri; zero-kappa
  style-off parity, SCD route baseline parity; M'nin loss/operator/gradienti
  değiştirmesi; zero/finite/shape/asset/state/RNG/PM/source guards.
- Knowledge artifact to update: bu planın dated execution eki ve findings.
- User review gate: CPU/source review sonrası teknik gate'e geç; rutin fix için
  tekrar izin isteme. Uzun GPU onayı henüz yoksa bilimsel faz başlamaz.
- Out of scope: global prior, graph ailesi/beta değişimi, default promotion.

### Batch2 — Native teknik gate ve kalibrasyon

- Goal: gerçek spectral-z gradient ve baseline host'unu kanıtlamak, bütçeyi ölçmek.
- Scope:32ID Gaussian/Background native gate; iki örnekli/two-scale directional
  finite difference; aynı-state baseline repeat sayısal varyansını raporla;
  sonra label-free b_M freeze. Teknik accuracy yöntemi seçmez.
- Touched areas: new fresh result/operator receipts; sampler yalnız kanıtlı fix.
- Stack context: kendi canonical-verified A100; weights/buffers frozen.
- Dependencies: Batch1/kullanıcının gerekli GPU süre onayı/setup gate'leri.
- Implementation notes: aynı input/h0/z0/noise/RNG exact; native outputs'un
  bitwise eşitliğini varsayma. FD tolerance GPU öncesi sabitlenir; öneri
  relative<=.10 veya absolute<=1e-5, iki perturbation ölçeği .001/.01*||z||.
- Verification: tolerance/nonzero finite gradient/source/state/graph/decoder
  geçer. Fail arşivlenir; toleransı sonucu görünce sessizce gevşetme. Baseline
  farkı varsa identity/trace incele, gerçek native farkı sakla; kayıtlı review
  olmadan guard'ı pas geçme. Frozen numeric kappa ve budget manifesti üretilir.
- Knowledge artifact to update: execution eki, evidence/operator manifesti.
- User review gate: teknik kapı ve güncel bütün-faz bütçesi geçtiğinde A/B'ye
  aynı onay kapsamında devam; material scope/duration değişimi kaydedilir.
- Out of scope: teknik accuracy ile candidate/ID/step seçimi, failure silme.

### Batch3 — Registered development A/B ve frozen winner

- Goal: bölüm4'ün20+kontrol ve sınırlı local-step taramasını tamamlamak.
- Scope/touched areas: yeni controller/config/immutable raw bundles/independent
  derived summary; inference source değişmeden A→B→freeze.
- Stack context/dependencies: Batch2passed; sealed IDs/calibration/grid/source.
- Implementation notes: finite failure unavailable; failed prefix macro15 değil.
  NaN/OOM atlayan retry/clip/fallback veya batch küçültme yok. Bilinen eski
  scale100/1000 failure'ları bu yeni grid'de otomatik retry olarak açılmaz.
- Verification: en fazla30tam development koşulu; biten .03 kolları reused,
  bütün seçimin input ve score SHA'ları kaydedilir; tie rule deterministic.
- Knowledge artifact to update: her material phase sonrası plan/findings/questions.
- User review gate: frozen selection ve bütçe kapısı sonrası kayıtlı olumlu
  development şartı sağlanırsa C'ye devam; yeni bir izin döngüsü kurma.
- Out of scope: sonuç sonrası M/rho/alpha/beta grid genişletme.

### Batch4 — Confirmation, maliyet ve bağımsız kabul

- Goal: winner'ı subset-overfit etkisinden ve local-step değişiminden ayırmak.
- Scope/touched areas: C'nin3–4kol/üç seed'i, bağımsız NumPy1 yeniden hesap,
  kaynak/asset/ID/RNG/graph/PM kanıtları ve raw/derived arşivler.
- Stack context/dependencies: winner manifest frozen; confirmation ID disjoint.
- Implementation notes: operational graph-free C_base ile adil timing; common
  eigensystem cache üretim maliyetini de adayın end-to-end maliyetine kat.
  Sadece cache-replay wall'dan deployment hızlanması çıkarma. Winner freeze
  sonrası32ID Gaussian/Background, C_base ve winner, üç fresh timing repeat:
  en çok2kol*2bozulma*32ID*3repeat=384event. Warmup ayrı; raw preprocessing,
  encoder/graph/local/probe/classifier maliyeti dahil; encoder/eigensystem
  cache hit'iyle bunu fresh diye adlandırma. Timing tekrarları accuracy/step
  seçimi için kullanılmaz; strict fiziksel paired accuracy fazıyla karıştırma.
  Normal/BG ayrı, model loading/hash/IO ayrı; memory ve eigensolve kaydedilir.
- Verification: per-corruption/seed/PM/count/transition/cluster CI, bütün guard
  ve ZIP path/CRC/SHA, gerçek Drive→local roundtrip; raw değişmez/derived ayrı.
- Knowledge artifact to update: new results note, findings/method/questions,
  headline değişirse README; sonuç ve seçilen/excluded scope ile entegrasyon.
- User review gate: complete/null/failure ile raporla; yeni scan/full-set/test
  turunu otomatik açma. Sadece gerekli code/tests/knowledge skill'e göre
  seçilerek integration'a normal push; kullanıcı default seçmedikçe değiştirme.
- Out of scope: raw/weights/credential Git'e ekleme; peer/own runtime teardown
  için sessiz varsayım. Kendi runtime durumunu raporla, açık closure iznini bekle.

## 7. Artifact ve devam sözleşmesi

Fresh task root önerisi `result/preparation/scdh_spectralz_search_20261101/`;
Drive `MyDrive/thesis/result/scdh_spectralz_search_20261101/`. UTC run_id,
tam command/config/environment/stdout/summary/per-corruption/notes yanında
per-example logits/labels/IDs, preparation tensors/RNG, applied gradients/
norm/cosine, graph source/eigenvalues/weights/identity ve sealed source bulunur.
Büyük dense eigensystem'i her adayda tekrar arşivleme: source/graph config ve
fiziksel h0/reference tek ortak cache'te; eigenvalue/weight/actual-basis hash ve
per-example applied norm/cosine bütün koşularda saklanır. Gerçek basis ve
full-gradient/clean-prediction vektörleri teknik finite-difference örneklerinde
bir kez fiziksel saklanır. Salt hash'ten basis kurtarıldığını veya source+h0
ile bitwise native eigensystem garantisi olduğunu iddia etme. Tam-basis bütün
veriye arşivlenecekse depolama/Drive transferi bütçeye ayrıca dahil edilmeli.
Kalibrasyon/grid/selection ve tüm failure/partial receipt'ler arşivlenir.
Local results ve Drive
hash'leri kabul edilmeden raw'ın yerine yalnız özet bırakma.

Clean-z mesafesi istenirse yalnız var olan exact-ID/input/eps/state eşlenmiş
oracle'dan posthoc CPU tanısıdır; yoksa unavailable yaz, yeni clean inference
scope'u ekleme. Ana başarı ölçütü frozen-classifier paired accuracy'dir.

[Current status] Bu dosya yalnız delegation planıdır. Yeni entrypoint/Colab/
ID manifesti/numeric kappa/sonuç/GPU approval receipt henüz üretilmedi.
Sonraki agent'ın ilk işi Batch1 + gerçek durum uzlaştırmasıdır. Tam notebook
çalıştırmak veya eski deneyi baştan tekrarlamak bir başlangıç adımı değildir.
