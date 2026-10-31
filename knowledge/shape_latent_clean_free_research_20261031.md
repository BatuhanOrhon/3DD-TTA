# Temiz veri olmadan faydalı shape latent: oracle sonrası araştırma

2026-10-31. **Literatür/kaynak incelemesi ve mevcut artifact üzerinde CPU analizi tamamlandı; yeni yöntem uygulanmadı, yeni GPU deneyi başlatılmadı.**

Kaynak worktree `/Users/batuhan/.codex/worktrees/7bf0/3DD-TTA`, branch
`integration/chamfer-guidance-variants-20261030`, HEAD
`a82200adbe117ce56a134dc3c22601cb73dc2654` ve korunmuş dirty araştırma dosyaları.
Branch değiştirilmedi. Başlangıçta fetch edilen ortak hedef
`origin/thesis-integration`: `a0023150fb15f449af1a0c3dd10844fa8412ddc3`.
**[User report]** Kullanıcı oracle sonuçlarından sonra bu araştırmayı başlatmayı istedi.
Bu rapordaki aday sırası araştırmacı önerisidir; kullanıcı bir algoritma seçmiş
veya yeni bir parametre taramasını onaylamış sayılmaz.

Etiketler: **[Paper]** birincil yayın, **[Code]** kaynak, **[Run]** hesap/deney,
**[User report]** kullanıcı beyanı, **[Inference]** yorum/öneri, **[Open]** açık soru.
[Kompakt kanıt](evidence/shape_latent_clean_free_research_20261031/README.md).

## 1. Araştırma sorusu ve mevcut kanıt

**[Run]** [Temiz-z oracle](shape_latent_oracle_results_20261031.md), aynı
64 development nesnesi/15 bozulma/seed0/Point-MAE üzerinde A %60.8333,
B temiz-z/eski-h %64.8958, C temiz-z/koşullu-h %63.0208 verdi. B−A +4.0625puan;
bu temiz bilgi kullanan teşhistir, kullanılabilir TTA yöntemi değildir.
C'nin B'yi geçmemesi, z−h uyumsuzluğunun başlıca engel olduğuna dayanak sağlamadı.

**[Run]** [Önceki global pilot](shape_latent_tta_results_20261031.md) tek düşük-noise
global öneri ve üç adımlı anchor guidance ile genel accuracy artışı bulmadı.
Yeni ve eski koşuların referansları farklıdır; burada cross-run accuracy farkı
hesaplanmaz. Önceki %5 küre bütün 960 gerçek temiz-z örneğini dışarıda bıraktı;
bu, daha büyük radius'un başarılı olacağını veya null sonucun nedenini kanıtlamaz.

**[Inference]** Soru artık “z yararlı mı?” değil, **“Temiz girdiye erişmeden,
aynı nesneye ait yararlı global düzeltme yönünü nasıl çıkarabiliriz?”**
Global diffusion operasyonel olarak kapalı kalır. Yeni aday bunun üzerine
karşılaştırılacak; salt radius/step sayısı büyütmek ilk tercih değildir.

## 2. Yeni yerel bulgu: farkın büyük kısmı posterior ortalamasında

**[Code/Run]** Oracle'ın fiziksel 30 posterior proof + 30 A cache'i okundu;
kayıtlı bütün tensör hash'leri ve Gaussian yeniden parametrelemesi doğrulandı.
Yeni model forward/backward yok. İki diagonal posterior için:

\[
W_2^2(q_c,q_d)=\|\mu_c-\mu_d\|^2+\|\sigma_c-\sigma_d\|^2.
\]

Ortalama payı, nesne başına yüzdelerin ortalaması değil,
`sum(mean_energy)/sum(mean_energy+scale_energy)` olarak hesaplandı.

| Kapsam | W2² içinde ortalama farkının payı | Medyan gerçek `||z_c-z_d||/||z_d||` |
|---|---:|---:|
| All15, 960 nesne–bozulma olayı | %97.8677 | %33.9909 |
| Gaussian | %95.6517 | %37.8764 |
| Background | %99.2760 | %36.5267 |
| Rotation | %97.1587 | %35.8705 |
| LiDAR | %98.8130 | %50.4486 |

**[Run]** Aynı eps kullanıldığından gerçek örnek farkı
`delta_z = delta_mu + delta_sigma * eps` olur. Gerçek örnek L2 karesinde
çapraz terim vardır; %97.8677, gerçek farkın “açıklanan varyans yüzdesi” değildir.
W2 diagonal Gaussian'lar için tanımlı dağılım uzaklığıdır; encoder belirsizliğinin
kalibre olduğunu veya bunu fiziksel bozulma hatası sayabileceğimizi göstermez.
Normalizasyon/interpolasyon ve örnekleme koşulları da posterior farkına dahildir.
64 nesne 15 kez gözlenmiştir; 960 bağımsız nesne veya ek seed yoktur.

**[Inference]** İlk kontrol edilecek değişken encoder'ın **ortalama yönüdür**.
Bu bulgu `z=mu` kullanmanın accuracy'yi artırdığını göstermez. Baseline
posterior örneklemesini tamamen kaldırmak başka bir müdahaledir; burada
onu önermiyoruz. Temiz-ortalama/bozuk-sigma oracle accuracy kontrolü de henüz yok.

## 3. Kaynakta ayrılması gereken iki mekanizma

**[Code/Paper]** LION `q(z|X) q(h|X,z)` ve `p(z) p(h|z) p(X|h,z)` yapısındadır.
Global prior X veya h'yi koşul olarak almaz; local prior temiz z'ye koşulludur.
LION Sec.3.2, VAE'nin latenti Gaussian'a yaklaştırması nedeniyle DDM'nin
geriye kalan dağılım farkını modellediğini açıklar.
[LION Sec.3, App.C.1](https://arxiv.org/html/2210.06978v1).

**[Inference]** Bir global kod toplu latent prior açısından makul olup bu
nesne için yanlış olabilir. Bu nedenle koşulsuz global prior + bozuk-z'ye
yakınlık, doğru nesnenin temiz z'sini tanımlamak için yeterli olmayabilir.
Bu bir tanımlanabilirlik hipotezidir; checkpoint'in score yönü ölçülmedi.

**[Code]** `models/vae_adain.py:105` global posterioru verir; global2style
mevcut all55 config'inde identity'dir. `tta.py` son local latenti ve ilk z'yi
decoder'a verir; `return_trajectory=True` final h'yi zaten sağlayabilir.
`decoded_to_geometry` classifier öncesinde frame değiştirip normalize eder.
**Yeni encoder girdisi classifier geometrisi/FPS çıktısı olmamalı:** mevcut
decoder'ın **2048×3 LION koordinatındaki** çıktısı saklanmalıdır. Böylece
ek normalizasyon, dönüş veya upsample, öneri yönüne karışmaz. Bu tercih
clean oracle'ın bağımsız normalizasyonuyla tam eşdeğer değildir; ayrı açık
preprocessing sınırıdır.

**[Code/Run]** Eski ham cache'lerde girdiler, posteriorlar ve logits var;
yeniden encode edilecek baseline decoder XYZ/final h vektörleri arşivlenmemiş.
Bu nedenle yalnız mevcut yerel NPZ'lerden yeni öneri/accuracy üretilemez.
Önceki global önerinin yalnız hash'i yön karşılaştırmasına yetmez.
Yeni teşhiste öneri vektörleri, mu/sigma ve final h/decoder XYZ fiziksel saklanmalı.

## 4. Güncel literatürden taşınabilecekler ve sınırları

**[Paper]** 31 Ekim 2026'da hedefli tarama yapıldı; yedi birincil makalenin
sabit sürüm HTML'leri yerelde indirildi, boyut/SHA256 kayıtları tutuldu.
Bu kapsama göre tam aynı LION/global-z TTA çözümü doğrulanmadı; kapsamlı
“böyle makale yok” veya yenilik iddiası yapılmıyor.

| Kaynak, incelenen bölüm | Bağlantı | Bizim koşulda sınır / karar |
|---|---|---|
| [PSLD, 2023, Sec.2 Eq.7 ve Sec.3](https://arxiv.org/html/2307.00619v1) | Encoder–decoder ve ölçüm tutarlılığını gluing ile birleştirir. | Bilinen lineer ölçüm ve özel teorik model; salt `E(D(z,h))≈z` temiz z garantisi değildir. Çevrim kontrolüne gerekçe, birebir uyarlama değil. |
| [RED-Diff, ICLR2024, Sec.3/Alg.1](https://arxiv.org/html/2305.04391v2) | Denoiser'ı stokastik optimizasyonda prior olarak kullanır; score Jacobian'ı gerekmez. | Global prior'ı uzun DDIM zinciri açmadan kullanmak mümkün bir ikinci yol; bizde veri terimi kalibre likelihood değil. |
| [DMPlug, NeurIPS2024, Sec.3.1–3.3](https://arxiv.org/html/2405.16749v2) | Reverse zinciri tek fonksiyon sayıp seed girdisini optimize eder; erken durdurmayı inceler. | Zincir boyunca backward ve çoklu dış iterasyon maliyeti. İlk yön teşhisi geçmeden uygulamak zayıf öncelik. |
| [SILO, ICCV2025, Sec.4.1–4.4](https://arxiv.org/html/2501.11746v1) | Bozuk girdinin encoder'a verilmesini inceler; latent ölçüm operatörü öğrenerek decoder maliyetini azaltır. | Ek operatör eğitimi ve bilinen degradation gerektirir; frozen/training-free şartımız için doğrudan aday değil. |
| [REED-VAE, 2025, Sec.3–4](https://arxiv.org/html/2504.18989v1) | Tekrarlı encode–decode çevrimlerinde hata birikimini gösterir, decoder'ı bunun için eğitir. | Biz eğitim yapmayacağız; çoklu çevrim yerine tek çevrim ve diffusion-off kontrolü önerisinin risk dayanağı. |
| [MCLC, 2026-01 sürümü, Sec.3](https://arxiv.org/html/2601.04791v1) | Ölçüm guidance'ı sonrasında, ölçüm yönüne dik Langevin düzeltmesi önerir. | Bizde fiziksel ölçüm operatörü ve kalibre likelihood yok; SCD'ye körce projeksiyon aynı yöntem/garanti değildir. Şimdilik ertelendi. |
| [DiLO, 2026-04 preprint, Sec.3.2/Alg.1](https://arxiv.org/html/2604.11375v1) | Fully denoised çıktıda fiziksel kayıp değerlendirip global seed'e geri türev alır. | PDE/neural-operator koşulu farklı ve pahalı. Tam denoised çıktıyı kullanma fikri ilgili; makaledeki manifold/bijeksiyon söylemi bizim sonuca garanti olarak taşınmaz. |

**[Inference]** Yeni makale bulmak tek başına yöntem seçme nedeni değil.
Bu kaynaklar, veri bağı korunmadan prior çekişi ile kör latent çevriminin
farklı risklerini gösteriyor. İkinci guidance için en somut hazır model
`p(h|z)`yi öğrenmiş local denoiser; dışarıdan eğitilecek operatör değil.

## 5. Üç araştırma adayı

### A — Önce denoise edilmiş gözlemden global ortalama yönü çıkar (ilk öneri)

**[Inference/Proposal]** Frozen baseline tek kez çalışır:

1. Başlangıç `mu_d, sigma_d, eps_z, z_d, h_d` ve RNG saklanır.
2. Aynı N8/B35 local zincirinden `h*` ve `X*=D(h*,z_d)` çıkarılır.
3. Frozen **global encoder** `q(z|X*)` yalnız bir kez çalışır; `mu_*` alınır.
4. İlk öneri `d = mu_* - mu_d`, `z_proposal = z_d + d` olur. Böylece
   ilk `sigma_d * eps_z` korunur; yeni Gaussian çekilmez. Global diffusion çağrısı **0**.
5. İlk küçük teşhis sadece yön/mesafe ve VAE çevrim kontrolünü ölçer.
   Yön umut verirse, eski **h_d**, eski local diffusion noise ve A-hXYZ
   Chamfer hedefiyle **bir** yeni local zincir çalıştırılır; z_proposal hem
   başlangıç koşulu hem ilk decoder z'si olur. Yerleşik style gradientleri
   mevcut baseline'daki gibi çalışır. `h` yeniden encode edilmez.

**Kontrol:** `X_AE=D(h_d,z_d)` diffusion olmadan decode edilir;
`d_AE = mu(E_z(X_AE)) - mu_d`. Aynı `eps_z`, aynı encoder frame ve eval modu
korunur. `d` ile `d_AE` eşdeğerse veya temiz ortalamaya yaklaşma yalnız AE
çevriminden geliyorsa, diffusion'un ek global bilgi ürettiği söylenemez.
Bu iki yön otomatik birbirinden çıkarılmaz; önce ayrı raporlanır.

**[Open]** Denoised X* hâlâ bozuk olabilir; encoder/decoder kendi yanlış
z'sini doğrulayan sabit noktaya dönebilir; `sigma_d` korunması optimum değildir.
Bu yüzden A başarı iddiası değil, daha pahalı guidance'dan önce yön teşhisidir.
Yeni %5 sınırı uygulanmaz: önce unclipped önerinin normu/finite durumu ölçülür.
Accuracy aşaması için blend/radius gerekiyorsa sonuçlar görülmeden ayrıca
tanımlanır; oracle hedefinden nesne başına katsayı seçilmez.

### B — Local denoiser ile z için koşullu uyum enerjisi

**[Inference/Proposal]** Dondurulmuş `h*` için ortak t/epsilon ile

\[
\mathcal E(z;h^*) = \mathbb E_{t,\epsilon}
 \left[w_t\|\epsilon_\theta^h(a_t h^*+b_t\epsilon,t,z)-\epsilon\|^2\right]
\]

hesaplanabilir; gradient yalnız z'ye gider. Bu, local prior'ın eğitim
gürültü-tahmin amacını kullanır. `h*` detach edilir; yeni h ve z birlikte
aynı enerjiyi aşağı çekip ölçümü anlamsızlaştırmamalı. `t` ve epsilon,
baseline z / öneri / temiz-z oracle teşhislerinde eşlenir; değerlendirme
gürültüsü optimizasyon gürültüsünden ayrı tutulur.

**[Inference/Open]** Seçilmiş birkaç t ve keyfî ağırlıkla bu bir **surrogate**
enerjidir; `-log p(h*|z)` veya exact posterior diye sunulamaz. Koşullu DDPM
bound bağlantısı için orijinal schedule/weight/end-point terimleri gerekir.
Bozuk h_d'yi hedef almak bozulmayı koruyabilir; h* de baseline yanlılığı taşır.
4. latent kanal eğitimdeki birimdedir; sırf XYZ/ham h mesafesiyle değiştirmek
başka amaçtır. B, otomatik olarak z−h uyumsuzluğunu gideren kanıtlanmış metrik değildir.

Global prior için RED-Diff-inspired bir terim daha sonra eklenebilir;
önce `-grad_z E` yönü clean-ortalama doğrultusuyla ve A yönüyle karşılaştırılmalı.
Bu iki bileşeni baştan birleştirmek, yararlı yönün kaynağını gizler.
Her enerji adımı ek local prior forward/backward ister; zaman ölçülmeden
“128 boyut olduğu için ucuz” denmez. İlk tercih A; B ikinci aday.

### C — Global zincirin girişini optimize et (DMPlug-inspired, ertelendi)

**[Inference/Proposal]** `z_clean=G_DDIM(u)` şeklinde sabit kısa global
zincirin noisy girişini u optimize etmek; veri terimini local denoiser veya
final decoded geometride tanımlamak mümkündür. Bu, önceki bozuk-z anchor
guidance'ından farklıdır. Zincir boyunca türev ve yerel veri terimi maliyeti
vardır; kısa zincirin bütün geçerli shape manifoldunu kapsadığı garanti değildir.
Faydalı veri yönü/enerjisi gösterilmeden C'nin yeni büyük deneyi önerilmiyor.

## 6. Bir sonraki küçük pilot için somut taslak

**[Inference/Proposal — henüz yürütme planı değil]** İlk aşama A'nın yön
teşhisi: mevcut **64 development ID/all15/seed0** ile uyumlu hazırlık;
tek baseline trajectory, AE-only decode ve iki ek global encoder çağrısı.
Mevcut cache hazırlığı kullanılır; eski accuracy yeniden yeni sonuç gibi sayılmaz.
Önce aynı32-ID Gaussian+Background teknik kontrol; ardından kapsam/bütçe
doğrulanırsa küçük all15. Tam2468/ek seed/backbone yok.

Clean X yalnız sonradan teşhis ölçümleri içindir; **d veya d_AE üretiminde
kullanılmaz**, clean etiket/latent ile aday seçen inference kapısı kurulmaz.
Bu oracle-teşhis hâlâ development verisine maruzdur; unbiased test değildir.

Raporlanacaklar: `||mu_*−mu_c||² / ||mu_d−mu_c||²`, aynı oran AE için,
`cos(d,mu_c−mu_d)`, adım normu, nonfinite oranı; 64 nesne bazında kümelenen
eşlenmiş belirsizlik. Sıfır payda/norm vakası ayrı kaydedilir. Aynı ölçümler
posterior örneği için de saklanır. Pozitif cosine tek başına mesafe/accuracy
iyileşmesi sayılmaz. Oracle metrikleri deployment seçicisi değildir.

Yön teşhisi desteklerse matched accuracy karşılaştırması A-baseline /
AE-mean-refresh / denoise-mean-refresh olur; aynı h_d, A-hXYZ anchor, local
noise, başlangıç RNG ve N8/B35 Point-MAE PM readout. Her adayın tüm önceki
baseline hazırlık+denoising maliyeti deployment süresine eklenir; tek ek
encoder'ın süresi toplam yöntem maliyeti gibi raporlanmaz. İkinci local zincir
normalde local reverse çağrılarını ikiye katlar. Yalnız decoder-z değiştirip
h*'yi tekrar decode etmek daha ucuzdur fakat ana koşullu-z testiyle aynı değildir.

**Falsifier:** A, AE-only'den daha iyi yön/mesafe göstermiyorsa
“local denoising global encoder'a ek yararlı bilgi verir” hipotezi bu pilotta
desteklenmez; otomatik çoklu çevrim yapılmaz. İyi latent uzaklığı olup accuracy
artmaması ayrı negatif sonuçtur; latent L2, accuracy'nin garantisi değildir.
Güçlü sonuç ancak daha sonra ayrılmış nesneler ve ek seed ile doğrulanabilir.

**[Decision/Open]** Yeni algoritma, sayısal blend/loss katsayısı ve yeni GPU
fazı henüz seçilmedi. A önerisi somutlaştırıldı; B ve C uygulamaları yok.
Yeni runtime yok; önceki hamlar yerelde tamdır. Kullanıcının ek Drive bağlantısı
gereksiz kararı korunur. Taze GPU gerekirse on setup hücresi yeniden doğrulanır;
bütün planlı GPU fazı60dakikayı aşacaksa süre onayı gerekir.

## 7. Doğrulama ve dosyalar

**[Run]** `analyze_posteriors.py` sentetik iki-nesne cebir kontrolü, tüm
posterior/cache tensör hash'leri, finite/pozitif sigma, 30proof/15×64 kapsam,
örnek yeniden parametreleme ve çapraz-terim eşitliğini geçti.
Ortam Python3.12/NumPy1.26.4; yeni accuracy/model inference yok.
Ham oracle dosyaları salt okundu, yeni türetilmiş çıktı ayrı dizine yazıldı:
`result/preparation/shape_latent_clean_free_research_20261031/posterior_analysis_v1/`.
Birincil HTML/text kaynakları aynı root altında `primary_sources/` içinde;
Git'e yalnız receipt, analiz kodu ve kompakt sonuç konur.
Yeni üretim kodu/config/dependency/model ağırlığı değişikliği yok.
