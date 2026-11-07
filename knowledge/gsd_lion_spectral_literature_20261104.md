# LION diffusion guidance için graf spektral yöntemler: yeniden tarama ve yöntem önerisi

**Tarih:** 2026-11-04. **Kanıt:** [Paper], [Code], [Run], [Inference], [Open].
**Durum:** Literatür ve matematik çalışması; önerilen sampler uygulanmadı, GPU deneyi yapılmadı.

## 1. Sonuç ve araştırma sorusu

**[Inference] En güçlü aday, bozuk bulutun düşük frekanslarını doğrudan hedeflemek yerine, LION'un temiz tahminine uygulanacak küçük yüzey düzeltmesini graf üzerinde düzenleyen bir proximal adımdır.** Düzeltme, sırasız gözlemlerle sağlam nokta–düzlem eşleşmesinden hesaplanır; yoğun özayrışım ve diffusion ağı üzerinden geri yayılım gerektirmez. LION decoder'ının koordinat skip bağlantısı, fiziksel düzeltmeyi yerel latent'e yaklaşık olarak taşıma imkânı verir. Bu, mevcut SCD'nin yerine denenebilecek matematiksel olarak tutarlı bir adaydır; daha yüksek accuracy veya daha düşük toplam süre henüz gösterilmemiştir.

**[Paper] Genel “diffusion temiz tahmini + graf Gaussian düzeltme + seyrek çözüm” fikri yeni değildir.** Eylül 2026 tarihli GPARA ön baskısı bunu açıkça kuruyor. Dolayısıyla olası tez katkısı; eğitimsiz LION uyarlaması, bilinmeyen bozulmada sırasız yüzey eşleştirmesi, decoder/scheduler ile doğru aktarım ve eşleştirilmiş accuracy–maliyet kanıtı düzeyinde aranmalıdır. GPARA'nın öğrenilen grafı ve bilinen ölçüm operatörü bizim problemimizde hazır değildir. [GPARA §3.1, Eş.6–13](https://arxiv.org/pdf/2609.34172).

Yanlışlanabilir hipotez: **Sabit classifier, aynı LION checkpoint'i, aynı başlangıç örnekleri ve aynı değerlendirme altında, güvenilir normal artıklarının graf düzenlemeli düzeltmesi SCD+PM'yi hem PM accuracy hem toplam süre bakımından geçebilir.** Grafın ayrı katkısı, aynı yöntemin graf katsayısı sıfır olan kontrolüne karşı da gösterilmelidir.

## 2. Kapsam, kaynaklar ve checkout

**[Code]** Kaynak taban `512b5aa9869673004d9b9b3724bb38dbdada5b76`, branch `research/gsd-physical-followup-20261104`. Kullanıcının belirttiği `gsd-guidance-math-audit-20261103` worktree'si okuma sırasında başka oturum tarafından `research/gsd-physical-accuracy-20261104` / `73e117a` durumuna geçirildi. Bu çalışmanın yazıları, istenen branch korunarak ayrı `gsd-lion-literature-20261104` worktree'sinde üretildi. Diğer oturumun dosyalarına veya runtime'ına müdahale edilmedi.

**[Paper]** Eski notları kaynakça yerine kullanmakla yetinmeden **33 ayrı çalışmanın 35 birincil PDF'si** yeniden indirildi; iki dosya ek materyaldir. Yöntem, varsayım, denklem ve ilgili kanıt bölümleri odaklı okundu. Bu sayı her ekin her kanıtının bağımsız doğrulandığı veya literatürün eksiksiz tarandığı anlamına gelmez. Özellikle GSDTTA, LION, 3DD-TTA, CloudFixer, posterior guidance, proximal/geri yayılımsız yöntemler, graf filtreleri ve nokta bulutu denoising kaynakları birlikte incelendi.

- [Kaynak ve okuma günlüğü](evidence/gsd_lion_literature_20261104/reading_log.md): çalışma başına okunan sayfa/bölüm, alınabilir fikir ve sınır.
- [Kaynak manifesti](evidence/gsd_lion_literature_20261104/sources.json): URL, PDF SHA256, boyut, sayfa sayısı ve erişim zamanı.
- Yerel ham kaynaklar: `result/preparation/gsd_lion_literature_20261104/` altında PDF, metin ve kritik denklem sayfalarının görüntüleri; Git tarafından dışlanır. Bu bir benchmark sonuç paketi değildir; Drive yedeği oluşturulmadı.
- [Yeniden çalıştırılabilir CPU denetimi](evidence/gsd_lion_literature_20261104/audit.py) ve [sonucu](evidence/gsd_lion_literature_20261104/audit.json).

**[Code/Run]** `knowledge/README.md`, tez kapsamı, ilgili paper/method/open-question/findings kayıtları; özellikle [1 Kasım spectral-z sonucu](scdh_spectralz_results_20261101.md), [3 Kasım matematik denetimi](gsd_guidance_math_efficiency_20261103.md), [fiziksel pilot](decoded_physical_spectral_results_20261103.md), [4 Kasım takip](decoded_physical_followup_20261104.md) gerçek `tta.py`, `decoded_spectral.py` ve `models/latent_points_ada.py` ile uzlaştırıldı. Eski tarihteki planlar tamamlanmış sonuçların yerine kullanılmadı.

## 3. Projenin amacı ve geçmiş deneylerin getirdiği kısıtlar

**[Code]** Tez hedefi yalnız daha düzgün point cloud üretmek değildir: test zamanında bozulmuş bulutu, yeniden eğitilmeyen classifier'ın daha doğru sınıflandıracağı biçimde iyileştirmek ve maliyeti kontrol etmektir. Çalışan operasyonel referans [N8 Chamfer+PM](current_baseline.md): normal bozulmalarda 8, Background'da 35 yerel reverse adımı; `gamma=.03`, style `eta=.003`, SCD `p=.95`, toplam DDIM çizelgesi 100, LION raw/eval ve EMA kapalı, son decode'da özgün `z0`. Bu politika bozulma ismine bağlıdır; tamamen bilinmeyen bozulmaya hazır politika diye sunulamaz.

**[Code]** Varsayılan SCD, decoder çıktısı ile ham X arasında değil, temiz yerel latent'in ilk üç kanalı ile ilk encode edilmiş yerel latent'in ilk üç kanalı arasında hesaplanır. İki yönlü nearest-neighbor uzaklıklarının en küçük p oranı tutulur; varsayılan reduction toplamdır. Baseline SCD backward'ı denoiser'dan geçer; decoder backward gerektirmez. Fiziksel spectral eklenti decoder yolunu ayrıca açar. Önerimiz fidelity alanını da değiştirdiğinden bunun katkısı graf katkısıyla karıştırılmamalıdır.

| Kayıt | Doğrudan kanıt | Bu öneri açısından sonuç |
|---|---|---|
| Eski smooth spectral+SCD ve projection | İlk dar pilotlardaki kazançlar genişletilince null/küçük negatif sonuçlara döndü | Yalnız Gaussian/Background veya küçük prefix sonucu yöntem kanıtı değildir |
| 1 Kasım SCD-h / spectral-z | 128 stage-disjoint nesne, 15 bozulma, 3 seed: S %65.5729, C %65.5903, style-off %65.4861. S−C −0.01736 pp; %95 nesne-küme CI [−0.39931,+0.36458] | Spectral-z üstünlüğü gösterilmedi; geliştirme kümesindeki +0.625 pp doğrulanmadı |
| Aynı çalışmanın taze B32 süreleri | Gaussian C 4.3341 s, S 14.4686 s; Background C 7.0235 s, S 19.7350 s | Mevcut spectral yol 3.338× / 2.810× maliyetli; spektral alan kendiliğinden hızlı değildir |
| 3 Kasım H8 fiziksel loss | İki Gaussian nesnede decoder residual'ını gradyanda dondurma, full güncellemeye çok yakın; küçük pilotta sınıflandırma tüm kollarda 2/2 | Yerel türev yaklaşımı için sınırlı destek; accuracy kanıtı değildir. H8 denoiser backward'ını kaldırmaz |
| 4 Kasım kontrollü native takip | H8/full 96/96 durum geçişi, min cosine .995612. Kontrollü tekrar en kötü logit farkı 5.58e−5; eski ayarda 2.4244 | Önce backend tekrarlanabilirliği; eski detach başarısızlığını tek başına detach'a bağlamak doğru değil |
| Aynı takipte clean oracle | 64 karışık kökenli satırın 42'sinde spectral loss azalırken clean Chamfer arttı; 1/4 gerçek GPU-F satırı, 41/60 CPU yeniden kurulan satır | Doğru türev, doğru hedef demek değildir. Bu oran popülasyon hata oranı veya 64 GPU doğrulaması değildir |

**[Run]** Son satırın sınırı önemlidir: iki CPU/GPU-F yeniden kurma kontrolü başarısız kaldı; bunlar silinmedi. Buna rağmen arşivlenmiş gerçek GPU-F ile bir karşı örnek mevcut: Gaussian ID459 terminal durumda loss `.01439452008 → .01408463612`, clean-CD `.050524493507 → .050544263416`. Bu sampler accuracy deneyi değil, “loss azalması temiz geometriyi mutlaka iyileştirir” önermesine karşı örnektir. [Takip protokolü ve ham köken ayrımı](decoded_physical_followup_20261104.md).

**[Inference]** Daha derin diffusion, sürekli re-encode, global prior veya daha çok spektral katsayıyı otomatik çözüm seçmiyorum. Önceki derinlik/yeniden kodlama çalışmalarında bunlar tutarlı kazanç sağlamadı. Clean-z oracle potansiyeli, gerçek bozuk girdiden iyi z elde ettiğimiz anlamına gelmiyor. Eski kaynak-model reproduksiyon boşlukları da yeni yöntemle kapanmış sayılmayacak.

## 4. Literatürün birlikte söylediği

### 4.1 GSDTTA'nın iddiası ile bu fork aynı değildir

**[Paper]** GSDTTA, koordinatları düşük spektral bileşenlerde parametrik kaydırır; classifier/pseudo-label/entropy-diversity ve geometrik terimler ile model adaptasyonu içeren bütün bir prosedürdür. Tek bir şekil üzerinde düşük frekansta yüksek koordinat enerjisi bulunması, sınıflandırma bilgisinin veya faydalı düzeltme yönlerinin tamamının orada olduğunu kanıtlamaz. Buradaki donmuş classifier + LION guidance, tam GSDTTA replikasyonu değil, GSD'den esinlenen bir sentezdir. [GSDTTA §3, Eş.11–18](https://openaccess.thecvf.com/content/ICCV2025/papers/Wei_3D_Test-time_Adaptation_via_Graph_Spectral_Driven_Point_Shift_ICCV_2025_paper.pdf).

**[Paper/Inference]** LION'un yerel diffusion prior'ı global latent'e koşulludur; clean shape prior ile bozuk gözlemin fidelity terimini ayırmak gerekir. CloudFixer'ın diffusion teacher'ını stop-gradient kullanması, büyük denoiser Jacobian'ı olmadan geometrik adaptasyon yapılabileceğini gösteren yakın örnektir. Fakat CloudFixer yine Chamfer kullanır, başka bir parametrizasyon/model ailesidir; doğrudan “SCD kaldırıldı ve LION hızlandı” kanıtı değildir. [LION §3](https://arxiv.org/pdf/2210.06978), [CloudFixer §4.2](https://arxiv.org/pdf/2407.16193).

### 4.2 Guidance gradyanı ile temiz tahmini düzeltmek farklı işlemlerdir

**[Paper]** DPS, Tweedie temiz tahmini üzerinden ölçüm likelihood'unu yaklaştırır; ölçüm gradyanını `x_t`'ye taşırken denoiser Jacobian'ı kullanılır. PiGDM'de ölçüm operatörünün türev ihtiyacı azalır ama denoiser VJP hâlâ vardır. Dolayısıyla “pseudoinverse = tamamen backward-free” sonucu yanlıştır. [DPS §3.1](https://arxiv.org/pdf/2209.14687), [PiGDM §3, Eş.4–8](https://jankautz.com/publications/PiGDM_ICLR23.pdf).

**[Paper]** DiffPIR, temiz tahmine proximal data-consistency uygular ve düzeltilmiş tahminden noise bileşenini yeniden hesaplar. DDS de temiz tahminde CG düzeltmesi yapar; ancak kullandığı taşıma ve manifold varsayımları farklıdır. Efficient DPS'nin Gaussian koşullu yaklaşımı, ölçüm ve temiz tahmin arasında pozitif tanımlı bir doğrusal çözüm verir. Bunlar ağdan geri yayılım yerine fiziksel alanda küçük çözüm düşüncesini destekler; gerçek LION posterior'ının tam çözüldüğünü göstermez. [DiffPIR §3 ve Alg.1](https://arxiv.org/pdf/2305.08995), [DDS §3](https://arxiv.org/pdf/2303.05754), [Efficient DPS §4, Eş.4.7–4.11](https://arxiv.org/pdf/2503.10237).

**[Paper]** DDNM/DDRM'deki null-space veya SVD, bilinen ölçüm operatörüne aittir; keyfî bir point-cloud grafının özvektörleriyle değiştirilemez. Özellikle `A=I` ile gürültülü gözleme hard consistency, gürültüyü geri koyabilir. ReSample latent decoder üzerinden tekrarlı optimizasyon; DAPS çok sayıda prior/likelihood değerlendirmesi gerektirebilir. Bunları mevcut 8 adımlık LION hattına düşük maliyetli alternatif varsaymak için gerekçe yoktur. [DDNM §3](https://arxiv.org/pdf/2212.00490), [DDRM §3.2](https://arxiv.org/pdf/2201.11793), [ReSample §3.1](https://arxiv.org/pdf/2307.08123), [DAPS §3](https://arxiv.org/pdf/2407.01521).

### 4.3 2026 çalışmaları yenilik alanını daraltıyor

**[Paper]** GPARA, `x0=x_hat+δ`, graf precision'ı `Q=τI+λL` ve ölçüm precision'ı ile `δ=(Q+HᵀR⁻¹H)⁻¹HᵀR⁻¹(y−Hx_hat)` biçiminde posterior residual düzeltmesi kuruyor. Önerimizin çekirdeği bununla örtüşür. Farklarımızın olası değeri; grafın yeniden eğitim olmadan üretilmesi, bilinen doğrusal H yerine donmuş yerel yüzey eşleştirmesi, sağlam ağırlıklar ve LION aktarımıdır. Bunlar henüz özgünlük veya üstünlük kanıtı değil, sınanacak uygulama katkılarıdır. [GPARA §3.1](https://arxiv.org/pdf/2609.34172).

**[Paper]** Yeni Graph Posterior Sampling çalışması graf-Tikhonov/TV'yi inverse problem guidance'ına katıyor; graph denoiser eğitimi ve diffusion üzerinden gradyan kullanımı sürüyor. Posterior DDIM'in yakınsama sonuçları bilinen doğrusal ölçüm, prior/predictor ve adım sınırı koşullarına bağlı; sekiz LION adımına doğrudan garanti vermez. FB-GDM'deki belirsizlik öğrenimi de bilinen ölçüm modeline dayanır ve Jacobian terimini korur. Üçü de burada yalnız ön baskı kanıtı olarak ele alındı. [Graph DPS](https://arxiv.org/pdf/2605.19621), [Posterior DDIM §3](https://arxiv.org/pdf/2607.19333), [FB-GDM §III](https://arxiv.org/pdf/2609.29216).

### 4.4 Grafın doğru rolü: şekli değil düzeltmeyi düzenlemek

**[Paper]** Graf denoising literatüründeki patch/manifold ve normal temelli yöntemler, doğrudan tüm XYZ koordinatlarına low-pass uygulamaktan daha yapısal fidelity/prior ayrımları yapar. GLR seyrek sistemler çözer; feature-graph yöntemi grafı geometriye göre öğrenir; normal-GTV, normallerin parçalı düzgünlüğünü kullanır. Robust ICP ise eşleşmeleri ve sağlam ağırlıkları dondurup yerel alt problem çözme konusunda yöntemsel örnektir; onun rigid-registration garantilerini bizim nonrigid düzeltmemize taşımıyoruz. [GLR §IV](https://arxiv.org/pdf/1803.07252), [Feature Graph Learning §V](https://arxiv.org/pdf/1907.09138), [Normal GTV §III](https://arxiv.org/pdf/1804.10831), [Robust ICP §5](https://arxiv.org/pdf/2007.07627).

**[Paper]** Graph Wiener filtresi için signal/noise PSD'si ve stationarity varsayımları gerekir. Tek bozuk bulut veya ardışık diffusion tahminlerinin uzlaşması bunları kendiliğinden tanımlamaz. Score-based denoising ve Noise2Score3D faydalı skor/Tweedie bağlantıları sunar; fakat ayrıca öğrenilmiş denoiser/score ağı kullanırlar. Noise2Score3D'nin unknown-noise seçimi de TV kriteridir, sınıflandırma doğruluğu veya gerçek noise varyansı garantisi değildir. [Stationary Graph Processing §4–5](https://arxiv.org/pdf/1601.02522), [Score-Based Denoising §3](https://arxiv.org/pdf/2107.10981), [Noise2Score3D §3](https://openaccess.thecvf.com/content/ICCV2025/papers/Wei_Noise2Score3D_Tweedies_Approach_for_Unsupervised_Point_Cloud_Denoising_ICCV_2025_paper.pdf).

## 5. Mevcut fiziksel spectral loss neden yetmiyor?

**[Code/Inference]** Güncel fiziksel deneyde, sabit bozuk referans grafında

\[
\ell(Y;X)=\frac1{3N}\operatorname{tr}[(Y-X)^\top F_X(Y-X)],\qquad
\nabla_Y\ell=\frac2{3N}F_X(Y-X).
\]

Bu cebir geçerlidir; aynı bazda GFT ve ters GFT kullanmak da geçerlidir. Fakat `F_X` low-pass ise düşük frekanslı bozulmayı temiz şekil sanabilir. İndeks bazlı `Y_i−X_i` ayrıca sırasız iki bulut arasında fiziksel eşleşme varsayar. İzotropik gürültü dışında bu varsayım özellikle zayıftır.

LION decoder'ı mevcut kodda

\[
D(h,z_0)=\Pi h+0.01R(h,z_0),\quad h\in\mathbb R^{N\times4},\quad z_0\in\mathbb R^{128}.
\]

H8 yaklaşımı decoder Jacobian'ını `Π` ile değiştirir. Oysa `h_hat(h_t)` üzerinden denoiser Jacobian'ı hâlâ zincirde kalır. Bu iki nedenle H8'in başarılı türev kontrolleri ne kötü referans sorununu ne ana geri yayılım maliyetini tek başına çözer.

**[Inference]** Heat filtresini Chebyshev ile yaklaşık uygulamak yoğun özayrışımı kaldırabilir, ama aynı yanlış hedefi daha hızlı optimize eder. Bu bir mühendislik kontrolüdür; accuracy hipotezinin kendisi değildir.

## 6. Öneri: sağlam yüzey eşleşmeli, graf düzenlemeli proximal düzeltme

### 6.1 Sabitler ve gözlem modeli

**[Proposal/Inference]** İlk adayda mevcut prior forward sayısı, başlangıç h/z/noise, DDIM çizelgesi ve classifier korunur. `z0` sabit tutulur; yalnız mevcut PM'nin üç ara temiz-tahmin noktasında düzeltme yapılır: normal politikada 2/4/6, Background'da 9/18/27. Son decode dördüncü PM gözlemidir. Style güncellemesini kapatmak ve guidance sıklığını azaltmak ayrı değişkenlerdir; deney tasarımı bunları graf kazancından ayırmalıdır.

Bir checkpoint'te

\[
\widehat h_0=(h_t-\sigma_t\epsilon_\theta(h_t,t,z_0))/a_t,
\quad Y=D(\widehat h_0,z_0),\quad a_t=\sqrt{\bar\alpha_t},\;\sigma_t=\sqrt{1-\bar\alpha_t}.
\]

`X`, classifier normalizasyonundan önceki, encoder'a verilen aynı fiziksel çerçevedeki gözlemdir. Yeni tekrar normalize etme veya iki buluta bağımsız ölçek uygulama yapılmaz.

1. **Graf:** Y üzerinde simetrik, self-edge içermeyen kNN grafı; negatif olmayan uzamsal ağırlıklar. İlk adayda tek graf ailesi kullanılır. Kombinatoryal `L=D_g−W_g`, ortalama aktif derece ile normalize edilir; `L1=0` korunur. İzole düğüme yapay 1000 cezası gerekmez. Normal yön uyumu eklemek ayrı ablation olabilir; ilk taramaya yeni bir serbestlik eklenmez.
2. **Yüzey:** Y'nin yerel PCA'sından birim normal `n_i`; komşuluk yetersiz veya normal güvenilmezse ilgili gözlem ağırlığı sıfırlanır. Buradaki 3×3 PCA özçözümleri, N×N graf özayrışımı değildir.
3. **Eşleşme:** Her `x_j` en yakın üretilmiş yüzey patch'ine `π(j)` atanır. `r_j=n_{π(j)}ᵀ(x_j−Y_{π(j)})`. Bu gözlemden yüzeye tek yönlü eşleşmedir; eksik bölgelerde üretilmiş her noktayı gözleme çekmez. Tanımlı eşit mesafe bağlama kuralı ve permütasyon kontrolü gerekir.
4. **Güven:** `q_j∈[0,1]`, normal güveni ve fiziksel mesafe kapısından gelir; patch doluluğuyla ağırlıkları normalize etmek tekrar örneklenen noktaların baskınlığını sınırlayabilir. Bu değerler kalibre edilmiş inlier olasılığı değildir. `σ>0`, etiketsiz yerel aralıkla ölçeklenebilen bir robust eşiktir; gerçek bozulma varyansı diye yorumlanmaz.

Bu graf, eşleşme, normal, güven ve ölçek **bir çözüm boyunca sabittir**. Sonraki seçilmiş checkpoint'te yeniden hesaplanabilir. Böylece tüm dinamik grafın türevini hesapladığımız veya global bir loss'u minimize ettiğimiz iddia edilmez.

### 6.2 Konveks yerel amaç ve seyrek çözüm

`δ∈R^{3N}` fiziksel koordinat düzeltmesi olsun. `A`'nın j. satırı yalnız `π(j)` noktasının üç bileşenine `n_{π(j)}ᵀ` uygulasın. Huber fonksiyonu `ρ_c(u)=u²/2` (|u|≤c), aksi halde `c|u|−c²/2` olsun. Yerel amaç:

\[
J_t(\delta)=\frac{\kappa_t}{2}\|\delta\|^2+
\frac{\lambda_t}{2}\delta^\top(L\otimes I_3)\delta+
\sum_jq_j\rho_c\!\left(\frac{(A\delta-r)_j}{\sigma}\right),
\quad\kappa_t>0,\;\lambda_t\ge0.
\tag{1}
\]

Graf terimi **Y'yi değil δ'yı** düzgünleştirir. LION'un şekli düzgün diye küçültülmez; artık yoksa `δ=0` olur. Başlangıç `δ=0` etrafındaki tek Huber-IRLS majorizer'ı

\[
W_{jj}=\frac{q_j}{\sigma^2}\min\left(1,\frac{c\sigma}{\max(|r_j|,\varepsilon)}\right)
\]

ile şu sistemi verir:

\[
\boxed{[\kappa_tI+\lambda_t(L\otimes I_3)+A^\top WA]\delta=A^\top Wr.}
\tag{2}
\]

Sıfır artık için Huber ağırlığının limit değeri `q_j/σ²` alınır. `κ>0` sistemin pozitif tanımlı olmasını, dolayısıyla tek çözüm bulunmasını sağlar; gözlem almayan veya izole düğümlerde de problem tanımlıdır.

**[Derivation]** `H` sol matris, `b=AᵀWr` ise donmuş kuadratik alt problem için

\[
J_Q(\delta_*)-J_Q(0)=-\tfrac12b^\top H^{-1}b\le0.
\tag{3}
\]

Huber majorizer'ı sıfırda gerçek amaca değdiği ve onu yukarıdan sınırladığı için alt problemde azalma, donmuş (1)'de de azalma sağlar. Bu yalnız o sabit eşleşme/normal modeli içindir: clean geometri, classifier accuracy veya tüm reverse süreç için monotonluk değildir.

**[Proposal]** Matris oluşturmadan CG/3×3 blok-Jacobi önkoşullandırma yeterlidir. İterasyon sınırı, bağıl residual ve gerçek alt-problem azalma kontrolü birlikte tutulur; başarısız çözümde düzeltme uygulanmaz. Güven yarıçapı gerekirse tüm vektör aynı `α∈[0,1]` ile ölçeklenir. Konveks amaçta bu ölçekleme azalmayı korur; bağımsız nokta clipping'i aynı garantiyi taşımaz. `κ_t` için diffusion SNR'sine bağlı bir çizelge denenebilir, fakat `κ0 a_t²/(σ_t²+ε)` ancak sezgisel schedule'dır; ölçülmüş posterior precision değildir.

### 6.3 Bunun GSD karşılığı

Özel olarak `A=I`, `W=wI`, `L=U diag(ν_k)Uᵀ` olduğunda

\[
\widehat\delta_k=\frac{w}{\kappa_t+w+\lambda_t\nu_k}\widehat r_k.
\tag{4}
\]

Bu, tüm modlarda tanımlı bir **rasyonel spektral filtredir**. Sert `M=400` kesimi veya eigenvector hizalaması gerekmez. Düşük frekansta da trust terimi vardır; yüksek frekanslı gözlem artığı bastırılır. Aynı operatör özvektör çıkarmadan seyrek çözümle uygulanır. [Graf filtreleme temelleri](https://arxiv.org/pdf/1211.0053), [filtreler ve operatör eşleşmesi](https://arxiv.org/pdf/2211.08854).

**[Derivation]** Gerçek nokta–düzlem probleminde `AᵀWA` yönlü ve uzamsal olarak değişkendir; genel olarak `L⊗I3` ile komüte etmez. Dolayısıyla (4)'ü doğrudan her frekansa uygulamak tam çözüm değildir; (2) çözülmelidir. `d_max`, kullanılan L'nin en büyük derecesi olmak üzere kaba bir sınır:

\[
\operatorname{cond}(H)\le1+\frac{2\lambda_td_{\max}+\max_i\sum_{\pi(j)=i}W_{jj}}{\kappa_t}.
\]

Bu, q/doluluk ve trust parametresinin yalnız denoising'i değil çözüm maliyetini de etkilediğini gösterir. Bir CG iterasyonu `O(|E|+M_obs)` iş ve `O(Nk+M_obs)` saklama gerektirir. **kNN/eşleştirme kurulumunun maliyeti buna dahil değildir**; kullanılan arama brute-force ise o aşama hâlâ kuadratiktir.

### 6.4 LION latent'ine aktarım

**[Code/Derivation]** Dördüncü yerel kanal ve z0 sabit bırakılarak `Δh=Πᵀδ` uygulanır. Bu decoder'ın tam tersi değildir:

\[
D(\widehat h_0+\Delta h,z_0)-(Y+\delta)
=0.01[R(\widehat h_0+\Delta h,z_0)-R(\widehat h_0,z_0)].
\tag{5}
\]

Eğer o bölgede R'nin Lipschitz sabiti `L_R` ise hata `≤.01 L_R‖δ‖`; **L_R ölçülmediği için .01 katsayısı tek başına küçük hata garantisi değildir.** H8'in 96/96 aynı-durum türev başarısı, sonlu düzeltmenin decoder tersi olduğunu veya denoiser Jacobian'ını yok saymanın aynı trajectory'yi üreteceğini kanıtlamaz. Bu yeni yöntemin ilk gerçek-model kapısı, (5)'in doğrudan ölçülmesidir.

### 6.5 DDIM'e doğru taşıma: iki farklı algoritmayı karıştırmamak

**[Code]** Checkout'taki DDIM `prediction_type=epsilon`, `clip_sample=False` kullanır; kabul edilen baseline son alpha'yı 1 alır. `s`, scheduler'ın gerçek bir sonraki timestep'idir; indis olarak `t−1` varsayılmaz. DDIM'in stokastik eta'sı sıfırken (bu, style öğrenme oranı `.003` ile aynı parametre değildir):

\[
h_s=a_s\widehat h_0+\sigma_s\epsilon_\theta.
\]

**[Derivation/Proposal]** DiffPIR benzeri biçimde düzeltilmiş temiz tahminden effective epsilon yeniden hesaplanırsa

\[
\epsilon'=(h_t-a_t(\widehat h_0+\Delta h))/\sigma_t,
\qquad
\boxed{h'_s=h_s+\left(a_s-\frac{\sigma_sa_t}{\sigma_t}\right)\Delta h.}
\tag{6}
\]

İlk aday için önerilen aktarım budur. Eski `gamma` ile ayrıca çarpılmaz; güç/trust (1)'de tanımlanır. `σ_t=0` durumunda bölme yapılmaz; terminal temiz state doğrudan ele alınır. Buna karşılık eski epsilon'u koruyarak ilerlemek `h'_s=h_s+a_sΔh` verir: DDS-benzeri başka bir algoritmadır, (6) ile eşdeğer değildir. Stokastik DDIM, clipping veya farklı parameterization eklenirse (6) yeniden türetilmelidir.

**[Proposal]** PM'nin mevcut *düzeltme öncesi* Y gözlemleri ve gerçek son decode kullanılırsa mevcut decode/head sayısı korunabilir. `Y+δ`, gerçek `D(h_hat+Δh,z0)` diye classifier'a sunulmaz. Düzeltme sonrası PM istenirse ek decoder çağrıları açıkça maliyete eklenir. Böylece hız tahmini sahte bir decoder tasarrufuna dayanmaz.

### 6.6 Sınırlar ve korunacak karşı örnek

**[Inference]** Nokta–düzlem fidelity, temiz yüzey üzerindeki teğetsel yeniden örneklemeyi düzeltme gereği saymaz; indis-MSE'nin bir sorununu kaldırır. Fakat kenar, ince parça, yanlış normal, uzak yanlış eşleşme ve sistematik düşük frekanslı bozulma hâlâ sorunludur. Tek yönlü eşleşme, LION'un ürettiği fazladan yanlış parçayı gözlemle mutlaka silmez. Grafı Y'de kurmak da prior'ın yanlış semantiğini pekiştirebilir.

**[Derivation/Run]** En basit karşı örnek `r=c1` ve `A=I`: (4)'te DC modunda `δ=w/(κ+w)c1`. Referansın bu sabit kayması yanlışsa loss azalırken temiz şekilden uzaklaşılır. Graf bunu silemez. CPU denetimi bu başarısızlığı özellikle içerir. Dolayısıyla öneri, 4 Kasım'da bulunan bozuk-anchor sorununu *tam çözmüş* sayılmaz; eşleşme/robustluk yoluyla azaltılması sınanır.

## 7. Alternatifler ve seçim gerekçesi

| Aday | Hangi maliyeti azaltır? | Accuracy beklentisinin dayanağı | Karar |
|---|---|---|---|
| Aynı heat loss + Chebyshev/H8 | Yoğun eig ve decoder backward'ın bir kısmı; denoiser backward kalır | Amaç değişmediği için eski anchor hatası kalır | Eşdeğerlik/hız kontrolü; ana accuracy önerisi değil |
| (1)–(6) sağlam graf proximal | Denoiser backward ve her adımda SCD; eski fiziksel spectral yola göre ayrıca yoğun eig/decoder backward | Sırasız yüzey fidelity, aykırı artıklara sınırlı etki, düzeltmeye prior | Öncelikli falsifiable aday |
| Graf önkoşullu RED-Diff/CloudFixer benzeri residual güncelleme | Tam Jacobian yerine frozen score ve seyrek önkoşullandırma | Farklı prior/observation dengesi | İkinci sıra; objective ve latent aktarımı daha fazla değişiyor |
| PSD/SNR-adaptive graph Wiener | Eigenfree mümkün | Ancak temiz sinyal/noise istatistiği tanımlıysa anlamlı | Mevcut veride belirsizlik kalibrasyonu yok; ilk aday değil |

**[Paper/Inference]** Chebyshev, seyrek L ile polinom filtreyi yaklaşık `O(K|E|)` uygular; düzgün filtrelerin graf perturbasyonlarına karşı davranışı sert rank kesimine göre daha kontrollüdür. Bu ifadeler keyfî graf/topoloji değişimi altında accuracy garantisi değildir. [Hammond §6.1](https://arxiv.org/pdf/0912.3848), [Defferrard §2.1](https://arxiv.org/pdf/1606.09375), [Levie Teorem 2](https://arxiv.org/pdf/1901.10524).

**[Paper]** RED-Diff'te score residual'ının backward-free kullanımı belirli variational türetim ve timestep ağırlıklarına dayanır; epsilon-MSE'nin türevini keyfî biçimde silmekle aynı değildir. Ayrıca herhangi bir denoiser residual'ını global bir enerji gradyanı saymak için Jacobian simetrisi gibi koşullar gerekir. Bu yüzden ikinci alternatifin ancak yerel/algoritmik yorumu savunulabilir. [RED-Diff §4, Önerme 2](https://arxiv.org/pdf/2305.04391), [RED açıklamaları Lemma 3/Teorem 1](https://arxiv.org/pdf/1806.02296).

## 8. Gerçek hız nereden gelebilir?

**[Inference]** N=2048, B=32 için yalnız yoğun float32 eigenvector matrisi yaklaşık 512 MiB tutar. Yeni çözüm bunu saklamaz. SCD baseline'a karşı asıl beklenen kazanç, 8/35 adımda denoiser backward yerine aynı prior forward'larını ve yalnız üç fiziksel düzeltme çözümünü kullanmaktır. Decoder backward tasarrufu ancak fiziksel spectral eklentiyle karşılaştırırken geçerlidir; saf SCD baseline'da zaten yoktur. Sparse GPU kernel verimi, kNN, normal hesabı ve senkronizasyon bilinmeden hız oranı verilemez.

\[
T_{\rm yeni}\approx T_{\rm encode}+nT_{\rm prior\ forward}
+T_{\rm mevcut\ PM\ decode/head}
+K(T_{\rm graph}+T_{\rm match}+T_{\rm normal}+J T_{\rm PCG}).
\]

Bu formül gerçek maliyet dökümüdür, ölçüm değildir. Önceki H8 B2 süreleri graph kurulumunu tam temsil etmez; yeni yöntem için kullanılmaz. Chamfer nearest-neighbor araması tamamen yok olmaz: eşleştirme üç checkpoint'te yine gerekir. SCD'ye alternatif amaç kullanmak, geometrik komşuluk aramasından bütünüyle kurtulmak anlamına gelmez.

## 9. Yapılan yeni kontrol ve yapılmamış deneyler

**[Run: synthetic CPU]** NumPy 1.26.4 / Python 3.12.10, seed `20261104`; 15 düğüm, 23 gözlem, bir izole düğüm ve sıfır güvenli gözlemler. `audit.py` üretim sampler'ından bağımsız bir cebir tanığıdır; kNN/normal kestirimi, gerçek LION, CUDA veya classification kullanmaz.

| Denetim | Sonuç |
|---|---:|
| Matris kurmadan H çarpımı / yoğun H farkı | 1.78e−15 |
| CG / doğrudan çözüm bağıl farkı | 1.55e−11; 20 iterasyon |
| En küçük H özdeğeri; κ=1.7 | 1.7 |
| Kuadratik amaç değişimi / analitik formül | −2.565614046; fark yaklaşık 1e−15 |
| Yönlü sonlu fark hatası | 1.34e−9 |
| Tek Huber majorizer sonrası gerçek donmuş amaç değişimi | −2.854535886 |
| Nokta/gözlem permütasyonu sonrası çözüm farkı | 1.11e−16 |
| Normal işareti değişimi | H ve b değişmiyor |
| İzotropik spektral formül / çözüm farkı | 7.22e−16 |
| Yeniden hesaplanan epsilon ile DDIM aktarım hatası | 3.83e−16 |
| Eski epsilon'u koruma ile fark | 0.56844; eşdeğer değiller |
| Oyuncak residual decoder'da lift hatası / üst sınır | .001108 / .002017 |
| Temiz düzlem üzerinde teğetsel yeniden örnekleme | normal residual 0; indis-MSE 1.7567 |
| Tutarlı yanlış DC gözlemi | objective azalırken clean-MSE +.01917 |

Normal işareti/permütasyon kontrolleri **donmuş operatör** içindir; gerçek eşleşme tie-breaking'inin uçtan uca invariant olduğu henüz sınanmadı. Oyuncak decoder tanh fonksiyonudur; LION için Lipschitz ölçümü değildir. Hiçbir satır speedup veya accuracy sonucu değildir.

## 10. En küçük anlamlı gerçek-model değerlendirmesi

**[Proposal/Open]** Aşağıdaki çalışma tasarımı uygulanmadı ve GPU bütçesi verilmiş bir run talimatı değildir. Eski sonuçları yeniden üretmek yerine yalnız yeni varsayımları sınamak gerekir.

1. **Aktarım kapısı:** Kontrollü native backend ile arşivli durumlarda (5)'i ölç; birkaç trust büyüklüğü için gerçek decoder düzeltmesinin yönü/normu ve (6)'nın actual scheduler çıktısı sınansın. H8/full türev testini tekrar etmek bunun yerine geçmez. Normal geçerliliği, izole düğüm, nokta sırası ve tekrar örneklemeye dayanıklılık da burada kontrol edilir.
2. **Amaç kapısı:** Geliştirmeden ayrılmış mekanizma nesnelerinde fiziksel clean oracle yalnız teşhis için kullanılsın. Robust-graf düzeltme, eşlenmiş λ=0 düzeltmeyle clean Chamfer/yüzey yönü bakımından karşılaştırılsın; clean veri veya test label'ları gerçek guidance girdisi olmasın. Oracle kötüleşmesi classifier başarısızlığına eşit değildir, ama yaygın ters yön bulgusu adayın gerekçesini zayıflatır.
3. **Nedensel kontroller:** Operasyonel C (.03/.003), z sabit C, aynı seyrek guidance sıklığıyla C, backward-free geometrik kontrol ve önerinin λ=0 / λ>0 halleri ayrıştırılsın. Tüm kolları büyük benchmark'a taşımak gerekmez; teknik/pilot kapılarında elenebilir. Yalnız C ile yeni yöntemi karşılaştırmak, kazancı graf, style veya azalan update sayısından hangisinin verdiğini açıklamaz.
4. **Kilitli doğrulama:** Geliştirmede seçilen tek konfigürasyon, önceden belirlenmiş ayrık nesne kümesinde 15 bozulma ve en az mevcut çalışma düzeyinde çoklu seed ile doğrulansın. Aynı ID, encode/noise cache, scheduler, fiziksel frame, classifier, PM okuma ve backend kullanılsın. Nesne-kümeli paired bootstrap raporlansın; seed×corruption satırları bağımsız örnek gibi sayılmasın. Önceden görülmüş benchmark bütünü “hiç kullanılmamış test” diye sunulmasın.
5. **Ortak başarı ölçütü:** C'ye karşı PM farkının %95 paired CI alt sınırı >0 ve toplam steady-state süre C'den düşük olmalı; örneğin önceden ilan edilen ≥%20 süre düşüşü pratik hedef olabilir, tahmin değildir. λ>0, λ=0'ı geçmezse GSD katkısı gösterilmemiştir. Daha hızlı ama daha düşük accuracy veya daha doğru ama daha yavaş sonuç ayrı bir Pareto sonucu olarak korunur.
6. **Tam maliyet:** B1 ve B32; warm-up sonrası senkronize ölçüm, median ve dağılım, peak bellek. Graph/normal/matching/solver/decode/head/preprocessing toplamda yer alsın; kurulum ayrıca dökülsün. Yalnız Gaussian ve Background zamanlarını bütün bozulmaların sonucu diye genellemeyelim. İlk yükleme maliyeti ayrı verilsin.

**[Open]** Mevcut 128 nesne/3 seed sonucunun CI genişliği yaklaşık ±0.4 pp'dir; çok küçük farklar için güç yetersiz olabilir. Yeni adayın ne kadar örneğe veya iterasyona ihtiyaç duyduğu şimdiden bilinmiyor. Parametre sayısını büyüten bant/normal/graf aile taraması, ancak bu dar adayın temel kapıları geçmesinden sonra anlamlıdır.

## 11. Karar ve tezde savunulabilecek ifade

**[Decision]** Baseline ve üretim varsayılanları değişmedi. Yeni sampler kodu, bağımlılık, classifier adaptasyonu veya GPU run'ı eklenmedi. Bir sonraki teknik adım tam benchmark değil, LION sonlu-adım lift ve scheduler aktarım kapısıdır.

**[Inference]** Şimdilik savunulabilecek sonuç: “Graf spektral bilgiyi bozuk koordinatlara mutlak fidelity olarak uygulamak yerine, diffusion temiz tahmininin sağlam yüzey düzeltmesine Gaussian/kuadratik prior olarak uygulamak; LION'un residual decoder yapısı sayesinde ağ geri yayılımı ve yoğun özayrışım olmadan gerçekleştirilebilecek bir aday sağlar.” Genel graf-posterior formülünün önceki çalışmalarda bulunduğu açıkça belirtilmelidir. Accuracy/hız üstünlüğü, bilinmeyen bozulma altında güvenilirlik ve özgün uygulama katkısı [Open] durumundadır.

## 2026-11-05 GPU follow-up supersedes feasibility assumptions

[Run/Inference/Open] [Four-hypothesis results](gsd_hypotheses_results_20261105.md) test the proposal on392states/group. Uniform graph benefit and direct identity lift are not supported; H1/H2 superiority remains unestablished. Correct sparse optimization and DDIM algebra do not establish clean-geometry/accuracy benefit. The literature and derivations above remain dated proposals, not validated deployment claims.
