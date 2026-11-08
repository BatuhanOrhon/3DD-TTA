# LION ve 3DD-TTA entegrasyonlarının makale ve kod karşılaştırması

Tarih: 8 Kasım 2026. İlk inceleme tabanı: `thesis-integration@117b7f9e807999ac8df74db79eaf74a32e5463f6`; son uzlaştırma, yeni all15 kaydını içeren `44af62acf4301b785c53b6819e9f5875698ff3e8` ile yapılmıştır.

**[Inference] Sonuç:** En güçlü yöntem katkısı adayı, LION'un nokta yapılı yerel latentinde iki yönlü kırpılmış Chamfer eşleşmelerini dondurup kapalı biçimli bir düzeltme çözmek ve bu düzeltmeyi aynı epsilon tahminiyle DDIM geçişine taşımaktır. Bu, incelenen görüntü-uzayı yöntemlerinin birebir uygulaması değildir. Ancak proximal güncelleme, denoiser backward'ını kaldırma ve timestep ensembling genel ilkeleri önceki çalışmalarda vardır. PM için savunulabilir iddia, mevcut donmuş 3B sınıflandırıcıyla seyrek decoded-clean yörünge tahminlerinin değerlendirilmesidir. Evrensel ilk kullanım veya güçlü bağımsız algoritmik yenilik bu incelemeyle kanıtlanmış değildir.

Bu rapor önce makalelerin ne yaptığını, sonra uygulanan aktarımı ve özgünlük sınırını ayırır. **[Paper]** birincil yayın, **[Code]** kaynak kod, **[Run]** kabul edilmiş deney kaydı, **[User report]** kullanıcı beyanı, **[Inference]** değerlendirme, **[Open]** henüz kurulmamış bağlantıdır. Yeni model inference'ı, GPU deneyi veya runtime işlemi yapılmadı. Kod ve ilgili yöntem bölümleri okundu; makalelerin bütün deneyleri yeniden üretilmedi.

## 1. Bir fikrin kaynağı ile sonradan bulunan öncül farklıdır

**[Code/User report]** 11 Ekim yörünge fikri, ara temiz nokta bulutlarını değerlendirme ve olasılık ortalaması alma biçiminde kullanıcının önerisi olarak kayıtlıdır. TEDM, Wang ve Li makaleleri 25 Ekim atıf incelemesinde ilgili öncüller olarak bulunmuştur. Bunların doğrudan ilk ilham olduğunu söylemek tarihsel kaydı değiştirir. Doğru ifade, yerel önerinin literatürdeki öncüllerle konumlandırılmasıdır; doğrudan dış esin kaynağı mevcut kayıtta belirtilmemiştir. [Tarihsel öneri ve atıf kararı](gsd_trajectory_attribution_20261025.md).

**[Code/Paper]** Buna karşılık 7 Kasım [LION adayları](guidance_lion_candidates_20261107.md), koddan önce Guess & Guide, CloudFixer, Piecewise Guidance, RPM-Net ve Readout Guidance ile ilişkileri açıkça kurar. Bunlar aynı rolü oynamaz: bazıları aktarım dayanağı, bazıları benzer ilkenin öncülü, bazıları yalnız yapısal uyarı veya kontrol gerekçesidir. Bir kaynakçadaki her makale uygulanmış bileşen değildir.

## 2. Ortak altyapı: LION ve orijinal 3DD-TTA

### 2.1 LION hangi yapıyı sağlıyor?

**[Paper]** [LION: Latent Point Diffusion Models for 3D Shape Generation](https://proceedings.neurips.cc/paper_files/paper/2022/file/40e56dabe12095a5fc44a6e4c3835948-Paper-Conference.pdf), NeurIPS 2022, §3:

1. VAE, nokta bulutunu global şekil latenti `z` ve `z`'ye koşullu nokta yapılı yerel latent `h` ile temsil eder.
2. Ayrı diffusion prior'ları global dağılımı ve koşullu yerel dağılımı modeller.
3. Decoder, `h` ve global şekil bilgisinden fiziksel nokta bulutunu üretir.

**[Code]** Kullanılan konfigürasyonda `h` açıldığında `B×2048×4`: üç XYZ-benzeri latent koordinat ve bir ek özellik kanalıdır. Bunlar fiziksel giriş koordinatlarıyla otomatik olarak özdeş değildir. `models/latent_points_ada.py:255` decoder'ı bütün kanalları tüketir; çıktı, latent XYZ skip yolu ile öğrenilmiş residual'ın toplamıdır. Dolayısıyla dördüncü kanalın doğrudan güncellenmemesi, modelde etkisiz olduğu anlamına gelmez.

**[Inference]** Bu mimari ve önceden eğitilmiş ağırlıklar LION'a aittir. Çalışmanın katkısı bunları yeniden tasarlamak değil, test-zamanı değişkenlerine nasıl müdahale edildiğidir.

### 2.2 3DD-TTA ne ekliyor?

**[Paper]** [Test-Time Adaptation of 3D Point Clouds via Denoising Diffusion Models](https://openaccess.thecvf.com/content/WACV2025/papers/Dastmalchi_Test-Time_Adaptation_of_3D_Point_Clouds_via_Denoising_Diffusion_Models_WACV_2025_paper.pdf), WACV 2025, §3:

1. Bozuk giriş LION ile encode edilir.
2. Yerel latente seçilen derinlikte Gaussian noise eklenir.
3. DDIM ile ters süreçte temiz yerel latent tahmin edilir.
4. Selective Chamfer Distance, büyük mesafeleri kırparak tahmini girişin encoded geometrisine bağlar.
5. Loss, denoiser üzerinden yerel noisy state ve global conditioning style'a geri yayılır.
6. Son latent decode edilip donmuş sınıflandırıcıya verilir.

**[Code]** `tta.py:174` prior çağrısı; `:175–176` DDIM ve predicted-clean; `:187–199` ilk üç kanalda SCD; `:229–234` backward, yerel `gamma` ve style `eta` güncellemesi; son decode `:256` sonrasındadır. Güncel PM gözlemcisi `:180–183`'te mevcut adımın guidance güncellemesinden önce çağrılır.

**[Inference]** LION'u 3B TTA için kullanma, SCD ve style conditioning güncellemesi zaten temel makalenin katkılarıdır. Bunlar yeni tez yönteminin özgün bileşenleri olarak yeniden sahiplenilemez. N5→N8 ise başlangıç noise derinliğini ve ters güncelleme sayısını birlikte değiştirir; yalnız sabit bir noise aralığını daha sık örneklemek değildir.

**[Paper/Code]** Paper yöntemi bu tur venue PDF'ye erişim hatası nedeniyle [birincil arXiv v1 §3.3/Eş.8–12/Alg.1](https://arxiv.org/html/2411.14495v1) üzerinden okundu. Repository comparator'ı tarihsel `p=.95`/summed-SCD kullanır; paper'ın `.96` ve point-count normalization ifadesiyle birebir özdeş diye sunulmaz. Kodda `gamma=local, eta=style`; paper ve bazı legacy host'larda isim eşlemesi farklıdır.

## 3. Probability Mean: makalelerden LION readout'una

### 3.1 Ortak önceki ilke

**[Paper]** [TEDM: Robust semi-supervised segmentation with timestep ensembling diffusion models](https://proceedings.mlr.press/v225/rosnati23a.html), ML4H 2023, §4, Eş. 8–9, basılı s.516: farklı timestep temsilleri ortak, eğitilen bir piksel predictor'ına verilir; timestep tahminleri ortalanır. Görev tıbbi görüntü segmentasyonudur ve predictor eğitimi vardır.

**[Paper]** [Time Is a Feature: Exploiting Temporal Dynamics in Diffusion Language Models](https://arxiv.org/html/2508.09138v3), §4.1, Eş.4: tek denoising yörüngesindeki ara cevaplar çıkarılır, semantik olarak gruplanır ve ağırlıklı oyla birleştirilir. Bu, aynı yörüngeyi tekrar kullanma öncülüdür; decoded 3B geometrinin sınıf olasılık ortalaması değildir.

**[Paper]** [Understanding Representation Dynamics of Diffusion Models via Low-Dimensional Modeling](https://papers.nips.cc/paper_files/paper/2025/file/9a3b1949b6da9b8facfdc28e1b4722ed-Paper-Conference.pdf), NeurIPS 2025, Ek B.4: farklı noise seviyelerindeki temsillere classification probing uygulanır ve tahminler logit düzeyinde birleştirilir. Logit ortalaması, softmax sonrası olasılık ortalamasıyla aynı işlem değildir.

Bu tur Ek B.4 yöntemi [birincil arXiv v3 HTML](https://arxiv.org/html/2502.05743v3) üzerinden doğrulandı: seçilen optimum timestep çevresinde iki daha ince ve iki daha kaba düzey, her düzeyde ayrı linear classifier, ardından logit ortalaması. Venue PDF s.28 eşlemesi önceki audit'tendir; sayfanın bu tur görsel kontrolü timeout nedeniyle tamamlanmadı.

### 3.2 Bizdeki adımlar ve değişiklikler

**[Code]** Güncel N8+PM sözleşmesi [current_baseline.md](current_baseline.md)'dedir:

1. Aynı LION ters yörüngesi bir kez yürütülür.
2. Normal N8 koşulunda 2/4/6. adımların `pred_original_sample` değerleri alınır; sıradan son çıktı ayrıca tutulur. Background N35 için ara adımlar 9/18/27'dir.
3. Her ara latent aynı orijinal encoded global latentle decode edilir; classifier'a uygun koordinat dönüşümü ve örnekleme uygulanır.
4. Her mevcut source head kendi başına, ağırlık ve buffer'ları sabitken dört geometriyi değerlendirir.
5. `p_k=softmax(f(D(h_k,z_0)))`; karar `argmax_c (1/4) Σ_k p_k(c)` olur.

`scripts/analyze_legacy_trajectory.py:17–37`, sayısal olarak kararlı softmax ve aday eksenindeki ortalamayı uygular. `shared_trajectory.py` ve `legacy_trajectory.py` probe sırasında RNG/buffer durumunu korur. Hız koşusunun hafif gözlemcisi `scripts/run_lion_guidance_speed.py:25–69`; gerçek PM indirgemesi `:349`'dadır.

**[Code]** Makalelerden farklı olarak yeni predictor eğitilmez, classifier/BN adaptasyonu yapılmaz, dört farklı modelin logits'i karıştırılmaz. Dört source model ayrı endpoint'lerdir. Bu, dört tam bağımsız diffusion koşusunun ensemble'ı da değildir. Ara adaylar mevcut adımın guidance'ından öncedir; terminal son güncellemeden sonradır. Eski legacy host'un style/decode sözleşmesi güncel N8 host'uyla aynı kabul edilmemelidir.

**[Inference]** Özgünlük: genel ortalama kuralında düşük; donmuş 3B classifier, decoded-clean seyrek adaylar ve tek yönlendirilmiş LION yörüngesinin birlikte kullanıldığı sistem uyarlamasında somut fark vardır. İncelenen öncüller bu tam pipeline'ın birebir kopyası olduğunu göstermiyor; bu da ilk kullanım kanıtı değildir.

### 3.3 Entropi seçimi ayrı bir koldur

**[Paper]** [Purge-Gate](https://arxiv.org/html/2509.09785v1), ICCV 2025, §3.3.1, Eş.20–21, farklı token-purging uzunluklarını minimum predictive entropy ile seçer. Bizde aday değişkeni purging uzunluğu yerine diffusion checkpoint'idir; classifier token yapısı değiştirilmez. Bu, bilinen seçim kriterinin farklı aday ailesine aktarımıdır. PM ve minimum-entropy kararları birbirine eklenmiş bir tek yöntem olarak sunulmaz.

## 4. Hızlandırma: Guess & Guide/Spin'den doğrudan latent Chamfer düzeltmesine

### 4.1 Makaledeki çözüm

**[Paper]** [Guess & Guide: Gradient-Free Zero-Shot Diffusion Guidance, v1](https://arxiv.org/html/2603.07860v1), 9 Mart 2026, §3/Alg.1 ve Ek B.1–B.2:

1. Bilinen ölçüm modeli `y=A(x)+noise` ve pretrained diffusion prior kullanılır.
2. Ara timestep'te denoise–data consistency–re-noise tekrarlarıyla warm start oluşturulur.
3. Tahmin decode edilir; fiziksel/görüntü uzayında ölçüm uyumsuzluğu ile temiz tahmine uzaklığı birleştiren problem çözülür.
4. Sonuç yeniden encode edilir; noise bilgisi korunarak noisy uzaya geri taşınır ve DDIM sürdürülür.

Ek B, bu data-consistency işleminin proximal/MAP yorumunu ve doğrusal `A` için kapalı biçimini verir. Optimizasyonun gradient'i gerektiğinde `A` üzerinden geçer; denoiser veya decoder VJP'si gerekmez. “Gradient-free”, bütün matematiksel işlemlerin gradientsiz olduğu anlamına gelmez.

**[Paper]** Aynı çalışma [v3'te Sparse Scheduled Diffusion Guidance for Inverse Problems (Spin)](https://arxiv.org/html/2603.07860v3), 28 Mayıs 2026 adını taşır. §3 ve Ek A.1–A.2, seyrek guidance, latent/pixel sınırı ve conservative stochastic re-noising'i açıklar. V1 ve v3 iki bağımsız öncül sayılmaz. Projede ilk dayanak v1 adıyla kaydedilmiştir; v3 sürüm farkı atıfta korunmalıdır.

### 4.2 LION'a aktarırken çözülen farklı problem

**[Code/Inference]** Bizde bilinen görüntü bozunum operatörü `A` yoktur. Referans, bozuk nokta bulutunun encoded yerel XYZ'sidir. Fiziksel uzayda her adım decode–optimize–encode yapmak yerine LION'un nokta yapılı latentine doğrudan müdahale edilir. `lion_guidance_pilot.py:7–18`, `:69–76`, `:113–172` ve çağıran `scripts/run_lion_guidance_speed.py:346` bu yolu uygular. Çalıştırılan `tau=.1`'dir; yardımcı fonksiyonun varsayılan argümanından bilimsel koşu katsayısı çıkarılmaz.

İşlem adımları:

1. Denoiser yalnız forward ile temiz latent `(q_hat,f_hat)` ve DDIM'in ordinary `prev_sample` değerini üretir.
2. `q_hat` ile corrupted encoded XYZ referansı `r` arasında iki yönlü nearest-neighbor eşleşmeleri bulunur. Her yönde en küçük `%95` mesafe tutulur.
3. Bu adım için eşleşmeler ve kırpılan kümeler sabitlenir. Ters yönden aynı tahmin noktasına gelen bütün referans noktaları `scatter_add` ile toplanır. İki yönde seçilen aynı kenarın çokluğu korunur.
4. Her nokta için kapalı biçimli proximal düzeltme hesaplanır.
5. Sadece temiz XYZ değişimi, actual previous cumulative alpha ile DDIM'in noisy previous state'ine aktarılır.
6. Sonraki adım yine bütün dört kanalı işleyen LION prior'ıyla yürür. Cheap-only kolunda guidance style güncellemesi yapılmaz; dördüncü kanal ve style model içinde kullanılmaya devam eder.

### 4.3 Hangi denklem bizim uygulamamızın çekirdeği?

**[Code/Inference]** Sabit seçilmiş iki yönlü kenar çokluğu `m_ij` için:

`Q(q) = (1/2) Σ_ij m_ij ||q_i-r_j||²`

`q_plus = argmin_q (1/2)||q-q_hat||² + tau Q(q)`

Nokta başına çözüm:

`q_plus_i = (q_hat_i + tau Σ_j m_ij r_j) / (1 + tau Σ_j m_ij)`.

Bu formül bir kuadratik problemin standart çözümüdür. Projeye özgü tasarım, iki yönlü kırpılmış Chamfer kenarlarının bu probleme dönüştürülmesi ve LION sampler'ına doğru ölçekte/sırada bağlanmasıdır. Eşleşmeler her adım yeniden kurulduğu için bu, bütün nonconvex Chamfer probleminin tek adımda global optimumu değildir.

**[Code/Inference]** Deterministik DDIM'de aynı `eps_hat` korunursa:

`h_prev = sqrt(alpha_bar_prev) h_clean + sqrt(1-alpha_bar_prev) eps_hat`

olduğundan aktarım:

`h_prev_plus = h_prev + sqrt(alpha_bar_prev) (q_plus-q_hat, 0)`.

Kod bunu `bridge_clean_delta` ile yapar; önceki alpha actual scheduler grid'inden, son geçişte scheduler'ın final-alpha değerinden alınır. Buradaki `0`, feature kanalına doğrudan ek correction verilmediğini gösterir. Bu cebirsel eşitlik, G&G/Spin'in stochastic posterior sampler'ını veya bütün teorik garantilerini yeniden üretmez.

**[Inference]** Dondurulmuş eşleşme/kırpma kuadratiği için descent açıklanabilir. Bunun decoder çıktısındaki clean-CD'yi, doğru sınıf olasılığını veya accuracy'yi her örnekte artırdığı sonucu çıkmaz. Orijinal SCD'nin denoiser Jacobian'ı üzerinden gelen dört-kanal/style yönü de aynen korunmuş değildir; hesap kazanımının yanında güncelleme mekanizması değişir.

### 4.4 CloudFixer ile doğrudan benzerlik ve fark

**[Paper]** [CloudFixer](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04603.pdf), ECCV 2024, §4.1–4.2/Eş.1: nokta bulutuna dönüş ve nokta başına displacement uygular; diffusion modelinden detached temiz hedef alıp dönüşüm parametrelerini Chamfer ile optimize eder. Denoiser backward'ı yoktur. §4.2, farklı stokastik adaptasyon sonuçlarının olasılıklarını oylamayı da içerir.

**[Code/Inference]** Bizde optimizasyon değişkeni fiziksel `R,Delta` değil yerel latent temiz XYZ'dir; hedef yönü corrupted latent referansa doğrudur, tek LION ters yörüngesi sürer ve kapalı çözüm doğrudan DDIM'e aktarılır. PM adayları ayrı stokastik adaptasyon koşuları değil aynı yörüngenin checkpoint'leridir. Bu farklar somuttur; “3B TTA'da denoiser backward'ını ilk kez kaldırdık” iddiası CloudFixer nedeniyle savunulamaz.

### 4.5 Piecewise Guidance ve denenmiş hybrid

**[Paper]** [Diffusion Models for Solving Inverse Problems via Posterior Sampling with Piecewise Guidance](https://arxiv.org/html/2507.18654v1), §III, bilinen doğrusal ölçüm modeli için noise seviyesine bağlı farklı conditional-score yaklaşımları kullanır. Düşük-noise yaklaşımı ile yüksek-noise yaklaşımı bir threshold ile ayrılır.

**[Code/Inference]** Projenin hybrid kolu `1/4/7/...` döngülerinde tam backward, aralarda cheap correction kullanır; sparse kol aynı tam-backward takviminde aradaki guidance'ı atlar. Bu modulo-3 takvim, makaledeki noise-threshold score formülü değildir. Genel pahalı/ucuz guidance bileşimi bilinir; bu particular hybrid'in accuracy faydası pilotta gösterilmedi. Nihai macro14 hız sonucu cheap-only kola aittir, hybrid başarısı diye aktarılmaz.

## 5. Sonuçların özgünlük değerlendirmesindeki doğru yeri

**[Run]** [8 Kasım macro14 raporu](lion_guidance_macro14_results_20261108.md): ModelNet40-C, Point-MAE, bütün 2.468 nesne, 14 bozulma, seed0, B128/final36; üç runtime'dan ayrı corruption grupları birleştirilmiştir.

| Yöntem | Accuracy | Batch-amortized süre |
|---|---:|---:|
| Orijinal N5 terminal SCD | %64.051285 | 80.383 ms |
| Ayarlanmış N8 terminal, aynı kaydedilmiş izden | %64.896388 | Ayrı ölçülmedi |
| Ayarlanmış N8+PM | %66.137995 | 148.318 ms |
| Cheap latent correction+PM | %66.088794 | 107.275 ms |

**[Run/Inference]** PM'nin kendi N8 terminaline katkısı `+1.241607pp`; N8 terminal−N5 terminal farkı `+.845103pp` ve depth+oran değişimidir. Cheap−N8PM `−.049201pp`, paired nesne-cluster %95 aralığı `[−.219958,+.124450]pp`; süre `%27.672109` azalır. Önceden seçilen `−.3pp` noninferiority sınırı ve en az `%15` süre tasarrufu koşulları geçer. Bu eşit accuracy ispatı veya B1 latency ölçümü değildir.

**[Run]** İki veri kümesi/dört head/üç seed üzerindeki [31 Ekim N8+PM transferi](chamfer_depth_transfer_results_20261031.md) ayrı kanıttır. Bu sekiz endpoint'in sonuçları, cheap hızlandırmanın sekiz endpoint'te de test edildiği anlamına gelmez. **[User report/Open]** Kullanıcının tüm sekiz koşulda pozitif hız/doğruluk beyanı korunur; bu incelemede bağlanan 80/148/107 ms kaydı yukarıdaki tek-head macro14 çalışmasıdır. Ek sekiz-koşul cheap arşivleri mevcutsa ayrı source/protocol bağıyla değerlendirme genişletilir; kullanıcı beyanı sessizce run-backed kanıta dönüştürülmez.

**[Open]** Cheap ile unguided, aynı sabit style ve aynı PM bütçesi altında tamamlanmış nihai karşılaştırma bu macro14 kaydında yoktur. Dolayısıyla sonucun ne kadarının özellikle nonzero proximal correction'dan, ne kadarının backward/style-guidance kaldırılmasından geldiği izole edilmiş değildir. Bu, operasyonel doğruluk–maliyet sonucunu ortadan kaldırmaz; mekanizma özgünlüğü iddiasının kapsamını belirler.

### 5.1 İnceleme sürerken gelen all15 güncellemesi

**[Run]** `44af62a` ile ayrı Background N35 fazı tamamlanıp [all15 sonuç kaydı](lion_guidance_all15_results_20261108.md) yayımlandı. Aynı ModelNet/Point-MAE/2468 nesne/seed0/B128 kapsamındaki composite:

| Yöntem | All15 accuracy | Batch-amortized süre |
|---|---:|---:|
| Orijinal N5/N35 terminal SCD | %63.854673 | 104.452 ms |
| Ayarlanmış N8/N35+PM | %66.015667 | 169.922 ms |
| Cheap+PM | %65.729335 | 119.689 ms |

Cheap−N8PM `−.286332pp`, %95 CI `[−.472717,−.097245]pp`; süre `%29.562521` azalır. Nokta tahmini .3pp içinde olsa da CI alt sınırı `−.3pp`'den aşağıdadır; önceden seçilen noninferiority koşulu all15'te geçmez. Background tek başına `−3.606159pp` kaybeder; macro14 koşulu geçmeye devam eder. Cheap−orijinal terminal all15 `+1.874662pp`, `%14.58755` ek süreyle elde edilir. Bu yüzden “bütün bozulmalarda accuracy korunarak hızlanma” iddiası desteklenmez; kapsamı belirtilmiş doğruluk–maliyet dengesi desteklenir.

**[Inference]** Bu yeni negatif sınır, makale–kod arasındaki özgün tasarım farklarını silmez. Fakat yöntem yararlılığı ile originality ayrı değerlendirilmelidir; özellikle sekiz model/veri kümesi için universal cheap üstünlüğü bu tek-head all15 sonucu üzerinden kurulamaz. Buradaki güncelleme peer'in tamamlanmış kayıtlarını okur; yeni GPU veya Background yeniden koşusu yapılmadı.

## 6. Savunulabilir özgünlük düzeyi

| Bileşen | Önceden bilinen kısım | Somut yerel fark | Karar |
|---|---|---|---|
| PM | Timestep prediction ensemble, softmax ortalaması, voting | Tek yönlendirilmiş LION yörüngesinden sparse decoded-clean + terminal, mevcut frozen 3B heads | Uygulama/deneysel katkı; genel ensembling buluşu değil |
| N8 ve oran ayarı | Horizon/noise/rate seçimi | Kontrollü karşılaştırma ve aktarılabilir bir operating point | Deneysel katkı; bağımsız algoritmik yenilik zayıf |
| Cheap correction | Proximal split, frozen target, backward kaldırma | İki yönlü %95-kırpılmış latent NN kuadratiği + kapalı çözüm + feature-zero DDIM bridge | İncelenenler içinde en somut yöntem uyarlaması; global ilk kullanım ve proximal-specific kazanç henüz kurulmuş değil |
| Bütün sistem | LION, SCD ve bilinen readout/guidance ilkeleri | Sabit modellerle doğruluk–maliyet dengesi sağlayan test-zamanı pipeline | Savunulabilir artımlı araştırma katkısı |

**[Inference]** Güçlü anlatım: “3DD-TTA'nın yerel latent yönlendirmesini, iki yönlü kırpılmış eşleşmeler üzerinde çözülen bir temiz-latent düzeltmesiyle ucuzlatıyor; bu düzeltmeyi DDIM geçişine taşıyıp aynı yörüngedeki seyrek tahminleri donmuş sınıflandırıcılarla birleştiriyoruz.” Bu cümle işlemin ne olduğunu açıklar; “yeni proximal optimizasyon”, “yeni ensemble prensibi” veya “ilk gradientsiz 3B TTA” iddiası içermez.

Başarı ölçüsü yalnız kaynak kodun başka yerde bulunmaması değildir. Bilinen bir işlemi farklı isimlendirmek yenilik sağlamaz; farklı değişken/amaç/bağlaşım üzerinde işe yarayan, gerekçeli ve karşılaştırılmış bir çözüm üretmek araştırma katkısı olabilir. Bu çalışmada ikinci tür somut tasarım farkları vardır. Kapsamlı öncelik taraması ve en yakın mekanizma kontrolleri olmadan “tamamen yeni algoritma” hükmü vermiyorum.

## 7. Kaynak ve sürüm denetimi

**[Code]** `lion_guidance_pilot.py`, gerçek macro14 bilimsel kaynak `af90b34` ile bu review tabanında byte eşittir: SHA256 `ce79086ae259682895274d2d3f8324098799665b0ffbbec6df3f740c8584c7d0`. Runner'daki sonraki farklar Background fazı ve protokol metadata'sıdır; macro14 işlevlerine ait continuation bridge ayrıca [AST tanığında](evidence/lion_guidance_speed_20261107/r4_continuation_source_ast_witness.json) kayıtlıdır. Kaynak dosya kimlikleri ve okunan primary URL/sürümleri bu raporun compact evidence dizinindedir.

Bu inceleme kod, matematik ve önceki kabul edilmiş deney kayıtları arasındaki bağlantıyı denetler. Ham macro14/transfer ZIP'lerinin tamamı yeniden hash'lenmedi; önceki raw acceptance/independent review kayıtları kullanıldı. Yeni deney, koşu sonucu, novelty garantisi veya yayın kabulü üretilmedi.

## Ek A. Spektral yönlendirme ve gradient composition

### A1. GSDTTA'dan LION latent fidelity'ye

**[Paper]** [3D Test-time Adaptation via Graph Spectral Driven Point Shift](https://openaccess.thecvf.com/content/ICCV2025/papers/Wei_3D_Test-time_Adaptation_via_Graph_Spectral_Driven_Point_Shift_ICCV_2025_paper.pdf), ICCV 2025, §3.1–3.2, Eş.7–18, PDF s.4–5:

1. Fiziksel XYZ üzerinde RBF-kNN/outlier grafı ve `L=U Lambda U^T` oluşturulur.
2. `C=U^T X` katsayılarının yalnız düşük M modu için `DeltaC` öğrenilir; `X_shift=X+U_M DeltaC` elde edilir.
3. Özellik/eigenmap descriptor'larından pseudo-label ve merkezler çıkarılır.
4. CE, entropy/diversity ve tek yön Chamfer ile input kaydırması; ayrı adımda classifier adaptasyonu yapılır.

**[Code]** Bizde `tta_gsd.py:398` sonrası corrupted local-latent XYZ'den baz kurulur; her temiz latent tahminine karşı bozuk latent referansın spektral farkı cezalandırılır. Kayıp SCD ile birlikte denoiser üzerinden noisy local/style'a türev verir. Classifier adaptasyonu, makaledeki descriptor/pseudo-label döngüsü ve doğrudan öğrenilen `DeltaC` yoktur.

Legacy heat için kodun hedefi:

`F=U diag(exp(-beta lambda_i) 1[i<M]) U^T`,

`L_spec = sum_b tr((Y-Q)^T F (Y-Q))/(3N)`.

Bu, temizlemek için düşük bantta hareket **öğrenmekten**, bozuk referanstan düşük-bant uzaklaşmayı **cezalandırmaya** geçiştir. Ortak sabit bazda gradient `2F(Y-Q)/(3N)` olduğundan açık bir inverse-GFT fonksiyon çağrısının bulunmaması hata değildir; dönüşüm türevde gerçekleşir. Fakat fidelity'nin minimumu `Y=Q`'dur; temiz geometriye yararlı hareket de cezalandırılabilir.

**[Code]** Dosya ismi yöntem kimliği değildir. `graph_spectral.py` modern self-excluded/active-graph hard ve normalized smooth profilleri taşır. `graph_spectral_v2.py` adına rağmen legacy self-query, raw-lambda/truncated-heat ve farklı threshold yoludur; `spectral_guidance_losses.py` bunu tüketir. `3r` ve `3N` reduction'ları, raw ve normalized lambda aynı yöntem sayılmaz. `tta_gsd_v2.py` legacy host'unda `eta=local, gamma=style`; modern/original host'ta ters isim eşlemesi vardır.

**[Inference]** GFT/low-band fikri doğrudan esindir. Latent-space fidelity ve diffusion-input gradient'i belirgin bir proje uyarlamasıdır; full GSDTTA reprodüksiyonu değildir. Makalenin öğrenilmiş shift/model-adaptation başarısı bu farklı objective'e taşınamaz.

### A2. PixelAsParam ve PCGrad'den local/style projection'a

**[Paper]** [PixelAsParam](https://proceedings.mlr.press/v202/dinh23a/dinh23a.pdf), ICML 2023, §4.2/Eş.12 ve §5/Eş.16/Alg.2: DDPM güncellemesini denoising, diversity ve classifier yönlerine ayırır; seçilen yön çiftlerinde negatif dot varsa karşı yöne dik projeksiyon uygular. Bütün çiftlere müdahale etmenin her zaman iyi olmadığı §6.4'te incelenir.

**[Paper]** Daha eski [Gradient Surgery for Multi-Task Learning / PCGrad](https://arxiv.org/html/2001.06782v4#S2.SS3), NeurIPS 2020, §2.3/Alg.1, model-weight task gradient'lerinde negatif dot halinde çatışan bileşeni çıkarır.

**[Code]** Tarihsel `53ba252:tta_pxp.py:225–266`, SCD yönünü spectral yöne karşı projekte eder; `tta_pxp_sym.py` iki yönü karşılıklı projekte eder. Batch bütünü flatten edilir. DDPM denoising/diversity/classifier ayrışımı uygulanmaz. Sonraki `gsd_composition.py:109–179` ise örnek başına local ve style bloklarını ayırır, weighted gradient'leri bir kez oluşturur ve symmetric projection sonrası norm cap uygular:

`c'=c-(c.q/||q||²)q`, `q'=q-(c.q/||c||²)c` (yalnız çatışmada),

`d=min(1,||c+q||/||c'+q'||)*(c'+q')`.

P_PC kolunda local yalnız SCD, style bu projection'ı kullanır. P_SUM ordinary sum, P_NORM aynı-state magnitude kontrolüdür. Bu blok routing genellikle tek ortak scalar objective'in gradient descent'i değildir. Phase5'in `86fb155:gsd_composition.py` dosyası current ile byte eşit doğrulandı; bütün tarihsel dosyalar eşit diye genellenmez.

**[Inference]** Özgün yerel unsur; PCGrad fikrinin örnek başına LION input/style bloklarına, norm kontrolü ve matched karşılaştırmalarla aktarılmasıdır. Projection formülü yeni değildir; PixelAsParam'ın tam algoritması da gerçekleştirilmiş değildir. Negatif cosine kendi başına zararlı accuracy mekanizması göstermez.

### A3. 4D, dinamik baz ve fiziksel loss neyi değiştirdi?

**[Code]** `use_4d_gft` grafın komşuluk uzayını değil GFT sinyalini XYZ'den XYZ+feature'a değiştirir; bu 4D fiziksel nokta veya 4D kNN değildir. Dynamic yolda baz predicted XYZ'den detach/no-grad ile yenilenir; hem referans hem live sinyal aynı güncel bazda karşılaştırılır. Graph-Jacobian alınmadığı için full geometry-dependent objective gradient'i değil frozen-metric adımıdır.

**[Code]** Sonraki `decoded_spectral.py`, grafı tam prepared encoder input'unda kurar ve loss'u decoded fiziksel XYZ residual'ına uygular. Böylece latent/fiziksel metric farkını hedefler. H8 surrogate, forward decoded değerini korurken decoder türevini latent XYZ identity ile değiştirir; denoiser Jacobian'ı hâlâ kalır. Hard-M halfmix ise yalnız PM checkpoint'lerinde `.5*SCD+.5*alpha*GSD` uygular; yoğun eigensolve'u kaldırmaz.

**[Run]** Bu kolların katkısı teşhis/karşılaştırmadır. Phase6 full all15/üç-seed P_PC−SCD yalnız `+.036917pp`; Nov1 spectral-style farkı `-.01736pp`, CI `[−.39931,+.36458]pp` ve daha yüksek maliyet. Nov4 fiziksel pilotta control/live0/full/H8 doğru sayıları `313/315/313/312` /480; numerical attribution kapısı başarısız. Nov7 M100 halfmix−half0 `+.8333pp` /16-nesne pilotunda CI sıfırı içerir; güncel 448-nesnelik kısmi kayıtta M100−SCD `-.178571pp`'dir. Bunlar başarılı PM/cheap mekanizmasının parçaları olarak sunulmaz. [Spektral retrospektif](graph_spectral_retrospective_audit_20261101.md), [composition](gsd_scd_gradient_composition_review_20261004.md), [physical](decoded_physical_accuracy_20261104.md), [halfmix](gsd_lowband_halfmix_results_20261107.md).

## Ek B. Sparse, semantik, density ve feature deneyleri

### B1. Adaptive Guidance ile fixed sparse backward aynı algoritma değil

**[Paper]** [Adaptive Guidance: Training-free Acceleration of Conditional Diffusion Models](https://arxiv.org/html/2312.12487v1#S5), §4.2/Eş.7 ve §5, conditional/unconditional CFG tahminlerinin cosine benzerliği eşik aşınca unconditional forward'ı kaldırır; timestep sayısını korur.

**[Code]** Ekim uygulaması cosine gate veya CFG branch'i kullanmaz. `chamfer_guidance_variants.py:35`, `G={1,4,7,...} union {T}` kurar; `tta_gsd_v2.py`'nin gerçek `original_0fd2581` snapshot'ı bütün prior forward'larını sürdürür, seçilmeyen adımlarda SCD/backward'ı atlar. T10/T30'da 4/11 backward kalır. Kasım `lion_guidance_pilot.py:114` kolu yalnız modulo-3 kullanır, son adımı zorla dahil etmez: N8 için 1/4/7. Bu iki sparse schedule eşit değildir.

**[Run/Inference]** Ekim dört-head ortalamalarında MN `−1.3571pp`, Scan `−.9683pp`; Gaussian+Background zamanında yaklaşık `%21` tasarruf, accuracy kaybıyla birliktedir. Özgün adaptive kriter değil; guidance redundancy fikrinin farklı hesap dalına aktarımıdır.

### B2. MAMVI'den fixed-X symmetric KL'ye

**[Paper]** [MAMVI: 3D Test-Time Adaptation via Masked Multi-View Point Clouds](https://arxiv.org/html/2606.12939v1#S3), §3.1–3.3/Eş.1–7: maskeli point-patch görünümleri oluşturur; detached prediction consensus, view entropy ve yönlü KL ile normalization parametrelerini confidence'a bağlı learning rate kullanarak adapte eder.

**[Code]** Bizim Ekim source snapshot'ı, bozuk X'in frozen head tahminini bir kez alıp `q=stopgrad softmax(F(X))` yapar. Seçilmiş decoded tahminde `p=softmax(F(D(hhat0,style)))` hesaplanır; `semantic_trajectory.py:33` hedefi `.5*(KL(p||q)+KL(q||p))` olarak sınıf ve nesneler üzerinde toplar. Live decoder girdisi clone edilir, detach edilmez; input-gradient denoiser üzerinden local/style'a ulaşır. q ek PM oyu değildir. Maskeli çoklu-view, normalization update ve adaptive LR taşınmaz.

**[Run/Inference]** MN `+.1627pp`, Scan `−.0516pp`; her iki CI sıfırı içerir ve süre artar. Bu symmetric divergence yeni değildir; frozen-model koşulunda latent guidance'a aktarım somuttur, MAMVI reprodüksiyonu veya doğrulanmış genel kazanım değildir.

### B3. GDA ve MEMO'dan nesne başına marginal entropy'ye

**[Paper]** [GDA: Generalized Diffusion for Robust Test-time Adaptation](https://arxiv.org/html/2404.00095v2#S3), §3/Eş.6–8/Alg.1, bir örneğin augmentasyon tahminlerini ortalar ve marginal entropy'yi diffusion guidance'a ekler; style/content amaçları da vardır. [MEMO](https://arxiv.org/html/2110.09506v3#S3.SS1), §3.1/Eş.1–3/Alg.1, aynı örneğin augmentasyonlarından `H(mean p)` hesaplayıp model parametrelerini adapte eder.

**[Code]** `semantic_trajectory.py:46` her örnek için decoded p ile fixed-X q'nun aritmetik ortalamasını kullanır:

`L_H=-sum_b sum_c [(p_bc+q_bc)/2] log[(p_bc+q_bc)/2]`.

Bu, `H(mean_b p_b)` batch sınıf dağılımı değildir; `mean(H(p),H(q))` de değildir. `H((p+q)/2)=.5H(p)+.5H(q)+JS(p,q)` olduğundan confidence ve agreement birlikte etkilenir. Modeller frozen, updated değişkenler local/style'dır. GDA'nın 16 AugMix, style/content ve seçim mekanizmaları; MEMO'nun weight update'i aktarılmaz.

**[Run/Inference]** MN dört-head ortalaması `+.3254pp`, Scan `+.0833pp`; yalnız MN aggregate CI sıfır üstündedir. Ek `%7–8` süre vardır. Bu küçük olumlu MN sonucu korunur, yeni genel entropy prensibi veya her modelde üstünlük denmez.

### B4. CloudFixer density ağırlığını başka loss'a taşıdık

**[Paper]** CloudFixer §4.2, yerel yoğunluktan türettiği inverse-kNN ağırlığını fiziksel displacement regularizer'ında `sum_j w_j||Delta_j||²` kullanır; izole noktaların daha rahat hareket etmesini hedefler.

**[Code]** Bizde `chamfer_guidance_variants.py:42–87` bozuk encoded latent XYZ'de nonself k10 squared-distance inverse'ini, epsilon `1e-6` ve nesne başına mean1 ile hesaplar. Ağırlık detach edilir; displacement regularizer'ı yerine iki yönlü SCD residual'ları reference noktasının ağırlığıyla çarpılır. Önce distances ile %95 trim, sonra weight uygulanır; weighted-distance ile trim sırası değiştirilmez. Dense latent noktalar yüksek ağırlık alır; bu inverse-density balancing değildir.

**[Run/Inference]** MN `−.7262pp`, Scan `−.5040pp`; negatif sonuç. Aynı neighborhood istatistiğini farklı amaç/değişkende kullanmak bir uyarlamadır; etkisinin CloudFixer ile aynı olduğu veya density'nin correctness confidence olduğu varsayılamaz.

### B5. RPM-Net'ten LION scalar-feature eşleştirmesine

**[Paper]** [RPM-Net: Robust Point Matching using Learned Features](https://arxiv.org/html/2003.13479v1#S5), §5.1–5.3/Eş.5–9, learned geometrik descriptor'lar, soft correspondences, Sinkhorn ve weighted rigid-transform/SVD çözümünü eğitimle birleştirir.

**[Code]** `lion_guidance_pilot.py:22–40,100–109,138–152` mevcut LION scalar feature'ını kullanır: XYZ-8NN shortlist içinde `d_xyz²/s_xyz² + beta(df/s_f)²` ile hard eşleşme seçer. Assignment detach edilir; canlı loss yalnız XYZ residual'ıdır, feature değerine fidelity cezası eklenmez. Ortak reference scale ve MAD/RMS fallback vardır. Learned descriptor, Sinkhorn ve rigid registration yoktur. Shuffled-feature, geometric-beta0 ve direct-4D kontrolleri ayrıdır.

**[Run/Inference]** Feature eşleşmeleri değiştirir, ancak Density+Occlusion/16 nesne pilotunda bütün kollar 22/32 doğrudur; native repeat kapısı 11/16 geçer. Fikrin özellik kanalını korumak yerine assignment'a sokması somut bir LION uyarlamasıdır; bu ayarda doğrulanmış accuracy katkısı yoktur.

### B6. Readout Guidance ile causal-PM feature repair ilişkisi

**[Paper]** [Readout Guidance: Learning Control from Diffusion Features](https://arxiv.org/html/2312.02150v1#S3.SS2), §3.2 ve §4.1–4.3, diffusion feature'larından pose/depth/correspondence çıkaran trained head'ler ve reference-distance gradient'iyle kontrol üretir. Causal PM teacher veya mevcut LION scalar kanalı bunun algoritması değildir; yakın feature/control öncülüdür.

**[Code]** `lion_guidance_pilot.py:43–65,178–227` hazır trajectory adaylarını değiştirir; sampler'a geri besleme yapmaz. İlk aday sabit kalır. Sonrakiler yalnız geçmiş kabul edilmiş tahminlerin detached PM'sini teacher alır; CE gradyanının yalnız feature kanalına RMS-relative `.01` adımı uygulanır. XYZ latent ve z0 sabittir. Tekrar decode sonrası fiziksel relative-RMS hareket `.001`'i aşarsa orijinal adaya fallback olur. Yeni trained readout/head yoktur.

**[Run/Inference]** 191/192 feature düzeltmesi kabul edilir ve 155'inde teacher CE azalır; bütün kolların son doğrusu 48/64 kalır. Ek readout maliyeti yaklaşık `%34.84`'tür. Somut bir teacher/routing/trust-limit sentezi vardır, ancak mevcut pilotta accuracy katkısı yoktur. [Pilot sonuçları](lion_guidance_pilot_results_20261107.md).

## Ek C. Okunan birincil kaynaklar ve erişim sınırları

Aşağıdaki locator'lar yöntem okumasına aittir; atıf listesine giren her makalenin bütün ekleri okunduğu veya bütün sonuçlarının yeniden üretildiği iddia edilmez.

| Kaynak | Doğrudan okunan yöntem | Rol |
|---|---|---|
| LION, arXiv2210.06978v1 / NeurIPS2022 | §3/Eş.5–7, Ek C.1–C.2; primary HTML | Mimari/altyapı |
| 3DD-TTA, arXiv2411.14495v1 / WACV2025 | §3.1–3.3/Eş.8–12/Alg.1; primary HTML, venue fetch403 | Temel comparator |
| TEDM, PMLR225/rosnati23a | §4/Eş.8–9, PDF s.5/basılı516 | Sonradan belirlenen PM öncülü |
| Time Is a Feature, arXiv2508.09138v3 | §4.1/Eş.4; primary HTML | Sonradan belirlenen tek-yörünge voting öncülü |
| Representation Dynamics, arXiv2502.05743v3 / NeurIPS2025 | Ek B.4 primary HTML; venue sayfa görüntüsü timeout | Sonradan belirlenen logit-ensemble öncülü |
| Purge-Gate, arXiv2509.09785v1 | §3.3.1/Eş.20–21; primary HTML | Minimum-entropy aday seçimi |
| Guess & Guide, arXiv2603.07860v1 | §3/Alg.1, Ek B.1–B.2 | Cheap correction tasarım dayanağı |
| Spin, aynı arXiv v3 | §3, Ek A.1–A.2; primary HTML | Aynı çalışmanın sonraki başlık/algoritma açıklaması |
| CloudFixer, ECCV2024 | §4.1–4.3/Eş.1, PDF s.5–7 | Density esini ve backward-free 3B TTA öncülü |
| Piecewise Guidance, arXiv2507.18654v1 | §III | Karma guidance için yakın ilke; aynı schedule değil |
| GSDTTA, arXiv2507.18225v1 / ICCV2025 | §3.1–3.2/Eş.7–18; venue PDF s.4–5 görsel kontrol | Belgelenmiş spektral esin |
| PixelAsParam, PMLR202/dinh23a | §4.2/Eş.12, §5/Eş.16/Alg.2, §6.4 | Belgelenmiş projection esini |
| PCGrad, arXiv2001.06782v4 | §2.2–2.3/Alg.1 | Projection cebrinin daha eski öncülü |
| Adaptive Guidance, arXiv2312.12487v1 | §4.2/Eş.7 ve §5 | Belgelenmiş sparse-guidance esini |
| MAMVI, arXiv2606.12939v1 | §3.1–3.3/Eş.1–7 | Belgelenmiş consistency esini |
| GDA, arXiv2404.00095v2 | §3/Eş.6–8/Alg.1 | Belgelenmiş semantic entropy esini |
| MEMO, arXiv2110.09506v3 | §3.1/Eş.1–3/Alg.1 | Marginal entropy öncülü; full algoritma taşınmadı |
| RPM-Net, arXiv2003.13479v1 | §5.1–5.3/Eş.5–9 | Feature-correspondence öncülü |
| Readout Guidance, arXiv2312.02150v1 | §3.2 ve §4.1–4.3 | Feature/control öncülü; trained head taşınmadı |

SeaLion, DragonDiffusion, RACER, GPARA ve daha geniş aday literatürü önceki kayıtların bağlamıdır; buradaki başarılı PM/cheap yönteminin uygulanmış bütün ataları diye sayılmaz. Bu raporun esas hükmü yukarıdaki çekirdek eşlemelere dayanır. Geniş aday listesi için [7 Kasım tasarım incelemesi](guidance_lion_candidates_20261107.md) ve [önceki uygulama/literatür denetimi](chamfer_guidance_implementation_literature_audit_20261031.md) korunur.
