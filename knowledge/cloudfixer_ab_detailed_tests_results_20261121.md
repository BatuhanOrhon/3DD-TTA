# CloudFixer A/B: ayrıntılı test envanteri ve sonuçlar

**Derleme: 21 Kasım 2026. GPU deneyleri: 18–20 Kasım 2026.**

[Run/Code] Bu belge tamamlanmış ve kabul edilmiş arşivlerin ortak Türkçe kaydıdır. Yeni model inference'ı, GPU koşusu, hiperparametre araması veya test etiketiyle yeniden seçim yapılmadı. Sayılar mevcut CSV/JSON'lardan toplandı; dört A/B ham ve analiz ZIP'inin boyut/SHA kimliği yeniden kontrol edildi. Tam sayısal derleme: [verified_report_tables.json](evidence/cloudfixer_ab_record_20261121/verified_report_tables.json).

**Ana sonuç:** A'nın tam Point2Vec karşılaştırmasında orijinal plain CloudFixer önde kaldı. B'nin Point-MAE doğrulamasında ise kilitli geç PM pencereleri, aynı proximal yörüngenin kendi terminalini üç ayarda da geçti. Bunlar farklı referanslar ve farklı deneylerdir; birlikte çelişki oluşturmazlar.

## 1. Ortak yöntem ve terimler

[Code] Her iki çalışmada donmuş **resmî CloudFixer XYZ diffusion** kullanıldı; bu bilimsel deneylerde **LION kullanılmadı**. Classifier ağırlıkları güncellenmedi. Bizim kollarımız clean XYZ tahminini gradient-free proximal SCD ile düzeltiyor. Resmî CF kolu, yazarın rotation/delta optimizasyonunu kullanıyor.

| Terim | Bu deneydeki anlamı |
|---|---|
| N | Ters diffusion güncelleme sayısı; classifier çağrı sayısı değildir. |
| K | 500 seviyeli resmî noise takvimindeki başlangıç native indeksi. |
| τ | Proximal düzeltme gücü; PM softmax sıcaklığı değildir. |
| retained=.95 | Proximal SCD'nin iki yönlü seçici eşleşmelerde tuttuğu oran. |
| Pre-clean probe | Denoiser'ın bir adımda verdiği temiz XYZ tahmini; o adımın proximal düzeltmesinden önce okunur. |
| Post-clean probe | Aynı temiz tahminin proximal düzeltmeden sonraki hali. |
| Pre-extra-t0 state | Son nonzero adımın transport işleminden sonraki, ek t0 çağrısından önceki state. Kalıntı gürültü içerir. |
| Gerçek terminal | Ek native t=0 denoiser çağrısından çıkan temiz XYZ; t0'da ayrıca proximal düzeltme yapılmaz. |
| PM | Seçilen adayların classifier softmax olasılık vektörlerinin eşit ortalaması; logit ortalaması veya tek-adım accuracy değildir. |

[Code] Her nonzero adımda akış şöyledir: `state → denoiser clean tahmini → pre gözlemi → proximal SCD → post gözlemi → sonraki state'e transport`. N güncelleme bittikten sonra ek `phi(state, t=0)` çağrısı gerçek terminali üretir. Bu tanım hem A'nın `9afa8436…` hem B'nin `eec1e8b2…` kaynaklarındaki `cloudfixer_diffusion.py` ile byte düzeyinde eşleşir.

**N32 gerçek terminalin açık tanımı:** K160 grid'i `160,155,…,5,0` olur. İlk 32 nonzero seviyede güncelleme, ardından ek t0 denoiser çağrısı vardır: **33 denoiser çağrısı**, nihai bulutta **tek classifier okuması**, **PM kapalı**. N8/K80 geç PM ise **9 denoiser çağrısı** ve **[20,10,0] üzerinde 3 classifier okuması** kullanır. Bu iki A kolunda N/K/τ/readout birlikte değişir.

PM için her adayda `p_t=softmax(logits_t)` hesaplanır, seçilen pencerede `ortalama(p_t)` alınır ve en büyük ortalama olasılığın sınıfı seçilir. A ve B'nin kayıtlı PM hesaplarında softmax float64'tür. Tahmin kuralı gerçek etikete bakmaz; etiketler doğruluk ve teşhis hesabında kullanılır.

## 2. Hangi testleri yaptık?

| Grup | Test/kontrol | Kapsam ve kabul edilen çıktı |
|---|---|---|
| A0 | Ortam/native/veri kontrolü | Yeni VM'lerde kanonik 10 hücre, ayrı cell1 restart/reconnect, CUDA/FPS/KNN/Chamfer/EMD/PVCNN, veri ve checkpoint kimlikleri. Yalnız bilinen cell2 fastrlock istisnası ertelendi; gerçek Python doğrulandı. |
| A1 | Point2Vec entegrasyonu | Resmî mimari, strict checkpoint yükleme, preprocessing/class order, bağımlılıklar; 180 state anahtarı ve [1,40] classifier çıktısı doğrulandı. Eski Point-MAE yolu korundu. |
| A2 | Teknik smoke ve recovery goldens | Background/lidar üzerinde 2'şer nesne; doğrudan resmî classifier, sparse/dense readout, model/input/RNG ve tarihsel XYZ replay kontrolleri. Bunlar tam doğruluk testi değildir. |
| A3 | Tam dört-kol karşılaştırma | 2,468 nesne × 15 corruption × s5 × seed2, B1; her kol 37,020 gözlem. Source, N8 geç PM, N32 terminal ve plain CF. |
| A4 | Kesinti sonrası devam | Her 32 nesnede atomik Drive payload/receipt, SHA ve RNG. 4 VM epizodu tek kampanyayı tamamladı; eski 35,000 gözlem yeniden skorlanmadı. |
| A5 | Sonuç kabulü | 1,170 blok, 60 corruption-sonuç satırı (4×15) ve 4 özet satırı; ham logits/labels/probability/entropy/logSNR/RNG/timing kontrolleri, bağımsız yeniden sayım. |
| B0 | Doğrulama nesneleri | 6 tarihsel arşivdeki 136 farklı ID ile doğrulama kümesinin 256 ID'si arasında sıfır kesişim. |
| B1 | Gözlemci teknik pilotu | Uniform, eski 128 nesne, seed0/B32; 4 batch × dense/sparse/terminal-only = 12 payload. Eski canonical/dense logits farkı 0. |
| B2 | Kilitli hipotez doğrulaması | 256 nesne × 15 corruption × 3 seed × 3 schedule; 1,080 batch birimi, 34,560 yörünge. |
| B3 | Erken/geç oylar | Kanonik, tüm-adım ve kilitli geç eşit PM; repair/harm ve sabit rakibe karşı probe marjları. |
| B4 | Entropy | Pooled ve aynı-yörünge AUC; 3 pencerede lineer ve 6 β'lı üstel ağırlık ailesi. |
| B5 | Birikimli düzeltme | N8/.1, N12/.1, N12/(.8÷12); Nτ, RMS, marj ve terminal geçişleri. |
| B6 | Gözlem yeri ve ek t0 | Pre/post PM, son post-clean, pre-extra state ve gerçek terminal eşleştirmeleri. |
| B7 | Offline readout ve arşiv kabulü | 31 kayıtlı readout adı; 279 özet, 4,185 corruption satırı, 479 paired kontrast; ham/analiz arşivi ve ikinci Drive mount geri okuması. |

## 3. A — tam ModelNet40-C / Point2Vec karşılaştırması

### 3.1 Protokol

[Run] Tam 2,468 test nesnesi, aynı 15 corruption, severity 5, yazar seed'i **2**, batch **1**. Aynı donmuş Point2Vec ve aynı resmî diffusion checkpoint'i kullanıldı. İki proximal kolda retained=.95 sabit; N8/K80/τ=.1 geç PM `[20,10,0]`, N32/K160/τ=.03 gerçek terminal. Kollar tam test etiketleri görülerek yeniden ayarlanmadı.

Source, yazar akışındaki adaptation öncesi classifier kontrolüdür. Plain CF ilk classifier çıktısıdır. Resmî ikinci vote1 forward'ı RNG akışını korumak için çalışır ve ayrıca teşhis olarak kaydedilir. Bunları bağımsız paper-reproduction source ölçümü veya aynı readout olarak birleştirmiyoruz.

### 3.2 Ana doğruluk, süre ve bellek

| Kol | Doğru / 37,020 | Doğruluk (%) | Kayıtlı süre (s) | Tepe ayrılmış bellek (MiB) |
| --- | --- | --- | --- | --- |
| Source kontrolü | 22947 | 61.985413 | 713.338 | 436.935 |
| N8/K80/τ=.1 geç PM | 27748 | 74.954079 | 9071.108 | 586.330 |
| N32/K160/τ=.03 gerçek terminal | 27111 | 73.233387 | 25822.797 | 586.283 |
| Orijinal CF — plain | 28596 | 77.244733 | 29013.512 | 586.407 |


[Run] Diagnostic vote1 **28,545/37,020 = 77.106969%**; ana plain CF sayısı **28,596**'dır. N8 PM'nin kayıtlı yöntem süresi CF'ninkinin yaklaşık **1/3.20**'sidir. Zamanlar CUDA synchronize ile ölçülür; diagnostic vote forward'ı, gate, I/O ve setup hariçtir. Bunlar bütün dağıtımın uçtan uca hızlanma veya fatura ölçümü değildir. `peak_gpu_memory_mb` alanının gerçek birimi MiB'dir.

### 3.3 Eşleştirilmiş doğruluk farkları

| CF referansına karşı | Δ puan | Nominal %95 aralık | Düzeltti / bozdu |
| --- | --- | --- | --- |
| N8/K80/τ=.1 geç PM | -2.290654 | [-2.663425, -1.925918] | 1240 / 2088 |
| N32/K160/τ=.03 gerçek terminal | -4.011345 | [-4.513844, -3.506213] | 1507 / 2992 |


[Run] 10,000 PCG64 bootstrap draw, seed20261118; 2,468 orijinal nesne tüm 15 corruption'ıyla birlikte yeniden örneklenir. Aralıklar A'nın sabit seed'i ve native gerçekleşmesine koşulludur. Seed/native koşular-arası varyansı vermez.

[Inference] **İki sabit proximal politika bu koşulda plain CF'ı geçmedi.** N8 PM, N32 terminalden 637 olay / **+1.7206915 puan** öndedir; farklı N/K/τ nedeniyle bu PM'nin tek başına nedensel etkisi sayılamaz. N8 PM'nin kendi N8 terminaline karşı referansı A'nın bu ana tablosunda ayrı bir kol değildir; aynı-yörünge readout hipotezi B'de Point-MAE ile test edilir.

### 3.4 Corruption bazında tüm sonuçlar

| Corruption | Source % | N8 PM % | N32 terminal % | CF plain % | PM−CF puan |
| --- | --- | --- | --- | --- | --- |
| background | 36.831442 | 65.316045 | 76.458671 | 76.094003 | -10.777958 |
| cutout | 83.995138 | 85.251216 | 82.333874 | 81.442464 | +3.808752 |
| density | 86.547812 | 79.659643 | 75.000000 | 79.335494 | +0.324149 |
| density_inc | 90.113452 | 86.912480 | 83.184765 | 87.763371 | -0.850891 |
| distortion | 75.810373 | 76.337115 | 72.325770 | 78.687196 | -2.350081 |
| distortion_rbf | 74.432739 | 79.376013 | 77.876823 | 80.753647 | -1.377634 |
| distortion_rbf_inv | 77.025932 | 81.969206 | 80.064830 | 83.427877 | -1.458671 |
| gaussian | 49.554295 | 90.842788 | 87.641815 | 91.572123 | -0.729335 |
| impulse | 65.599676 | 91.936791 | 89.262561 | 91.572123 | +0.364668 |
| lidar | 20.299838 | 40.842788 | 39.870340 | 38.371151 | +2.471637 |
| occlusion | 41.531605 | 41.369530 | 41.004862 | 41.166937 | +0.202593 |
| rotation | 46.920583 | 48.055105 | 44.935170 | 69.489465 | -21.434360 |
| shear | 79.416532 | 75.891410 | 72.730956 | 78.970827 | -3.079417 |
| uniform | 63.614263 | 90.437601 | 87.803890 | 90.964344 | -0.526742 |
| upsampling | 38.087520 | 90.113452 | 88.006483 | 89.059968 | +1.053485 |


[Run] PM CF'ı 6/15 corruption'da, N32 terminal 3/15 corruption'da geçer. PM'nin toplam 848 doğru-tahmin açığının 529'u rotation, 266'sı background kaynaklıdır: **795/848 = 93.75%**. Bu tablo bunları ana ortalamadan çıkarma veya test etiketleriyle yeni ayar seçme gerekçesi değildir. [Inference] Açığın bu iki bozulmada yoğunlaşması, geometrik neden veya rotation optimizasyonunun nedensel katkısını tek başına kanıtlamaz.

### 3.5 Smoke, yeniden başlatma ve bütünlük

[Run] Her replacement VM'de bilimsel kaynak/asset/environment imzaları ve fresh-empty-store goldens kontrol edildi. Source ve iki proximal kolun deterministik çıktıları tam eşleşti; geçmiş adapted XYZ'nin classifier replay'i tam eşleşti. Native KNN backward sayısal olarak deterministik değildir: r12 smoke'da 4 tahminin 1'i değişmişti; r13 smoke'da 0/4 argmax değişimi olsa da plain logits farkları background'da .433021486, lidar'da .419406652 idi. Bu, tam koşular-arası performans varyansını ölçmez.

| Epizot | Yeni blok | Yeni gözlem |
|---|---:|---:|
| r3 | 847 | 26,824 |
| r9 | 182 | 5,740 |
| r12 | 77 | 2,436 |
| r13 | 64 | 2,020 |
| Toplam | 1,170 | 37,020 |

[Run] Bunlar dört bağımsız seed deneyi değil, tek sabit kampanyanın devam epizotlarıdır. Son arşivde önceki **2,212 ham dosya / 35,000 gözlem** byte düzeyinde değişmeden korunur. Primary analyzer ve ikinci bağımsız ham-logit sayımı bütün doğru sayıları doğruladı.

## 4. B — aynı-yörünge PM/terminal hipotezleri

### 4.1 Doğrulama kapsamı ve bağımsızlık

[Run] Donmuş **Point-MAE**, aynı resmî XYZ diffusion; **256 ayrılmış orijinal nesne**, 15 corruption, s5, seed **0/1/2**, B32, K80, retained=.95. Üç schedule: N8/.1, N12/.1 ve N12/τ=.8÷12=.0666666667. Her schedule/seed 3,840; üç seed havuzunda her schedule **11,520** gözlem vardır. Toplam **34,560 schedule-özgü yörünge**, bağımsız 34,560 nesne değildir.

[Run] Geç pencereler **N8 `[20,10,0]`**, **N12 `[27,0]`** olarak önceki geliştirme logits'lerinden seçilip doğrulamadan önce kilitlendi. Altı tarihsel arşivdeki 136 ID ile yeni 256 ID'nin kesişimi sıfırdır. Bağımsızlık bu hipotezler için nesne ayrımına aittir; benchmark daha önce incelenmiş olduğundan bütün benchmark'ın hiç görülmediği iddia edilmez.

[Code/Run] 31 kayıtlı readout: 1 terminal + 3 eşit PM + 21 entropy-ağırlıklı PM + 3 post-PM + 3 tek-gözlem teşhisi. β=0 kolları eşit PM ile tam aynı sonucu verir; dolayısıyla 31 bağımsız algoritma değildir. İsimlerin tamamı ve 93 schedule/readout havuz sonucu JSON kanıtında yer alır. Aynı kayıtlı yörüngeler offline yeniden okunur; 31 yeni GPU koşusu yapılmaz.

### 4.2 Teknik pilot sonucu

[Run] Uniform/seed0 üzerinde aynı 128 eski geliştirme nesnesi dense, sparse ve terminal-only modlarında işlendi. Her modda gerçek terminal **112/128**. Eski canonical-four ve dense-nine logits ile maksimum mutlak fark **0**; input/model/RNG/transport/endpoint kontrolleri geçti. Süreler: dense **18.542214 s**, sparse **10.970161 s**, terminal-only **9.977568 s**. Bunlar küçük teknik pilotun gözlem maliyetidir; tam doğruluk kıyası değildir.

### 4.3 Erken, kanonik, tüm-adım ve geç PM

[Run] Kanonik pencere N8'de `[70,50,30,0]`, N12'de `[67,47,27,0]`; tüm-adım penceresi 9 veya 13 pre-clean/terminal adayını kullanır. PM olasılık ortalamasıdır. Tablodaki her satır aynı yörüngenin kendi terminaliyle eşleştirilmiştir. Her hücrenin paydası 11,520'dir.

| Ayar | Pencere | Terminal doğru (%) | PM doğru (%) | Düzeltti / bozdu | Δ puan | Nominal %95 aralık |
| --- | --- | --- | --- | --- | --- | --- |
| N8 / τ=.1 | Geç | 8295 (72.005208) | 8399 (72.907986) | 213 / 109 | +0.902778 | [+0.564236, +1.241319] |
| N8 / τ=.1 | Kanonik | 8295 (72.005208) | 8233 (71.467014) | 241 / 303 | -0.538194 | [-1.093750, +0.008681] |
| N8 / τ=.1 | Tüm adımlar | 8295 (72.005208) | 8229 (71.432292) | 236 / 302 | -0.572917 | [-1.137153, -0.026042] |
| N12 / τ=.1 | Geç | 8112 (70.416667) | 8311 (72.144097) | 315 / 116 | +1.727431 | [+1.276042, +2.170139] |
| N12 / τ=.1 | Kanonik | 8112 (70.416667) | 8265 (71.744792) | 389 / 236 | +1.328125 | [+0.729167, +1.892361] |
| N12 / τ=.1 | Tüm adımlar | 8112 (70.416667) | 8242 (71.545139) | 398 / 268 | +1.128472 | [+0.503472, +1.727431] |
| N12 / τ=.8÷12 | Geç | 8286 (71.927083) | 8362 (72.586806) | 212 / 136 | +0.659722 | [+0.295139, +1.032986] |
| N12 / τ=.8÷12 | Kanonik | 8286 (71.927083) | 8226 (71.406250) | 225 / 285 | -0.520833 | [-1.032986, -0.017361] |
| N12 / τ=.8÷12 | Tüm adımlar | 8286 (71.927083) | 8237 (71.501736) | 251 / 300 | -0.425347 | [-0.920139, +0.069444] |


[Run] Geç pencere bütün üç ayarda terminalin üstündedir; üç pooled aralık da sıfırın üstünde. Kanonik ve tüm-adım PM'nin işareti schedule'a göre değişir. Geç−kanonik farkı N8'de **+1.440972 puan**, N12/.1'de **+0.399306**, eşlenmiş N12'de **+1.180556**; N12/.1 geç−kanonik aralığı sıfırı içerir. N8 geç pencere aynı zamanda t20/t10'u eklediği için tüm fark yalnız erken adayları çıkarmaya bağlanamaz.

### 4.4 Seed bazında geç PM doğrulaması

| Ayar | Seed | Terminal / 3,840 | Geç PM / 3,840 | Δ puan | Nominal %95 aralık |
| --- | --- | --- | --- | --- | --- |
| N8 / τ=.1 | 0 | 2770 | 2790 | +0.520833 | [-0.052083, +1.067708] |
| N8 / τ=.1 | 1 | 2768 | 2816 | +1.250000 | [+0.755208, +1.770833] |
| N8 / τ=.1 | 2 | 2757 | 2793 | +0.937500 | [+0.494792, +1.406250] |
| N12 / τ=.1 | 0 | 2710 | 2772 | +1.614583 | [+0.989583, +2.291667] |
| N12 / τ=.1 | 1 | 2699 | 2769 | +1.822917 | [+1.145833, +2.500000] |
| N12 / τ=.1 | 2 | 2703 | 2770 | +1.744792 | [+1.119792, +2.369792] |
| N12 / τ=.8÷12 | 0 | 2773 | 2782 | +0.234375 | [-0.390625, +0.859375] |
| N12 / τ=.8÷12 | 1 | 2760 | 2791 | +0.807292 | [+0.260417, +1.354167] |
| N12 / τ=.8÷12 | 2 | 2753 | 2789 | +0.937500 | [+0.390625, +1.510417] |


[Run] Dokuz seed nokta farkının hepsi pozitiftir; yedisinin nominal aralığı sıfırı dışlar. N8 seed0 ve eşlenmiş N12 seed0 aralıkları sıfırı içerir. B aralıkları 10,000 paired draw ile 256 orijinal nesneyi bütün corruption ve seed'leriyle birlikte örnekler; çoklu karşılaştırma düzeltmesi yapılmamıştır.

### 4.5 Corruption bazında terminal / geç PM

[Run] Her hücre 256 nesne × 3 seed = 768 gözlemden hesaplandı.

| Corruption | N8 terminal % | N8 geç PM % | N12/.1 terminal % | N12/.1 geç PM % | N12 eşlenmiş terminal % | N12 eşlenmiş geç PM % |
| --- | --- | --- | --- | --- | --- | --- |
| background | 35.156250 | 36.718750 | 35.156250 | 38.281250 | 35.026042 | 36.979167 |
| cutout | 83.333333 | 83.723958 | 83.723958 | 83.333333 | 83.593750 | 82.942708 |
| density | 83.593750 | 84.375000 | 83.723958 | 84.244792 | 82.942708 | 83.333333 |
| density_inc | 88.932292 | 89.583333 | 89.583333 | 89.192708 | 89.192708 | 89.322917 |
| distortion | 73.046875 | 73.697917 | 69.531250 | 72.265625 | 73.437500 | 73.307292 |
| distortion_rbf | 78.906250 | 80.729167 | 74.088542 | 78.125000 | 78.125000 | 80.729167 |
| distortion_rbf_inv | 80.598958 | 81.380208 | 77.604167 | 79.557292 | 79.947917 | 80.989583 |
| gaussian | 91.536458 | 92.187500 | 88.151042 | 91.666667 | 91.406250 | 91.927083 |
| impulse | 92.968750 | 93.098958 | 91.796875 | 92.187500 | 92.708333 | 93.229167 |
| lidar | 40.885417 | 42.317708 | 37.369792 | 39.713542 | 41.276042 | 42.057292 |
| occlusion | 39.062500 | 39.192708 | 38.802083 | 39.192708 | 39.583333 | 40.364583 |
| rotation | 33.333333 | 35.937500 | 32.161458 | 34.895833 | 32.942708 | 34.375000 |
| shear | 74.348958 | 74.869792 | 73.697917 | 74.609375 | 73.828125 | 74.479167 |
| uniform | 92.317708 | 92.968750 | 89.062500 | 92.057292 | 92.447917 | 92.447917 |
| upsampling | 92.057292 | 92.838542 | 91.796875 | 92.838542 | 92.447917 | 92.317708 |


[Run] N8 geç−terminal 15/15 corruption'da pozitiftir. N12/.1'de 13/15 pozitif, cutout ve density_inc'de üçer olay negatiftir. Eşlenmiş N12'de 11/15 pozitif, uniform eşit; cutout −5, distortion −1, upsampling −1 olaydır. Hiçbir corruption'a özel policy seçilmedi.

### 4.6 Erken yanlış oylar geç doğru kararı bastırıyor mu?

[Run] Kanonik PM'deki harm grubunda yanlış rakip PM'nin seçtiği sınıftır; repair grubunda terminalin seçtiği yanlış sınıftır. Her probe için aynı rakibe karşı `p(doğru)−p(rakip)` ölçülür. Aşağıdaki marjlar grup ortalamalarıdır; gruplar sonuca göre tanımlandığından bu bir fiziksel nedensellik deneyi değildir.

| Ayar | Grup | n | Ortalama p(doğru)−p(sabit rakip), erken → t0 | Doğru sınıf top-1 sayıları |
| --- | --- | --- | --- | --- |
| N8 / τ=.1 | harm | 303 | -0.530 / -0.482 / -0.135 / +0.352 | 17 / 17 / 94 / 303 |
| N8 / τ=.1 | repair | 241 | +0.295 / +0.460 / +0.369 / -0.225 | 147 / 202 / 188 / 0 |
| N12 / τ=.1 | harm | 236 | -0.524 / -0.483 / -0.071 / +0.330 | 8 / 12 / 81 / 236 |
| N12 / τ=.1 | repair | 389 | +0.390 / +0.534 / +0.465 / -0.290 | 276 / 335 / 318 / 0 |
| N12 / τ=.8÷12 | harm | 285 | -0.538 / -0.484 / -0.056 / +0.352 | 12 / 20 / 112 / 285 |
| N12 / τ=.8÷12 | repair | 225 | +0.310 / +0.479 / +0.374 / -0.236 | 143 / 201 / 184 / 0 |


[Run] N8'de terminal doğruyken kanonik PM'nin bozduğu **303** olayın ilk probe'unda **252** kez nihai yanlış rakip, yalnız **17** kez gerçek sınıf top-1'dir. Erken iki negatif marj, terminalin pozitif marjını olasılık ortalamasında bastırabilir. Buna karşılık erken doğru oylar da terminal hatalarını onarır: N8'de 241 repair, N12/.1'de 389 repair vardır. [Inference] Erken adaylar koşulsuz zararlı değildir; net denge schedule'a bağlıdır.

### 4.7 Entropy aynı nesnenin doğru adımını seçebiliyor mu?

| Ayar | Pooled AUC | Aynı yörüngede çift-ağırlıklı AUC | Yörünge-ağırlıklı AUC | Yanlış / tüm yüksek güvenli aday |
| --- | --- | --- | --- | --- |
| N8 / τ=.1 | 0.867319 | 0.646667 | 0.657621 | 5825 / 58903 |
| N12 / τ=.1 | 0.868328 | 0.640281 | 0.658420 | 8196 / 84677 |
| N12 / τ=.8÷12 | 0.867040 | 0.635492 | 0.653963 | 8399 / 85344 |


[Run] Skor negatif entropy'dir; düşük entropy'nin doğru adayı öne alması sınanır. Pooled AUC farklı nesne/adımları karıştırır; aynı-yörünge AUC aynı nesne/corruption/seed/schedule içindeki doğru-yanlış çiftlerini karşılaştırır, tie yarım sayılır. Yaklaşık .867–.868 pooled ilişki, aynı yörüngede .635–.647'ye düşer. Yörünge başına eşit ağırlıkta yaklaşık .654–.658'dir. `max p≥.9` adaylar arasında binlerce yanlış tahmin bulunur; yüksek güven doğru etiket garantisi değildir.

**Ağırlık testleri:** `w_linear∝1−H/log(40)` (eşit fallback) ve `w_exp∝exp(−βH)`; β = 0,.5,1,2,4,8. Aşağıdaki değerler aynı penceredeki eşit PM'ye göre **yüzde puan farkıdır**.

| Ayar | Pencere | Lineer Δ | β=0 | β=.5 | β=1 | β=2 | β=4 | β=8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N8 / τ=.1 | canonical | +0.008681 | +0.000000 | +0.000000 | -0.052083 | -0.112847 | -0.138889 | -0.182292 |
| N8 / τ=.1 | all | -0.017361 | +0.000000 | -0.043403 | -0.060764 | -0.234375 | -0.295139 | -0.486111 |
| N8 / τ=.1 | late | -0.052083 | +0.000000 | -0.043403 | -0.069444 | -0.086806 | -0.052083 | -0.095486 |
| N12 / τ=.1 | canonical | +0.026042 | +0.000000 | +0.008681 | +0.043403 | -0.164931 | -0.199653 | -0.338542 |
| N12 / τ=.1 | all | -0.078125 | +0.000000 | -0.095486 | -0.173611 | -0.364583 | -0.538194 | -0.711806 |
| N12 / τ=.1 | late | +0.034722 | +0.000000 | +0.052083 | +0.017361 | +0.017361 | -0.052083 | -0.069444 |
| N12 / τ=.8÷12 | canonical | +0.026042 | +0.000000 | +0.034722 | +0.043403 | -0.043403 | -0.173611 | -0.243056 |
| N12 / τ=.8÷12 | all | -0.121528 | +0.000000 | -0.138889 | -0.147569 | -0.260417 | -0.416667 | -0.607639 |
| N12 / τ=.8÷12 | late | -0.017361 | +0.000000 | -0.008681 | -0.026042 | -0.034722 | -0.034722 | -0.043403 |


[Run] Lineer ve β1, geç pencerede üç ayarda da tutarlı kazanç sağlamadı; bu nominal geç-pencere aralıkları sıfırı içerdi. β8, tüm-adım PM'yi her ayarda düşürdü. Aile bütünü raporlanır; doğrulama etiketlerine göre kazanan β seçilmez. [Inference] Entropy kısmi sıralama bilgisi taşıyor, fakat basit ağırlıklandırma burada güvenilir ek fayda göstermedi; her olası entropy seçicisinin imkânsız olduğu sonucu çıkmaz.

### 4.8 N artınca birikimli düzeltme terminali bozuyor mu?

| Ayar | Nominal Nτ | Düzeltme RMS toplamı | Terminal−girdi RMS | İlk geç → terminal logit marjı | Ek t0 hareket RMS |
| --- | --- | --- | --- | --- | --- |
| N8 / τ=.1 | 0.800 | 0.099893 | 0.351872 | 3.164 → 2.946 | 0.001994 |
| N12 / τ=.1 | 1.200 | 0.133604 | 0.352437 | 3.075 → 2.700 | 0.001973 |
| N12 / τ=.8÷12 | 0.800 | 0.103603 | 0.352092 | 3.131 → 2.920 | 0.001863 |


| Terminal karşılaştırması | Düzeltti / bozdu | Δ puan | Nominal %95 aralık |
| --- | --- | --- | --- |
| N12 / τ=.1 − N8 / τ=.1 | 231 / 414 | -1.588542 | [-2.152778, -1.032769] |
| N12 / τ=.8÷12 − N12 / τ=.1 | 410 / 236 | +1.510417 | [+0.954861, +2.074653] |
| N12 / τ=.8÷12 − N8 / τ=.1 | 180 / 189 | -0.078125 | [-0.416667, +0.243056] |


[Run] Sabit τ=.1 ile N8→N12, nominal Nτ'yı .8→1.2 ve düzeltme RMS toplamını .099893→.133604 yükseltti; terminal **−1.588542 puan** düştü. N12'de τ=.8÷12 yapılınca **174** net doğru tahmin geri geldi: **+1.510417 puan**. Eşlenmiş N12 terminali N8'den yalnız **−.078125 puan** farklı; aralık sıfırı içerir.

[Inference] Bu sınırlı kontrol terminal açığının birikimli düzeltme gücüne duyarlı olduğunu destekler. Aynı Nτ, aynı zaman ayrıklaştırması veya aynı yörünge anlamına gelmez. İlk geç doğru/terminal yanlış grubu ile tersi grubun kalan düzeltme RMS'leri birbirine çok yakındır: N8 .020849/.020877; N12/.1 .035668/.035222; eşlenmiş N12 .029027/.028957. Dolayısıyla tek nesnede bozulmayı yalnız kalan düzeltme büyüklüğüyle açıklayan genel overshoot kanıtı yoktur.

### 4.9 Pre/post gözlem ve ek t0 neyi değiştirdi?

| Ayar | Karşılaştırma | Düzeltti / bozdu | Δ puan | Nominal %95 aralık |
| --- | --- | --- | --- | --- |
| N8 / τ=.1 | Post geç PM − pre geç PM | 93 / 131 | -0.329861 | [-0.572917, -0.078125] |
| N8 / τ=.1 | Son post-clean − gerçek terminal | 139 / 113 | +0.225694 | [-0.060764, +0.512153] |
| N8 / τ=.1 | Gerçek terminal − ek t0 öncesi state | 90 / 57 | +0.286458 | [+0.078125, +0.494792] |
| N12 / τ=.1 | Post geç PM − pre geç PM | 98 / 124 | -0.225694 | [-0.512153, +0.060764] |
| N12 / τ=.1 | Son post-clean − gerçek terminal | 127 / 111 | +0.138889 | [-0.147569, +0.416667] |
| N12 / τ=.1 | Gerçek terminal − ek t0 öncesi state | 105 / 72 | +0.286458 | [+0.043403, +0.529514] |
| N12 / τ=.8÷12 | Post geç PM − pre geç PM | 86 / 74 | +0.104167 | [-0.104167, +0.312500] |
| N12 / τ=.8÷12 | Son post-clean − gerçek terminal | 110 / 106 | +0.034722 | [-0.234375, +0.303819] |
| N12 / τ=.8÷12 | Gerçek terminal − ek t0 öncesi state | 84 / 72 | +0.104167 | [-0.121528, +0.329861] |


[Run] Post PM, seçilen nonzero adımların düzeltme sonrası clean bulutlarını ve aynı gerçek terminali kullanır. Son post-clean bulut transport öncesidir; pre-extra state transport sonrasıdır; gerçek terminal ek t0 denoiser çağrısının çıktısıdır. Bu üç farklı nesne birbirinin yerine adlandırılmaz.

[Inference] Ek t0 çağrısı bu üç ayarda net pozitif fark verdi; eşlenmiş N12 aralığı sıfırı içerdi. Post gözleme geçmek N8 geç PM'yi nominal olarak azalttı; N12'lerde işaret/kararlılık ortak değildir. Evrensel “post daha iyi” veya “son t0 çağrısı bozuyor” sonucu desteklenmedi. Bu deneyler LION decoder'ına ait bir üstünlüğü test etmez.

### 4.10 Süre, bellek ve tekrar sayımı sınırı

[Run] B doğrulama fazı **8,414.28089 s (2 saat 20 dakika 14 s)** sürdü. Schedule başına üç seed'in tanısal süre toplamları N8 **1,872.13 s**, N12/.1 **2,619.04 s**, eşlenmiş N12 **2,618.91 s**; tepe ayrılmış bellek sırasıyla **9,287.39 / 9,293.40 / 9,293.40 MiB**. Bu pipeline pre/post classifier gözlemlerini ve bütün offline readout'lara ortak teşhis maliyetini içerir. 31 readout'un CSV sürelerini toplayarak 31 ayrı GPU koşusu maliyeti üretmek hatalıdır. A/B süre ve belleği farklı backbone/batch/observer kapsamları nedeniyle doğrudan verimlilik kıyası değildir.

## 5. A ile B birlikte ne söylüyor?

| Soru | Kanıtın desteklediği sonuç | Desteklenmeyen daha güçlü iddia |
|---|---|---|
| Seçilen paket Point2Vec'te CF'ı geçti mi? | A'da iki sabit proximal politika CF'ın altında kaldı. | Proximal/PM her backbone ve ayarda kötüdür. |
| Geç PM kendi terminalini geçebilir mi? | B'de kilitli geç pencereler üç pooled ayarda pozitiftir; 9 seed nokta farkı da pozitiftir. | Her pencere/her seed aralığı veya her corruption kesin pozitiftir. |
| Erken oylar bastırıyor mu? | Bazı canonical harm gruplarında erken yanlış sınıf lehine güçlü oylar var. | Erken adımlar her zaman zararlıdır veya geometry bozulmasının nedeni kanıtlandı. |
| Entropy güvenilir seçim mi? | Pooled ilişki güçlü, aynı-yörünge ilişki daha zayıf; sabit ağırlıklarda tutarlı ek kazanç yok. | Entropy kusursuz seçici ya da bütünüyle bilgisizdir. |
| Nτ mekanizması? | N12'de gücü azaltmak terminal açığının çoğunu kapattı. | Eşit Nτ eşit dinamik veya nesne bazında kanıtlanmış overshoot'tur. |
| Ek t0 / pre-post? | Okuma yeri önemlidir; ekstra t0 yardımcı olabilir; ortak bir pre/post üstünlüğü yok. | Son probe, son post-clean, noisy state ve gerçek terminal aynı şeydir. |

## 6. Arşivler, kaynaklar ve operasyonel kapanış

[Run] A ham/analiz ve B ham/analiz arşivleri yeniden yazılmadı. Kaynak/etiket/model/RNG/SHA/CRC kabul kayıtları ve özgün analizler korunur. A'da PM penceresi önceki geliştirme keşfinden kilitlendi; full test tüm geliştirme nesnelerini de içerir. B nesne ayrımı bu hipotezlere özgüdür. A tek seed; B üç sabit seed'e koşulludur; tekrar seed ve corruption'lar bağımsız nesne sayısını artırmaz.

- **A ham:** [20261120-184955_full.zip](../result/modelnet40_c/cloudfixer_point2vec/20261120-184955_full.zip); 136,139,832 byte; SHA256 `b2f0685cb1f0c3e0a053dff6e19f71503e9b793ba2ed340ff1eb0c6f6081e42d`.
- **A analiz:** [20261120-184955_full_recount.zip](../result/modelnet40_c/cloudfixer_point2vec/20261120-184955_full_recount.zip); 192,235 byte; SHA256 `8c6f9da76d509213d31de0da88f555ff28acdf98c95588350e123750e80960aa`.
- **B ham:** [20261118-093757_confirmation.zip](../result/modelnet40_c/cloudfixer_hypotheses/20261118-093757_confirmation.zip); 395,520,996 byte; SHA256 `80ad0b9192690cc4959fafe31fcb53ba6a127a1831da4e44787cf9ed2a57ffb0`.
- **B analiz:** [20261118-093757_confirmation_recount.zip](../result/modelnet40_c/cloudfixer_hypotheses/20261118-093757_confirmation_recount.zip); 320,749 byte; SHA256 `0de7f49d376afece37e5d132fa243e2f5d9746494d13d2efe6b5731d8169ec9e`.

[Run] A'nın son oturumu `tta-cloudfixer-point2vec-recovery-a100-20261120-r13`, endpoint `gpu-a100-s-usw4b0-3turcmp9s9iyb`; scientific/preservation kabulünden sonra resmî Drive flush/unmount ve açık kullanıcı talimatıyla stop yapıldı. **20 Kasım 20:35:58 UTC** sunucu envanteri oturumun yokluğunu, 0 aktif atama ve 0 CU/saat gösterir. 21 Kasım kayıt kontrolünde de sunucu envanteri boş bulundu. B oturumu `tta-cloudfixer-pm-hypotheses-a100-20261118-r13` daha önce dışarıdan kaybolmuştu; agent tarafından kapatılmış gibi sunulmaz. Bilimsel çıktıları bu kayıptan önce ikinci mount ile korunmuştu.

Kaynaklar: A `9afa8436b1770e38e148f9093d3edd85a4d9770a`; B `eec1e8b2fe272e9d1f42b6a628cacd5341f1c8c8`; resmî CloudFixer `f043e24414fcf4165fdc892abff21215c9989141`. Point2Vec, Point-MAE, diffusion, label ve environment SHA ayrıntıları ayrı A/B raporlarında ve arşiv config/environment dosyalarında bulunur. Operasyonel duraklamalar, başarısız tahsisler, bütçe değişiklikleri ve kurtarma kaynakları [yürütme günlüğünde](cloudfixer_two_colab_execution_20261118.md) saklanır; başarısız geçmiş silinmez.

## 7. Tam kayıtların okuma sırası

1. Bu ortak envanter: hangi testin hangi soruyu cevapladığı ve karşılaştırma sınırları.
2. [A tam Point2Vec raporu](cloudfixer_point2vec_full_results_20261120.md): tam benchmark, bütün corruption'lar, recovery ve kapanış.
3. [B doğrulama raporu](cloudfixer_pm_hypotheses_confirmation_20261118.md): ilk ayrıntılı hipotez kaydı, tablolar ve özgün evidence bağlantıları.
4. [A sayım/CI dosyaları](../result/modelnet40_c/cloudfixer_point2vec/20261120-184955_full_recount/summary.csv), [B 279 özet satırı](../result/modelnet40_c/cloudfixer_hypotheses/20261118-093757_confirmation_recount/summary.csv), [B 479 kontrast](../result/modelnet40_c/cloudfixer_hypotheses/20261118-093757_confirmation_recount/paired.csv), [B diagnostics](../result/modelnet40_c/cloudfixer_hypotheses/20261118-093757_confirmation_recount/diagnostics.json), [B canonical contribution](../result/modelnet40_c/cloudfixer_hypotheses/20261118-093757_confirmation_recount/canonical_contributions.json).
5. [Bu derlemenin kompakt sayısal kanıtı](evidence/cloudfixer_ab_record_20261121/verified_report_tables.json) ve kaynak kimliği makbuzu.

[Open] A'da seed/native varyansı, Point2Vec'te aynı N/K/τ ile PM–kendi terminali ayrımı ve rotation/background açığının fiziksel nedeni çözülmüş değildir. Bunlar gelecekte bağımsız geliştirme protokolü gerektiren bilimsel sorulardır; bu tamamlanmış görevde bekleyen test veya yeni GPU izni olarak kaydedilmez.
