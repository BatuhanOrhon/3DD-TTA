# Sabit denoiser hedefinde doğrudan SCD rotasyon araması

## Ön kayıt · 21 Kasım 2026

**[Code]** Amaç, t80'de tahmin edilen ilk temiz nokta bulutu q ile gözlem X arasındaki geometrik uyumu doğrudan arayarak doğru pozu ve sınıflandırmayı iyileştirip iyileştiremeyeceğimizi ölçmek. N8/K80/τ=0.1 ve PM[20,10,0] sabit. Üç kol rotasyonu kapalı N8, aynı q üzerinde üç Kabsch ve doğrudan SCD. Hepsi aynı ön işlemeyi, gürültüyü, FPS tohumlarını ve donmuş Point2Vec ile CloudFixer ağırlıklarını kullanır. Her politika dokuz denoiser ve üç sınıflandırıcı geçişi yapar.

**[Code]** Doğrudan SCD, P=I ile başlar. Her iki yönde en yakın komşuları yeniden hesaplar, kareli mesafelerin en küçük %95'inin yön başına ortalamasını alır ve ikisini toplar. Geçersiz veya tekrarlanan gözlem indeksleri poz hesabının dışında kalır. İlk tur P'yi üç eksende ±15°, ikinci tur güncel P'yi ±10° döndürerek altı adayı karşılaştırır. Yalnız kesin skor düşüşünde en iyi adımı kabul eder. Başlangıç dahil 13 skor, toplam en çok 25°. Sınıf etiketi veya sınıflandırıcı güveni poza girmez. Ham X·P sonraki sekiz proksimal düzeltmenin referansıdır. Spektral filtre kullanılmaz.

**[Paper]** Pozu ve yer değiştirmeyi donmuş difüzyon tahminine göre birlikte güncelleme fikri CloudFixer §4.2'den gelir. Yazarlar simetrik Chamfer ile gradyan alır. Burada projedeki seçici eşleşme ve iki yönlü kırpma kullanılır. 3DD-TTA seçici Chamfer'ın, Chetverikov vd. kırpılmış eşleşmelerin, Kabsch kapalı form kontrol kolunun kaynağıdır. Altı eksen yönüyle türevsiz tarama Kolda, Lewis ve Torczon Alg. 3.1'den esinlenir. İki sabit adım ve denoising zincirine entegrasyon bu deneyin seçimidir. [CloudFixer §4.2](https://arxiv.org/html/2407.16193#S4.SS2) · [3DD-TTA](https://arxiv.org/html/2411.14495) · [Trimmed ICP](https://doi.org/10.1109/ICPR.2002.1047997) · [Kabsch](https://doi.org/10.1107/S0567739476001873) · [Direct Search](https://doi.org/10.1137/S003614450242889)

**[Code]** ModelNet40-C S5, temiz + 15 bozulma, B1, tohumlar 0/1/2. Önceki geliştirme, doğrulama ve spektral pilot nesneleri çıkarıldı. Kalan 2468-indeksli havuzdan UTF-8 `cf-scd-20261121:{id}` SHA-256 değerine göre ilk 12 nesne seçildi. [Sabit indeksler ve eşikler](../cfgs/cloudfixer_scd_rotation_pilot.json). Eşlenmiş 576 giriş × 3 yöntem. Ayrıca tohum 0'da temiz ve Background örneklerine bilinen birim, x5°, bileşik (15°,−15°,15°) ve y60° rotasyonları uygulanır. Doğru ters pozdan kalan açı ve referans şekle kırpılmamış Chamfer ayrı ölçülür. Her nesnenin üç tohumu ile tüm koşulları birlikte yeniden örnekleyen eşzamanlı %95 bootstrap aralıkları raporlanır.

**[Decision]** Daha geniş bir doğrulamaya geçiş için üç Kabsch'a göre Rotation ve Background kayıpsız, temiz ve Rotation dışı 14 ortalamasında en fazla 2 yüzde puan kayıp, ortalama B1 süre oranı ≤1.20 olmalı. Bir bağın olumlu sonuç sayılmasını önlemek için ek olarak Rotation'da N8'e karşı net artış ve temiz bileşik sentetik pozda pozitif ortalama açı kazanımı aranır. Bu eşikler keşif amaçlıdır. Temiz sonuç all15 dışında tutulur.

**[User report/Code]** Kullanıcı küçük GPU pilotunu onayladı. Deney dalı `experiment/cloudfixer-rotation-pilot`, temel `1a4974f`. Aynı görev için hazırlanmış `tta-cloudfixer-rotation-a100-20261121-r9`, uç `gpu-a100-s-usw4b0-3355ke025p8dj`, A100-SXM4-40GB. Drive, kanonik ortam ve Point2Vec hazırlık işaretleri 21 Kasım'da yeniden doğrulandı. Paket manifestosu model, kaynak ve veri kimliklerini içerir. Operatör sırası `scd_deploy`, `scd_smoke`, `scd`. Model ve veri yükleme dahil yaklaşık 8–15 dakika, 6.73 CU/saatte 0.9–1.7 CU tahmini. İki hesaplama evresi toplam 1500 saniye kalıcı bütçe ile sınırlı. Hatalı durum/parite/özet veya hash, OOM ve bütçe aşımında dur. Her bloğu Drive'a kaydet, bağlantı koparsa süreç ve doğrulanmış blokları denetle, aynı kampanyada son kayıttan sürdür. Drive kökü `MyDrive/thesis/cloudfixer_scd_rotation_20261121`, uzakta çalışma dizini `/content/cloudfixer_scd_rotation_20261121`, yerel arşivler `result/modelnet40_c/cloudfixer_rotation`.

## GPU sonucu

**[Run]** Temiz kaynak `ba13d815b5a04faa5e1ead98f7f50a7eff2cf8bb`, mühürlü koşu `20261121-193927_scd`. GPU ön kontrolü iki örnekte N8 eşdeğerliğini bit düzeyinde gösterdi. 48 blokta 576 eşlenmiş girdi, 1.728 benchmark ve 288 sentetik politika değerlendirmesi tamamlandı. SHA-256, ZIP CRC, 48 blok hash'i, logitlerden ortalama olasılık ve tahmin, isabet/geçiş sayıları, iki turun 13 SCD skoru ve SO(3) doğrulandı. Yerel ZIP `result/modelnet40_c/cloudfixer_rotation/20261121-193927_scd.zip`, Drive kopyası aynı baytları içeriyor. Ham modeller, girdi dosyaları, ortam ve paket kimlikleri manifestoda ve arşivde yer alıyor. [Arşiv kaydı](evidence/cloudfixer_scd_rotation_pilot_20261121/archive_receipt.json)

Doğru tahmin sayıları aşağıda, yüzde puan farkları aynı nesne ve tohumlar üzerinde hesaplandı. Her tekil koşul 12 nesne × üç tohum içerir. Bütün koşulları ve tohumları nesne kümeleri olarak yeniden örnekleyen eşzamanlı %95 aralığın yarıçapı 8.33 yüzde puan. [Eşlenmiş farklar, onarımlar ve kayıplar](evidence/cloudfixer_scd_rotation_pilot_20261121/paired.csv) · [Tohum başına sayılar](evidence/cloudfixer_scd_rotation_pilot_20261121/per_corruption.csv)

| Koşul | N8 | Üç Kabsch | Doğrudan SCD | SCD − N8 (yp) | SCD − Kabsch (yp) |
|---|---:|---:|---:|---:|---:|
| clean | 33/36 | 33/36 | 33/36 | +0.00 | +0.00 |
| background | 26/36 | 25/36 | 26/36 | +0.00 | +2.78 |
| cutout | 32/36 | 32/36 | 32/36 | +0.00 | +0.00 |
| density | 31/36 | 31/36 | 31/36 | +0.00 | +0.00 |
| density_inc | 33/36 | 33/36 | 33/36 | +0.00 | +0.00 |
| distortion | 32/36 | 32/36 | 32/36 | +0.00 | +0.00 |
| distortion_rbf | 34/36 | 34/36 | 34/36 | +0.00 | +0.00 |
| distortion_rbf_inv | 33/36 | 32/36 | 33/36 | +0.00 | +2.78 |
| gaussian | 34/36 | 34/36 | 34/36 | +0.00 | +0.00 |
| impulse | 34/36 | 33/36 | 34/36 | +0.00 | +2.78 |
| lidar | 18/36 | 18/36 | 18/36 | +0.00 | +0.00 |
| occlusion | 18/36 | 18/36 | 18/36 | +0.00 | +0.00 |
| rotation | 21/36 | 21/36 | 21/36 | +0.00 | +0.00 |
| shear | 27/36 | 26/36 | 27/36 | +0.00 | +2.78 |
| uniform | 33/36 | 34/36 | 33/36 | +0.00 | -2.78 |
| upsampling | 35/36 | 35/36 | 35/36 | +0.00 | +0.00 |
| macro14 | 420/504 | 417/504 | 420/504 | +0.00 | +0.60 |
| all15 | 441/540 | 438/540 | 441/540 | +0.00 | +0.56 |

**[Run]** Doğrudan SCD, Rotation’da 8/36 girdiyi döndürdü ve bunlardan üçünde en yüksek olasılıklı sınıf değişti. Bu üç tahmin de yanlış kaldı. Background ve temiz koşulda her örnekte birim poz kaldı. SCD ile N8 bütün 576 girdide aynı doğru/yanlış sonucunu verdi. Kabsch’a göre SCD dört yanlış tahmini düzeltti (Background, `distortion_rbf_inv`, Impulse ve Shear) ve Uniform’da bir doğru tahmini kaybetti. Background’daki +2.78 yp farkın eşzamanlı aralığı [−5.56, +11.11] yp, all15’teki +0.56 yp farkın aralığı [−7.78, +8.89] yp.

Aşağıda sentetik testlerde bilinen ters poza kalan ortalama açılar yer alıyor. Her hücre 12 nesne içeriyor. Kimlik satırında sıfıra yakın değer, yöntemle eklenen istenmeyen pozu ölçüyor. [Nesne sayıları ve SCD etkinliği](evidence/cloudfixer_scd_rotation_pilot_20261121/decision.json)

| Girdi | Bilinen dönüş | N8 kalan açı | Üç Kabsch kalan açı | Doğrudan SCD kalan açı |
|---|---|---:|---:|---:|
| Temiz | Birim | 0.00° | 0.54° | 0.00° |
| Temiz | x5° | 5.00° | 4.15° | 5.00° |
| Temiz | Bileşik | 24.74° | 24.01° | 23.34° |
| Temiz | y60° | 60.00° | 60.32° | 60.00° |
| Background | Birim | 0.00° | 0.48° | 0.00° |
| Background | x5° | 5.00° | 4.29° | 5.00° |
| Background | Bileşik | 24.74° | 23.91° | 23.24° |
| Background | y60° | 60.00° | 60.16° | 60.00° |

SCD bileşik dönüşü temiz girdide 4/12, Background’da 5/12 nesnede azalttı. Ortalama kazanım 1.41° ve 1.50°. x5° kontrolünde iki aday adımı da uygulanmadı. y60° kontrolünde 25° bütçe tam terslemeyi zaten kapsamıyor ve seçilen poz birim kaldı.

| Yöntem | Ortalama B1 (ms) | Medyan B1 (ms) | B1 p95 (ms) | Tahsis edilen tepe GPU belleği (MiB) |
|---|---:|---:|---:|---:|
| Rotasyon kapalı N8 | 245.95 | 238.19 | 349.90 | 554.32 |
| Üç Kabsch | 259.46 | 251.48 | 362.97 | 554.35 |
| Doğrudan SCD | 253.47 | 245.53 | 358.04 | 802.41 |

**[Run]** SCD arama bileşeni, CUDA olaylarıyla 6.70 ms ortalama aldı. Kabsch eşleme ve SVD bileşenleri toplamı 5.19 ms. SCD’nin senkronize politika süresi N8’e göre %3.06 yüksek ve Kabsch’a göre %2.31 düşük. Altı adayın paralel mesafe hesabı N8’e göre 248.09 MiB tepe tahsis farkı yarattı. Üç kol da 9 denoiser ve 3 sınıflandırıcı çağrısı yaptı. Pilotun başlangıcından arşivin doğrulanmasına 9 dk 29 sn geçti. Hesap bakiyesi hazırlık, ön kontrol, pilot ve aradaki süreler boyunca 59.87 → 58.17 CU değişti. Görünen oran 6.73 CU/saat. [Koşul ve bileşen başına GPU maliyeti](evidence/cloudfixer_scd_rotation_pilot_20261121/gpu_cost.csv)

**[Decision/Inference]** Ön kayıt, N8 üzerinde pozitif bir Rotation farkı gerektiriyordu. Dört doğruluk toleransı, süre ve temiz bileşik poz koşulu karşılandı. Rotation’da N8 karşısındaki sıfır net fark genişletme koşulunu karşılamadı. Bu adayın şu anki kapısı kapalı. t80 hedefinin yön sinyalini iki sabit açı ve 12 nesnede ölçtük. Küçük düzeltmeler 10° aday adımının altında kalabilir ve t80 hedefi girdi yönünü büyük ölçüde koruyabilir. Poz aramasının yararına ilişkin karar, ayrı nesnelerde gerçek Rotation onarımları, Background kayıpları ve açı iyileşmesi birlikte ölçülürse değişebilir. GPU koşusu kapandı ve r9 Colab ataması doğrulama anında canlı ve boşta kaldı.

**[Paper]** 3DD-TTA, SCD’yi latent noktalarda hesaplar ve yön başına kırpılmış toplamı o yöndeki tüm noktaların sayısına böler. Bu pilot, XYZ’de geçerli referans noktalarını seçip her yönde tutulan mesafelerin ortalamasını kullanır. [3DD-TTA, Denklem 11](https://arxiv.org/html/2411.14495#S3.SS3)

**[Code]** Paylaşılan entegrasyon, deney dalındaki `ba13d81` yöntem kodunu, ön kaydı, sonuç notunu ve doğrulanmış özetleri içerir. Ham arşiv ve Colab geçici operatör dosyaları yerel sonuç dizinlerinde ve belirtilen Drive arşivinde tutuldu. Entegrasyon öncesi ilgili 49 pytest kontrolü ve Git boşluk kontrolü geçti.
