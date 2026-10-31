# Shape latent TTA: ilk beş kollu pilot sonucu — 2026-10-31

[Run/Inference] İlk sabit pilotta yeni shape-latent yöntemlerinden hiçbiri
mevcut N8/Background35 Chamfer + probability-mean referansını all15 doğrulukta
aşmadı. Bu, bütün global diffusion yaklaşımlarının işe yaramadığının kanıtı
değildir. Denenen sabit parametrelerle, bu küçük ve daha önce açığa çıkmış
nesne grubunda iyileşme kanıtı yoktur. Yöntemler varsayılan yapılmadı.

## Eşleşmiş deney

[Code/Run] ModelNet40-C severity5, Point-MAE, seed0, 64 orijinal nesne,
15 bozulma, B32. Her kol 960 tahmin olayı; bunlar 960 bağımsız nesne değildir.
Aynı orijinal z/h/local epsilon, girdiler ve hazırlık sonrası RNG kullanıldı.
64 nesne önceki depth-development128 içinden etiketsiz SHA sıralamasıyla
seçildi; teknik smoke32 bunun altkümesidir. Bağımsız confirmation değildir.

Local DDIM normal8/Background35; gamma=.03, eta=.003, p=.95. Primary readout
normal clean2/4/6+terminal8, Background clean9/18/27+terminal35 olasılık
ortalamasıdır. CSV'nin terminal alanı primary yerine kullanılmadı; ham
trajectory logits'ten PM yeniden hesaplandı. LION raw/eval/EMAoff; sınıflandırıcı
ve LION ağırlıkları sabit. PointNet2 burada CUDA FPS/gather kütüphanesidir;
deneyde PointNet++ sınıflandırıcısı kullanılmadı.

| Kol | Müdahale | All15 PM% | Referansa fark, puan | Background PM% | Macro14 PM% |
|---|---|---:|---:|---:|---:|
| baseline | Orijinal z ile clean/final decoder | 60.937500 | — | 53.1250 | 61.495536 |
| updated | Güncellenmiş style ile clean/final decoder | 60.729167 | −0.208333 | 51.5625 | 61.383929 |
| anchor | updated + her style güncellemesine %5 uzaklık sınırı | 60.312500 | −0.625000 | 50.0000 | 61.049107 |
| proposal | anchor + t20'de tek global temiz tahmin | 60.312500 | −0.625000 | 54.6875 | 60.714286 |
| guided3 | anchor + global DDIM20/10/0 ve temiz tahmin üzerinden guidance | 60.416667 | −0.520833 | 51.5625 | 61.049107 |

[Run] Global proposal/guided3 aynı bağımsız Gaussian gürültüyü paylaşır;
global model zamanları sırasıyla21 ve21/11/1'dir. Guided3 kaybı temiz z tahmini
ile orijinal bozuk z arasındaki nesne başına128-boyut MSE'dir; nesneler
toplanır, katsayı1. Noisy z'ye türev global denoiser içinden geçer; her adımın
düzeltmesi en fazla .01||z_original||, son global tahmin ve sonraki bütün
style güncellemeleri .05||z_original|| küresine projekte edilir. Orijinal h
korunur; yeni z ile h yeniden encode edilmez.

## Eşleşmiş farklar ve belirsizlik

| Karşılaştırma | Fark, puan | Düzeltilen / bozulan tahmin | Koşullu %95 aralık, puan |
|---|---:|---:|---:|
| updated − baseline | −0.208333 | 12 / 14 | [−1.041667, +0.625000] |
| anchor − updated | −0.416667 | 13 / 17 | [−1.458333, +0.625000] |
| proposal − anchor | 0.000000 | 15 / 15 | [−1.145833, +1.145833] |
| guided3 − anchor | +0.104167 | 12 / 11 | [−0.729167, +0.937500] |
| guided3 − proposal | +0.104167 | 12 / 11 | [−0.833333, +1.145833] |
| guided3 − baseline | −0.520833 | 12 / 17 | [−1.666667, +0.625000] |

[Run/Inference] Aralıklar 10.000 eşleşmiş orijinal-nesne bootstrap tekrarıdır
(seed20261031); nesnenin15 bozulması birlikte örneklenir. Tek diffusion seed'i,
sabit bozulmaları ve açığa çıkmış nesne havuzunu koşullar; çoklu karşılaştırma
düzeltmesi yoktur. Bütün aralıklar sıfırı içerir. Üç adımlı kolun anchor'a göre
net bir olayı kazanması güvenilir fayda kanıtı sağlamaz; proposal karşılaştırması
hem zincir uzunluğunu hem guidance'ı değiştirir, saf guidance ablation'ı değildir.

[Run/Inference] Bozulma bazında proposal Background'da referanstan bir olay
iyidir (35/64 karşı34/64). Guided3 distortion_rbf'de iki olay iyi, Gaussian'da
üç olay kötüdür; rotation bütün kollarda17/64'tür. Bu küçük farklardan yöntem
seçimi veya bozulma yönlendirmesi yapılmadı.
[Tam bozulma tablosu](evidence/shape_latent_tta_study_20261031/pilot_per_corruption.csv).
Bu kohortun60.9375%'i başka kohorttaki66% veya68.741319% ile doğrudan
karşılaştırılamaz; burada referans da aynı64 nesne üzerinde yeniden çalıştı.

## Latent mekanizmasından öğrendiklerimiz

[Run] Güncellenmiş style'ın orijinal z'ye göre medyan son uzaklığı Gaussian'da
%0.1206, Background'da%6.7474; Background maksimumu%18.8361'dir. Anchor kolu
Background medyanını%4.9767'ye, maksimumunu kayan nokta toleransıyla%5'e
indirir; fakat doğruluk artmaz. Yalnızca büyük style drift'ini sınırlamak bu
pilot için yeterli değildir.

[Run/Inference] İlk global temiz tahminin medyan uzaklığı Gaussian'da%9.8530,
Background'da%10.2440'tür. Proposal/guided3 son uzaklığı her ikisinde de yaklaşık
%5'tir. Gaussian'da doğal local-style hareketi yaklaşık%0.12 iken global
müdahalenin yaklaşık%5 hareket getirmesi, sınırın her koşul için küçük bir
müdahale sayılmadığını gösterir. Bunun doğruluk kaybının nedeni olduğu bu
deneyle ayrıştırılmadı; temiz-z oracle veya yeniden-encode-h kontrolü yoktur.

[Run] Guided3 gradientleri bütün incelenen adımlarda sonlu ve sıfırdan
büyüktür. Gaussian medyan düzeltme||z|| oranı%0.1518, Background%0.1577'dir;
bu iki bozulmada maksimumlar%0.2108/%0.1981, %1 tavanının altındadır.
Guidance çalışmıştır; gözlenen null sonucu kopmuş Jacobian'a bağlayacak
kanıt yoktur. Anchor yönü bozuk başlangıç z'sini korur; semantik doğru yöne
hareket garantisi sağlamaz.

## Doğrulama, süre ve kaynak

[Run] Teknik32×2 smoke beş kolu geçti; orada da referans70.3125%, proposal
67.1875%, guided3 68.75% idi. İlk controller, tamamlanmış iki-bozulma çıktısını
yanlışlıkla full15 status şartına bağladı; bu idari başarısızlık korundu.
Versiyonlu guard gerçek coverage'ı doğruladı ve başarılı baseline byte'ları
tekrar çalıştırılmadan kullanıldı. Bilimsel kaynak/parametre değişmedi.

[Run] Pilot bütün beş kolda30 batch, toplam150 batch shard ve30 fiziksel ortak
cache ile tamamlandı. Yerel bağımsız NumPy1 yeniden hesabı uzak acceptance ile
birebir aynı: ID/sıra/PM, ham kaynak SHA, donmuş model state'leri, fiziksel
cache tensor/RNG hash'leri, global RNG değişmezliği,1/3 çağrı sayıları,
%5 sınırlar ve30 proposal/guided3 noise SHA eşleşmesi geçer. ZIP CRC ve SHA
yerelde doğrulandı; Drive kopyasının SHA kontrolü controller'da yapıldı.

[Run] Araştırma 13:02:35.879UTC başladı; pilot dahil kayıtlı elapsed1337.696s
(22.295dk), başlangıçtaki idari stop/inceleme/aktarımı da kapsar. Setup ayrı
kaydedildi; araştırma55dk tavanını aşmadı. Pilot13:14:56–13:24:53UTC civarıdır.
Reconstruction+terminal ölçümleri baseline86.6322s, updated81.0645s,
anchor81.0350s, proposal81.2596s, guided3 82.1225s; clean-probe classifier
hesabı reconstruction içindedir, arşivleme/worker startup bu toplam değildir.
Baseline cache üretir, diğerleri replay eder: bunlardan baseline'a göre
uçtan uca hızlanma sonucu çıkarılamaz. Replay kolları içinde guided3/anchor
yaklaşık1.0134 oranı da tekrarsız instrumented ölçümdür.

[Run] Artifact I/O dahil worker süreleri aynı sırayla117.9443s,112.4127s,
112.3185s,112.4935s,113.5556s'dir. Kaydedilen bütün-process tepe GPU bellek
25218.42–25221.38MiB aralığındadır; model yükleme ve probe'ları içerir.

[Code/Run] GPU kaynağı a82200adbe117ce56a134dc3c22601cb73dc2654 + uncommitted
334-dosya snapshot'tır: manifest
`e9a75f8b90374b96b6fd49bb59a0ba646747e7e98a01d01d4604ddf373fc5786`,
source ZIP `d7c360debc73597c7a8449f456e8aaad5580ae5694538a080bbf620e7282de3d`.
Daha yeni operational-baseline integration ağacının GPU ile denenmiş olduğu
iddia edilmez. CPU repository testleri502passed/1historical-fixture skip ve
373subtest; vendor kök collection desteklenmez. Ayrı4 operator kontrolü geçer.

Pilot ham ZIP95791160byte, SHA
`c2a58d37641e549cf3fc5403143c913483f92dd6bba912efe8d715f23d688ec3`.
Smoke ZIP19218443byte, SHA
`4e96bdd77293832645a9f9647ccf0b76a4bcf9742975ca69d4d68263b2507b3a`.
Drive: `MyDrive/thesis/result/shape_latent_tta_study_20261031/phase-evidence/`.
Yerel hamlar `result/preparation/shape_latent_tta_study_20261031/operator/`
altında korunur. [Kompakt kanıt ve yeniden hesap](evidence/shape_latent_tta_study_20261031/README.md).
[Plan ve başarısız deneme geçmişi](shape_latent_tta_study_20261031.md).

[Decision/Open] Ek grid, seed, ScanObjectNN, conditional-h/oracle veya tam
benchmark başlatılmadı. Yeni yöntem baseline'a yükseltilmedi. İlerideki çalışma
önce global önerinin temiz shape bilgisi sağlayıp sağlamadığını ve h-z
uyumunu ayırmalıdır; bu pilot tek başına daha uzun zinciri gerekçelendirmez.
Kendi `tta-shape-latent-a100-20261031-r1` runtime'ı korunur; kapatma izni yoktur.

[Run] Türetilmiş analiz158311byte ZIP olarak Drive'a kopyalanıp geri indirildi;
byte-identik SHA `7b2310c398fc32636e02aafe19c317f43c0063e9f9411738bf14677ae47b8ad7`.
[Gerçek roundtrip receipt](evidence/shape_latent_tta_study_20261031/derived_archive_receipt.json).
Sonradan hazırlanmış latest-target entegrasyonunda518CPUtest/412subtest ve
ayrı60NumPy1test geçer;61skip'in60'ı bu ayrı ortamda karşılanır, biri tarihsel
source fixture'dır. Bu kontroller GPU accuracy deneyi değildir.
