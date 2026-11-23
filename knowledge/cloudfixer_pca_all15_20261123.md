# Denoiser PCA geri beslemesi için ModelNet40-C all-15 pilotu

[User report/Decision, 23 Kasım 2026] Kullanıcı, önceki odak pilotundaki clean kaybına rağmen 15 bozulmanın ortalama etkisinin ölçülmesini istedi. Önceki karar ve 192 bloğun sonuçları aynen korunur. Yeni soru, 15 eşit ağırlıklı bozulmanın ortalaması ile Rotation dışındaki 14 bozulmanın ortalamasıdır. Temiz veri bu ortalamaların dışında. Dondurulan konfigürasyon cfgs/cloudfixer_output_rotation.json içindeki daha önce hiç değerlendirilmemiş other_ids 128 nesne; önceki tarihsel tam test bütün özgün ID'leri içerir. Yazar ModelNet40-C s5, seed 0, B1, dondurulmuş Point2Vec ve XYZ, N8/K80, aynı üç PM okuması. Üç eşlenmiş kol identity, pose_pca_t50, pose_pca_all. 1.920 koşul-nesne, 5.760 yöntem tahmini, 480 adet dörder nesnelik blok. Önceden seçilen 15° t50 alt eşiği aynı kayıtlardan ikincil, zamanlamasız yeniden sayılabilir. Başlıca ölçüler all15 / kalan14 / her bozulma PM ve terminal doğruluğu, eşlenmiş nesne-kümeli %95 aralık ve B1 p50/p95. Yöntemler ve eşikler pilot sonucuna göre ayarlanmaz. Başlangıç d93d9c4 (bilimsel kaynak 7304679 ile aynı rotasyon kodu).

[Decision/Open] Ayrı Drive MyDrive/thesis/pca_all15_20261123, yerel result/modelnet40_c/pca_all15_20261123. A100 tercih korunur. Önceki r4 oturumu 08:15 UTC envanterinde yok; yalnız başka görevin tahsisi görünüyor. Yeni ad tta-pca-all15-a100-20261123-rN, operatör bu görevin CLI istemcisi. Tahsis öncesi hesap toplamı 440,49 CU. Yeni oturumda kanonik notebook 1/yeniden bağlan/2/Drive/3–10, ek model ve native doğrulama, ayrı smoke, ardından scripts/run_cloudfixer_rotation.py --phase feedback_all15. Beklenen yeni tahsis süresi ~1,5–2,5 saat (A100 6,73–8,88 CU/saat varsayımıyla ~10–23 CU); kesintiler dahil çalışma üst sınırı 4 tahsis saati / yaklaşık 36 kendi CU. Hesap toplamı paylaşımlı. Veri, checkpoint, native, sayım, akış veya kayıt hatasında tanı ve doğrulanmış bloklardan devam. Her dört nesnede SHA doğrulamalı Drive yayını; host aynaları her 16 nesne, nihai ZIP tekrar indirilir. Oturum kapatma ayrı izin gerektirir.

[Code] Kaynak 9225740, 216 CPU testi geçti. 25 araştırma ve 513 yazar kaynak dosyası tek tek doğrulandı. Paket research.zip SHA-256 d2335b8f9fe3bf3ace417ed41b6fcb6a69f490654db178851d690601c5836468, official.tar c7817cc2d04bdf1f8a054ec69831afb10f96d27068a65a0b86820232f314fdcd. Aynı 128 other_ids, önceki odak ve smoke ID'lerinden ayrık. Kaynak ağırlıkları Point2Vec 6c1d4884… ve XYZ b64841d0…, her A100'de yeniden doğrulanır. Geri kazanım için ayrı klasörde feedback_all15_deploy, feedback_all15_restore, feedback_all15_smoke, feedback_all15 sırası korunur. Rapor tek seed'e koşullu 10.000 nesne-kümeli bootstrap dağılımından iki taraflı nominal %95 aralık ve araştırma betiğinin eşzamanlı aralığını içerir.

[Run] Üç kapasite reddinden sonra 2026-11-23T08:20:05.926607+00:00 başlangıçlı tta-pca-all15-a100-20261123-r4, gpu-a100-s-usc1b0-21ic5pfeiqb4w, Standard A100. Yalnız bu görevin tarifesi 6,73 CU/saat (paylaşımlı hesap 13,46). Kendi dört saatlik sınırın tarife karşılığı ~26,92 CU. Yerel denetleyiciler result/preparation/pca_all15_20261123 altında allocate_new.py, start_setup.py, guarded_setup.py, run_science.py, audit_results.py. Hücre 1 geçti; hücre 2 aynı tahsiste.

[Run/Open] Hücre 1 ve aynı VM'de hücre 2 doğrulandı. Yalnız kayıtlı fastrlock derleme hatası ertelendi. 08:26 UTC'de yeni VM için Google'ın Drive izin akışı açıldı. Kullanıcı onayı sonrası sürücü mount ve KNN dosyasını kontrol eder, hücre 3–10, üç kollu kısa kontrol, her dört nesnede Drive kaydı yapan all15 ve bağımsız yeniden sayım sırayla çalışır. Bu aşamada all15 tahmini sıfır.

[Run] Drive mount ve 5.717 baytlık KNN kaynak dosyası 08:28 UTC'de aynı VM'de doğrulandı. Hücre 3 dosyayı kopyalayıp kendi tamamlanma işaretini verdi; kalan setup ve araştırma sürücüsü kullanıcı etkileşimi gerektirmeden devam ediyor.


## All-15 pilot sonucu

[Run] Aynı 128 nesne ve tek sabit seed ile üç yöntemin 1.920'şer tahmini tekrar sayıldı. PM, t20/t10/t0 sınıf olasılıklarının eşit ortalaması. Aşağıdaki ilk 15 satırın her biri 128 nesne içeriyor. Son iki satırda sırasıyla 1.792 ve 1.920 eşlenmiş gözlem var. Clean ortalamaya dahil edilmedi. [Sayısal kanıt](evidence/pca_all15_20261123/result.json) · [Ham arşivden yeniden sayım betiği](evidence/rotation_feedback_20261123/recount.py).

| Bozulma | Baseline PM | PCA t50 PM | PCA sekiz adım PM | t50 farkı (puan) | Sekiz adım farkı (puan) |
|---|---:|---:|---:|---:|---:|
| background | %69,53 | %64,06 | %64,06 | −5,47 | −5,47 |
| cutout | %87,50 | %82,03 | %82,81 | −5,47 | −4,69 |
| density | %80,47 | %73,44 | %72,66 | −7,03 | −7,81 |
| density_inc | %89,84 | %85,16 | %84,38 | −4,69 | −5,47 |
| distortion | %80,47 | %75,00 | %76,56 | −5,47 | −3,91 |
| distortion_rbf | %81,25 | %78,12 | %76,56 | −3,12 | −4,69 |
| distortion_rbf_inv | %85,94 | %83,59 | %83,59 | −2,34 | −2,34 |
| gaussian | %91,41 | %89,84 | %90,62 | −1,56 | −0,78 |
| impulse | %92,97 | %90,62 | %90,62 | −2,34 | −2,34 |
| lidar | %35,94 | %32,81 | %33,59 | −3,12 | −2,34 |
| occlusion | %36,72 | %32,03 | %32,81 | −4,69 | −3,91 |
| rotation | %42,19 | %62,50 | %61,72 | +20,31 | +19,53 |
| shear | %78,12 | %74,22 | %75,78 | −3,91 | −2,34 |
| uniform | %92,19 | %88,28 | %88,28 | −3,91 | −3,91 |
| upsampling | %91,41 | %89,84 | %89,84 | −1,56 | −1,56 |
| Rotation dışındaki 14 | %78,12 | %74,22 | %74,44 | −3,91 | −3,68 |
| All-15 | %75,73 | %73,44 | %73,59 | −2,29 | −2,14 |

[Run] Tek t50 Rotation'da 54 doğrudan 80'e çıktı (+26, +20,31 puan), diğer 14'te 1.400'den 1.330'a indi (−70). Sekiz adım Rotation'da 79 doğru (+25, +19,53 puan), diğer 14'te 1.334 doğru (−66) verdi. Her iki PCA yönteminin accuracy'si Rotation dışındaki her satırda bazın altındaydı. All-15 olaylarında t50 38 yanlışı düzeltti, 82 doğruyu bozdu. Sekiz adımda aynı sayılar 39 ve 80. Bu sayımlar sınıfları ya da bozulma kimliklerini yöntem seçiminde kullanmayan dondurulmuş politikalar içindir.

| Yöntem | All-15 PM doğru / 1.920 | All-15 PM farkı ve nominal %95 aralık (puan) | All-15 terminal accuracy | B1 p50 | B1 p95 |
|---|---:|---:|---:|---:|---:|
| N8/K80 baseline | 1.454 | referans | %75,21 | 242,7 ms | 352,9 ms |
| PCA t50 | 1.410 | −2,29 [−4,22, −0,63] | %72,66 | 245,2 ms | 355,3 ms |
| PCA sekiz adım | 1.413 | −2,14 [−3,91, −0,52] | %72,66 | 253,0 ms | 362,9 ms |

[Run/Inference] Nominal aralıklar sabit seed'de 128 nesneyi eşlenmiş olarak 10.000 kez yeniden örnekledi. Çoklu karşılaştırma ailesi için eşzamanlı all-15 %95 aralıkları t50 için [−10,10, +5,52], sekiz adım için [−9,95, +5,68] puan. Nominal aralıklar kayıp yönünde, eşzamanlı aralıklar sıfırı içeriyor. Bu panelde ölçülen genel doğruluk daha düşük; tam 2.468 nesneye veya seed'ler arası etkiye kesin büyüklük atfedilemez. Pooled B1 p95, baseline'a göre +%0,67 ve +%2,82; en büyük satır bazlı p95 artışları da +%2,64 ve +%4,80. Hız tüm satırlarda hedef içinde.

[Run/Secondary] Önceki odak çalışmasında seçilen 15° alt eşik bu ayrık nesnelerde aynı GPU çıktılarından yeniden oynatıldı. All-15 1.418/1.920 = %73,85 (−1,88 puan), Rotation 79/128 = %61,72 ve diğer14 1.339/1.792 = %74,72. Bu tekrar sayımının ayrı bir süre ölçümü yok. Eski t50'nin kimlik kararı verdiği tüm olaylarda üç ham logit vektörünün bazla birebir eşitliği kontrol edildi.

[Decision] Genel N8/K80 + geç PM baseline korunur. Global PCA ile her bozulmayı aynı şekilde düzeltmek, bu pilotun gerçek Rotation kazancına rağmen, 15'li ortalamayı düşürdü. Önceki clean sonucu ayrı 128 nesne ve üç seed üzerinde kaldı. Bozulma türünü test sırasında bilmeyen bir rotasyon seçicisinin, ancak ayrı nesnelerde hem all-15 hem Rotation dışı ortalamayı koruyabildiği ölçülürse baseline değişikliği tekrar değerlendirilebilir.

[Artifact] 40 GB A100, bilimsel kaynak 9225740. 480 all-15 bloğundaki 5.760 yöntem-örneğinin yanı sıra ayrı smoke'ta 2 blokta 6 yöntem-örneği doğrulandı. Üç kolun iki smoke koşulundaki üçer logit vektörü eski 80 GB oturumundaki eşleriyle aynıydı. 482 bloğun yerel ayna kayıtları ve payload SHA-256 değerleri nihai ZIP'lerle eşleşti. Her iki nihai arşiv ayrıca bağlı Drive'dan doğrudan tekrar indirildi ve VM'den alınan kopya ile bayt bayt aynı bulundu. Arşivler Drive'da MyDrive/thesis/pca_all15_20261123/archives, yerelde result/modelnet40_c/pca_all15_20261123 altında. Ana arşiv 20261123-091333_feedback_all15.zip (6.316.277 bayt), SHA-256 a2cf85604c94477ec3671958bc0ba534bbac4a2256714ca3a1785a818638e153. Smoke SHA-256 3198f98d2ee20601593bee346ce17ef53f95c2cbf2a0fee8f36cdba8f1be26b9. Tahsis başlangıcından ana arşiv teslimine 80,43 dakika; bu oturumun 6,73 CU/saat tarifesiyle ~9,02 CU. Hesap toplamı eşzamanlı iki oturumu kapsadığı için bu, süre × tarife tahminidir.
