# Yüzey onaylı t50 PCA, GPU doğrulaması ve koşullu tam test

[User report/Decision] Kullanıcı 23 Kasım'da büyük GPU doğrulamasını ve başarılı olursa sadece yeni yöntemin ModelNet40-C tam testini yetkilendirdi. Mevcut baseline dondurulmuştur. Başlangıç 06876dd, oturum sahibi bu deney, çalışma ağacı surface512-20261123. 18:06 UTC envanteri boş, bakiye 400,77 CU. Yeni isim tta-surface512-a100-20261123-rN. Paket, kullanıcı hesabı consent'i ve harcama çalışma alanı result/preparation/surface512_20261123. Drive MyDrive/thesis/surface512_20261123, bilimsel arşiv result/modelnet40_c/surface512_20261123. Bir A100; 6,73–8,88 CU/saat üzerinden kurulum ve doğrulama tahmini 3–5 saat (20–45 CU), koşullu tam test ek 3–5 saat (20–45 CU), kesintiler dahil süre sınırı 16 tahsis saati / ~142 CU. Kapatma ayrı izin gerektirir.

[Code/Open] Önce kanonik Colab notebook hücre 1, aynı VM'de 2, Drive onayı ve 3–10, sonra pinned Point2Vec/XYZ ortamının geri yüklenmesi, smoke, surface512 ve yalnız karar geçerse surface_full. Yazar checkpoint SHA'ları 6c1d4884…/b64841d0…, ModelNet40-C s5, B1. Yeni aday t50 PCA + ham gözlemde iki nokta grubunda düzlemsellik ağırlıklı eksen uyumu >0,015; sabit 256 merkez, 16 komşu. Eski pilot ve geliştirme ID'lerinin birleşimi hariç tutularak sabit liste cfgs/cloudfixer_surface512.json üretildi. Validation 512×(15+clean)×3 = 24.576 aday tahmini. Aynı nesne ve seed'de dönüş reddedildiğinde iki yol eşittir. Arşivde tam bu RNG ile eksik olan kontrollerden yalnız kabul edilen olaylarda ve liste konumu 0,16,…,496 olan önceden seçilmiş hız altgrubunda baseline ölçülür. Bu altgrupta kol sırası dönüşümlü, kararsızlık varsa kimlik/logit eşitliği smoke ve eş zamanlı kontrol ile doğrulanır. CloudFixer yeniden çalıştırılmaz.

[Decision] Rotation ≥+3 puan, all15 >0, clean ve her bir diğer tür ≥0 puan, B1 p50/p95 oranları ≤1,05. Nesne-kümeli belirsizlik ayrıca raporlanır; nokta ölçütü geçmiş olması baseline terfisi anlamına gelmez. Teknik arıza, bozuk veri/çıktı, ≤0 ağırlıkta kimlik yolundan sapma, bütçe veya hız sorununda dur, sonuçlara göre eşik değiştirme. Başarılıysa s5 2468×15 = 37.020 nesne-bozulmada yalnız aday, tam seed 2, tarihsel frozen Point2Vec/CF arşivlerinden karşılaştırma; farklı RNG olduğunda açıkça belirt. Her dört nesnede Drive'a SHA kontrollü blok; yerel ayna ve son ZIP doğrulama. Oturum kaybolursa aynı kaynak paketini tekrar doğrula, tamamlanan blokları yükle, eksikleri çalıştır. Ayrı run ID kullan. Tüm bozulma satırlarında PM, terminal, düzelen/bozulan, uygulanma ve B1 gecikme kaydedilir.

[Code] Tam tarama kapısı geçerse, aynı aday/seed2/N8/birebir ID-adresli RNG ile ölçülen doğrulama örneklerinden 512×15=7.680 adayı kaynak manifesti ve ham SHA'ları doğrulayarak kullanır. Geriye 29.340 yeni GPU tahmini kalır. Tam raporda her 37.020 aday gözlemi yer alır.

[Code/Artifact] Bilimsel kaynak f44dc30, 29 araştırma ve 513 yazar kaynağı doğrulandı. research.zip SHA-256 a05b3f378f17511e7f1e9eece9a8d10a48a790068ef1d0540edc75db682708c9. official.tar c7817cc2d04bdf1f8a054ec69831afb10f96d27068a65a0b86820232f314fdcd. Önceki kayıtlardan 875 ID dışlandı, 512 yeni ID seçildi. Ayrı eski GPU kontrolleri Rotation 321 kabul/1146 veto, Shear 321 kabul, Density 321 veto. 257 CPU testi geçti. Karar analizinin tek bir korunan tür düşerse başarıyı reddettiği ayrıca sınandı. Operatörler result/preparation/surface512_20261123/{allocate_new.py,start_setup.py,guarded_setup.py,operate.py}; ana izleme kayıtları allocation.log, setup_progress.log, science.log.

[Run] Yedi A100 kapasite reddi ardından 2026-11-23T18:16:21.407680+00:00 UTC başlangıçlı tta-surface512-a100-20261123-r8, gpu-a100-s-ass1c0-3ck9nrgygida9, Standard A100, bu göreve ait tarife 6,73 CU/saat ve başka görünür oturum yok. 16 saatlik üst sınır bu tahsiste 107,68 CU. Hücre 1 ve kernel restart işareti doğrulandı. Makale notu ve panel kaydı paylaşılan thesis-integration dalında befa0eb; aktif yöntem varsayılanları değişmedi.

[Run/Open] 18:24 UTC — Hücre 2 aynı r8 tahsisinde hem uzak .done dosyası hem de doğrudan CLI tamamlanma satırıyla doğrulandı. Bu VM için Google Drive kullanıcı onayı açıldı. Denetleyici mount ve KNN dosyası kontrolü sonrası hücre 3–10, deployment/restore/smoke/512 ve yalnız geçerse tam aday safhasına otomatik devam eder. Bu aşamada yeni GPU doğruluk sonucu yoktur.

[Run] 18:34 UTC — Kullanıcı Google onayını verdi. İlk mount kontrolü başarısız oldu. Yetki aynı endpoint’e yeniden aktarıldı, Colab mount işlemi tamamlandı ve bağımsız CLI listelemesi ile KNN dosyası görüldü. İkinci doğrulama hücresi mount ve 5717 baytlık KNN arşivini onayladı. Yerel resume_after_mount.py mevcut oturumda hücre 3–10 sıralamasına geçti. Önceki iki hücre tekrarlanmadı.

[Run] 19:20 UTC — 10/10 kanonik hücre, CUDA native testleri, 15 veri dizisi, Point2Vec/XYZ checkpoint doğrulaması ve ayrı cf_point2vec ortamı tamamlandı. Smoke 20261123-191936_feedback_surface_smoke.zip SHA-256 2f9cb88c4cbb3e363de51355219af6368325e6d30b998cf04223645a2152bf89, 6 blok. Dört önceden kilitli eski karar ve logit karşılaştırması geçti. Aynı ZIP doğrudan Drive'dan tekrar indirilip SHA, CRC ve ham logitleri yeniden kontrol edildi. 512 taraması başladı; 6144 blok hedefi, her dört olayda Drive kaydı.

[Run, teknik ara kontrol] İlk sabit 128 nesne × 16 koşul × 3 seed = 6144 kayıt tamamlandı. Yerel blok kopyalarındaki SHA-256, eksiksiz/tekil kayıtlar, 9/3 çağrıları ve reddedilen müdahalelerde üç logit eşitliği doğrulandı. Önceden seçilen 8 zamanlama nesnesinde all15 için 360 eşli ölçüm, aday/baseline B1 p50 247,83/243,82 ms (+%1,65), p95 356,12/353,33 ms (+%0,79). Bu aşamada doğruluk üzerinden karar verilmedi.

[User decision, 20:30 UTC] Kullanıcı GPU işleri tamamlandığında bu oturumun kapatılmasını istedi. 512 ölçütleri geçilirse son GPU işi tek adaylı tam taramadır. Ölçütler geçilmezse son GPU işi doğrulama panelidir. close_after_results.py son kapsamın bütün Drive ve host ZIP’lerini doğrulayıp resmî flush_and_unmount yaptıktan sonra isim ve endpoint eşleşmesiyle yalnız r8 oturumunu kapatır, envanterde yokluğunu ve hesap kullanımını kaydeder.


## 512 nesnede doğrulama sonucu

[Run/Decision, 24 Kasım 2026 TSİ] Yüzey onaylı tek t50 PCA, Rotation PM doğruluğunu **%46,09’dan %64,58’e** çıkardı. All-15 **%73,90’dan %75,14’e** yükseldi. Clean’in 1.536 eşli kararında doğru/yanlış sonucu değişmedi. Diğer 14 bozulmada toplam iki ek doğruya rağmen dokuz bozulmanın kendi satırında düşüş görüldü. Önceden kaydedilen dört ölçütten Rotation, all-15 ve ortak hız geçti. Her bir bozulmada doğruluğu koruma koşulu geçilmedi. Bu sebeple koşullu 2.468 × 15 tam aday taraması başlatılmadı. Tez baseline’ı N8/K80 SCD + geç Probability Mean olarak kaldı.

512 yeni nesnenin her biri 15 bozulma ve clean altında seed 0, 1 ve 2 ile ölçüldü. Her tekil satırda 1.536, all-15’te 23.040 eşli gözlem var. Üç seed aynı nesneleri tekrar ölçtü. All-15 temiz veriyi dışarıda bırakır. Tabloda fark aday eksi baseline ve birimi yüzde puandır. Aralıklar 512 nesne üzerinden 10.000 tekrarlı kümeli bootstrap ile hesaplanan satır bazında nominal %95 aralıklardır. Düzelen ve bozulan sütunları PM’nin doğru/yanlış geçişlerini, son sütun kabul edilen dönüş sayısını verir.

| Koşul | Baseline PM (%) | Aday PM (%) | Fark (puan) | %95 aralık (puan) | Düzelen / bozulan | Uygulanan dönüş |
|---|---:|---:|---:|---:|---:|---:|
| Clean | 91,34 | 91,34 | 0,00 | [0,00, 0,00] | 0 / 0 | 13 |
| Background | 62,89 | 62,57 | -0,33 | [-0,98, +0,13] | 1 / 6 | 13 |
| Cutout | 83,59 | 83,33 | -0,26 | [-0,78, +0,13] | 1 / 5 | 13 |
| Density | 79,43 | 79,36 | -0,07 | [-0,20, 0,00] | 0 / 1 | 12 |
| Density Increase | 86,07 | 85,87 | -0,20 | [-0,59, 0,00] | 0 / 3 | 15 |
| Distortion | 73,76 | 73,96 | +0,20 | [-0,26, +0,78] | 5 / 2 | 62 |
| Distortion RBF | 78,19 | 78,12 | -0,07 | [-0,20, 0,00] | 0 / 1 | 27 |
| Distortion RBF Inverse | 79,43 | 79,36 | -0,07 | [-0,52, +0,46] | 3 / 4 | 15 |
| Gaussian | 89,65 | 89,65 | 0,00 | [0,00, 0,00] | 0 / 0 | 8 |
| Impulse | 91,02 | 90,82 | -0,20 | [-0,59, 0,00] | 0 / 3 | 17 |
| LiDAR | 41,73 | 41,54 | -0,20 | [-0,52, 0,00] | 0 / 3 | 22 |
| Occlusion | 42,12 | 41,99 | -0,13 | [-0,59, +0,20] | 1 / 3 | 34 |
| Rotation | 46,09 | 64,58 | +18,49 | [+15,36, +21,74] | 284 / 0 | 564 |
| Shear | 75,52 | 76,95 | +1,43 | [+0,46, +2,54] | 26 / 4 | 115 |
| Uniform | 89,52 | 89,52 | 0,00 | [0,00, 0,00] | 0 / 0 | 9 |
| Upsampling | 89,45 | 89,45 | 0,00 | [0,00, 0,00] | 0 / 0 | 3 |
| Rotation dışındaki 14 | 75,88 | 75,89 | +0,01 | [-0,18, +0,16] | 37 / 35 | 365 |
| All-15 | 73,90 | 75,14 | +1,24 | [+0,95, +1,52] | 321 / 35 | 929 |

[Run] Rotation’da seed 0, 1 ve 2 için kazanç sırasıyla +19,34, +18,36 ve +17,77 puan. Üçünde de baseline’ın doğru bildiği bir tahmin kaybedilmedi. Rotation’da toplam 284 düzeltme 117 ayrı nesneden geldi. All-15 kazancı üç seed’de +1,30, +1,17 ve +1,25 puan. Aynı örneklerde terminal okuması da Rotation’da %45,57’den %64,06’ya, all-15’te %73,46’dan %74,70’e çıktı. Adayda PM, terminalin all-15 doğruluğuna 0,44 puan ekledi.

[Run/Inference] Rotation dışındaki 14 türdeki 37 düzeltmeye 35 kayıp eşlik etti. Shear’da net 22 ve Distortion’da net 3 ek doğru elde edildi. En büyük kayıplar Background’da net 5, Cutout’ta net 4 olaydı. Dokuz olumsuz satırın her birinin nominal aralığı sıfırı içerir. Kayıpların büyüklüğü için belirsizlik sürüyor, ancak önceden yazılan karar kuralı gözlenen satır farklarına dayanıyor. Aynı nesneler üzerinden bütün satırlar için birlikte hesaplanan %95 aralık Rotation’da [+15,30, +21,68], all-15’te [−1,95, +4,43] puandır.

[Run/Inference] Rotation dışındaki 35 kayıp dokuz ayrı nesnede toplandı. Bunların 15’i tek bir nesneye ait ve dokuz nesnenin üçünde rotasyon bozulması altında da düzeltme görüldü. Gelecek geometri çalışmasında ilk tanı, bu nesnelere hangi yüzey yamalarının dönüş onayı verdiğini ve aynı nesnenin Rotation örneğinde nasıl davrandığını incelemektir. Bu panel üzerinde geliştirilen bir kuralı değerlendirmek için ayrı nesneler gerekir. [Nesne düzeyinde sayım](evidence/surface512_20261123/affected_objects.json)

[Run] Önceden ayrılan 32 nesne × 15 bozulma × 3 seed üzerinde A100 B1 için 1.440 eşli zamanlama alındı. Baseline → aday p50 245,25 → 250,55 ms (+%2,16), p95 354,45 → 358,33 ms (+%1,10). Rotation’a özgü p95 255,00 → 272,03 ms (+%6,68, 96 çift). Kayıtlı %5 sınırı all-15 p50 ve p95 içindi. Adayın en yüksek kayıtlı GPU bellek kullanımı 554,46 MiB. Her aday tahmininde dokuz denoiser ve üç sınıflandırıcı çağrısı doğrulandı.

[Artifact] Ham arşiv `result/modelnet40_c/surface512_20261123/20261123-192004_feedback_surface512.zip`, Drive konumu `MyDrive/thesis/surface512_20261123/archives/`. 51.753.848 bayt, SHA-256 `a273d78157fa9e45a2aa841c541ae8cfc4929b9d3c345aea3283ee3379bab807`. 6.144 blokta bütün ID/seed/bozulma kombinasyonları birer kez görüldü. Drive’dan bağımsız indirilen ZIP, GPU arşiviyle aynı SHA ve geçerli CRC verdi. Üç logit okumasından PM tahminleri yeniden hesaplandı ve yöntem kararı ham sayılarla doğrulandı. CloudFixer bu 512 panelinde çalıştırılmadı. [Karar, hız, güven aralıkları ve seed bazında sayılar](evidence/surface512_20261123/result.json)

[Run/Close] Drive yazımları tamamlanıp disk ayrıldı. Colab CLI, yalnız bu deneye ayrılan `tta-surface512-a100-20261123-r8` oturumunu 23 Kasım 21:41 UTC’de (24 Kasım 00:41 TSİ) kapattı. Bağımsız sunucu ve kullanım sorguları sıfır aktif oturum, 0,00 CU/saat ve 378,00 CU bakiye gösterdi. Tahsis süresi yaklaşık 3 saat 25 dakika. Başlangıç ve bitiş bakiyelerinin farkı 22,77 CU. Kaynak experiment/surface512-20261123 üzerinde f44dc30, sonuçların paylaşım kapsamı isteğe bağlı deney uygulaması, konfigürasyon, test ve kompakt kanıtlardır. Ham ZIP’ler ve geçici operatörler sonuç dizinindedir. İlgili 257 yerel test geçti. [Kapatma ve iki arşivin sağlama toplamları](evidence/surface512_20261123/closure.json) · [Straub vd., literatür ve projedeki uyarlama](papers/straub2014_manhattan_frames.md)
