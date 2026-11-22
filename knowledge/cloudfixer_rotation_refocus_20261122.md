# Rotation için araştırma kararı ve sonraki deney

**22 Kasım 2026.** [User decision] Araştırmanın önceliği Rotation. [Inference] İlk aday, nesnenin baskın düz yüzeylerinden tek bir bileşik dönüş kestirip N8'den önce uygulamak. Mevcut K80, sekiz adımlı gradient-free SCD ve t20/t10/t0 Probability Mean karşılaştırma temelidir. İlk karar, yön düzeltmesinin bağımsız nesnelerde sınıflandırmaya geçip geçmediğidir.

## Doğruluktaki açığın büyüklüğü

[Run] Donmuş Point2Vec, ModelNet40-C severity 5, 2.468 özgün nesne, seed 2. Rotation'da kaynak sınıflandırıcı 1.158 doğru (%46,92), N8 terminal 1.168 (%47,33), N8 geç PM 1.186 (%48,06), N32/K160 terminal 1.109 (%44,94), plain CloudFixer 1.715 (%69,49). N8 PM'nin CloudFixer'a 529 net doğru farkı, tüm bozulmalardaki 848 net farkın %62,38'i. Diğer 14 satır sabitken bu açığı kapatmak genel doğruluğa +1,43 puan getirir. [Tam koşu](cloudfixer_point2vec_full_results_20261120.md) · [Ham olasılıklardan yeniden sayım](evidence/cloudfixer_background_20261121/counts.json).

[Run/Arithmetic] Mevcut üç sınıf-olasılığı vektöründe, 1.183 nesnede aynı yanlış sınıf doğru sınıfın önünde. Bu nesneler her konveks ağırlıklı ortalamada yanlış kalır. Kalan tüm nesneler düzeltilse bile bu üç vektörü yeniden ağırlıklandırmanın iyimser tavanı 1.285/2.468 = **%52,07**. Bu, mevcut %48,06'nın en fazla +4,01 puan üstü ve plain CloudFixer'ın 17,42 puan altıdır. Yeni geometri veya yeni sınıflandırıcı girdisi yeni bir vektör üretebilir. [Karşılaştırma tanımı ve kodu](evidence/cloudfixer_background_20261121/recount.py).

## Önceki deneylerden kalanlar

[Run] Aşağıdaki farklar aynı deney içindeki eşlenmiş kontrolle karşılaştırılmıştır. Farklı pilotların mutlak doğrulukları farklı nesneleri ve rastgelelik akışlarını kullanır.

| Yöntem ve kontrol | Rotation | Tüm 15 | Ölçülmüş süre ve yorum |
|---|---:|---:|---|
| Tek Kabsch / N8, 64 nesne × 3 seed | +1,56 puan | −0,21 puan | 257,1 / 252,2 ms, ortalama |
| t80 sabit hedefle üç Kabsch / N8, aynı grup | +3,65 | −0,03 | 265,5 / 252,2 ms. Rotation'da 7 düzeltme, 0 kayıp. Eşzamanlı %95 aralık [−7,81, +15,10] puan |
| t80/t50/t20 kademeli Kabsch / N8, aynı grup | +3,13 | +0,17 | 265,8 / 252,2 ms. 6 düzeltme, 0 kayıp. Ayrı 32 nesnede Rotation farkı 0 |
| Yedi pozdan güvenle seçim / N8, 64 × 3 | +3,13 | −3,68 | Çıkış seçimi 375,9 / 252,2 ms. Seçimi zincire geri vermek Rotation'da ek kazanç getirmedi |
| Spektral ön filtre / her kolun kendi filtresiz Kabsch'ı, 12 × 3 | −2,78 / +2,78 / 0 | −0,56 / +0,93 / −0,56 | Tek, sabit, kademeli sırası. Sabit hedefte +1 doğru. Ek süre %1,31–2,39 |
| t80 hedefine doğrudan SCD ile poz araması / N8, ayrı 12 × 3 | 0 (21/36) | 0 (441/540) | Ortalama +%3,06. Rotation'da 8 örnek döndü, doğru/yanlış dengesi değişmedi |

[Run] Ayrı 32 nesneli pilotta ağırlıklı SCD + kademeli Kabsch birleşimi Rotation'da +3,12, tüm 15'te +0,21 puan verdi. Buna karşılık tek filtreli t10'un 256 nesnelik panelinde Rotation üç tohumda −0,65 puandı. Önceden kayıtlı doğrulama koşullarını geçen bir Rotation yöntemi henüz yok. [Kabsch ve güven seçimi](cloudfixer_rotation_pilot_20261121.md) · [Spektral eşleştirme](cloudfixer_spectral_rotation_pilot_20261121.md) · [SCD araması](cloudfixer_scd_rotation_pilot_20261121.md) · [Birleşik kol](cloudfixer_background_pilot_20261122.md) · [Tek t10](cloudfixer_background_followup_20261122.md).

## Teşhis

[Code/Paper] Yazarların üretecinde severity 5, üç eksenin her biri için işareti ayrı seçilen 12,5°–17,5° dönüşleri sırayla uygular ve sonra normalize eder. Eski sınıflandırıcı araması kimlik ve altı tek-eksenli ±20° aday deniyordu. Örneğin üç açının büyüklüğü tam 15° olduğunda sekiz işaret bileşiminde bozulmanın geodezik açısı 24,74° veya 27,02°. Eski yedi aday arasından gerçek ters dönüşe en yakınını seçebilsek bile kalan açı sırasıyla 18,39° veya 21,30°. Hesap satır vektörü için uygulanan B ve aday R ile arccos((tr(BR)−1)/2). Bu geometrik kısıt sınıflandırmanın doğru olması için açının sıfırlanması gerektiğini göstermez. [Resmî üreteç](https://github.com/jiachens/ModelNet40-C/blob/fad786c430c007cfbb7b4fe90ed1c9873d960921/data/generate_c.py#L16-L30) · [Kullandığımız arama](../cloudfixer_rotation.py).

[Run] Önceki 64 nesne ve üç seed için dondurulmuş yedi t20 logitini yeniden saydık. Rotation'da döndürülmemiş t20 **73/192**, güven ve açı cezasıyla seçilmiş t20 **76/192**. Yedi adaydan en az biri **110/192** olayda doğruydu. Başlangıçta yanlış olan 37 olayda doğru bir aday vardı, seçici bunların sekizini buldu ve beş eski doğruyu kaybetti. 82 olayda yedi adayın tamamı yanlıştı. Clean'de aynı üç sayı 166, 166 ve 177. Bunlar tek t20 tahminleri. Gerçek etiketi kullanarak sayılan 110 erişilebilir yöntem doğruluğu değildir ve bütün PM için üst sınır oluşturmaz. Arşiv bilimsel kaynağı 1367d39, yerel/Drive deney yolu result/modelnet40_c/cloudfixer_rotation/20261121-161227_validation.zip (Drive archives/20261121-161227_validation.zip). [Sayım ve seed tablosu](evidence/cloudfixer_rotation_refocus_20261122/t20_pose_coverage.json) · [Yeniden üretim](evidence/cloudfixer_rotation_refocus_20261122/recount_views.py).

[Inference] Sabit hedefli Kabsch, eski 16 nesnelik sentetik kontrolde 24,74° bileşik açıyı ortalama 23,76°'ye indirdi. Bu sonuç ve t20 arşivi birlikte iki farklı darboğaza işaret ediyor. Mevcut denoiser tahminiyle en yakın komşu eşleşmesi büyük bir yön değişimi üretemiyor ve sınıflandırıcı güveni mevcut doğru sınıfı veren adayları güvenilir biçimde seçemiyor. CloudFixer 30 iterasyonda hem global dönüşü hem noktasal yer değiştirmeleri optimize ediyor. Bizim tam test farkı bunların katkısını tek başına ayırmıyor. Makalenin Tablo 3'ünde de parametreleme seçiminin etkisi var. [Sentetik Kabsch tanısı](evidence/cloudfixer_rotation_pilot_20261121/development_pose_recount.json) · [CloudFixer §4.2 ve §5.3](https://arxiv.org/html/2407.16193v1).

## Hız hedefiyle ilk aday

[Paper/Inference] Nesnenin düz bölgelerinde yüzey normalleri ortak doğrultularda toplanabilir. Bu doğrultulardan, eğitim ve sınıflandırıcı sorgusu gerektirmeden, kimliğe yakın bir bileşik rotasyon kestirilebilir. Manhattan frame literatürü bu ilkeyi yapılandırılmış sahnelerde kullanır. Buradaki varsayım, bazı ModelNet nesnelerinde bu doğrultuların kaynak sınıflandırıcının beklediği eksenlerle örtüşmesi. Yuvarlak, eğik veya simetrik şekillerde bu varsayım zayıflar. [Straub vd., CVPR 2014, §3](https://openaccess.thecvf.com/content_cvpr_2014/papers/Straub_A_Mixture_of_2014_CVPR_paper.pdf).

[Code/Run] Ön kayıt 0abb521. Kaynağı doğrulanan temiz data_original.npy üzerinde yalnız önceki 16 geliştirme nesnesine bilinen dönüşler uygulandı. 256 yerel yamada 16-komşulu kovaryansın en küçük özvektörü yüzey normalini verdi. Düzlemsellikle ağırlıklandırılan eksen uyumu 79 küçük düzgün dönüşte puanlandı. İki yama grubunun kararları uyuşur ve iki grupta da uyum yeterince artarsa tek poz uygulanır. Aksi halde sonuç kimliktir. Model, GPU ve sınıf etiketi kullanılmadı. Sentetik dönüşte normaller aynı dönüşle taşındı. Ham temiz XYZ ile yapılan bu kontrol, yazar loader'ı sonrası ölçülen eski Kabsch deneyiyle eşlenmiş bir yöntem karşılaştırması değildir. [Dondurulmuş tarif](evidence/cloudfixer_rotation_refocus_20261122/frame_design.md) · [CPU kontrol kodu](evidence/cloudfixer_rotation_refocus_20261122/frame_check.py) · [Nesne bazında sonuç](evidence/cloudfixer_rotation_refocus_20261122/frame_check.json).

| Bilinen dönüş | Başlangıç açısı | Kalan ortalama açı | İyileşen / kötüleşen / aynı nesne |
|---|---:|---:|---:|
| Temiz kimlik | 0° | 0° | 0 / 0 / 16 |
| Tek eksende küçük dönüş | 5° | 5° | 0 / 0 / 16 |
| Bileşik (+15°, −15°, +15°) | 24,74° | 10,43° | 12 / 0 / 4 |
| Farklı bileşik (−8°, +21°, +11°) | 24,32° | 18,51° | 8 / 0 / 8 |
| Tek eksende büyük dönüş | 60° | 83,75° | 0 / 13 / 3 |

[Inference] Büyük açıda iki yama grubu yanlış eksende de anlaşabiliyor. Eksen işareti ve sırası yüzey geometrisinden her durumda belirlenemez. Yöntem genel SO(3) düzelticisi olarak kullanılamaz. Ham nokta bulutlarındaki açı sonuçları, sınıf doğruluğu veya GPU gecikmesi ölçmüyor.

## Önerilen deney sırası

1. **Gerçek poz düzeltmesinin sınıflandırma değerini ölç.** Pilotlara ayrılmamış bir geliştirme grubunda aynı donmuş N8/PM için üç kol çalıştır. Girdi olduğu gibi, bilinen sentetik dönüş zincir başlamadan geri alınmış ve aynı ters dönüş yalnız üç sınıflandırıcı girdisine uygulanmış. Bozma/tersleme, normalizasyon sırası, örneklenen indeksler, başlangıç gürültüsünün koordinat sistemi ve FPS çekilişlerini eşle. Sentetik kimlik ve ters-dönüş kontrollerinde temiz eşin yeniden oluştuğunu doğrula. Arşivdeki gerçek Rotation–clean çiftlerinde yalnız metrik için güvenilir eşleme ve küçük uyum artığı doğrulanırsa gerçek açı çıkar. Aksi halde açıyı yalnız bilinen sentetik dönüşlerde raporla. Aynı küçük panelde yeterli bütçe olursa yazar CloudFixer'ında tam, R sabit ve noktasal Δ sabit kollarını aynı RNG ile çalıştır. Böylece orijinal yöntemin üstünlüğünün hangi bileşenle ilişkili olduğunu ayır. Bilinen ters dönüşle dahi N8 düzelmiyorsa sınıflandırma ve ön işleme tanısına dön.
2. **Tek adayla kontrollü pilot yap.** N8'den önce kabul edilmiş dönüşü bir kez hem buluta hem SCD referansına uygula, görüntüleri aynı düzeltilmiş çerçevede sınıflandır. N8/K80/τ=.1/%95 ve üçlü PM sabit kalsın. Ek denoiser ve sınıflandırıcı çağrısı hedefi sıfır. Karşılaştırmalar N8, sabit hedefli üç Kabsch ve yüzey normalleri adayı. Aynı kabul edilen örneklerde eş-açılı rastgele eksen kontrolü yaparak yön seçiminin katkısını ölç. Test sırasında bozulma türü, sınıf ve temiz eş kullanılmasın. Yöntem her nesnede aynı kuralla çalışsın. İlk ölçümde tüm süreye komşu arama ve CPU–GPU aktarımı dahil olsun.
3. **Sınırları önceden kilitle.** Öneri olarak pilotlardan ayrı 128 özgün ID, aynı ID'lerde seed 0/1/2, Rotation s5 ve clean. Kimlik listesi ve eşikler sonuçlardan önce dondurulsun. Bilinen farklı eksen/açı ve 60° stres testini de raporla. İlk eleme için Rotation en az +3 puan, temiz kaybı en fazla 1 puan, B1 p50 ve p95 en fazla 1,10×. Uygulanma oranı ve düzeltilen/bozulan tahmin sayısı da verilsin. İlk eleme geçilirse aynı politikanın tam 15 bozulmada genel doğruluk ve kalan 14 ortalamasını değerlendirmek için ayrı bir doğrulama grubu kullan. Önerilen benimseme ölçütleri genel doğrulukta pozitif fark, kalan 14'te en fazla 0,5 puan kayıp ve maliyet sınırı. Nesne bazında eşlenmiş güven aralıkları kazancı veya korumayı göstermiyorsa aday keşif düzeyinde kalsın. Tüm pilot adayları geçmiş tam testte yer aldığı için tek başına bu benchmark geniş SO(3) genellemesi kanıtlamaz.
4. **Büyük açılarda başarısızlık sürerse** ayrı bir çalışma olarak donmuş XYZ önselinin kimlik ve geometrik aday arasındaki tercihini sınamak mümkün. Aynı gürültü ve zamanla iki tahmini karşılaştıran test önce bilinen açılı küçük grupta yanlış yön seçimini azaltmalı, sonra bütçesini karşılamalı. Öğrenilen kanonikleştirme de literatürde var, ancak ek eğitim gerektirir ve mevcut eğitimsiz protokol için önerilen ilk deney kapsamını aşar. [Kaba vd., ICML 2023](https://proceedings.mlr.press/v202/kaba23a.html).

[Decision] Genel baseline N8/K80 + geç PM. Bu kayıtta yapılanlar eski logitlerin CPU üzerinde yeniden sayımı ve ayrı bir geometrik fizibilite kontrolüdür. Yeni Colab sınıflandırma koşusu yapılmadı.
