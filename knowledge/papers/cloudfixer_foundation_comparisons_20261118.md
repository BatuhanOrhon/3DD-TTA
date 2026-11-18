# CloudFixer görsel-dil temel modellerinde nasıl karşılaştırıldı?

**[Paper/Inference] Uni-Adapter, CloudFixer’ın yalnızca girdiyi düzelten sürümünü, aynı Uni3D-Large sınıflandırıcıyla yapılan üç deneyde de geçiyor.** ModelNet40-C’de yayımlanan fark **+13,61 yüzde puan**, ScanObjectNN-C’de **+11,90 puan**. ShapeNet-C’de yön aynı, büyüklük belirsiz. Yazarların ortalama sütunu **+5,17 puan**, yayımlanan 15 bozulmaya eşit ağırlık verilince **+7,38 puan** veriyor. Üç veri kümesinin her birinde CloudFixer’ın kazandığı iki bozulma da var. Point-Cache aynı üç tabloda CloudFixer’ın üzerinde. Bu karşılaştırmanın kaynağı Tamjidi ve arkadaşlarının deneyidir. [AAAI 2026, Tablo 1–3](https://ojs.aaai.org/index.php/AAAI/article/download/37888/41850#page=5)

## Aynı sınıflandırıcı, aynı bozulmalar

[Paper] Tamjidi ve arkadaşlarının *Adapt-As-You-Walk Through the Clouds* çalışmasında kaynakçada Shim, Kim ve Yang’ın ECCV 2024 makalesi yer alır. Yazarlar CloudFixer† için üretici modeli ve geometri kılavuzlamasını kullandıklarını, sınıflandırıcıyı sabit tuttuklarını belirtir. Uni-Adapter da kodlayıcıyı sabit tutar, ayrıca önceki etiketsiz örneklerden öğrendiği sınıf prototiplerini kullanır. Üç deney de örnek başına 1.024 nokta, 15 bozulma, şiddet 5 ve tek örneklik test partileriyle yürütülmüştür. Ek materyal, diğer bazı çevrimiçi yöntemler için varsayılan parti büyüklüğünü 16 verir, CloudFixer, 3DD-TTA, Point-Cache ve Uni-Adapter için bir örnek kullanıldığını ayrıca açıklar. Ölçü top-1 nesne sınıflandırma doğruluğudur. [Ana makale, deney düzeni ve sonuçlar](https://ojs.aaai.org/index.php/AAAI/article/download/37888/41850#page=5) · [Ek, PDF s. 12](https://arxiv.org/pdf/2511.15311#page=12)

| Yazarların bildirdiği ortalamalar (%) | ModelNet40-C · Tablo 1 | ShapeNet-C · Tablo 2 | ScanObjectNN-C · Tablo 3 |
|---|---:|---:|---:|
| Kaynak Uni3D-Large | 59,15 | 57,92 | 38,07 |
| CloudFixer† | 56,09 | 57,24 | 34,43 |
| 3DD-TTA | 56,06 | 55,30 | 30,78 |
| Point-Cache | 66,73 | 57,70 | 42,13 |
| Uni-Adapter | 69,70 | 62,41 | 46,33 |
| Uni-Adapter − CloudFixer† (puan) | **+13,61** | **+5,17** | **+11,90** |
| Point-Cache − CloudFixer† (puan) | +10,64 | +0,46 | +7,70 |
| CloudFixer† − kaynak (puan) | −3,06 | −0,68 | −3,64 |

[Paper/Inference] **ShapeNet-C satırının aritmetik kontrolü.** CloudFixer†’ın 15 hücresinin toplamı 825,44, ortalaması **55,0293**. Uni-Adapter için aynı hesap 936,20 / 15 = **62,4133** verir. Basılı CloudFixer† ortalamasının hücre ortalamasından 2,21 puan sapması iki ondalık basamağa yuvarlamayla açıklanamaz. Aynı değerler konferans PDF’sinde ve yazarların proje sayfasında görülüyor. ModelNet40-C’de ayrı bir uyumsuzluk 3DD-TTA’nın CloudFixer† ile sırasını da etkiler. 3DD-TTA’nın basılı ortalaması 56,06 iken hücreleri 849,13 / 15 = **56,6087** verir. CloudFixer†’ın 841,33 / 15 = **56,0887** değeri basılı 56,09 ile uyumludur. ShapeNet-C’de de 3DD-TTA’nın hücre ortalaması 55,3047 ile CloudFixer†’ın 55,0293 değerini az farkla aşar. Ham tahminler veya yazar düzeltmesi, hangi değerlerin özgün koşuyu temsil ettiğini aydınlatabilir. [Konferans makalesi, Tablo 1–2](https://ojs.aaai.org/index.php/AAAI/article/download/37888/41850#page=5) · [Yazarların sonuç tabloları](https://mehran-tam.github.io/Uni-Adapter/)

### Aynı tabloda başka hangi yöntemler CloudFixer†’ı geçiyor?

[Paper/Inference] Aşağıdaki değerler **her yöntemin basılı ortalaması eksi aynı tablodaki CloudFixer† ortalaması** olarak yüzde puan cinsinden verilmiştir. Pozitif değer o yöntemi öne çıkarır. Çalışmanın kendi katkısı Uni-Adapter’dır. Diğer on iki yaklaşım yazarların bu görsel-dil modellerine uyguladıklarını bildirdikleri karşılaştırmalardır. 3DD-TTA, CloudFixer ve Point-Cache nokta bulutları için geliştirilmiştir. Geri kalanlar 2B alandan aktarılmıştır. Nokta bulutuna veya görsel-dil modeline aktarılan bu deneyler, on iki özgün yöntemin yazarlarının ayrı ayrı bildirdiği CloudFixer karşılaştırmaları olarak sayılamaz. [Ana makale, “Baselines” ve Tablo 1–3](https://ojs.aaai.org/index.php/AAAI/article/download/37888/41850#page=5)

| Yöntem − CloudFixer†, basılı fark (puan) | ModelNet40-C | ShapeNet-C | ScanObjectNN-C |
|---|---:|---:|---:|
| TENT | +3,39 | −0,01 | +3,91 |
| SHOT | +3,46 | 0,00 | +4,62 |
| SAR | +3,92 | −0,50 | +3,47 |
| DUA | +3,41 | −0,16 | +3,57 |
| MEMO | +3,11 | +0,68 | +3,91 |
| TPT | +4,93 | +1,86 | +4,53 |
| LAME | +3,08 | +0,77 | +4,62 |
| 3DD-TTA | −0,03 | −1,94 | −3,65 |
| T3A | +8,03 | −10,35 | +7,71 |
| TDA | +7,54 | +2,19 | +6,19 |
| Point-Cache | +10,64 | +0,46 | +7,70 |
| Uni-Adapter | +13,61 | +5,17 | +11,90 |

[Inference] ModelNet40-C ve ScanObjectNN-C’de basılı ortalamalara göre CloudFixer†’ın altında kalan tek yöntem 3DD-TTA’dır. ShapeNet-C’de MEMO, TPT, LAME, TDA, Point-Cache ve Uni-Adapter’ın basılı ortalamaları daha yüksek, SHOT’unki eşittir. ShapeNet-C’de **tüm yöntemler için 15 basılı hücre yeniden ortalanırsa** yalnız T3A CloudFixer†’ın gerisinde kalır. 3DD-TTA − CloudFixer† hücre farkları ModelNet40-C, ShapeNet-C ve ScanObjectNN-C sırasında **+0,5200, +0,2753 ve −3,6587** puandır. İlk iki veri kümesinde basılı ve hesaplanan sıralamalar çeliştiği için bu karşılaştırma 3DD-TTA’nın üstünlüğünü teyit etmez. [Tablo 1–3, karşılaştırma satırları](https://arxiv.org/html/2511.15311v1#Sx3.T1)

### Hangi bozulmalar belirleyici?

[Inference] Aşağıdaki farklar her tablonun aynı bozulma sütununda Uni-Adapter doğruluğundan CloudFixer† doğruluğu çıkarılarak hesaplandı. ModelNet40-C’de en büyük kazanç örtülmede. ScanObjectNN-C’de LiDAR bozulması ters yöndeki en büyük farkı veriyor. ShapeNet-C LiDAR’da ise ikisi de yaklaşık %2,5 doğrulukta kalıyor ve 0,01 puanlık fark ancak basılı hassasiyette görülebiliyor. [Tablo 1–3, bozulma hücreleri](https://arxiv.org/html/2511.15311v1#Sx3.T1)

| Bozulma | ModelNet40-C | ShapeNet-C | ScanObjectNN-C |
|---|---:|---:|---:|
| Uniform gürültü | +1,38 | +1,32 | −1,55 |
| Gauss gürültüsü | −0,32 | −3,07 | +4,47 |
| Arka plan | +14,42 | +13,23 | +17,21 |
| Darbe gürültüsü | +4,14 | −0,91 | +11,36 |
| Nokta sayısını artırma | −0,73 | +0,50 | +1,38 |
| RBF deformasyonu | +12,80 | +12,40 | +14,62 |
| Ters RBF deformasyonu | +12,80 | +10,62 | +22,03 |
| Yoğunluk azaltma | +24,63 | +10,87 | +15,14 |
| Yoğunluk artırma | +17,42 | +17,02 | +11,69 |
| Kesme dönüşümü (shear) | +19,69 | +12,46 | +22,88 |
| Döndürme | +18,12 | +11,37 | +22,03 |
| Bölge çıkarma (cutout) | +19,13 | +9,28 | +14,79 |
| Çarpıtma | +21,20 | +15,05 | +19,10 |
| Örtülme (occlusion) | +27,83 | +0,61 | +10,15 |
| LiDAR | +11,66 | +0,01 | −6,89 |

### Uni-Adapter bir örneği nasıl sınıflandırıyor?

1. **[Paper] Başlangıç tahmini.** Dondurulmuş nokta kodlayıcısı nesneyi bir özellik vektörüne çevirir. Sınıf adlarının metin vektörleriyle kosinüs benzerliği sıfır-örnekli sınıf puanlarını ve en olası sınıfı verir. [Denklem 2](https://arxiv.org/html/2511.15311v1#Sx3.SSx2.SSSx1)
2. **[Paper] Prototipi güncelleme.** Tahmin edilen sınıfta boş yer varsa örnek yeni bir merkez olur. Yer doluysa kosinüs benzerliği en yüksek merkez güncellenir. Yeni örneğin entropisine ve merkezin önceki güncelleme sayısına bağlı ağırlıklar, az güvenilen bir tahminin etkisini azaltır. Böylece bir sınıfın farklı görünümleri ayrı merkezlerde tutulabilir. [Denklem 3–5](https://arxiv.org/html/2511.15311v1#Sx3.SSx2.SSSx1)
3. **[Paper] Etiket düzeltme.** Yöntem bütün sınıflardaki prototiplerden bir benzerlik grafı kurar. Zayıf bağlantıları eleyip normalize graf Laplasyenini hesaplar. İlk sınıf olasılıklarına yakınlıkla komşu prototiplerin etiket uyumunu birleştiren doğrusal sistemi eşlenik gradyanla çözer. Her prototipe çözümde en yüksek skoru alan sınıfı atar. Bu grafın düğümleri nesne prototipleridir. [Denklem 6–10](https://arxiv.org/html/2511.15311v1#Sx3.SSx2.SSSx2)
4. **[Paper] Son tahmin.** Örneğin prototiplere benzerlikleri düzeltilmiş sınıfa göre ortalanır. Makaledeki Denklem 13, kaynak ve bellek sınıf puanlarını kendi tahmin entropileri düşük olana daha çok ağırlık vererek birleştirir. Tablo 6’da yalnız çevrimiçi prototiplerle ModelNet40-C %68,48, grafla düzeltme eklendiğinde %69,70’tir. Aynı ekleme ShapeNet-C’de +0,62, ScanObjectNN-C’de +1,21 puan verir. [Denklem 11–13 ve Tablo 6](https://arxiv.org/html/2511.15311v1#Sx5.T6)

[Code/Open] Yazar deposunun `cf0d8b1` sürümünde her bozulma için ayrı, sıralı bir veri yükleyicisi kurulur. Bellek her bozulmanın başında boşaltılır ve mevcut örnek nihai puanlamadan önce belleği günceller. Dolayısıyla bu yürütme geçmiş örneklerden faydalanır, bozulmaların rastgele karıştığı kesintisiz bir akışı sınamaz. Açık sürüm ayrıca Denklem 13’teki logitleri doğrudan birleştirmek yerine iki softmax **olasılık** vektörünü ters entropiyle ağırlıklandırır. Makalede güven katsayısı β=10 verilirken depo ayarları β=150 kullanır. Bu kaynak kodunun yayımlanan tabloları üreten tam sürüm olduğu doğrulanmış değildir. [Deney döngüsü](https://github.com/Mehran-TAM/Uni-Adapter/blob/cf0d8b17aefccbb8860643a24f3df050c7c2d844/main_test-time.py#L66-L95) · [Bellek ve birleştirme kodu](https://github.com/Mehran-TAM/Uni-Adapter/blob/cf0d8b17aefccbb8860643a24f3df050c7c2d844/Uni_Adapter.py#L218-L293) · [Uygulama ayarları](https://github.com/Mehran-TAM/Uni-Adapter/blob/cf0d8b17aefccbb8860643a24f3df050c7c2d844/utils/hyperparams.py) · [Makalede deney ayarları](https://arxiv.org/html/2511.15311v1#Sx4)

[Paper/Open] CloudFixer† için ekte “Step Size” olarak verilen adaylar {1, 5, 10, 20, 30, 50}, rotasyon parametresi adayları {0,01, 0,02, 0,03}. Tablo 7 hangi birleşimin seçildiğini vermiyor. Tablo 9’daki Uni-Adapter–CloudFixer† eşleştirilmiş t-testi p-değerleri ModelNet40-C, ShapeNet-C ve ScanObjectNN-C sırasında 1,48 × 10⁻⁵, 5,00 × 10⁻⁴ ve 6,47 × 10⁻⁵. Bunlar yazarların bildirdiği testlerdir. Tablo 8’in ULIP-2 ve OpenShape sonuçları arasında CloudFixer† yoktur. Tablo 5 Uni3D için sıfır-örnekli çıkarımı 39,19, Uni-Adapter’ı 36,93 ve Point-Cache’i 9,73 örnek/s olarak verir. CloudFixer burada zamanlanmamıştır. [Ek, Tablo 7–9](https://arxiv.org/pdf/2511.15311#page=12) · [Ana makale, Tablo 5](https://arxiv.org/html/2511.15311v1#Sx4.T5)

## Point-Cache ve BayesMM için kanıt nereden geliyor?

**[Paper] Point-Cache, CVPR 2025.** Sun ve arkadaşları CloudFixer’a girişte ve ilgili çalışmalar bölümünde [46] ile atıf verir. Donmuş nokta ve metin kodlayıcılarından önce sınıf tahmini çıkarılır. Tahmin edilen sınıf için küresel özellik ve sözde etiket belleğe alınır. Kapasite dolunca daha düşük entropili örnek, en yüksek entropili örneğin yerini alabilir. Yerel bellek, aynı örneklerin yama özelliklerini k-means merkezleriyle özetler. Yeni örneğin küresel ve yerel benzerlikleriyle ağırlıklandırılan sözde etiketler, sıfır-örnekli sınıf puanlarına eklenir. Çalışmanın kendi Tablo 1’i, ULIP ile Ren ve arkadaşlarının **yedi** bozulmalı ModelNet-C verisinde, şiddet 2’de %47,52 kaynak ve %53,70 hiyerarşik bellek doğruluğu verir. Ana metinde veya ek tablolarda CloudFixer doğruluk satırı bulunmadı. Yukarıdaki %66,73 ise **Uni-Adapter yazarlarının** Uni3D-Large, 15 bozulma, şiddet 5 deneyidir. [Point-Cache, §2–4 ve Tablo 1](https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_Point-Cache_Test-time_Dynamic_and_Hierarchical_Cache_for_Robust_and_Generalizable_CVPR_2025_paper.pdf) · [Ek materyal](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Sun_Point-Cache_Test-time_Dynamic_CVPR_2025_supplemental.pdf)

**[Paper] BayesMM, CVPR 2026.** Zhu ve arkadaşları §2 ve kaynakçada CloudFixer’ı [22] olarak tanımlar. Her sınıf için metin istemlerinden bir Gauss dağılımı ve dilsel önsel oluştururlar. Sıradaki nokta bulutunun özelliği geldikçe geometrik dağılımın ortalama ve kovaryansını çevrimiçi güncellerler. İki dağılımın sınıf tahminlerini Bayesçi model ortalamasıyla birleştirirler. Ek A, ModelNet-C’de 64 genel isteme ek olarak her sınıf için 50 üretilmiş betimleme kullanıldığını belirtir. Ana ve ek tablolar sıfır-örnekli model, Point-Cache ve BayesMM’yi karşılaştırır. CloudFixer satırı bulunmadı. [BayesMM, §2–4 ve Tablo 1](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhu_Adapting_Point_Cloud_Analysis_via_Multimodal_Bayesian_Distribution_Learning_CVPR_2026_paper.pdf) · [Ek A ve C](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Zhu_Adapting_Point_Cloud_CVPR_2026_supplemental.pdf)

[Paper/Inference] İki çalışmanın ULIP karşılaştırması, paydanın önemini gösterir. BayesMM’nin Tablo 1’inde kaynak %48,60, Point-Cache %55,02, BayesMM %59,42. Bu üç ortalamaya temiz veri sütunu da girer. Point-Cache’in %47,52 ve %53,70 değerleri aynı kaynak ve bellek satırlarındaki yedi **bozuk** sütunu ortalar. Üç ScanObjectNN bölümü de ayrı tutulmalıdır. Bu bağımsız deney düzenleriyle BayesMM − CloudFixer farkı hesaplanamaz. [Point-Cache, Tablo 1](https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_Point-Cache_Test-time_Dynamic_and_Hierarchical_Cache_for_Robust_and_Generalizable_CVPR_2025_paper.pdf#page=6) · [BayesMM, Tablo 1](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhu_Adapting_Point_Cloud_Analysis_via_Multimodal_Bayesian_Distribution_Learning_CVPR_2026_paper.pdf#page=5) · [Önceki 48 eşleşmenin denetimi](../pointcloud_tta_reporting_audit_20261025.md)

## Yakın tarihli yayınlar ve kapsam sınırı

| Yayın | Birincil kaynaktan doğrulanan ilişki |
|---|---|
| Huang vd., PTA, ICML 2026 | Nokta bulutlarında ULIP ve Point-Cache ile aynı yedi bozulmalı protokol. Tablo 3’te ModelNet-C ortalamaları sırasıyla %47,52, %53,70 ve PTA için %55,13. İncelenen tam metin ve kaynakçada CloudFixer veya Shim görünmüyor. [ICML yayını](https://proceedings.mlr.press/v306/huang26cc.html) · [Yazar sürümü, Tablo 3](https://arxiv.org/html/2604.21360v1#S4.T3) |
| Jiang vd., TTAD, Ekim 2026 | CloudFixer’a §2.2.1’de ve kaynakçada [27] ile atıf veriyor. Tahmini gürültü artıklarından örneğe özel gürültü dağılımı çıkarıyor ve denoiser’ı oluşturduğu çiftlerle güncelliyor. Ölçüt geometrik gürültü giderme kalitesi. [Yöntem ve deneyler](https://arxiv.org/html/2610.27410v1#S4) |
| Mohammadi vd., MELD, 13 Kasım 2026 ön baskısı | §1 ve §6’da Shim vd. (2024) atfı doğrulandı. Etiketsiz hedef LiDAR taramasında PointNet++’ın dört özellik seviyesine ayrı difüzyon modelleri eğitiliyor. Teras segmentasyonunda bölgeler arası F1 ölçülüyor. CloudFixer karşılaştırma tablosunda yer almıyor. [§4–5 ve kaynakça](https://arxiv.org/html/2611.18033v1#S5) |
| ATTA-4DSeg, AAAI 2026 | İlgili çalışmalar paragrafında CloudFixer adı geçiyor, fakat yanında Zhang vd. (2021) yazıyor. Kaynakçada Shim, Kim ve Yang’ın makalesi saptanmadı. Ad eşleşmesi doğrudan bibliyografik atıf olarak sayılamaz. [AAAI makalesi](https://ojs.aaai.org/index.php/AAAI/article/download/39277/43238) |

[Open] Bu inceleme nesne düzeyindeki sınıflandırma karşılaştırmalarını 18 Kasım 2026 itibarıyla kapsar. Üç Uni3D-Large tablosunda nesne bazında eşli ham tahminler, seçilmiş CloudFixer† ayarları ve sınıflandırıcıyı güncelleyen CloudFixer-O koşulu doğrulanmadı. Bunlar, farkın ne kadarının aynı geometri düzeltmesiyle ve başka kaynak modellerinde korunduğunu sınamak için gereklidir. Tez deposunun burada kullanılan taban revizyonu `9538768`. Hesaplar basılı değerlerden türetilmiştir.
