# Aday PM nasıl çalışıyor? 25 noktayla t50 dönüşü

28 Kasım 2026. [Code/Run] Bu not, Point2Vec tam testinin ardından sorulan iki soruyu yanıtlar. Birincisi aynı nesnede dönüşün nasıl seçildiği, ikincisi bu seçimin çıkarım süresine etkisidir. Geometri örneği CPU üzerinde `OutputPose.update`, `correct` ve gerçek yüzey seçicisiyle doğrulandı. ModelNet tahminleri ise ayrı, arşivlenmiş deneylerden geliyor. Başlangıç `origin/thesis-integration@59a390e`. [Tam test ve baseline kararı](cloudfixer_surfacefull_gpu_20261124.md#tam-test-sonucu) · [Yüzey yönleri için literatür kaydı](papers/straub2014_manhattan_frames.md)

## 1. Noktalardan ana eksenlere

PCA, merkezden çıkarılmış nokta koordinatlarının hangi yönlerde en fazla ve en az yayıldığını bulur. Birim bir yön boyunca her noktanın izdüşümünün karesini alıp ortalarsak o yöndeki varyansı elde ederiz. Kovaryans matrisinin en büyük özdeğerine karşılık gelen özvektör bu varyansı en yüksek yapan yönü verir. Diğer özvektörler ona dik olan diğer ana yönlerdir.

Sayısal örnekte uzun, düz bir levhanın 25 noktasını kullanıyoruz. `u ∈ {−6, −3, 0, 3, 6}` ve `v ∈ {−2, −1, 0, 1, 2}` için birer `(u,v,0)` noktası var. Önce u, sonra v artacak şekilde sıraladık. XZ kesitinde levha 20° yukarı eğilmiş olsun. `c = cos(20°)` ve `s = sin(20°)` yazalım. Gözlenen ham nokta `X = (cu, 0,9v, su)`. Denoiser’ın t50 temiz tahminini örnek için `q = (cu, v, su)` alıyoruz. Böylece hem yön hem SCD’nin düzelteceği küçük bir koordinat farkı var. Eksene paralel levha yalnız örneğin kuruluşunda kullanılır. Algoritmaya verilen iki bulut X ve q’dur.

q’nun merkezi sıfırdır. u’nun karelerinin ortalaması 18, v’ninki 2, uv’nin ortalaması sıfırdır. Dolayısıyla `C = qᵀq/25` şöyledir.

```text
             x        y        z
x        15,8944      0     5,7851
y         0           2     0
z         5,7851      0     2,1056
```

x ve z birlikte arttığı için çapraz terim pozitiftir. En büyük özdeğer 18 ve uzun yön `(c,0,s)`, ikinci özdeğer 2 ve yönü `(0,1,0)`, üçüncü özdeğer 0 ve düzlem normali `(−s,0,c)`. Bir normalin işaretinin ters çevrilmesi aynı düzlemi tarif eder. Kodun kullandığı ölçeklenmemiş `qᵀq` özdeğerleri 450, 50 ve 0 verir. Yönler ve eşik için kullanılan oran aynıdır. [PCA hesabı](../cloudfixer_rotation.py#L37)

## 2. t50’de dönüş önerme ve ham yüzeyden onay alma

N8 zinciri 80 → 70 → 60 → 50 → 40 → 30 → 20 → 10 → 0. Dördüncü geçişte denoiser gürültü tahmini ε verir ve `q = (z₅₀ − σ₅₀ ε)/α₅₀` temiz şekil tahmini elde edilir. PCA bu SCD öncesi tahmini kullanır.

1. PCA eksenlerinin işaret ve sıralama belirsizliğinden gelen 24 sağ-elli olasılıktan mevcut duruşa en yakın dönüş seçilir. Örnekte bu, eğimi geri alan 20° dönüşüdür. Nesne sınıfı ve gerçek bozulma açısı seçime girmez.
2. Açı 5°–30° aralığındadır. Sıralanmış özdeğerlerin en küçük ardışık farkının en büyük özdeğere oranı `min(2−0,18−2)/18 = 0,1111` olup 0,05 sınırını geçer. Koddaki sabit sıra tohumu 20261122 ile 13 ve 12 noktaya ayrılan q’da iki ayrı PCA önerisi arasında 9,38° vardır. İzin verilen üst sınır 10°’dir.
3. Girdi X’te tekrarlar kaldırılır. En çok 256 merkez, her merkezde kendisi dışındaki 16 en yakın nokta kullanılır. Bu komşuluğa da PCA yapılır. Küçük özdeğerin özvektörü yerel yüzey normalidir. Artan özdeğerleri μ₀, μ₁, μ₂ olarak yazarsak yüzeyin ağırlığı `(μ₁−μ₀)/μ₂` olur (kod sıfır payda için koruma içerir). Her normal için hizalanma skoru, en iyi hizalandığı koordinat eksenindeki bileşenin karesidir.
4. Levhada her yerel normal `±(−s,0,c)` olur. Mevcut pozda skor `c²=0,88302`, dönüşten sonra `1`. Merkezleri ikiye ayırınca her iki grubun ağırlıklı skoru 0,11698 artar ve 0,015 eşiğini geçer. Kod kararı `accepted`. Aynı levhayı XY düzleminde 20° döndürerek kontrol ettik. Ana eksenler yine 20° önerdi ve yarım bulutlar yine 9,38° ayrıştı. Fakat normaller başlangıçta da Z’ye paraleldi. Her yarının skoru 1 → 1 oldu ve kod `surface_veto` verdi. [Geometrik kontroller](../cloudfixer_rotation.py#L102) · [Yüzey skoru](evidence/cloudfixer_rotation_refocus_20261122/frame_check.py#L24)

## 3. Tek noktada SCD, dönüş ve PM

`u=3, v=1` için q = (2,81908, 1, 1,02606), referans X = (2,81908, 0,9, 1,02606). Bu eşleşme %95 çift yönlü kırpma sonrasında her iki yönde de tutuldu. Proksimal SCD’de τ = 0,1. Bu noktanın yeni y değeri `(1 + 0,1×(0,9+0,9))/(1 + 0,1×2) = 0,98333`. x ve z zaten eşleştiği için aynı kalır. Genel kural, korunan hedef koordinatların toplamını ve her noktanın gerçek eşleşme sayısını kullanır. [SCD güncellemesi](../pointcloud_proximal.py#L5)

Kabul edilen dönüş, satır vektörleri için `x′ = cx+sz`, `y′ = y`, `z′ = −sx+cz` verir. SCD sonrası örnek nokta (3, 0,98333, 0) olur. Referans noktası (3, 0,9, 0) olur. Gürültü tahmini ε da aynı dönüşü alır. Sonraki durum `z₄₀ = α₄₀(q_SCD R) + σ₄₀(ε R)` olarak kurulur. İlerideki SCD de döndürülmüş referansı kullanır. Böylece yerel nokta düzeltmesi ile nesnenin tamamına uygulanan rijit dönüş ayrı işlemlerdir. [Sampler](../cloudfixer_diffusion.py#L134)

PM’de Point2Vec, t20 ve t10’daki SCD öncesi temiz tahminleri ve gerçek t0 terminalini okur. 40 sınıfın her biri için üç softmax olasılığı eşit ağırlıkla ortalanır. En yüksek ortalama son sınıfı verir. [Sınıflandırma ve ortalama](../cloudfixer_rotation.py#L454)

## 4. Gerçek Point2Vec örnekleri ve süre

[Run] 512 nesneli eşli pilotun Rotation / seed 2 kayıtlarında ID 1450’nin gerçek sınıfı yataktı. 22,81745° kabul edildi. Aşağıdaki değerler aynı nesne ve seed için kaydedilmiş üç logitten hesaplandı.

| Yöntem | Yatak t20 (%) | Yatak t10 (%) | Yatak t0 (%) | Yatak PM (%) | En yüksek PM |
|---|---:|---:|---:|---:|---|
| Baseline | 10,59 | 15,06 | 12,55 | 12,73 | Merdiven |
| Aday | 83,34 | 82,78 | 83,94 | 83,35 | Yatak |

ID 633’te (klozet) yüzey kontrolü öneriyi reddetti, arşivde üç logit vektörü eşit ve iki PM de %80,71. Kaynak `result/preparation/surface512_20261123/drive_verified/20261123-192004_feedback_surface512.zip`, `raw/block_1218.npz` ve `raw/block_1219.npz`, pilot SHA-256 `a273d78157fa9e45a2aa841c541ae8cfc4929b9d3c345aea3283ee3379bab807`.

| A100 B1 eşli zamanlama | Çift | Baseline medyan ms | Aday medyan ms | Baseline P95 ms | Aday P95 ms |
|---|---:|---:|---:|---:|---:|
| 15 bozulma | 1.440 | 245,2541 | 250,5456 | 354,4494 | 358,3336 |
| Rotation | 96 | 244,4519 | 249,1247 | 254,9987 | 272,0314 |

All-15 için medyan +%2,16 ve P95 +%1,10. Rotation için medyan +%1,91 ve P95 +%6,68. Bu oranlar, aynı panelde yöntem başına hesaplanan yüzdeliklerden gelir. Tam kümenin yeni 29.340 aday kaydında medyan 247,99 ms ve P95 352,44 ms ölçüldü. Tam taramada baseline yeniden zamanlanmadı. 37.020 tam test girdisinin 1.433’ünde, Rotation’daki 2.468 girdinin 903’ünde dönüş kabul edildi. Diğer source modeller için sonuç bu aşamada açıktır. [Pilot ölçümleri](cloudfixer_surface512_gpu_20261123.md) · [Tam test sayımları](evidence/surfacefull_20261124/full_comparison.json)
