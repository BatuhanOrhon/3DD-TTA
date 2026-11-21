# CloudFixer XYZ baseline kararı ve rotation/background araştırma yönü

21 Kasım 2026. [User report/Decision/Run/Code/Inference/Open]

## 1. Yeni çalışma referansı

[Decision] Sonraki yöntem geliştirmesinin başlangıcı **CloudFixer XYZ diffusion +
gradient-free proximal SCD + geç eşit Probability Mean**, kısa adı
`cloudfixer_xyz_proximal_pm_n8_k80_late_v1` olarak sabitlenmiştir. Bu ad yeni bir
cache/manifest sürümü değildir; mevcut ve GPU'da çalıştırılmış `proximal_pm`
kolunun araştırma referans adıdır. **N8/K80/τ=.1/retained=.95; pre-clean native
t=[20,10] ve gerçek t0 terminalinin eşit olasılık ortalaması [20,10,0].**
Dokuz denoiser, üç classifier çağrısı; float64 softmax ve probability mean.
Classifier ve prior donmuş; LION, spectral terim ve entropy ağırlığı yoktur.

[Decision] Bu, kendi yöntemimizi geliştirmek için seçilen **operasyonel baseline**.
Başarı kıyasının ana referansı hâlâ **orijinal plain CloudFixer**: aynı frozen
Point2Vec/XYZ checkpoint'leri ve yazar seed2/B1 akışında %77.244733. Başlangıç
yöntemimiz %74.954079 ile geridedir. Daha düşük sonucu referans seçmek CF'a
üstünlük iddiası değildir. Source kontrolü ve N32/K160/τ=.03 gerçek terminal
negatif/alternatif kontroller olarak korunur. CF'ın ikinci vote1 forward'ı
diagnostic; ana karşılaştırma ilk plain çıktıdır.

[Code/Decision] Eski `lion_proximal_pm_n8_v1` kodu ve sonuçları tarihsel ayrı
prior yolu olarak korunur. Yeni araştırmanın giriş noktası
`scripts/run_cloudfixer_point2vec.py`, `POLICIES['proximal_pm']` ve
`proximal_policy()` fonksiyonudur. Eski LION CLI'sini çalıştırmak yeni baseline'ı
çalıştırmaz. Bu kararda GPU'da kullanılan sampler/runner/registration baytları
değiştirilmedi; eski arşivlerin kimliği ve RNG devamı korunur. Yeni bir sampler,
genel wrapper veya etikete göre corruption yönlendirmesi eklenmedi.

## 2. A ve B bu kararı nasıl destekliyor?

[Run] A: tam ModelNet40-C, 2468 nesne ×15 corruption ×s5; seed2/B1, frozen
Point2Vec. Source 22947/37020=%61.985413; N8 geç PM 27748=%74.954079;
N32 terminal 27111=%73.233387; plain CF 28596=%77.244733. N8−CF
−2.290654 puan, original-ID paired nominal %95 GA [−2.663425,−1.925918].
N32−CF −4.011345 [−4.513844,−3.506213]. Tek seed ve dört VM'de devam etmiş
tek native gerçekleşme; aralıklar seed/optimizer varyansını kapsamaz.

[Run/Inference] N8 PM, N32 terminalden +1.720691 puan ve daha düşük kayıtlı
yöntem süresiyle öndedir: 9071.108s vs25822.797s; CF29013.512s. CF'a göre
yaklaşık3.20× süre oranı setup/I/O/gate/diagnostic dışıdır; uçtan uca hız sonucu
değildir. A'nın N/K/τ/readout farkı PM'nin izole katkısını ölçmez.

[Run] B bunu ayrı backbone/protokolde izole eder: Point-MAE, ayrılmış256
nesne, tüm15/s5, seed0/1/2/B32, K80. Kilitli geç PM kendi terminaline karşı:

| Schedule | Terminal % | Geç PM % | Δ puan, nominal %95 GA |
|---|---:|---:|---|
| N8/τ=.1, [20,10,0] |72.005208|72.907986|+.902778 [.564236,1.241319]|
| N12/τ=.1, [27,0] |70.416667|72.144097|+1.727431 [1.276042,2.170139]|
| N12/τ=.8÷12, [27,0] |71.927083|72.586806|+.659722 [.295139,1.032986]|

[Run/Inference] Dokuz seed nokta farkı pozitiftir; yedi nominal aralık sıfırı
dışlar. N8 geç PM15/15 corruption'da kendi terminalinden yüksek. Erken kanonik
ve tüm-adım PM ayara göre zarar verebilir. N8 kanonik PM'nin bozduğu303
terminal-doğru olayın ilk probe'unda252 kez nihai yanlış rakip top1'dir; erken
doğru oylar da241 terminal hatasını onarır. Bu nedenle geç eşit pencereyi koru,
yeni bir entropy seçicisi veya daha çok erken oy ekleme.

[Run/Inference] Pooled entropy AUC≈.87, aynı yörünge içinde≈.64–.66;
entropy-ağırlıklı geç PM tutarlı ilave kazanç vermedi. N12'de τ'yı Nτ=.8
olacak şekilde küçültmek terminali +1.510417 puan geri kazandırdı; N8'e göre
−.078125 puanlık farkın aralığı sıfırı içerir. Bu eşdeğer dinamik veya genel
optimum kanıtı değildir. Ek t0 çağrısı N8/N12/.1'de kendi gürültülü girdisine
göre +.286458 puan kazandırdı; t0'ı kaldırmak varsayılan çözüm değildir.

[Decision/Limit] N8, burada test edilmiş kendi politikalarımız içindeki
accuracy/maliyet dengesi ve B'deki kilitli readout desteğiyle çalışma referansı
seçilmiştir. Geç pencere geliştirmede posthoc seçilmiş, B'de önceden kilitli
nesne-ayrık doğrulama görmüştür; global optimum veya Point2Vec'te izole PM
kanıtı değildir. N12 güçlü-PM kazancı mutlak en yüksek doğruluk anlamına gelmez.

## 3. Rotation ve background: sayılar ve aşılması gereken eşik

[Run] Her satırın paydası2468. Tam tablo ve bütün15 bozulma
[A/B ayrıntılı raporunda](cloudfixer_ab_detailed_tests_results_20261121.md).

| Bozulma | Source doğru (%) | N8 geç PM doğru (%) | N32 terminal doğru (%) | CF plain doğru (%) | N8−CF net doğru |
|---|---|---|---|---|---:|
| rotation |1158 (46.920583)|1186 (48.055105)|1109 (44.935170)|1715 (69.489465)|−529|
| background |909 (36.831442)|1612 (65.316045)|1887 (76.458671)|1878 (76.094003)|−266|

[Run/Inference] Toplam net açık848; iki satır795, yani **%93.75**'ini oluşturur.
Kalan13 bozulmanın net açığı53'tür (PM6/15 bozulmada CF'ın üstünde). Bu oran
net sayılara aittir; bütün CF-correct/PM-wrong olayların %93.75'i demek değildir.
Bu iki bozulmada düzelme olasılığı, iyileşme yönü veya nedensel katkı ölçmez.

[Inference/Arithmetic] Diğer13 sonuç aynı kalır ve bu iki accuracy tam CF'a
eşitlenirse PM27748+795=28543/%77.101567 olur; CF28596/%77.244733'ün hâlâ
53 olay/0.143166 puan gerisindedir. **Yalnız bu iki bozulmayı değiştirerek
mevcut CF nokta sonucunu kesin geçmek için ≥849 net ek doğru gerekir.**
İki bozulmanın toplamı2798'den en az3647/4936=%73.885737'ye çıkmalı;
ortalama iyileşme ≥17.200162 puan. Bu, ikisinin CF toplamı3593'ten54 fazla
doğru demektir. Başka bozulmalarda her net kayıp bu eşiği aynı miktarda artırır.
Yalnız rotation iyileşirse toplamda ≥2035/2468 doğruya ulaşması gerekir;
yalnız background iyileşirse ≥2461/2468 gerekir. Bunlar koşullu aritmetik
hedeflerdir, uygulanabilirlik ya da istatistiksel üstünlük kanıtı değildir.

[Run/Inference] Background'da N32, CF'ı9 olay/.364668 puan geçer; rotation'da
N32 source'un49 olay altındadır. Bu, sabit horizon/başlangıç/güç/readout
etkileşiminin bozulmalar arasında heterojen olduğunu gösterir; hangisinin
nedensel etkeni olduğunu ayırmaz. Etiketten/corruption adından N32-background
ve N8-diğer şeklinde sonradan oracle politika oluşturulmaz.

## 4. Geometrik açıklamalar: kodda ne var, ne henüz kanıtlanmadı?

[Code] Resmî `adapt.cloudfixer()` (authorf043e244…): her örnekte6D rotation
ve per-point delta için30 Adamax güncellemesi; `(x+delta) @ R`. Denoiser
hedefi randomt∈[.02,.12] üzerinde `no_grad`; bidirectional Chamfer matching
ve kNN yoğunluğundan `1/mean_squared_distance^pow` ağırlıklı delta cezası,
λ10→1. Rotation ve delta farklı öğrenme hızlarıyla açıkça optimize edilir.
Kaynak ayrıntısı [protokol auditinde](cloudfixer_paper_protocol_audit_20261115.md).

[Code] Bizim `cloudfixer_diffusion.correct()` yolumuz her pre-clean q için
**gözlenen bozuk x'e** en yakın komşuları kurar, her yönde%95 mesafe tutar ve
frozen quadratic'ı kapalı biçimde çözer. Ayrı bir global SO(3) değişkeni veya
kNN-yoğunluklu anchor cezası yoktur. Mask yalnız veri/loader'ın geçerli saydığı
referansları dışlar; bilinmeyen background noktaları için öğrenilmiş/ölçülmüş
bir temiz-inlier oracle'ı değildir. %95 trimming, background'ı kusursuz ayırma
garantisi vermez; yakın outlier'lar kalabilir, temiz uç noktalar elenebilir.

[Inference/Hypothesis R] Pose düzeltmesi olmadan bozuk/orientasyonu hatalı
anchor'a geri çekilmek, prior'ın global yön düzeltmesiyle çatışabilir.
Rotation'da N8'in source'a yalnız28 olay eklemesi, CF'ın557 eklemesi bu
hipotezle uyumludur. Fakat CF−N8 farkı rotation optimizasyonunun izole etkisi
değildir: amaç, randomt, optimizer, delta ve zincir de farklıdır.

[Inference/Hypothesis B] Background anchor'ları ve normalizasyon/FPS'deki
etkileri, düzeltmeyi outlier'ları korumaya zorlayabilir. Yoğunluk destekli
anchor güveni, asimetrik/robust eşleşme veya ayrı bir pose+inlier yaklaşımı
incelenebilir. Büyük N32 background artışı tek başına bu açıklamayı kanıtlamaz;
ön işleme, prior ve sampling etkileri ayrılmalıdır.

[Open] Prior'ın pose duyarlılığı, classifier'ın pose duyarlılığı, ilk anchor'ın
kirlenmesi ve düzeltmenin prior'a karşı çalışması ayrı sorulardır. Nihai
accuracy bunlardan hangisinin baskın olduğunu belirleyemez. PCA/Procrustes
tek başına canonical yön garantilemez; simetri, eksen permütasyonu/işareti,
eşleşme ve reflection sorunları ayrıca kontrol edilmelidir.

## 5. Sonraki aşama için sınanabilir inceleme

[Proposal/Open] Bu kayıt yeni GPU fazı başlatmaz. Öncelik en küçük kontrollü
geometri teşhisidir; yeni yöntem başarısı henüz ölçülmedi.

1. **Girdi sözleşmesi:** aynı frozen prior/head/veri/normalizasyon/FPS/RNG ile
   bozuk-input, clean-pair (yalnız teşhis) ve pre/post q bulutlarını takip et.
   Clean etiket/pose/mask yalnız değerlendirme-oracle teşhisinde; inference'a
   girmez. Gerçek corruption mask'i varmış gibi outlier kimliği üretme.
2. **Rotation ayrıştırması:** yazar CF'ın full/rotation-off/delta-off kollarını
   kontrollü karşılaştır. Kendi N8 hattında pose-off/pose-on ve τ=.1/τ=0
   kontrollerini eşleştir; yalnız pose bileşenini değiştir. SO(3) constraint,
   rigid mesafe/centroid kontrolleri ve simetriye duyarlı pose metrikleri kullan.
   Parametreleri prior hedefi/etiketsiz geometriyle belirle; sınıf etiketinden
   yön ya da örnek başına iyi adım seçme. FullCF rotation-off da zinciri
   değiştireceğinden sonuç yalnız o algorithm içindeki ablation'dır.
3. **Background ayrıştırması:** preprocessing'i sabit tutarak mevcut%95
   anchor kuralı ile tek bir etiketsiz density/robust-anchor alternatifini
   karşılaştır; τ=0 ve temiz-anchor oracle yalnız teşhis kontrolleri olsun.
   Inlier/outlier korunumu (bilinen sentetik eşleşme varsa), komşu çakışması,
   kalan nokta/yoğunluk dağılımı, displacement ve prior-fit'i ölç. Geometri
   kaybının azalmasını sınıflandırma kazancı sayma.
4. **Etkileşim:** baseline / yalnız pose / yalnız robust-anchor / ikisi
   birlikte. N/K/τ/geç PM kilitli, ek çağrıların maliyeti dahil. Pose ve
   anchor'ı aynı anda değiştirip tek neden iddia etme. Bütün15'te ortak
   inference kuralı; corruption adı veya severity üzerinden routing yok.
5. **Doğrulama:** A tam testinin sonuçları artık görülmüştür; yeni ayarların
   geliştirmesi kaynak temiz örneklerden ayrılmış nesneler ve bağımsız
   sentetik rotation/background ailesi üzerinde yapılmalı. Eğitim/pretraining
   maruziyetini dürüstçe kaydet; kullanılabilir ayrım doğrulanmadan disjoint
   iddiası kurma. Ayar/seed0,1,2/probe/bütçe ve ID manifestini önce dondur.
   Son all15/s5 ölçümü paired original-ID bootstrap, seed SD, repair/harm,
   süre/bellek ve kalan13'te gerilemeyle birlikte raporla. Aynı benchmark'ın
   yeniden ölçümü dış veri üzerinde bağımsız genelleme kanıtı değildir.

[Falsifier] Pose-only kontrollü kol rotation'ı iyileştirmezse açık SO(3)
eksikliği açıklaması zayıflar. Robust-anchor, geometriyi değiştirip background
accuracy/repair-harm dengesini iyileştirmezse anchor-kirlenmesi çözümü
desteklenmez. İki hedef kazanımı kalan13 kayıplarıyla silinirse all15 üstünlük
iddiası reddedilir. Nokta skorunu geçmek yeterli değil: önceden kilitli ortak
politikada CF'a karşı paired aralık, seed davranışı ve maliyet görülmelidir.

## 6. Kod, kanıt ve yeniden çalıştırma sınırı

[Code/Run] Tam uygulama zinciri: `pointcloud_proximal.py` →
`cloudfixer_diffusion.py` → `scripts/run_cloudfixer_point2vec.py`;
Point-MAE kontrolleri `scripts/run_cloudfixer_transfer.py`,
`scripts/run_cloudfixer_exploration.py`, `scripts/run_cloudfixer_hypotheses.py`.
`cloudfixer_store.py` atomik32-nesnelik payload/SHA/receipt ve RNG devamını
sağlar. Point2Vec asset manifest'i sınıf sırası/180 state anahtarı/strict
checkpoint kimliğini bağlar. Setup ve bütün analyzer/test dosyaları birlikte
aktarılır. Tek checkpoint yolunu değiştirmek backbone desteği sayılmaz.

[Code] Paketleme (CPU; yeni/boş output gerekir):

```bash
python scripts/prepare_cloudfixer_transfer.py \
  --official-checkout /Users/batuhan/Desktop/projects/CloudFixer \
  --classifier point2vec --output /tmp/cloudfixer_point2vec_baseline_package
```

Bu komut GPU çalıştırmaz. Point2Vec runner `--phase smoke` ve `--phase full`
ile dört dondurulmuş kolu birlikte çalıştırır; yalnız-baseline CLI olduğu
iddia edilmez. Gerçek koşu kanonik10 setup, native/checkpoint/data smoke,
task/environment/Drive receipt, boş kimliği doğrulanmış output/cache ve
kaydedilmiş bütçe/kurtarma planı gerektirir. Eski manifest'e yeni kod takılmaz.

[Run] Kabul edilmiş kaynaklar: A `9afa8436b1770e38e148f9093d3edd85a4d9770a`,
B `eec1e8b2fe272e9d1f42b6a628cacd5341f1c8c8`; iki sampler SHA'sı aynı
`3e3c66246af5dff79603f9263ab784040a32c7d0c0bde0fe398900ca862d6f27`.
Detaylı kayıt kaynak HEAD'i `4de6fdbd17a759e72c1d483b7e0c66828c19aaa2`.
Author commit `f043e24414fcf4165fdc892abff21215c9989141`.
Tam checkpoint/veri ve dört arşiv kimliği
[kaynak receipt'inde](evidence/cloudfixer_ab_record_20261121/source_receipt.json).

[Run] Sayısal eşik ve politika sözleşmesi
[recount.py](evidence/cloudfixer_baseline_20261121/recount.py) ile yeniden
sayılabilir; [karar ve sayılar](evidence/cloudfixer_baseline_20261121/decision.json).
Yeni sonuç/tuning veya hata maskesi türetilmedi; A/B arşivleri değişmeden kalır.
