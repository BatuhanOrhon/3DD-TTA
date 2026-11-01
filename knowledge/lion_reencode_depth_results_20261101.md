# LION ikinci geçiş N5/N8 sonuçları — 2026-11-01

[Run/Inference/Decision] İkinci geçişi uzatmak bu pilotta accuracy kazancı sağlamadı. Sabit ilk geçiş PM referansı %60,520833 iken taze N3 ikinci PM %57,916667, N5 %56,562500 ve N8 %55,312500 oldu. Birleşik PM hiçbir koşulda ilk geçişi aşmadı. Baseline değişmedi; ek GPU taraması başlatılmadı.

## Sabit karşılaştırma

ModelNet40-C/severity5, önceden açığa çıkmış aynı64 orijinal nesne ×15 bozulma, seed0, Point-MAE/B32: her koşul960 nesne-bozulma olayıdır,960 bağımsız nesne değildir. N3 pilotunun tam N8/Background35 ilk çıktıları, yeniden kodlanan z1/h1, posterior Gaussians, local Gaussian ve ikinci geçiş öncesi RNG fiziksel arşivden aynen kullanıldı. İlk geçiş/encoder yeniden çalıştırılmadı. İlk PM ve cycle logits arşiv referansıdır; yeni inference yalnız ikinci local zincirdir.

İkinci decoder z1’i sabit kullanır; global diffusion kapalı, SCD hedefi yeni h1XYZ, gamma=.03/eta=.003/p=.95, DDIM100/final-alpha1, raw/eval LION/EMAoff ve frozen classifier korunur. N3 timestep20/10/0, N5 40/30/20/10/0, N8 70..0 by10. Başlangıç noise katsayıları yaklaşık .07915/.14224/.23464’tür. Dolayısıyla bu horizon/noise derinliği ve guidance sayısı karşılaştırmasıdır; sabit noise seviyesinde yalnız sayısal çözüm adımı sayısı değişmemektedir.

Dört eşit PM adayı: N3 clean1/2/3+terminal3, N5 clean2/3/4+terminal5, N8 clean2/4/6+terminal8. N3 clean3 pre-guidance ve terminal3 post-guidance farklıdır. Birleşik PM her geçişe .5 ağırlık verir. İkinci terminal sonuçları PM değişiminden ayrı tutuldu. Entropy, ek seed veya başka horizon seçilmedi.

## Accuracy ve eşlenmiş farklar

Sabit ilk PM581/960=%60,520833; ilk terminal574/960=%59,791667.

| İkinci geçiş | İkinci terminal % | İkinci PM (doğru/960) | Birleşik PM (doğru/960) |
|---|---:|---:|---:|
| Taze N3 kontrolü |57,187500|57,916667 (556)|59,895833 (575)|
| N5 |56,041667|56,562500 (543)|59,375000 (570)|
| N8 |54,375000|55,312500 (531)|59,479167 (571)|

| Fark | Puan | Koşullu %95 aralık | Düzelten / bozan |
|---|---:|---|---:|
| N5 ikinci PM − ilk PM |−3,958333|[−7,291667; −0,833333]|31 /69|
| N8 ikinci PM − ilk PM |−5,208333|[−8,958333; −1,770833]|34 /84|
| N5 ikinci PM − taze N3 |−1,354167|[−2,916667; +0,104167]|16 /29|
| N8 ikinci PM − taze N3 |−2,604167|[−4,895833; −0,520833]|23 /48|
| N5 birleşik PM − ilk PM |−1,145833|[−3,125000; +0,729167]|20 /31|
| N8 birleşik PM − ilk PM |−1,041667|[−3,020833; +0,833333]|20 /30|

Aralıklar10,000draw/seed20261101 ile64 orijinal nesneyi,15 bozulmayı birlikte tutarak cluster bootstrap eder. Tek seed, exposed cohort, fixed corruptions koşulludur; native tekrar varyansı ve çoklu karşılaştırma düzeltmesi içermez. N5−N3 ve birleşik PM aralıkları0 içerir. Evrensel zarar veya benchmark anlamlılığı iddiası yapılmaz.

N5, taze N3’e göre3 bozulmada artar/5 eşit/7 düşer; N8 için1/5/9. N5’in en büyük kayıpları Upsampling ve Density-inc−6,25; N8’de Density-inc−7,8125 ve Shear−6,25 puandır. [Tam bozulma tablosu](evidence/lion_reencode_depth_20261101/per_corruption.csv), [karşılaştırmalar](evidence/lion_reencode_depth_20261101/depth_comparison.json).

## Teknik kontrol ve sayısal tekrar sınırı

İlk32-ID Gaussian/Background N3 kontrolünün logit eşleşme kapısı başarısız oldu. Bütün başlangıç posterior/noise/noisy-latent/style/RNG hash’leri aynıyken ilk prior tahmininden itibaren fark vardır. Başarısız7.47MB ZIP ve orijinal failed ledger korunur; başarısız sonuç tamamlanmış gibi yeniden etiketlenmedi.

Aynı noisy input/style ile dört prior forward çağrısında, ağırlıklar/RNG/eval modu değişmeden farklı çıktılar ölçüldü: ilk çağrıya göre max_abs .001667/.002379/.002379. Tek bir CUDA operator nedeni izole edilmedi. Bu kanıttan sonra operator v2 exact fiziksel/host kapılarını korur, bütün sayısal uyuşmazlıkları raporlar ve64-ID taze N3 kontrolünü ekler. Başarılı v2 technical3/5/8 ve pilot3/5/8 yeni UTC dizinlerindedir; original failure veya bütçe sıfırlanmadı.

Taze N3 PM %57,916667, arşivli N3 %58,020833’ten−.104167puan farklıdır.960 olayda PM7, terminal14, aday düzeyinde61 etiket uyuşmazlığı vardır; PM accuracy açısından1 düzelme/2 bozulma kalanı yanlış-sınıf değişimidir. Birleşik PM %59,895833 vs eski %59,583333 de tekrar farkıdır. Yeni tabloda eski %58,020833 taze kontrolün yerine kullanılmadı.

Exact cache/posterior/Gaussian/RNG/frame/noisy-state/scheduler/decoder-z/new-h-anchor/ID/label/hash/frozen-weight/shared-Python/resolved-config/native kimlikleri kabul edildi. Remote ve bağımsız yerel NumPy1 kabul JSON’ları altı başarılı fazın tamamında exact eşittir.3 technical ZIP’in her biri122,3 pilot ZIP’in her biri219 manifest üyesiyle SHA/CRC/path/type kontrolünden geçti. Başarısız ilk technical118 üye olarak ayrıca doğrulandı.

Generic shared-worker `preparation_mode=fresh_encode` alanı yeni cache yolunun başlangıçta bulunmamasını yansıtır; bu replay entrypoint’inde encoder çağrısı yoktur. `shared_reconstruction`, `reencode_replay`, `reference_files` ve yürütülen kaynak gerçek second-only davranışı açıkça kaydeder. Ham alan değiştirilmedi.

## Yorum ve karar

Yeni z’nin temiz referansa MSE’sindeki %5,140506 azalma encoder köprüsünde oluşmuştu; üç koşulda aynı z kullanıldığı için bu sayı burada tekrar bir z iyileşmesi değildir. Root-MSE biriminde %2,604161 azalma ve olayların %56,875’inde yakınlaşma, görev accuracy’sine dönüşmüyor. İkinci terminal de adım sayısıyla düşer; yalnız PM checkpoint seçimini kaybın tek nedeni sayamayız.

[Inference/Decision] Bu ayarlarda üç adımın az kalması açıklaması desteklenmedi. Daha derin ikinci noise/denoising politikası kaybı gidermedi. Yeniden kodlanan h, anchor reset, local prior/noise veya global/local eşleşmenin tek nedensel katkısı izole edilmedi. z-only/h-only/anchor ablation, sabit-noise çözüm yoğunluğu, ek seed/unseen doğrulama ve entropy weighting yapılmadı. N5/N8 baseline’a alınmaz; daha yüksek horizon kendiliğinden taranmaz.

## Kaynak, maliyet ve koruma

Bilimsel runner/source commit c2a8eb8f18a0e187d6d3c72f7269ff1069666c97; assetmanifest8fb687147d87317f1ed1b18b43549b2a06a19ce81124b6f44cdf1b36f3c24d28; payload68998c2db1ef2795e701416114cd7ab282eba039e76a51566fd1b31398e09b58. Üç pilotun executed-source kimliği e7ef8a4967bdce3bb11e68297cf9b8c9fd225adfb9aa0fd7c069f65dbb9c571d, birebir aynıdır. Operator v2 SHA31417d83b175cc3ce301a44863496c400b8341dd0733debbb293471854e2a309; v2 analiz/driver kaynakları her phase ZIP’inde, original/v2 operator paketleri ayrıca korunur. Son raporlama/knowledge commit’i bilimsel source commit’inden ayrıdır.

8yeni kontrat, tests/648pass+412subtests/1historical skip; comparison self-check nesne cluster’ını korur. İki reviewer guard kusuru GPU öncesinde RED→GREEN düzeltildi; runtime tekrar tanısı sonrası raporlama regresyonu da RED→GREEN geçti. Bare pytest’in daha önce belgelenen vendored Hydra/CUDA/Python3.13 collection sınırları değişmedi; yeni broad vendored inference yapılmadı. Canonical ortam aynı A100 VM’de yeniden kullanıldı; strict CPU/no-forward/no-CUDA, pinned native ve data/checkpoint deployment kanıtı passed. Yeni VM veya setup replay gerekmedi.

N3/N5/N8 second-only reconstruction/probes/cache/geometry toplam46,259856/56,307062/71,470274s; worker+artifact IO83,334869/93,238171/108,313963s. İlk geçiş/encoder/model loading bu reconstruction sürelerinde yoktur; tam pipeline hız karşılaştırması değildir. Bilimsel wall ledger, failed ilk technical ve tanı/recovery dahil911,685022s=15,194750dk;10–20dk tahmin/55dk cap içindedir. Setup ve sonraki yerel CPU/transfer/Git işleri ayrıdır.

Yerel hamlar result/preparation/lion_reencode_depth_20261101/operator_v2/{smoke3,smoke5,smoke8,pilot3,pilot5,pilot8}_raw.zip; ilk failure operator/smoke3_raw.zip. Derived analizler ayrı *_local_analysis dizinlerindedir. Canonical head ZIP’leri result/modelnet40_c/3dd_lion_reencode_n{3,5,8}_replay/ altında değişmemiş hardlink’lerdir. External N3 ilk geçiş/posterior335MB arşivi önceki çalışmada korunmuş ve bu kayıt içinde pinlenmiştir.

Drive MyDrive/thesis/result/lion_reencode_depth_20261101/phase-evidence; source-v2 altında exact asset/operator paketleri. [Makbuzlar ve üye kabulü](evidence/lion_reencode_depth_20261101/member_acceptance.json).

| Pilot | Byte | SHA-256 |
|---|---:|---|
|N3|71,101,883|20b0a95f6c3746f740b1a73dde8b44d7255a98afc9f19b10a8fb0cd29314b74c|
|N5|71,102,017|daecbffaa130a632e00a619b8bce77138efae09d8ec01a0ed7a6b8214f5f25c3|
|N8|71,095,508|76df8381de9a0471468ff13486851ef772cf679a2a837c35c2560865eecbfb41|

Integration scope: source branch research/lion-reencode-20261101, base6ff2e49; yeni opt-in replay script/test, bağımsız analiz/kompakt kanıt, tarihli knowledge/plan. Temporary wrappers, setup/native/raw/cache/weights Git dışında yerel+Drive’da korunur. Kullanıcı/diğer agent çalışma alanları alınmaz. Normal fast-forward origin/thesis-integration hedeflenir; oluşan commit/push/final Drive capsule roundtrip makbuzları result/preparation/lion_reencode_depth_20261101/publication/ altında ve final Drive kapsülünde kaydedilir. Runtime tta-lion-reencode-a100-20261101-r1/A10080GB/High-RAM korunur; completion+close izni yoktur.
