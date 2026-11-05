# H5 güvenilirlik tanısı — 2026-11-05/06

**Durum [Run/CPU/Inference]:** Ham paket ve üç kontrol koşusu kabul edildi;
dar doğrulama düzeltmesinden sonra `analysis_v2` ve sabit bootstrap tamamlandı.
**H5 adayı ilerlemez:** native tekrar kapıları başarısız, combined AUROC0.2713
ve95% cluster aralığı[0.1376,0.3861]. İlk başarısız `analysis_v1` korunuyor.
Feature/bant/epsilon/tolerans/skor/advance kuralı veya ham arşiv değiştirilmedi.

## Kapsam ve ham kanıt

[Run] ModelNet40-C, PointMAE/LION, seed0, s5, B2; N8, Background B35.
Sekiz tarihsel olarak açığa çıkmış mekanizma ID'si:
459,545,1229,1435,1441,1557,1881,2095. Gaussian/Background/Shear/LiDAR;
üç yeni process; mevcut üç clean-PM ve terminal geometrileri. Control/weight0;
H5 q veya ek guidance uygulanmadı. PM sayımları yalnız tanısal kontrol
okumalarıdır; yöntem accuracy kazancı veya heldout doğrulaması değildir.

[CPU] Yerel ham paket:
`result/preparation/gsd_h5_reliability_20261105/h5_capture_raw_package.zip`.
1,391,073,853 byte;
SHA256 `ee58f23c92546858b7c82220e5c353f4aae68a6997df8e69289e3541dadbe105`.
Drive/local makbuzları ve altı parçanın her byte/SHA kimliği eşleşti.
Mevcut `checked_members/extract_checked` ile CRC/yol/tip/boyut kontrolleri,
47 ZIP üyesi ve46 manifest kimliği doğrulandı; taze `downloaded/` açıldı,
çıkarılan46 dosya yeniden hash'lendi. Ham ZIP yalnız okundu.

[CPU] Üç sealed koşu 20:48:33/20:50:51/20:53:10 UTC sırasıyla kabul edildi.
Her birinde176 executed-source kimliği, sabit model before/after, backend,
scheduler/protocol ve eşlenmiş prepared/reference/cache/RNG kontrolleri geçti.
Altı gerçek native ELF'nin byte/SHA'ları bütün config/bootstrap bağlarıyla
eşleşti.16 ayrı paired cache üç koşu için denetlendi. Kaynak overlay249 dosya,
`b766e61` H5 snapshot'ıdır; sonraki comprehensive overlay farklı kaynaktır.

## Native tekrar tabanı ve PM sayımları

[Run/CPU] Eski sayısal kapılar değiştirilmedi:
logitmax0.05, nesne/checkpoint RMS0.01, ilk decode RMS1e-4.
Üç repeat çiftinin tamamı başarısız; pair×B2 toplam27/48 geçer.

| Bozulma | Geçen pair×B2 | En büyük logit farkı | En büyük logit RMS | En büyük ilk-decode RMS |
|---|---:|---:|---:|---:|
| Gaussian |10/12|1.401933|0.651914|0.00425076|
| Background |0/12|4.939466|1.897221|0.06105568|
| Shear |10/12|0.641572|0.283533|1.34220e-7|
| LiDAR |7/12|0.449781|0.136723|2.17540e-5|

| Yeni kontrol repeat'i | Gaussian | Background | Shear | LiDAR | Toplam |
|---|---:|---:|---:|---:|---:|
|0|6/8|6/8|7/8|4/8|23/32|
|1|6/8|5/8|7/8|4/8|22/32|
|2|6/8|7/8|7/8|4/8|24/32|

[CPU] Background'da hem PM hem terminal için dört uyuşmayan nesne-pair
gözlemi vardır; diğer üç bozulmanın bu tekrarlarında sınıf tahminleri eşleşir.
Pair karşılaştırmaları aynı koşuları paylaşır, bağımsız denemeler değildir.
[Inference] Native gate tek başına kayıtlı H5 advance kuralını kapatır;
olası olumlu AUROC aralığı bunu onarmayacaktır.

## Korunan ilk analiz hatası

[Run/CPU] `analysis_v1` `make_bands` içindeki katı q sıralama kontrolünde
`Invalid normalized eigenvalues` ile durdu. Yalnız inputs/failure/native-repeat
gate dosyaları üretildi; feature freeze ve temiz-oracle okumasına ulaşılmadı.
Komut, stdout/stderr ve source kimlikleri acceptance_v1 altında korunuyor.

[CPU]96 archived spektrumun24'ünde hata: LiDAR'ın8 nesnesi×3 repeat.
Hepsi sonlu ve negatif olmayan float32 q; toplam174 komşu sıralama tersliği,
en büyüğü3.5762786865e-7. Mevcut normalize bant eşik toleransları yaklaşık
0.0025–0.0028'dir. Salt-okuma özgün anchored-block tanısı hiçbir ters çiftin
mevcut tekrar-eigenspace bloğu dışına taşmadığını gösterir. Ham q/U sırası ve
kimlikleri değiştirilmedi. Dar doğrulama düzeltmesi önce ebeveyne bildirildi
ve onaylandı; bilimsel bant/feature/oracle/epsilon/tolerans retuning'i yapılmadı.

[Code/CPU] Yalnız CPU validator değişti: her özgün anchored bloktaki HER
q değeri aynı `atol+rtol*|q_anchor|` içinde kalmak zorunda; artık katı komşu
monotonluğu koşulu yerine bu mevcut tekrar-eigenspace koşulu uygulanır.
Material backward tail/cross-block disorder reddedilir. Sıralama, U/q rewrite,
yeni tolerans, bant üyeliği/kenarı/formül değişimi yoktur. Float32 tiny-reversal,
material tail reddi ve strictly-sorted veride eski72e0eb4 fonksiyonuyla TAM aynı
çıktı testleri geçti; core/bootstrap13test başarılı. Parent son diff incelemesini
yapar. GPU b766 ve sonraki comprehensive8095 overlay'leri değiştirilmedi.

Önceki CPU analyzer SHA:
`10c97d9e3ccb45b29cc6b0077da0160a64d99f88ce6e1c54a23573c62c279a19`.
Doğrulama düzeltmeli CPU SHA:
`23893606655d3d84b14636915c097a06143ffb7e41cdb53e404e6b53739b3daa`.
Fark/kimlik/test makbuzu `acceptance_v1/analyzer_validation_fix_v2.*` içinde.

## Dondurulmuş feature ve set-oracle sonucu

[Run/CPU]2304 satırın tamamı saklandı:32 nesne×bozulma koşulu,
3repeat×4checkpoint×6band. Her exact X için ilk repeat'in archived GPU bazı
diğer repeat/zamanlara ortak uygulandı; eşleşmemiş eigenvector coefficient'leri
kıyaslanmadı. Arşiv q'su normalize/clamp Laplacian eigenvalue'udur, H5 q ağırlığı
değildir. Tüm features.json yazılıp hash'lendikten sonra temiz NPZ okundu.
Temiz set yalnız bozuk örneğin sabit center/scale/fiziksel frame'ine taşındı;
registration, classifier renormalization veya clean slot correspondence yoktur.

| Bozulma | Satır | Oracle uygun | Fayda | Zarar | Boş bant |
|---|---:|---:|---:|---:|---:|
|Gaussian|576|516|353|163|60|
|Background|576|480|358|122|96|
|Shear|576|504|336|168|72|
|LiDAR|576|480|216|264|96|
|Toplam|2304|1980|1263|717|324|

[CPU] Overshoot/tiny-residual/tiny-denominator/tie sayısı bu örneklerde0;
324 unavailable satırın tamamı boş banttır. Sabit epsilon .001||Y|| küçültülmedi.
Bu küçük output-space adımların clean-CD faydası sampler/accuracy kazancı değildir.

## Ortak-maskeli sıralama ve8-cluster bootstrap

[Run/CPU] Üç ana skor aynı1485 satırı kullanır:977fayda/508zarar.
İlk checkpoint'teki v/s yokluğu üçlü karşılaştırmadan495 uygun satırı çıkarır;
başka score'a farklı coverage verilmez. Ortak sıralı row-key SHA:
`2a5377f2a34de592143573fd77dc1e8713938d53173cd003a4b3788bb567d40c`.

| Önceden sabit güvenilirlik skoru | AUROC |95% object-cluster CI | Spearman |
|---|---:|---:|---:|
|Frequency −mean(lambda)|0.299841|[0.171317,0.419095]|−0.277350|
|Residual −r|0.260924|[0.131263,0.370684]|−0.409164|
|Combined −s, s=r/(1+v)|0.271341|[0.137627,0.386096]|−0.401091|

[CPU] Sabit2000draw/seed0, sekiz original-ID cluster replacement ile örneklenir;
her cluster bütün uygun repeat/bozulma/checkpoint/bant satırlarıyla çoğaltılır.
Pooled nonlinear AUROC her draw'da yeniden hesaplanır; object-AUC ortalaması
kullanılmaz. Üç skor aynı draw/maskeyi paylaşır.2000/2000 geçerli,0tek-sınıf,
0boş,0invalid draw. Baseline CI'ları aynı kayıtlı ham draw AUROC'larından
hesaplandı; yeni resampling/hiperparametre seçimi yapılmadı.

| Eşlenmiş AUROC farkı | Nokta farkı |95% cluster CI |
|---|---:|---:|
|Combined − Frequency|−0.028500|[−0.062963,+0.006332]|
|Combined − Residual|+0.010417|[+0.005726,+0.017766]|

[Run/CPU] Bozulma içindeki ortak-maskeli AUROC'lar da saklanır:

| Bozulma | Ortak n | Frequency | Residual | Combined |
|---|---:|---:|---:|---:|
|Gaussian|387|0.251119|0.251279|0.246483|
|Background|360|0.000000|0.000000|0.000000|
|Shear|378|0.354089|0.335725|0.333142|
|LiDAR|360|0.403199|0.369248|0.367845|

[Inference] Bu gelişimsel örneklerde kayıtlı güvenilirlik sıralamalarının yönü
oracle faydasını desteklemiyor. Combined, residual'a küçük bir fark gösterse
de kendi AUROC'su0.5 altında; frequency karşısındaki fark aralığı0 içeriyor.
Skor işareti sonradan çevrilmedi, farklı score/feature aranmadı veya q uygulanmadı.
Bu sonuç bütün gelecekteki reliability algoritmalarını evrensel olarak çürütmez.

Kayıtlı advance kuralının sonucu **not_advanced**:
native_all_pass=false; combined altCI0.137627≤0.5;
combined−frequency altCI−0.062963≤0; yalnız combined−residual altCI>0.
Yeni applied-q/accuracy deneyi bu koşudan destek almaz. Sekiz tarihsel olarak
açığa çıkmış cluster heldout/population confirmation değildir; satırlar bağımsız
deneme sayılmaz. Bu kural bootstrap/gerçek veri öncesi sabitlenmişti.

[CPU] Native feature variation ayrıca saklanır: Background r için uygun40
object-band özetinin9'unda repeat-range medyanı adjacent temporal-change
medyanını aşar. Diğer r/v/s özetlerinde bu median-dominance işareti yoktur;
bu, başarısız ham native kapıları geçerli veya bütün feature'ları stabil yapmaz.
Her r/v/s feature'ı için165 uygun object-band özeti,27 boş-bant özeti vardır.

## Tekrar üretim ve açık sınırlar

[CPU] Ortam: kalıcı root CPU venv, Python3.12.10/NumPy1.26.4/Torch2.5.1/
SciPy1.14.1, macOS arm64. GPU veya Colab çağrısı yapılmadı. Operasyonel runner
`downloaded/acceptance_v1/execute_frozen_analysis.py`; kaynak analyzer ve
bootstrap scriptleri sabittir. Ayrı komut/stdout/stderr/SHA makbuzları saklanır.

Ham kabul kanıtları:
`downloaded/acceptance_v1/package_acceptance.json`,
`raw_runs_acceptance.json`, `native_floor_summary.json`,
`normalized_q_validation_failure.json`, `q_order_anchored_block_audit.json`.
İlk başarısız analiz: `analysis_v1/failure.json`; tamamlanan analiz:
`analysis_v2/{summary.json,features.json,rows.jsonl,...}`; bootstrap:
`bootstrap_v1/{summary.json,bootstrap_values.npz,inputs.json}`.
V2 komutu47.247s, bootstrap21.407s CPU sürdü; iki stderr dosyası da boştur.
Ham H5 GPU fazının arşiv makbuzundaki süresi416.558s'dir; kaynak kurulum/
sonraki comprehensive faz ile karıştırılmaz, hız kazancı iddiası değildir.

[CPU] Bağımsız son recount2304 benzersiz condition key, aynı frozen feature
değer/sıra/hash,1485 ortak maske, seed0 draw-ID matrisi, eşlenmiş AUROC farkları
ve CI'ları doğruladı. Ham1.39GB paketin aynı SHA'sı tekrar kontrol edildi.
Kompakt sonuç: [CPU kabul ve ölçümler](evidence/h5_reliability_20261105/cpu_acceptance_v2.json).
Tam çıktı/source/komut hash manifesti:
`downloaded/acceptance_v1/output_artifact_manifest.json`.
Önceki bütün baseline/negatif sonuçlar ve başarısız V1 korunur; yeni GPU işi,
varsayılan yöntem değişimi veya bilimsel eşik gevşetmesi yapılmadı.
