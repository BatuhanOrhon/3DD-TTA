# Conditional global-z guidance: iki başlangıç da test edildi — 2026-10-31 UTC

[Run/Inference/Decision] Encoder z'si ve son SCD style/z'si ayrı ayrı test
edildi. Gerçek CUDA condition/global-chain gradient kontrolü geçti, fakat
bu sabit üç-adımlı guidance **güvenilir accuracy kazancı veya başlangıca göre
temiz-z yakınsaması göstermedi**. SCD başlangıcının üstünlüğü de gösterilmedi.
Operasyonel global diffusion kapalı kalır; prototip defaulta alınmadı.
Türkiye yerel tarihine göre sonuçlar2026-11-01'e taşmıştır; run/path adları
UTC2026-10-31'i korur. [Başlamadan kaydedilen tasarım ve kronoloji](shape_latent_conditional_pilot_20261031.md).

## Hangi z kullanıldı?

[Code/Run] `tta.py` içindeki iki ayrı tensör:

1. **encoder**: `SharedTrajectory.original_style`, encoder posterior örneği.
2. **scd**: `SharedTrajectory.updated_style`, ilk local geçişin son SCD
   style güncellemesi sonrası değer. All55 config'te `global2style` identity
   olduğu için bu128vektör global prior'a doğrudan verilebilir.

Her nesne/bozulma için bir ortak ilk local geçiş, detached terminal **h***
ve son SCD z'sini üretir. İki başlangıç aynı h* ile değerlendirilir; yeni
z için h yeniden encode edilmez. İkinci local geçiş orijinal h0/local noise/
hazırlık-sonrası RNG ve h0XYZ Chamfer anchor'ından yeniden başlar. Her kolun
kendi seçilmiş z'si clean probe ve final decoder'a gider. Legacy metadata'da
`decoder_style=original`, bu ikinci geçişe sağlanan **seçilmiş başlangıç z**
anlamındadır; bütün kolların encoder z'siyle decode edildiği anlamına gelmez.

[Code/Run] İlk encoder z/h0/noise, önceki fiziksel hazırlık kayıtlarından
kimlikleri doğrulanarak alınır; her kolda tekrar posterior örneklenmez.
ModelNet40-C severity5,32exposed orijinal nesne × gaussian/background/
rotation/lidar, seed0, Point-MAE. Her kolda128olay vardır; bunlar128bağımsız
nesne değildir. N8/Background35, gamma=.03, eta=.003, p=.95, total100;
raw/eval/frozen/EMAoff. Primary PM clean2/4/6+terminal8; Background9/18/27+35.

## Accuracy: aynı pilotun kendi referansları

| Başlangıç / kol | Doğru /128 | PM accuracy % | Kendi global-off kontrolüne fark, puan |
|---|---:|---:|---:|
| İlk ortak geçiş |55|42.968750|ayrı referans|
| Encoder / global-off replay |54|42.187500|0|
| Encoder / prior3 |56|43.750000|+1.562500|
| Encoder / guided3 |55|42.968750|+0.781250|
| Son SCD z / global-off |55|42.968750|0|
| Son SCD z / prior3 |55|42.968750|0|
| Son SCD z / guided3 |53|41.406250|−1.562500|

[Run] En yüksek nokta tahmini yalnız-prior/encoder kolundadır; bu da
güvenilir iyileşme iddiası değildir. Encoder-guided ilk ortak geçişle aynı
accuracy'dedir. İlk geçiş/encoder-off replay farkı ve aşağıdaki teknik-pilot
tekrar farkları, bir-iki olaylık farkın yorumunu özellikle sınırlar.

| Önceden belirlenen eşlenmiş karşılaştırma | Fark, puan | Koşullu %95 CI | Düzeltti / bozdu |
|---|---:|---:|---:|
| Encoder guided − encoder off |+.781250|[−2.343750,+4.687500]|3 /2|
| SCD guided − SCD off |−1.562500|[−4.687500,+1.562500]|1 /3|
| Encoder guided − encoder prior |−.781250|[−3.906250,+2.343750]|2 /3|
| SCD guided − SCD prior |−1.562500|[−5.468750,+2.343750]|2 /4|
| SCD guided − encoder guided |−1.562500|[−4.687500,+1.562500]|1 /3|
| SCD off − encoder off |+.781250|[−2.343750,+3.906250]|3 /2|
| Encoder off replay − ilk geçiş |−.781250|[−3.906250,+1.562500]|1 /2|

[Run/Inference] 10,000draw/seed81031 original-object cluster bootstrap;
aynı32nesnenin bütün bozulmaları birlikte örneklenir. Yedi aralık da0içerir.
Aralıklar seed/native sayısal tekrar varyansını ve unseen transferi kapsamaz.
Bu exposed development pilotu, confirmation veya tam benchmark değildir.

| Bozulma | İlk geçiş | Enc.off | Enc.prior | Enc.guide | SCD.off | SCD.prior | SCD.guide |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gaussian |56.250|56.250|65.625|62.500|62.500|65.625|62.500|
| Background |59.375|56.250|53.125|53.125|53.125|53.125|53.125|
| Rotation |28.125|31.250|28.125|31.250|28.125|25.000|28.125|
| LiDAR |28.125|25.000|28.125|25.000|28.125|28.125|21.875|

[Run] SCD-guided'in kendi-off kontrolüne net kaybı LiDAR'da yoğunlaşır.
Gaussian'daki encoder-guided nokta artışı, yalnız-prior kontrolünü aşmaz.
[Ham sonuçlardan bağımsız yeniden hesap](evidence/shape_latent_conditional_pilot_20261031/acceptance.json)
ve [per-corruption CSV](evidence/shape_latent_conditional_pilot_20261031/per_corruption.csv).

## Gradient temiz z yönünü taşıyor mu?

[Code/Run] Global epsilon DDIM20/10/0 (model21/11/1),1000train/100infer,
linear beta .0001→.02, clipoff/finalalpha1. Conditional energy tüm8192local
kanalda unit-weight MSE; dört sabit training index99/299/599/899. Ortak
private noise bankası üç global adım boyunca korunur; held-out Gaussian
bankası ayrıdır. Her global adımda dört local condition-VJP ve global
clean-estimate VJP; weights frozen. Güncelleme normu her nesne için sabit
.02*||encoder z||; iki başlangıç da aynı ölçeği kullanır. %5ball ve oracle
step/target seçimi yoktur. Katsayılar teknik accuracy'ye göre değiştirilmedi.
[Literatür ve formül](shape_latent_gradient_research_20261031.md).

| Kol | Toplu clean-z MSE | Kendi başlangıcına oran |
|---|---:|---:|
| Encoder off |.1283593803|1|
| Encoder prior3 |.1355415059|1.055953|
| Encoder guided3 |.1341945359|1.045460|
| SCD off |.1278786496|1|
| SCD prior3 |.1349668895|1.055429|
| SCD guided3 |.1337302257|1.045759|

[Run] SCD başlangıcı encoder'a göre MSE'yi yalnız%.374519 azaltır;
encoder→SCD hareketinin temiz hedefe cos medyanı .024282'dir. Son SCD
style'ı temiz global temsil yerine koymak bu ölçümle desteklenmez.

[Run/Inference] Guidance, kendi prior3 kontrolüne göre MSE'yi encoder'da
%.993769, SCD'de%.916272 azaltır; ancak kendi başlangıcına göre sırasıyla
%4.545952 ve%4.575882 **arttırır**. Held-out denoising enerjisi yaklaşık
%.130633/%.133040 düşer. Dolayısıyla enerji azalması, bu üç-adımlı zincirin
sonucunun temiz z'ye daha yakın veya daha doğru sınıflandırılır olmasını
sağlamadı. Aynı Gaussian-noise prior kontrolü bu ayrımı görünür kılar.

[Run] Negatif **clean-condition gradientinin**, o adımdaki prior-clean
tahminden paired clean z'ye yönle cosine medyanları:

| Başlangıç | İlk adım | İkinci | Üçüncü | İlk adımda pozitif yön payı |
|---|---:|---:|---:|---:|
| Encoder |.141144|.069202|.006827|%69.53125|
| Son SCD |.126003|.065907|−.013772|%64.84375|

[Inference/Open] İlk adımda zayıf bir temiz-yön bilgisi var; sonraki adımlarda
zayıflıyor. Bu, exact clean-score veya yakınsama kanıtı değildir. Pozitif
cosine tek başına MSE düşüşü sağlamaz: hareket büyüklüğünün karesi de yeni
uzaklığa eklenir. h* eski z/SCD yanlılığını taşıyabilir; finite-t MSE kalibre
−log p(h*|z) değildir. Clean z ve etiketler yalnız posthoc değerlendirmede,
objective/step/radius/selector girdisi olarak kullanılmadan açıldı.
[Fiziksel yön analizi](evidence/shape_latent_conditional_pilot_20261031/direction.json).

## Doğrulama ve tekrar sınırı

[Run/Code] CPU sentetik condition/dense-gradient/batch-scale/finite-difference/
global-VJP/private-RNG/scope kontrolleri geçti. Sonra gerçek A100'de bir
Gaussian nesnesi üzerinde iki başlangıç × condition/global-chain × iki
finite-difference ölçeği: sekiz denemenin tamamı, önceden yazılan%10relative
veya1e−5absolute toleransını geçti. Gözlenenrelative fark%3.16–6.49;
model/RNG değişmedi. Bu küçük gate, bütün nesnelerde tam Jacobian doğrulaması
veya faydalı semantik gradient garantisi değildir.
[Native rapor](evidence/shape_latent_conditional_pilot_20261031/native_gradient_report.json).

[Run] GB teknik/pilot aynı source/model/input/h0/noise/RNG ile yine farklı
tahminler verebilir: firstpass Gaussian19→18/32 (2prediction değişti),
Background18→19 (3); encoder-off19→18 (3) ve19→18 (1);
encoder-prior Gaussian21→21 (1), Background19→17 (3). Bu üç kontrolün
seçilmiş z/initial hash'leri de eşittir. SCD/guided kollarında ilk geçişten
türetilen z/h* ayrıca değişebilir; onların bütün farklarını sabit-latent
native varyans diye adlandırmayız. Kesin native neden belirlenmedi; yeni
repeat/GPU açılmadı. [Mevcut izlerin karşılaştırması](evidence/shape_latent_conditional_pilot_20261031/technical_pilot_repeatability.json).

[Run] Teknik807dosya/14shard ve pilot874dosya/28shard manifest/CRC/SHA;
fiziksel cache/paired RNG, private Gaussian, global forward/DDIM/correction
normları, selected-z clean/final decoder, frozen model/source ve PM hesabı
Drive+yerelde bağımsız kabul geçti. Remote acceptance JSON, outcomes ve
paired-latents yerel NumPy1 yeniden hesapla exact eşleşir. Analiz hamları
yeniden yazmadan ayrı output dizinine yapılır.

## Kaynak, başarısızlıklar, süre ve arşiv

[Code] Worktree7bf0, branch integration/chamfer-guidance-variants-20261030,
HEAD a82200adbe117ce56a134dc3c22601cb73dc2654 dirty korunur. Yedi denetlenmiş
önceki production dosyası byte-identiktir; yeni iki modül yalnız sealed
araştırma prototipidir. Kaynak ZIP SHA
2b3982cdf21aa60894df368f186b5dba01931a01658ba5474176ccda26b8a5f7;
asset manifestd6231c4aa807ddd72af17c5c9482ef26ead1f9a179124ae66fe9d26655d70054;
actual executed source4f20bd289925050d5561d65dfb865c12a73404bae842da4ddb445d81bb5ab6de.

[Run/Code] Fresh ten-cell setup/native/data/weights geçti; yalnız izin verilen
cell2 fastrlock hatası ertelendi. İki operator API hatası korunur:
CPU'da scheduler helper yolu ve ilk native raporda str→Path dosya kimliği.
İlki yalnız CPU resume ile; ikincisi yalnız native proof yeniden çalıştırılarak
düzeltildi. Tamamlanmış GB firstpass teknik recovery'ye byte-identik taşındı;
başarısız raw/receipt/ledger silinmedi veya üstüne yazılmadı. Scientific
algoritma/ayar/source seal bu düzeltmelerde değişmedi. Assetv1'in eksik
nested-manifest bildirimi de deployment öncesi yakalandı; v2 fiziksel üyeleri
doğrulanarak kullanıldı. V1 hiçbir inference çalıştırmadı.

[Run] Scientific ledger970.088193s=16.168min; başarısız teknik deneme,
review/recovery, geçişler ve iki başarılı fazın remote raw arşivi dahil.
Remote araştırma bitişi21:06:01.059334UTC. Kayıt başlangıcından3239.514669s
(53.9919dk), prospective59min whole ve1165s research cap altında. İlk45dk
tahmini aşıldı;55→59dk timing-only değişiklik pilot öncesi kaydedildi.
Bu, yöntemin deployment süresi değildir: worker replay ilk ortak local
geçişi dışarıda tutar; gerçek kullanım iki local trajectory gerektirir.
Sonraki CPU/Git/türetilmiş arşiv işi bu GPU-faz ledger'ına eklenmez.

| Ham arşiv | Byte | SHA256 |
|---|---:|---|
| Başarısız ilk teknik |36783790|9ecdeec539eff2d7923ba30c9d639e9b72d68c10ef16db48b1fd0d5d4cd074b4|
| Recovery teknik |186869678|262e39804c3947d2a1021a6d1ea9884a7dbf6eb1f94223f32686d2033242747f|
| Dört-bozulma pilot |325899956|13ab784c339e5514b2a897cc9e45a5ed4754cd8b280fd04c84d29a394669b609|

Drive: `MyDrive/thesis/result/shape_latent_conditional_pilot_20261031/phase-evidence/`.
Yerel: `result/preparation/shape_latent_conditional_pilot_20261031/operator/`.
Pilot beş64MiB-or-smaller parçadan orijinal ZIP SHA'sına exact birleştirildi;
ham ZIP değişmedi. [Kompakt kanıt](evidence/shape_latent_conditional_pilot_20261031/README.md).
Kendi `tta-shape-conditional-a100-20261031-r1` / gpu-a100-hm-euw4a0-2jl6gm4ux3xzj
runtime'ı korunur; monitor85488, teardown yok. Son inventory/arşiv/yayın
makbuzları tamamlandıkça bu kayda bağlanır; mevcut bilimsel sonuç değişmez.

[Decision/Open] Mevcut sonucu yüksek-accuracy veya temiz-z yakınsaması diye
sunma. Önceki64/all15 temiz-z oracle +4.0625puan sonucu ayrı protokoldür;
bu32/dört sonuçla mutlak accuracy olarak karıştırılmaz. Sonraki ayırıcı soru,
aynı conditional enerji yönünü global prior olmadan küçük bir clean-z adımında
ölçmek olabilir; bu yeni kontrol uygulanmadı/başlatılmadı. Daha güçlü step/grid,
ek seed, unseen confirmation veya yöntemleri birleştirme otomatik devam değil.

## Ek arşiv doğrulaması — 2026-10-31 21:18 UTC ve sonrası

[Run] Kullanılan source payload ve tüm operator/CPU-deneme/setup baytları
`final-archive-20261031T211816Z/` altında Drive'a doğrulandı. Operator/CPU/setup
ZIP1,564,492byte SHA f9678499b29f3a239de3950cbeaf325cb469dd7e10111a331f14c09d37f23894;
gerçek Drive download roundtrip ve57üye SHA/CRC geçti. İlk CLI prelude
bağlantısı koptu; uzakta iş başlamadığı denetlendi. Retry sırasında yerel
istemci sonlansa da uzak kopyalama tamamlandı; receipt bağımsız console
ile bulunarak ikinci bir kopyalama başlatılmadı. Bu CPU arşiv işi yeni GPU
deneyi değildir. [Roundtrip makbuzu](evidence/shape_latent_conditional_pilot_20261031/extra_backup_roundtrip_receipt.json).
Üç ham ZIP ve135/807/874üye posthoc analizden sonra tekrar exact doğrulandı;
raw/extracted baytlar değişmedi. Türetilmiş capsule/yayın makbuzları ayrıca
eklenir; bunlar bilimsel tabloyu veya algoritma ayarını değiştirmez.

[Run] Türetilmiş/source/operator capsule1,132,781byte SHA
64f1150e5354c490f0dc287fbe45b9859e9855cca4b95946895b5b0a20d21264
aynı final-archive dizininde Drive'a doğrulandı; gerçek download roundtrip,
143üye ve CRC/SHA geçti. Capsule, son yayın/capsule makbuzlarından önceki
rapor snapshot'ını korur; son sidecar makbuzları pending yayın metnini
supersede eder. [Final roundtrip](evidence/shape_latent_conditional_pilot_20261031/derived_roundtrip_receipt.json).

[Run] Yayın öncesi son server inventory, aynı A100/High-RAM endpoint atamasını
ve read-only monitor85488 sürecini doğruladı. GPU araştırması tamamlandı;
oturum korunur. [Inventory makbuzu](evidence/shape_latent_conditional_pilot_20261031/final_runtime_inventory.json).

[Run/Decision] Seçili45knowledge/evidence dosyası, önceki kabul edilmiş ortak
kayıtlar korunarak normal fast-forward ile `origin/thesis-integration` dalına
`8c1d544f65fc56a0360967234bbbf6c933d81d72` commit’iyle yayınlandı; uzak SHA ve
temiz yayın checkout’ı doğrulandı. Orijinal branch/HEAD ve yedi production
kimliği korundu. `operator/publication_receipt.json` aynı final-archive Drive
dizininden geri indirildi;5,448byte SHA
`bad5777e9bf12632628fb35e7aef82e4db26da0e152f2135bf6cd9b7cc8ef388` exact.
Bu final sidecar, capsule içindeki yayın-pending snapshot’ını supersede eder.

## Referans accuracy kapsamı yeniden denetlendi

[Run/Code/Inference] Kullanıcının %60+ SCD referansı sorusu üzerine önceki
refresh baseline’ının kabul edilmiş outcome matrisi yeniden toplandı; yeni
inference/GPU çalıştırılmadı. Aynı classifier/LION state kimlikleri doğrulandı.

| Aynı önceki baseline tahminlerinin kapsamı | Doğru/toplam | Accuracy % |
|---|---:|---:|
| 64 nesne, 15 bozulma |585/960|60.937500|
| Aynı 64 nesne, yalnız yeni pilotun 4 bozulması |117/256|45.703125|
| Yeni pilotla aynı 32 nesne, 15 bozulma |283/480|58.958333|
| Yeni pilotla aynı 32 nesne, aynı 4 bozulma |55/128|42.968750|
| Yeni pilot ilk SCD geçişi, aynı 32×4 |55/128|42.968750|

[Inference] %60.9375→%42.96875 farkı eski tahminler üzerinde yalnız kapsam
değiştirilerek de aynen oluşur. Önceki64 nesnede dört bozulmaya geçiş
−15.234375puan, ardından32 nesneye geçiş −2.734375puan: toplam −17.96875puan.
Bu sıralamaya bağlı betimsel ayrıştırmadır; nedensel corruption etkisi değildir.
Rotation/LiDAR yeni ortalamanın yarısıdır; all15’te toplam2/15 ağırlıktadır.
Son %42.97 tüm ModelNet40-C SCD accuracy’si diye sunulmamalı; küçük32×4
pilotunun kendi matched referansıdır. Önceki %60.94 de64-object development
sonucudur, tam veri benchmark’ı değildir. Aggregate eşitlik tek tek tahminlerin
veya native hesapların byte-identik olduğu anlamına gelmez.
[Hesap, ID/row eşlemesi ve kaynak SHA’ları](evidence/shape_latent_conditional_pilot_20261031/reference_scope_recheck.json).

## Kullanıcı isteğiyle tamamlandı ve kapatıldı — 2026-10-31 21:34 UTC

[User report/Run/Decision] Kullanıcı Colab oturumunu kapatmamızı, gözlemleri
knowledge kayıtlarına ekleyip yalnız knowledge dosyalarını integration dalına
birleştirmemizi istedi. Tamamlanan `tta-shape-conditional-a100-20261031-r1`
oturumuna CLI stop uygulandı; exit0/Session terminated ve21:34:18UTC server
envanterinde atamanın yokluğu doğrulandı. Kendi read-only monitor85488 durdu.
Önceki canlı/koruma notları tarihsel durumdur; bu kapanış onları supersede eder.
[CLI ve envanter kapanış makbuzu](evidence/shape_latent_conditional_pilot_20261031/closure_receipt.json).

[Run/Decision] Ham/source/operator/türetilmiş ve önceki yayın Drive roundtrip
kabulü kapanıştan önce yeniden kontrol edildi. Kanıtlar yerelde ve Drive'da
korunur. Bu kapanış yeni deneme değildir; iki başlangıçtaki null sonuç ve
native varyans sınırları değişmedi. Temiz-z yakınsaması hâlâ açık araştırma
sorusudur. Sonraki saf-enerji/ek-seed/tam-veri denemesi otomatik başlatılmadı.
Yalnız gerekli knowledge kayıtları seçilir; implementation/test/config veya
büyük ham dosyalar bu follow-up entegrasyonuna alınmaz. Orijinal branch/HEAD
ve mevcut kod değişiklikleri korunur.
