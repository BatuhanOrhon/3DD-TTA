# Temiz-z oracle pilotu: global bilgide fayda var — 2026-10-31

[Run/Inference] Küçük eşleşmiş pilot, temiz girdiden alınan shape latent'in
doğruluğa yardımcı olabildiğini gösteriyor. Eski h korununca +4.0625puan;
bozuk girdiden temiz z'ye koşullu h yeniden encode edilince +2.1875puan.
Dolayısıyla önceki global diffusion null sonucunu “shape latent düzeltmesi
faydasız” diye genellemek desteklenmiyor. Bununla birlikte mevcut global
denoiser'ın bu temiz bilgiyi üretebildiği gösterilmedi; operasyonel global
diffusion kapalı kalır. Bu sonuç temiz veri kullanan bir mekanizma tanısıdır,
uygulanabilir yeni TTA yöntemi veya benchmark kazanımı değildir.

## Sabit küçük deney

[User report/Decision] Kullanıcı dört kollu küçük pilotu onayladı, başlamadan
önce mevcut durumun knowledge'a kaydedilmesini istedi ve tam veri seti
taramasını dışladı. [Başlangıç kaydı](shape_latent_oracle_pilot_20261031.md)
kod ve yeni VM başlatılmadan önce yazıldı. Aynı64açığa çıkmış orijinal nesne,
ModelNet40-C severity5/all15, Point-MAE, seed0, B32 kullanıldı. Her kol960
nesne-bozulma olayı içerir; bunlar960bağımsız nesne değildir.

[Code/Run] Global diffusion dört kolda da kapalıdır. Temiz z, aynı orijinal
nesnenin `data_original.npy` girdisinin VAE global encoder posterior örneğidir;
global denoiser tahmini değildir. Clean/corrupt global posterior'lar aynı
standart Gaussian eps_z'yi; local posterior'lar aynı eps_h'yi kullanır.
Local diffusion epsilon ve hazırlık sonrası RNG bütün kollarda aynıdır.
Modeller eval/frozen; encoder veya sınıflandırıcı ağırlıkları güncellenmez.

[Code/Run] Local DDIM8/Background35, gamma=.03, eta=.003, p=.95, total100,
set-alpha-to-one=True, raw LION/EMAoff korunur. Primary PM normal clean2/4/6
+terminal8, Background clean9/18/27+terminal35 eşit olasılık ortalamasıdır.
Mevcut conditioning-style gradient güncellemesi sürer; clean/final decoder
o kola sağlanan başlangıç z'sini kullanır. CSV terminal değerleri primary
yerine alınmadı; trajectory logits'ten PM bağımsız yeniden hesaplandı.

| Kol | Başlangıç z / h | All15 PM% | A'ya fark, puan | Background PM% | Macro14 PM% |
|---|---|---:|---:|---:|---:|
| A — baseline | Bozuk z / bozuk X'ten kendi z'siyle h |60.833333|—|56.2500|61.160714|
| B — clean_z_fixed_h | Temiz z / A-h aynen korunur |64.895833|+4.062500|62.5000|65.066964|
| C — clean_z_reencode_h | Temiz z / bozuk X'ten temiz z'ye koşullu h |63.020833|+2.187500|64.0625|62.946429|
| D — clean_pair | Temiz z / temiz X'ten temiz z'ye koşullu h |77.708333|+16.875000|56.2500|79.241071|

[Run] Doğru olay sayıları A584, B623, C605, D746 /960'tır. Referans bu
eşleşmiş hazırlıkta yeniden çalıştı; önceki global-prior pilotunun585/960,
60.9375%'i karşılaştırmaya taşınmadı. Başka kohortlardaki66%/68.741319%
değerleriyle doğrudan karşılaştırma yapılmaz.
[Bozulma tablosu](evidence/shape_latent_oracle_pilot_20261031/per_corruption.csv).

## Eşleşmiş belirsizlik ve hipotez

| Karşılaştırma | Fark, puan | Düzeltilen / bozulan olay | Koşullu %95 aralık, puan |
|---|---:|---:|---:|
| B−A |+4.062500|56 /17|[+1.979167, +6.354167]|
| C−A — ana karşılaştırma |+2.187500|39 /18|[+0.625000, +3.854167]|
| C−B |−1.875000|19 /37|[−3.958333, 0.000000]|
| D−C |+14.687500|179 /38|[+9.895833, +19.583333]|
| D−A |+16.875000|196 /34|[+11.875000, +21.875000]|

[Run/Inference] Aralıklar10.000eşleşmiş orijinal-nesne bootstrap tekrarıyla
(seed20261031) hesaplandı; her nesnenin15bozulması birlikte örneklenir. Tek
diffusion seed'i, sabit bozulmalar ve açığa çıkmış64nesneye koşulludur; çoklu
karşılaştırma düzeltmesi yoktur. C−B'nin üst sınırı kayan nokta hesabında
8.78e−17'dir, tabloda0'a yuvarlandı. Kesin negatif bir interval gibi sunulmaz.

[Inference] Temiz global bilgi bu host altında fayda sağlayabilir: B−A ve
önceden ana karşılaştırma olarak belirlenen C−A pozitiftir. İkisi de önerilen
+1puanlık pratik inceleme eşiğini aşar; bu eşik significance testi değildir.
“Temiz z bile yardımcı olmuyor” sonucu çıkmadı.

[Inference] “Eski h ile yeni z'nin uyumsuzluğu ana engel” açıklaması bu pilotta
desteklenmedi. B, koşullu yeniden encoding yapan C'den ortalamada1.875puan
yüksektir; C ek kazanç sağlamaz. Bu, her türlü h−z uyumsuzluğunu veya ikinci
guidance fikrini çürütmez. Conditional encoder'ın h'yi yeni z ile üretmesi,
sınıflandırma açısından daha iyi bir başlangıç sağlayacağını garanti etmez.

[Run/Inference] B rotation'da25→34.375%, lidar'da28.125→37.5%, Background'da
56.25→62.5% olur; distortion'da60.9375→56.25% düşer. C Background'da64.0625%
ile B'den bir olay iyidir. D normal bozulmalarda güçlüdür fakat Background'da
A ile56.25%te kalır. Tek tek satırlardan etiketli corruption routing yapılmadı.

## Chamfer hedefi, normalization ve D'nin sınırı

[Code/Decision] Kaynak denetiminde mevcut TTA Chamfer hedefinin ham X değil,
başlangıç h'nin XYZ bileşeni olduğu görüldü. İlk konuşmadaki “bozuk girdi
hedefi” ifadesi bu açıdan eksikti. h'yi değiştirince hedefi de değiştirmemek
için dört kolun tamamında **A-h XYZ** sabit tutuldu. `tta_reconstruct`a opt-in
`chamfer_reference` eklendi; None iken eski math/RNG ve decoder davranışı aynı.
Gerçek target hash'i bütün kollarda aynı; CPU entegrasyon testi her adımı
kontrol eder. Yeni spectral/semantic/compatibility guidance eklenmedi.

[Code/Inference] Clean ve corrupt girdiler kendi bbox normalization'larıyla,
aynı LION ölçek/rotasyon kuralıyla hazırlanır. ModelNet çıkışı yeniden normalize
edilir; corruption center/scale'iyle inverse transform uygulandığı şüphesi
bu veri seti için doğrulanmadı (Scan yolu farklıdır). C−A, aynı bozuk local
girdi/anchor altında temiz global bilgi ve buna koşullu yeniden encoding'i
sınar. D−C bütün canonical local girdiyi, normalization ve interpolation dahil,
değiştirir; saf local-denoising veya uyumluluk ablation'ı değildir.

[Inference] D, bozuk A-h hedefi ve aynı8/35denoising altında tutulduğu için
formal doğruluk üst sınırı değildir. D'nin Background'daki null sonucu,
başka gözlem/hedef politikalarında temiz çiftin faydasızlığını kanıtlamaz.
Temiz h ve bozuk h'nin nokta sırası/interpolation'ı farklı olabildiğinden
ham pointwise h L2 farkı geometrik hata veya uyumsuzluk metriği sayılmaz.

## Latent tanısı

[Run] Aynı posterior Gaussian altında temiz-z örneğinin A-z'ye medyan
göreli L2 uzaklığı Gaussian%37.8764, Background%36.5267, rotation%35.8705,
lidar%50.4486'dır. C-h'nin A-h'ye medyan göreli farkı aynı bozuk nokta
sırasında sırasıyla%1.6859, %1.2159, %1.0003, %1.5691'dir.
[Tam betimsel kayıt](evidence/shape_latent_oracle_pilot_20261031/diagnostics_summary.json).

[Run/Inference] Ölçülen960/960temiz-z örneği önceki%5trust ball'ın dışındadır;
o kısıt bu deneydeki oracle değişimlerini dışlar. Bu uzaklık
corruption'ın saf hata normu veya optimum latent düzeltme mesafesi değildir;
encoder/normalization/resampling etkilerini de içerir. Daha büyük radius'un
gerçek global denoiser'ı iyileştireceği sonucu çıkarılamaz. Oracle yönünü
temiz veri olmadan bulabilmek ayrıca kanıtlanmalıdır.

## Fiziksel kanıt, süre ve kaynak

[Run] Teknik32×2kapı ve pilot64×15 dört kolu tamamladı. Pilot120batch shard,
30ortak baseline cache,30tam posterior proof ve120seçilmiş-kol cache içerir.
Yerel NumPy1 recount; SHA/CRC/güvenli extraction, ID/sıra/label/logit PM,
µ+σeps denklemleri, gerçek ortak eps_z/eps_h, temiz/bozuk prepared girdiler,
branch seçimleri, ortak diffusion noise/Chamfer anchor, serialized RNG,
donmuş model state'leri ve actual source hash'lerini doğruladı. Yerel
acceptance uzak JSON ile birebir aynı; outcome matrisleri de aynıdır.
Kimlik eşlemesi, yayımlanmış ModelNet40-C dizilerinin ortak orijinal satır
sırası sözleşmesine dayanır; aynı seçili sıra program içinde doğrulanır.
Temiz/bozuk noktalar arasında pointwise correspondence varsayılmaz.

[Code/Run] `Normal.sample()`ın ikinci dönüşü inplace işlem nedeniyle orijinal
Gaussian değil, örnekle aliased rho'dur. Yeni oracle hazırlığı bunu ilk CPU
testlerinde yakaladı; eps_z/eps_h açıkça çekilip saklanır. Bağımsız Gaussian
testi RED→GREEN geçti. Eski encode ikinci dönüşü kullanmadığından önceki
accuracy sonuçları bu bulguyla geçersiz sayılmaz. Diğer review bulguları,
doğru bilimsel metadata ve iki prepared girdinin fiziksel kaydıyla karşılandı.

[Run] Yeni own `tta-shape-oracle-a100-20261031-r1`, endpoint
`gpu-a100-hm-ass1c0-1b1f398ijpuro`, A100-SXM4-80GB. Önceki runtime sunucuda
listelenmiyordu; tarafımızdan kapatılmadı. Kullanıcı yeni oturumla devamı
onayladı. Canonical10-cell setup/native/data/checkpoint doğrulaması ve
strict/no-forward/no-CUDA CPU deployment receipt geçer. Yalnız önceden
izinli Cell2fastrlock hatası ertelendi. Yeni Drive onayı ilk timeout'tan
sonra CLI retry ile başarıyla mount edildi; auth URL'leri kanıta yayınlanmaz.

[Run] Araştırma14:41:56.363042UTC'de başladı; teknik+pilot+arada aktarım ve
son uzak arşiv dahil ledger907.6917s (15.1282dk),55dk tavanının altında.
Yerel indirme/analiz daha sonra tamamlandı. İlk A bütün oracle posterior'ları
üretip kaydeder; B/C/D ortak hazırlığı replay eder. Reconstruction+terminal
A118.5197s, B91.3891s, C91.4337s, D91.5256s; worker-wall149.9599/122.7704/
122.8011/122.7585s. Tepe GPU bellek25221.17–25221.92MiB. Bu ölçümlerden
deployment hızlanması veya yeniden encoding'in ücretsizliği iddia edilmez.

[Code/Run] Kaynak a82200adbe117ce56a134dc3c22601cb73dc2654 +338dosyalı dirty
snapshot; manifest `85057eeba260c7f2608f8ee38b3f9df7948af0754c444ba211654cf851a5e3bb`,
sourceZIP `42d7f973b5897e63c813fb36684bfcbe83eddb15b45819694910a1a341263b55`.
Root CPU519test/373subtest, bir tarihsel-fixture skip; izole latest-target
entegrasyonunda535test/412subtest, ayrı60NumPy1test geçer. Entegrasyon
ağacının GPU ile yeniden çalıştığı iddia edilmez.

[Run] PilotZIP621835687byte SHA
`52bffe3f220bc76c3e9cdd486f033f954148b240515d3f33ad659795aa9ce033`;
teknikZIP52073193byte SHA
`d89420d72f2e53ccd260195a97969ef4c482a430763490a6ea31928b8c9d60f9`.
Drive `MyDrive/thesis/result/shape_latent_oracle_pilot_20261031/phase-evidence/`;
yerel ham/derived `result/preparation/shape_latent_oracle_pilot_20261031/operator/`.
[Kompakt kanıt ve yeniden hesap](evidence/shape_latent_oracle_pilot_20261031/README.md).

[Decision/Open] Tam veri, ek seed/başka classifier/dataset veya yeni guidance
çalıştırılmadı; otomatik devam yok. Global diffusion operasyonel olarak kapalı
kalır. Shape latent araştırmasını “faydasız” diye kapatmak için gerekçe yok;
öncelik, faydalı temiz global bilgiyi temiz veri olmadan üretebilmek olmalıdır.
h-reencoding veya uyumluluk guidance'ının gerekliliği bu sonuçla kanıtlanmadı.
Kullanıcı sonuçları inceledikten sonra sonraki adımı belirleyecek. Yeni own r1
korunur; task-complete+close izni verilmedi.

### Final archival/runtime observation — 15:20 UTC

[Run/Failure/Open] After all scientific phases and raw retrieval/acceptance,
final derived upload returned Session not found. Read-only server inventory
then reported no active runtime. No stop/release/teardown command was executed;
loss time/cause is unknown. User's teardown authorization was never inferred.
All raw phase archives remain SHA-verified on Drive and locally; results and
independent acceptance are complete. This is not loss of the scientific data.

[Run/Open] Extra derived/source/operator capsule is locally CRC-validated,
2311574bytes SHA70eb6a04c9a18011129cf993ffcc68dd8fa399083b130a44c1fb8ebd2f5dc926.
Its additional Drive copy/roundtrip is pending, not claimed successful. Browser
Drive fallback was denied by browser security policy (user declined); no retry
or workaround. Existing authorized Git publication preserves compact analysis/
code/knowledge; raw ZIPs and local capsule are retained. No new GPU allocated
solely for this extra copy. [Archival status](evidence/shape_latent_oracle_pilot_20261031/derived_archive_status.json).

[User report/Decision] User explicitly accepts local retrieval as sufficient
and says there is no need to reconnect Drive. Extra derived Drive copy is
deferred by user decision; all Drive/browser recovery stops. Full local raw
ZIPs, independent analyses and source/operator capsule are retained. No
scientific result or verification remains pending for this pilot.
