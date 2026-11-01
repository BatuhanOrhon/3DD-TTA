# LION iki geçişli yeniden kodlama: N3 pilot sonucu — 2026-11-01

[Run/Inference/Decision] Bu sabit ayar accuracy artışı sağlamadı. Aynı koşudaki baseline PM %60.520833, ikinci geçiş PM %58.020833 (−2.5puan), iki geçişin eşit PM birleşimi %59.583333 (−.9375puan). Yeniden kodlanan z, arşivli temiz referansa toplu MSE bakımından %5.140506 yaklaşsa da sınıflandırma iyileşmedi. Operasyonel baseline değişmedi; yeni yöntem opt-in araştırma koludur.

## Sabit deney ve okuma kuralları

Kullanıcının onayıyla ilk geçiş normal8/Background35 olarak korundu. Decoder’ın tam2048×3 LION-frame çıktısı, ek normalization/FPS/rotation olmadan global ve local encoder’a verildi; h1 yeni z1’e koşulludur. Global diffusion kapalıdır. İkinci local zincir tüm bozulmalarda3adımdır: timestep20/10/0 ve kendi alpha20 başlangıç noise seviyesi. Her geçiş kendi başlangıç z’siyle decode edilir. İlk ve ikinci posteriorlarda aynı açık eps_z/eps_h, local diffusion’da aynı Gaussian kullanılır. İkinci SCD hedefi yeni h1XYZ’dir; dolayısıyla bu yalnız-z katkısını izole eden deney değildir. gamma=.03, eta=.003, p=.95, DDIMtotal100/finalalpha1, raw/eval LION/EMAoff, frozen Point-MAE/B32 korunur.

ModelNet40-C/severity5/all15, önceden açığa çıkmış aynı64orijinal nesne, seed0: kol başına960nesne-bozulma olayı vardır;960bağımsız nesne yoktur. Birinci PM normal2/4/6+terminal8, Background9/18/27+terminal35; ikinci PM clean1/2/3+terminal3. Son clean3 ve terminal3 pre-guidance/post-guidance olarak farklıdır. Birleşik PM sekiz adaya eşit, iki geçişe .5/.5 ağırlıktır. Entropy ağırlıklandırma/tuning/early exit/ek seed yoktur. CSV ve NPZ prediction ikinci PM, terminal_prediction/logits terminal tanısını korur.

## Aynı koşudaki accuracy

| Okuma | Doğru /960 | Accuracy % |
|---|---:|---:|
| İlk terminal | 574 | 59.791667 |
| İlk PM (baseline) | 581 | 60.520833 |
| İlk çıktı → encode → decode, ek diffusion yok | 569 | 59.270833 |
| İkinci terminal | 550 | 57.291667 |
| İkinci PM (ana aday) | 557 | 58.020833 |
| İki geçiş birleşik PM | 572 | 59.583333 |

| Eşlenmiş fark | Puan | Koşullu %95 aralık | Düzelten / bozan |
|---|---:|---|---:|
| İkinci PM − baseline PM (birincil) | -2.500000 | [-5.416667, +0.106771] | 31 / 55 |
| Birleşik PM − baseline PM | -0.937500 | [-2.708333, +0.833333] | 19 / 28 |
| Yalnız çevrim − ilk terminal | -0.520833 | [-1.979167, +0.937500] | 25 / 30 |
| İkinci terminal − ilk terminal | -2.500000 | [-4.791667, -0.312500] | 24 / 48 |

Aralıklar10,000draw/seed20261101,64orijinal nesne cluster bootstrap’ıdır; bütün15bozulma birlikte taşınır. Tek seed/exposed cohort/fixed corruptions koşulludur; native tekrar varyansı ve çoklu karşılaştırma düzeltmesi içermez. Birincil ve birleşik PM aralıkları0içerir. İkinci terminal aralığı negatiftir; bu ikincil tanıdan evrensel zarar sonucu çıkarılmaz.

İkinci PM, ilk PM’ye göre2bozulmada artar,4ünde eşit,9unda düşer. Background+3.125 ve Upsampling+3.125puan; Density-inc−10.9375, Uniform/Shear/RBF/Distortion/Occlusion/LiDAR−4.6875puan. Birleşik PM kısmen kaybı azaltır fakat baseline’ı aşmaz. [Tam tablo](evidence/lion_reencode_20261101/per_corruption.csv).

Bu fresh ilk-PM referansı %60.520833’tür; önceki refresh %60.9375 ve oracle %60.8333 farklı koşuların kendi referanslarıdır. Aynı nesne/draw/model kimliği native inference’ın bitwise tekrarını sağlamaz; farkın tek operator nedeni izole edilmedi. Önceki accuracy bu tabloya taşınmadı.

## Bilgi kaybı ve temiz-z tanısı

Yalnız encode–decode çevrimi ilk terminale göre25yanlışı düzeltip30doğruyu bozar: net−.520833puan, aralık0içerir. Ek çevrim bazı örneklerde yararlı karar bilgisini bozabilir; bütün accuracy kaybının zorunlu VAE bilgi kaybından geldiği kanıtlanmadı. Nokta tahmininde ikinci terminalin çevrime göre ek farkı−1.979167puandır; burada yeniden noise ekleme, local prior ve SCD birlikte değişir. h1 yenileme veya SCD anchor’ını tek neden saymak için ablation yoktur.

Temiz referans yalnız posthoc CPU analizindedir. Yeni30fiziksel batch’in tamamı eski clean-z oracle ile original-ID batch adı, prepared input, eps_z/eps_h bakımından exact eşleşir; LION before-state ve checkpoint kimlikleri eşittir. Arşivli clean encoder farklı runtime’dandır; küçük native farklar/normalization etkileri korunur. Gerçek kaydedilmiş ilk z kullanılır, mu+sigma*eps yeniden hesabı başlangıç yerine geçmez.

Toplu clean-z MSE 0.1046791633→0.0992981250; oran0.948594943, azalma%5.140506; olayların%56.875inde mesafe azalır. Bu fiziksel corruption hatası veya accuracy garantisi değildir. Nokta sırası değişebildiğinden h pointwise L2 yorumu yapılmadı. Çevrim öncesi/sonrası simetrik squared Chamfer LION-frame mean0.0021457994, medyan0.0020107217; bu yalnız geometrik değişim ölçüsüdür, temiz şekle hata değildir.

## Kaynak, doğrulama, maliyet ve arşiv

Source commit05e599dd7aec0ad9a68408b8f676274719b62a73; assetmanifest e202f5985c5ff9963f3ecabeb94ad0f8a8b220a40c39c9541ce0abeabc6f91dd; sourceZIP593d953cf7d761a260a544ccd3182d6477e3bba41e7caa401d286d3e1a3d1af3; actual executed-source c19849148f8f5ea12d65e999daaf814181ddb4d8337816be024e162c70d9abf2. Mevcut production baseline/model/preprocessing dosyaları değiştirilmedi; yeni opt-in runner/observer/posterior bridge ve testler eklendi.

12yeni CPU kontratı ve tests/640test+412subtest geçti,1tarihsel fixture skip’i korundu. Bare pytest vendored PointNet2 hydra, extensions/third_party EMD CUDA ve Chamfer importlib.find_loader yollarında dört collection hatası verir; bu macOS CPU ortamının sınırıdır. Bağımsız code review late-policy source-seal ve nonfinite-cycle-logit kusurlarını GPU öncesi yakaladı; ikisi de RED→GREEN ile düzeltildi.

Canonical10setup + gerçek CUDA/native/data/weight verifier passed; yalnız izinli Cell2fastrlock hatası ertelendi. Point-MAE strict/no-forward/no-CUDA deployment, dört pinned no-build native binding ve fresh-native preservation Drive/local kanıtı geçti. CLI prelude bağlantı hatası sonrası ledger/phase/process yokluğu doğrulanarak aynıVM’de retry yapıldı; REQUEST_TIMEOUT60 kullanıldı. Hata ve retry kaynakları korunur. Bilimsel source/ayarlar değiştirilmedi.

Smoke32/Gaussian+Background ve pilot64/all15 fiziksel cache/posterior/noise/RNG/frame/scheduler/own-z/anchor/frozen-model/source/label/PM kontrolleri remote ve bağımsız yerel NumPy1 yeniden hesapta geçti; kabul JSON’ları exact eşittir. Pilot30batch/30original cache/30posterior bridge proof,8,640first/second/cycle head-prediction olayı içerir. Tüm ZIP/member/SHA/CRC/path kimlikleri doğrulandı.

Pilot shared reconstruction/probes/cycle/cache süresi153.352696s; worker+artifact IO190.181432s. Bunlar tam iki-geçişli pipeline ölçümüdür; ilk encoder hazırlığı observer timer’ından önce yapıldığından ilk-geçiş dispatch timer’ı tam tek-pass deployment karşılaştırması sayılmaz. Fresh matched end-to-end tek-pass hız karşılaştırması yoktur. İlk denenen CLI prelude başlangıcından son remote raw arşive535.326741s=8.922112dk (55dk cap); bunun dışındaki canonicalsetup, local download/recount ve sonraki CPU/Git işleri ayrıca ayrılır.

Raw ZIPler: smoke25,178,429bytes SHA78c6748a1318a7edd673c151f959ad26e418ba0d2b22593e9fccdd0126d30223; pilot335,216,850bytes SHA31309af96daa6a19286756599dc3d76203e10d7ac995e92c675786b15b4229af. Drive MyDrive/thesis/result/lion_reencode_20261101/phase-evidence/. Yerel hamlar result/preparation/lion_reencode_20261101/operator/{smoke,pilot}_raw.zip; extracted hamlar değişmedi, local_analysis ve bridge_diagnostics ayrı türetilmiştir. Canonical head ZIPler result/modelnet40_c/3dd_lion_reencode_n3_pilot/20261101-092420_pointmae-seed0.zip (technical) ve20261101-092702_pointmae-seed0.zip (pilot).

## Karar ve açık sorular

[Decision/Open] Bu N3 varyantını baseline’a alma. Temiz-z’ye ortalama yaklaşma ile accuracy kazancı ayrı şartlardır; ikinci geçiş bu pilotta fayda göstermedi. N5 veya farklı ikinci noise derinliği, z-only/h-only/anchor kontrolleri, ek seed/unseen cohort ve maliyet-eşlenmiş uzun tek-zincir karşılaştırması yapılmadı. Bu sonuç bütün yeniden kodlama yöntemlerini çürütmez. Entropy ağırlıklandırma ile sonucu kurtarma/tuning denenmedi. Kullanıcının sonraki kararı olmadan ek GPU işi başlatılmadı. Kendi tta-lion-reencode-a100-20261101-r1 runtime’ı korunur; task-complete+close izni verilmedi. Son arşiv doğrulaması ve yayın kapsamı aşağıdadır.


## Son koruma ve entegrasyon kaydı

[Run/Code] Drive final-archive-20261101T094001Z/lion_reencode_final_capsule_v1.zip,
2,859,620bytes/SHA774708ddeb95b69de1bc390a1c4e42fb5658e534a3fa663adc105053aabe15a7,
Drive yolundan yeniden indirildi; CRC/path/type ve75manifest üyesi doğrulandı.
[Roundtrip makbuzu](evidence/lion_reencode_20261101/archive_roundtrip.json).
Bu kapsül son yayın metadatasından önceki snapshot’tır; ham bilimsel ZIPler değişmedi.

Source branch research/lion-reencode-20261101, source05e599d, base dbbdbee.
Seçilen kapsam opt-in observer/runner,12kontrat, bağımsız analiz ve kompakt kabul
kanıtları ile tarihli knowledge/plan kayıtlarıdır. Geçici setup/operator dosyaları,
ham ZIPler, posterior/cache ve ağırlıklar boyut/tekrar/denetim ayrımı nedeniyle
Git dışında yerel+Drive’da korunur. Başka çalışma alanlarının değişiklikleri alınmadı.
640test+412subtest/1skip, bağımsız review ve raw recount kanıtı geçerlidir;
bu doğrulamadan sonra scientific source/test byte’ları değiştirilmedi.
Normal fast-forward hedefi origin/thesis-integration’dır; oluşan commit ve
push doğrulaması yerel operator/publication_receipt.json ve Drive yayın
sidecar’ında kaydedilir. Runtime için kapatma izni yoktur; korunur.


[Run/Decision] Sonraki kullanıcı onayıyla N5 ve N8 test edildi; ilk bu rapordaki bekleyen horizon sorusu için [2026-11-01 follow-up sonucu](lion_reencode_depth_results_20261101.md) geçerlidir. Orijinal N3 sayıları ve arşivleri değiştirilmedi.
