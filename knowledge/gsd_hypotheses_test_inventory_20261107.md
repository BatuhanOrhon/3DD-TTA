# GSD/LION bağımsız H1–H4 çalışması: test envanteri ve eksikler

[User report/Run/Code/Inference/Open] 7 Kasım 2026'da kullanıcı, oturumdaki bütün testlerin adım adım açıklanmasını, başarısız/yarım kalan ve başlanmayan testlerin ayrılmasını, ardından knowledge kayıtlarının integration dalına yayımlanmasını istedi. Bu envanter 4–5 Kasım çalışmasının `research/gsd-physical-followup-20261104@acf78fb` kayıtlarını özetler; yeni GPU sonucu değildir.

**Kapsam:** Bu çalışmanın yerel H1=anchor, H2=robust surface, H3=proximal graph, H4=latent lift etiketleri, [daha sonraki kapsamlı H0–H8 kampanyasının](gsd_comprehensive_results_20261106.md) etiketleriyle aynı değildir. Örneğin buradaki H4, sonraki sparse-heat H4 testi değildir. Aşağıdaki “başlanmadı” ifadeleri yalnız bu bağımsız önerinin 5 Kasım kapanışındaki kapsamıdır; 6 Kasım'da ayrıca tamamlanan küçük ortak-durum classifier/decoder/graph/native testlerini yok saymaz.

[Genel 3–6 Kasım sohbet envanteri](gsd_session_test_inventory_20261107.md) ile [7 Kasım sonraki guidance adayları](guidance_lion_candidates_20261107.md) ayrı güncel kayıtlardır; bu import her ikisini korur.

[Sonuçlar ve matematik](gsd_hypotheses_results_20261105.md), [ayrıntılı yürütme/failure geçmişi](gsd_hypotheses_gpu_20261105.md), [literatür sentezi](gsd_lion_spectral_literature_20261104.md).

## 1. Tamamlanan testler, yürütme sırasıyla

| Aşama | Kapsam ve sonuç | Bilimsel anlamı |
|---|---|---|
| Literatürden matematiksel öneri | 33 çalışma/35 PDF; odaklı okuma bölümleri, URL ve SHA kayıtları | Generic graph-posterior residual formülü için yenilik iddiası yok; gerçek model faydası ayrıca sınanmalı |
| Sentetik CPU: sparse cebir | 15 düğüm/23 gözlem; açık matris–matrix-free, SPD, CG/doğrudan çözüm, gradient/FD, Huber amaç azalımı ve sınırlı adım | CG relative çözüm hatası1.55e-11; doğru optimizasyon, doğru hedef garantisi değildir |
| Sentetik CPU: değişmezlik/spectral özel durum | Permütasyon, normal-sign, sıfır gözlem, izole düğüm; izotropik spectral çözüm, nonuniform precision ile komütasyon sınırı | Spectral skaler bölme yalnız özel koşullarda geçerli |
| Sentetik CPU: aktarım ve karşı örnekler | Recomputed-epsilon DDIM; toy residual decoder sınırı; düzlemde tangential resampling; coherent wrong anchor | Toy lift geçmesi LION lift'ini sertifikalamaz; amaç azalırken temiz-MSE artabilir |
| Gerçek uygulamanın CPU tanıkları | İki düğümlü tam delta[3/8,1/8], sıfır gözlem, düzlem normalleri/normal-sign, bounded Huber influence; graph-free bloklar tek PCG iterasyonu | Kenar işareti/ağırlık, normal yönü ve preconditioner kontrolleri geçti |
| Colab hazırlama | İki standart A100; canonical10cells, native EMD/Chamfer/PointNet++/PVCNN, veri/checkpoint son verifier | Yaklaşık83GiB host/40GB GPU; B2 mekanizma koşularında OOM yok; bilinen fastrlock ertelemesi ve pip uyarıları saklandı |
| Teknik model pilotu | B2, IDs44/91, Gaussian/Background/Shear/LiDAR;8durum/grup | A'nın ilk8durumunda üç yön de clean-CD azalttı; yöntem üstünlüğü kabul edilmedi, matris genişletildi |
| Ana matris | 8exposed nesne,15bozulma,severity5,seeds0/1/2:360ana durum/grup | Tek noisy-to-clean tahmin; normal t70/Background t340; style sabit, classifier/PM/full reverse yok |
| Ek timestep matrisi | 4nesne/seed0,t10 ve alternatif t70/t340:32durum/grup | Teknik durumlar tekrar sayılmadan toplam392durum/grup;18yeni koşul+teknik pilot=19paket/grup |
| H1 anchor | Aynı F_X/adım sınırı: raw X versus encode/decode R; reconstructed−raw CD+2.711e-6,95%CI[-1.458e-5,+1.904e-5] | Üstünlük gösterilmedi;247/360başlangıca göre iyileşme, raw'dan daha iyi olma anlamına gelmez |
| H2 robust surface | Unordered nearest-surface/PCA normal/Huber; robust−raw CD−5.332e-5,CI[-1.704e-4,+4.685e-5] | Ortalama umut verici, belirsizlik sıfırı içeriyor; hem eşleşme hem amaç değiştiğinden tek bileşene atfedilemez |
| H3 graph katkısı | Aynı operatörde kappa1/lambda0 versus1; graph1−graph0 CD+3.859e-5,CI[+1.002e-5,+7.011e-5] | Sabit graph bu kohortta ortalamada daha kötü; graph0/1 başlangıca göre249/234durumda iyileşir |
| H3 sayısal çözüm | 784PCG/784yakınsama, en çok9iterasyon/9.98e-7gerçek relative residual; her capped adım frozen-Huber amacı azaltır | Amaç/sayısal başarı ile temiz-geometri faydası ayrı ölçülür |
| Bağımsız sparse oracle | 16teknik lambda0/1 çözümü SciPy CSR/CG ile | Delta max5.84e-7, optimum amaç relative3.55e-11 fark; gerçek sistem doğrulandı |
| H4 finite latent lift | İlk3latent kanala fiziksel delta; style/dördüncü kanal sabit;rho.001/.01 | 360ana durumda median relative hata149.8%/61.4%; identity-lift varsayımı desteklenmedi |
| Actual DDIM ve repeat kontrolleri | Recomputed-epsilon actual scheduler; ayrıca t340/70/10/0terminal; aynı girdide prior/decoder tekrarları | Scheduler cebri geçer; küçük frozen-state repeat RMS tüm trajectory kararlılığı anlamına gelmez |
| H4 posthoc CPU clean geometry | Arşivlenmiş changed çıktısı versus ideal Y+delta ve başlangıç;784lift kaydı | Gerçek hareket idealin faydasını kaybediyor; iki rho'da actual−baseline CI sıfırı içeriyor, her örnekte zarar iddiası yok |
| Ek109GPU diagnostic | 4scheduler+8yeniden kurulan graph permütasyonu+96decoder alanı+1coherent wrong-anchor | 8durum×4alan×3rho: öteleme çoğunlukla doğru, yerel alanlar sapıyor; adımı küçültmek genel çözüm olmadı |
| Permütasyon ve yanlış anchor | H2observation-order392durum max5.18e-16; yeniden Y-graph+PCG8durum max1.78e-14; yanlış düzlemde amaç−.714475,clean-MSE+.000786519 | Gözlem sırası kontrolü geçer; coherent yanlış gözlem bias'ını graph/Huber otomatik çözmez |
| Bileşen maliyeti | 8cached durum,B1,1warmup+3senkron tekrar;56kayıt | Fkurma51.89ms fakat cached uygulama.095ms; graph1.64ms,observation.47ms,PCG0/1 2.34/6.11ms,robust hazır-RHS.085ms; full-SCD speedup değildir |
| Bağımsız CPU/arşiv doğrulaması | 38ana paket, CRC/member/packageSHA, actual F/yön/Huber/decode/scheduler, count/source;784cosine yeniden hesabı | Geometry max2.63e-7, yön9.10e-9;392durum sekiz bağımsız nesnenin ötesine genellenmez |
| Kurtarma/yedek/kapanış | A-r3beklenmedik kaybından11sealed koşul kurtarıldı, yalnız kalan11–17B'de tamamlandı; final218member Drive roundtrip | Ana matris eksiksiz; B/own monitor kapatıldı,5Kasım19:57UTCabsence/peer preservation doğrulandı; bugün yeniden runtime sorgulanmadı |

CD farkları hazırlanmış koordinatlardaki squared Chamfer birimleridir; accuracy yüzdesi değildir. Aralıklar10000eşlenmiş8-object-cluster bootstrap ile yalnız bu kohorta koşulludur. Bozulma altanalizleri keşifseldir. A/B grupları farklı taze model durumları içerir; gruplar arası mutlak ortalama sıralaması yapılmaz.

Ek timestep grubunda dört nesne vardır; H2ortalamasının işaret değiştirmesi yalnız timestep'in nedensel etkisi olarak yorumlanmaz. Ana prior/decoder repeat kontrolleri sonraki Nov6native localization sonuçlarını geçersiz kılmaz.

## 2. Başarısız, kısmen doğrulanmış veya kurtarılamayan işler

| İş | Son durum |
|---|---|
| İlk payloadv2 checkpoint yüklemesi | use_ema API hatası; örnek hesaplamadan durdu. V3gerçek raw-loader API'siyle tamamlandı; ilkfailure korundu |
| İlk recovery launcher | Eski archival-source yolu eksikti; CPU/science başlamadan durdu. __file__ düzeltmesiyle fresh retry tamamlandı |
| CLI result-channel takılmaları | Gerçek process/completion marker kontrolüyle çözüldü; tamamlanmış native build yeniden başlatılmadı |
| Gerçek VJP/backward | **FAILED.** Hazırlama checkout'ındaki eval PVCNN devoxelization gerekli backward tensörlerini saklamıyordu |
| Central finite difference | Aynı no-grad forward ile tamamlandı; **kesin VJP'nin tamamlanması değildir**; model train moduna/native forward patch'e geçirilmedi |
| Supplemental central-slope ve graph permutation bağımsız recount | **Eksik kanıt.** Plus/minus decode ve permuted sparse diziler saklanmadı; sonuçlar GPU kaydıdır. Finite hareket/scheduler NPZ'leri bağımsız doğrulandı |
| A-r3unsealed condition11 cache ve console-only setup logları | **UNAVAILABLE.** VM kaybından geri alınamadı; ilgili fixed koşul B'de tamamlandı |
| İlk CPU analyzer | Float64 Y+delta değerlendirmesi actual float32 işlemle aynı değildi; V1korundu, V2aynı float32 toplamayla yeniden hesaplandı |
| fastrlock/pip genel tutarlılığı | Bilinen fastrlock hatası kullanıcı kararıyla ertelendi; resolver warning'leri saklandı; bütün bağımlılık tutarlılığı sertifikalanmadı |

**Ana392durumluk matrislerde yarım koşul yoktur.** Başarısız VJP, eksik intermediate kanıtlar ve kaybolan orijinal ara cache bununla karıştırılmaz. Kaynak hazırlama dev@a458cd4 ile yerel araştırma512b5aa farkı arşivlidir; tam güncel üretim pipeline replikasyonu iddia edilmez.

## 3. Bu bağımsız öneri için başlanmayan testler

| Test | Kapsam sınırı |
|---|---|
| Tam paired SCD/on/off ve classifier accuracy | Bu çalışmada classifier çalışmadı; H1raw fiziksel spectral kolu standart latent-SCD benchmark'ı değildir |
| Tam reverse diffusion ve PM | Tek temiz tahminde ölçülen düzeltmenin bütün akıştaki etkisi bilinmiyor |
| Nesne-ayrık validation/confirmation | Sekiz nesne previously exposed;392durum,392nesne veya heldout benchmark değildir |
| Tüm ModelNet40-C nesneleri/diğer severity'ler | All15bozulma kapsandı; tam nesne dosyaları ve severity1–4kampanyası yok |
| B1/B32 end-to-end hız/toplam bellek |56bileşen kaydı tam sampler+backward+classifier+PM maliyeti değildir |
| ScanObjectNN/farklı classifier-backbone aktarımı | Bu yeni robust proximal/lift önerisi için yapılmadı |
| Geniş lambda/graph/normal-confidence kalibrasyonu | Kappa1,lambda0/1 ve sabitheuristic ayarlar; optimum ya da kalibre Bayesian model yok |
| Decoder sonrası physical refinement | Yeni sonraki aday; uygulanıp accuracy açısından sınanmadı |
| Decoder-aware düşük boyutlu lift | Ölçülmüş decoder cevabına göre inverse tasarımı yalnız hipotez; implementasyon/sonuç yok |

[Run/Inference] Sonraki [6Kasım ortak-durum/native/heat kampanyası](gsd_comprehensive_results_20261106.md) kendi küçük classifier/probe sonuçlarını içerir; bu tabloda ona “yapılmadı” denmiyor. Aynı şekilde peer [hard100/200 halfmix çalışması](gsd_lowband_halfmix_study_20261105.md) bu göreve katılmadı ve bu yayın onu başlatmaz/durdurmaz.

[Decision/Open] Sparse optimizasyon ve DDIM cebri doğru çalıştı. H1/H2üstünlüğü unestablished, uniform graph katkısı bu kohortta negatif, identity lift unsupported. Operasyonel Chamfer+PM baseline değişmedi. Yeni yöntem, accuracy/hız üstünlüğü veya GSDTTA replikasyonu iddiası yoktur.
