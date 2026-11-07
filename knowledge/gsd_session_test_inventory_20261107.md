# GSD sohbeti: deney sırası, sonuçlar ve çalıştırılmayan aşamalar

Kayıt tarihi: 2026-11-07. Deneyler: 3–6 Kasım 2026.
Kanıt etiketleri: [Code], [Run], [CPU], [Inference], [Open].

[Code/Run] Kullanıcının sohbet içindeki adım adım özetini knowledge'a kaydetme
isteği üzerine, güncel `origin/thesis-integration@ecba1e4` ile uzlaştırıldı.
Bu kayıt mevcut kabul edilmiş raporların sentezidir; yeni GPU koşusu, yeniden
ham arşiv kabulü, hiperparametre seçimi veya yeni bilimsel sonuç değildir.
[5–6 Kasım toplu sonuçları](gsd_comprehensive_results_20261106.md) ile
[yayın hazırlığı değerlendirmesi](publication_readiness_20261107.md) korunur.

**[Decision] Tamamlanan kapsam zorunlu tanı aşamalarıdır; planlanan bütün
accuracy/optimizasyon kollarının çalıştığı anlamına gelmez.** Eşzamanlı accuracy
ve hız kazancı gösterilmedi. Başlamayan bir koşu başarısız accuracy deneyi
olarak etiketlenmez. Buradaki GSD, projedeki GSD-inspired guidance yoludur.

## 1. Matematik ve kod denetimi

[Code/CPU] Tam ortonormal bazda `U(UᵀY)=Y`; sabit U tek başına inverse hatası
değildir. Kesilmiş baz izdüşümdür. Tensor yerleşimi/transpose hatası bulunmadı.
Sentetik kontroller, geometry-dependent grafın tam türevi ile frozen-metric
türevinin farklılığını ve decoder'ın latent bant enerjisini başka bantlara
aktarabileceğini gösterdi; bunlar gerçek LION başarısı olarak sunulmadı.
Legacy/component yolunda ölçüm için tekrar hesaplanan spectral/Chamfer forward
çağrıları adım başına2→1 indirildi; invariant metric uygulanan cutoff ile
eşlendi. Son guidance yolunun iki kat hızlandığı iddiası yoktur.
[Matematik,11birincil kaynak ve CPU tanıkları](gsd_guidance_math_efficiency_20261103.md).

## 2. Encoder girdisi ile PM decoder çıktısının fiziksel spectral loss'u

[Run] Encoder'a giren hazırlanmış X'ten sabit graf kuruldu; mevcut PM decoder
çıktısı Y aynı metrikte karşılaştırıldı ve SCD'ye eklendi. İlk Gaussian44/91,
B2/seed0 testinde decoder türevi iki sabit ölçekte FD ile uyuştu: göreli hatalar
yaklaşık%2,25/%5,99; bağımsız tekrar da geçti. İlk dokuz koşul aynı iki nesnede
PM2/2 verdi. Teknik uygulanabilirlik gösterir; accuracy artışı veya tüm Jacobian
için sertifika değildir. [İlk native deneyler](decoded_physical_spectral_results_20261103.md).

## 3. H8 yön yaklaşımı ve reddedilen V5 optimizasyonu

[Run] Frozen decoder residual ile backward maliyetini azaltan H8'in ilk altı
denoiser-sonrası yön karşılaştırması geçti; minimumcosine.999839. Küçük B2
reconstruction/live0/full/H8 süreleri1.869/2.017/1.896s idi; graf kurulumu hariç,
tek ölçüm. Decoder forward grafını da kaldıran V5'in ilk RMS farkı1.2423e-5,
sabit limit1e-5 olduğundan kabul edilmedi. Optimizasyon geri alındı; ona bağlı
üç applied/timing koşusu başlamadı. Sonraki kontrollü backend bulguları bu
reddedilmiş koşuları geriye dönük başarılı yapmaz.

## 4. P0: backend ve tekrar kontrolü

[Run] Aynı Gaussian çiftinde legacy/controlled backend için üçer process:
en kötü logit farkı2.4244→5.57899e-5, ilk decoded RMS.01074→1.12835e-7.
Kontrollü ayarlar bu örneklerde bütün kontrolleri geçti. [Inference] Birlikte
değiştirilen bayraklardan tek sorumlu ayrıştırılmadı; custom CUDA işlemleri
deterministik ilan edilmedi. Sonraki Background başarısızlığı bu dar Gaussian
sonucunu geçersiz kılmaz; genellenemediğini gösterir.

## 5. P1/P2: geniş H8 yön kontrolü ve anchor karşı örneği

[Run]8nesne×4bozulmadaki H8 yön tanıkları96/96geçti; minimumcosine.995612.
Temiz bulut yalnız sonradan oracle olarak kullanıldı.64küçük çıktı-uzayı adımı
spectral loss'u azalttı;42adımda temiz Chamfer uzaklığı arttı. Bunların1/4'ü
doğrudan archived-GPU-F satırlarında,41/60'ı yalnız scalar-kontrolü geçmiş
CPU-reconstructed filtre satırlarındaydı;64bağımsız nesne veya64GPU-filtre
sertifikası değildir. İki CPU/GPU F rekonstrüksiyonu sabit1e-6limitini geçemedi.
Actual-GPU-F karşı örneğinde spectral loss.01439452→.01408464 iken temiz
CD.05052449→.05054426 oldu. [Inference] Bozuk X'e spectral sadakat temizliği
garanti etmez; bu bir sampler/accuracy zarar teoremi değildir.
[P0/P1/P2 raporu](decoded_physical_followup_20261104.md).

## 6. Tam diffusion akışıyla gerçek GPU accuracy pilotu

[Run]32farklı development nesnesi×15bozulma, s5/seed0/B2; kol başına480
değerlendirme. Bu bütün test kümesi değildir. PM önceden birincil seçildi.

| Kol | PM doğru/480 | PM accuracy | Terminal accuracy |
|---|---:|---:|---:|
|Operasyonel SCD+PM|313|%65,2083|%64,5833|
|Live observer, spectral ağırlık0|315|%65,6250|%64,5833|
|SCD+full physical|313|%65,2083|%65,4167|
|SCD+H8|312|%65,0000|%65,4167|

[Run]Full−live0 PM farkı−.4167yüzde puan,95%32nesne-clusterCI
[−1.6667,+.8333]; kazanç/zarar kurulmadı. Control/live0 sayısal kapısı45/240B2
durumda başarısız, Background16/16dahil; spectral ağırlık0iken iki PM sınıfı
değişti. Fiziksel filtre hash'leri240/240aynıydı. H8 full'e göre reconstruction
süresini%11,66azalttı; operasyonel SCD'ye göre graf+reconstruction%9,30,
worker-wall%20,10yüksekti. Terminal artışını PM başarısı yerine koymadık.
Bu eşlenmiş-cache pilot zamanları deployment throughput değildir.
[Accuracy,CI,maliyet ve başarısız kontrol](decoded_physical_accuracy_20261104.md).

## 7. Aynı observer altında sıfır spectral ağırlıklı tekrarlar

[Run]Gaussian44/91 ve Background495/512, control/live0, üçer yeni process:
12koşu. Gaussian within-arm3/3ve cross-arm9/9geçti; Background karşılıkları
0/3ve0/9. SCD aktifti, ek spectral ağırlık0. PM sınıfları bu seçili tekrarlarda
aynı kaldı. [Inference] GSD grafı veya observer değiştirmek Background
kararsızlığı için gerekli değil; bu yeni accuracy kaybı ölçümü değildir.
[Aynı-observer tekrarları](gsd_zero_repeat_20261104.md).

## 8. H5: güvenilir bantları sıralama

[Run/CPU]8nesne×4bozulma×3process×4checkpoint×6bant=2304satır.
1980uygun oracle adımı:1263fayda/717zarar;324boş bant. Skorlar ortak1485satır
üzerinde,2000bootstrap draw ile8özgün nesneye göre kümelenerek değerlendirildi.

| Skor | AUROC |95% nesne-clusterCI |
|---|---:|---|
|Frekans|.299841|[.171317,.419095]|
|Artık|.260924|[.131263,.370684]|
|Birleşik|.271341|[.137627,.386096]|

[Inference/Decision] AUROC sınıflandırma accuracy'si değildir; yararlı adımları
zararlılardan sıralama ölçüsüdür. Birleşik skorun CI'ı.5altında; frekans ve artık
karşısında gerekli iki üstünlük yok. Native tekrar kapıları da başarısız; ek
spectral ağırlık0kontrollerinde PM23/32,22/32,24/32. H5 NOT_ADVANCED; işaret
ters çevirme/skor tuning'i veya applied-q accuracy koşusu yapılmadı.
[H5 feature/oracle ve native kontrol](h5_reliability_results_20261105.md).

## 9. H2/H3: gerçek graf eskimesi ve decoder bant aktarımı

[Run]4nesne×4bozulma×ilk/son clean-PM durumları=16B2durumu. H2'de X/Y
grafının aynı artık üzerindeki filtre-yönü medyan cosine'ları.335–.523,
göreli farkları.946–1.077. Graf eskimesi ölçüldü; gerçek refresh guidance faydası
test edilmedi. H3'te iki bant/iki büyüklük/iki işaret ile128B2perturbasyonu,
256bağımlı nesne gözlemi. Düşük-latent perturbasyonun fiziksel other-active
enerji medyanı identity-XYZ'de%16,38, decoder sonrasında%50,60; sampled native
repeat floor üzerinde. Other-active, fiziksel düşük bandın tümleyenidir;
yalnız fiziksel top100 veya izole vertex enerjisi değildir. [Inference] Gerçek
decoder latent düşük frekansı fiziksel düşük frekans olarak korumuyor; bu inverse
implementasyon hatası ya da accuracy zararı kanıtı değildir.

## 10. H1/H6/H7: ortak durumda guidance yönü ve classifier okuması

[Run]Zero/C/S_lat/S_phys/P100_C/CS_lat/CS_phys yönleri iki sabit adım oranında
denendi. C FD3/15, latent7/15, fiziksel8/15geçti; bir başka durum forward'da
durdu. FD başarısızlığı yanlış autograd ile özdeş değildir; sonlu epsilon,
NN/retained-set değişimi ve native farklar ayrıştırılmadı, tolerans değişmedi.
224olası B2bulutun66'sı mevcut;158anahtar unavailable/failed nedeniyle korunur.

[Run/CPU]Bütün kolların ortak karşılaştırılabildiği3B2durumu4özgün nesne,
6bağımlı nesne-durum satırı içerir. Her nonzero kol iki oranda da4/6doğru;
doğru/yanlış örnekleri de aynı. P100_C XYZ-norm/feature/style sözleşmesi
doğrulandı, fakat iki oranda tutarlı geometri faydası yok. Physical heat C'yi
temiz-CD ortalamasında iki oranda da geçemedi; bazı CS yönleri temiz-CD'yi
iyileştirip sınıf marjını kötüleştirdi. Faydalı C'yi CS'nin bozduğu H1 örneği
bu uygun altkümede görülmedi. Bu tek sampler güncellemesi+sonraki tahminin
tanısıdır; tam trajectory/PM benchmark veya SCD/2+GSD/2 değildir.
[Ortak-durum kolları, kapsam ve marjlar](gsd_shared_probe_results_20261106.md).

## 11. H4: eigenfree aynı-heat loss ve native maliyet

[Run/CPU]Gerçek aynı graph/support ile dört bozulmada K8/16/32. K16/32loss
kontrolleri32/32geçti, gradyan0/32geçti; göreli hata1.619e-6–4.465e-6,
sabit limit1e-6. FP64 archived-operator incelemesinde U/q ve saklanan L
arasındaki fark yüksek-K truncation hatasına baskındı; wrong-backward kanıtı
veya tolerans gevşetmesi yok. B32 zamanları açık birimle:

| Kapsam | Exact U/q | Sparse K16 |
|---|---:|---:|
|Graf/filtre kurulumu|1,568–1,646 saniye|0,301–0,304 saniye|
|Hazır filtreyle loss+artık VJP|11,72–12,36 ms|386,64–402,99 ms|

[Inference/Decision]Kurulum ucuzladı, tekrar kullanılan loss/gradyan yavaşladı.
Mevcut Python/seyrek recurrence implementasyonu hız çözümü değil; bütün sparse
yöntemler için sonuç değildir. B32 iki nesnenin timing için tekrarından oluşur,
32bağımsız accuracy nesnesi değildir. Bütün-akış adayı NOT_ADVANCED.
[Native eşdeğerlik,FP64 ayrıştırma ve maliyet](h4_sparse_heat_results_20261106.md).

## 12. H0: ilk farkın yerelleştirilmesi ve exact-RNG prior tekrarı

[Run]6yeni trajectory koşusunda5mevcut sınır kaydedildi.6repeat çiftinin ilk
farkı loop1 local-prior noise_pred'de, exact giriş/RNG ile SCD/backward/PM'den
önceydi. Sonra aynı process içinde G/B için üçer actual prior forward ile
yaklaşık1e-6fark yeniden üretildi. İlk covered exact-input numeric leaf farkı
`sa_layers.0.0.voxelization` feature çıktısında; koordinatlar bitexact kaldı.
[Code/Inference]Kaynak zincirindeki FP32 atomicAdd(feat/count), toplama
sırasına bağlı rounding için makul açıklama.64MiB reference cap ve hook
zamanlama etkileri var; tek kernel nedeni, bütün amplification zinciri veya
deterministik tamir kanıtlanmadı. [H0 ve modül tanığı](h0_native_trace_results_20261105.md).

## Başlamayan veya sınırlı kalan çalışmalar

| İş | Kayıtlı durum / neden |
|---|---|
|HardM100/M200, clean-PM'de SCD/2+GSD/2|Ayrı agentta CPU kod/test hazır;4allocation başarısız, sonra kullanıcı bekleme talimatı. GPU setup/teknik test/alpha kalibrasyonu/beş-kol pilot/hız/accuracy başlamadı. NOT_RUN; yöntem başarısızlığı değil.|
|Applied H5 q ve nesne-ayrık doğrulama|H5 sıralama/native önkoşulları başarısız; NOT_ADVANCED.|
|Gerçek H2 graph refresh guidance|Yalnız operator drift ölçüldü; faydalı refresh/sham karşılaştırması çalışmadı.|
|Geniş H6/H7 accuracy|Native geçerlilik/tutarlı iki-oran faydası yok; NOT_ADVANCED.|
|Chebyshev bütün diffusion akışı/fresh deployment|Eşdeğerlik geçmedi, cached maliyet olumsuz; çalışmadı.|
|V5 applied/timing üç koşu|Forward gate reddi sonrası başlamadı; optimizasyon geri alındı.|
|Bozuk X yerine alternatif anchor ile accuracy|Bu kampanyada applied accuracy yok. Ayrı [Nov5standalone frozen-state anchor A/B](gsd_hypotheses_test_inventory_20261107.md) tamamlandı; üstünlük gösterilmedi. Bu mekanizma testi tam trajectory accuracy karşılaştırması değildir.|
|Yeni geometrik correspondence'ı guidance'a koymak|Bu kampanyada permütasyon/set-CD kontrolleri yapıldı. Ayrı [Nov5robust surface yönleri](gsd_hypotheses_test_inventory_20261107.md)392frozen durumda ölçüldü; ortalama fayda belirsiz, applied accuracy deneyi yine yok.|
|H3 feature/style'a ayrı perturbasyonlar|Bu kapsamın actual perturbasyonları latent XYZ bantlarıydı; ayrı feature/style kampanyası yok.|
|Deterministik voxelization ve yeniden accuracy|Yalnız module-level localization; kernel rewrite/onarım ve sonrasında accuracy koşusu yok.|
|Yeni adayın full-test/multi-seed confirmation'ı|Geliştirme önkoşulları sağlanmadı; başlamadı.|

[Code/Open]Ayrı halfmix'in son kayıtlı durumuna dayanan bu envanter, yeni
allocation/setup izni veya peer çalışmasını iptal değildir.
[Halfmix kodu,allocation ve kullanıcı beklemesi](gsd_lowband_halfmix_study_20261105.md).
Önceki spectral-z ve reencoding çalışmalarının sonuçları bağlam olarak
kullanıldı; bunları bu sohbetin yeni GPU koşuları gibi saymıyoruz.

## Başarısız analizler, yazılım kontrolleri ve kapanış

[Code/CPU]H5 tiny-float32-eigenvalue reversal kontrolü ile native-trace source
alias kontrolünün ilk analizleri başarısızdı; dar validation onarımı ve taze
çıktı dizinleriyle tamamlandı. Ham q/U sırası, bilimsel tolerans ve feature/skor
formülleri değiştirilmedi. CPU/GPU F1e-6ve V5 reddi ise başarısız olarak kaldı.

[CPU]Son temiz integration kontrolü807pass/2skip/49.60s. İki skip tarihsel
sealed-source ve Git dışı ham reencode fixture'ına bağlıydı; GPU accuracy
koşusu sayısı değildir. Yeni documentation kaydı bu suite'i yeniden çalıştırmaz.
[Kabul edilmiş test makbuzu](evidence/gsd_comprehensive_20261106/integration_validation.json).

[Run]Ownr4,5Kasım22:04UTC/6Kasım01:04Türkiye saatinde kapatıldı; monitor da
durdu. Kapanıştan önce18ham/türetilmiş arşiv ile nihai195manifest dosyası
Drive/yerel roundtrip doğrulandı. Bugünkü envanter bunları yeniden doğrulamış
gibi sunmaz. [Kapanış](evidence/gsd_comprehensive_20261106/closure.json),
[ham indeks](evidence/gsd_comprehensive_20261106/artifacts.json),
[nihai analiz paketi](evidence/gsd_comprehensive_20261106/final_backup.json).

[Inference]Sabit U inverse kusuru bulunmadı; hedef/reference yanlılığına karşı
örnek, graph drift, decoder band mixing ve native nonrepeatability için ayrı
kanıtlar var. Tek bir nedenin bütün accuracy sonucunu açıkladığı veya bir
onarımın kazanç getirdiği kanıtlanmadı. Baseline/default korunur; yeni GPU yok.


[Run/Inference, later Nov7 update] The independently owned [hardM100/M200 halfmix study](gsd_lowband_halfmix_results_20261107.md) subsequently completed all34planned jobs in40.33minutes;33rawruns/136actualcaches accepted. Frozen alpha and all five16-ID/all15controls are available. Native zero-guidance gate fails30/120B2cases; accuracy remains descriptive and no default is promoted. Earlier halfmix-unrun/user-wait entries are historical inventory snapshots.
