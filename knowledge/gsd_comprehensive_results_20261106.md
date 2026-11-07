# GSD kapsamlı tanı kampanyası — 5–6 Kasım 2026

[Run/CPU/Inference] Bu kampanyada **eşzamanlı accuracy ve hız artışı gösterilmedi**.
H5'in sabit güvenilirlik skoru beklenen yönü sıralamadı; kalan ortak-durum
deneyleri geniş accuracy koşusuna geçmek için gereken önkoşulları sağlamadı.
Bu, bütün spektral yöntemlerin başarısız olduğu iddiası değildir.

## Okuma sırası ve kapsam

- [Sohbetin tamamı: 3–6 Kasım deney sırası ve başlamayan testler](gsd_session_test_inventory_20261107.md).
- [Önceden dondurulan uygulama/ilerleme kuralları](gsd_comprehensive_execution_20261105.md).
- [H5: ham kabul, sayısal doğrulama ve nesne bootstrap sonuçları](h5_reliability_results_20261105.md).
- [H0: ilk prior/SCD/gradyan sınırı tanısı](h0_native_trace_results_20261105.md).
- [Ortak-durum H1/H2/H3/H6/H7 sonuçları](gsd_shared_probe_results_20261106.md).
- [H4: native eşdeğerlik ve bileşen maliyeti](h4_sparse_heat_results_20261106.md).
- [Birincil literatür ve matematik denetimi](gsd_guidance_math_efficiency_20261103.md).

[Run] ModelNet40-C s5/seed0; PointMAE ve LION raw/eval/EMA-off, B2,
N8/Background35, DDIM100/final-alpha1, gamma.03/eta.003/p.95.
H5: sekiz önceden açığa çıkmış ID, dört bozulma, üç yeni process.
Ortak-durum: bunların dört ID'si ve ilk/son clean-PM durumları.
H0: daha önce kabul edilmiş Gaussian44/91 ve Background495/512 çiftleri.
H4: H5 ilk tekrarının459/545 çifti, dört bozulma, gerçek A10080GB.
Bu koşular yeni heldout benchmark veya seed tekrarı değildir.

## Matematiksel sonuç ve kullanıcının decoder fikri

[Code/Inference] Aynı ortonormal tam bazda `X=U(UᵀX)` hâlâ doğrudur.
Noktaları güncelledikten sonra aynı U'yu kullanmak kendi başına ters dönüşüm
hatası oluşturmaz. Kesilmiş baz bir izdüşümdür; farklı graf bazlarının
katsayılarını doğrudan karşılaştırmak ise aynı metrik değildir.

Encoder öncesi X grafını sabitleyip PM decoder çıktısı Y ile karşılaştırmak
uygulanabilir; fiziksel denemeler bunu zaten yaptı. Sabit heat metrik için:

\[
 F_X=U_X\operatorname{diag}(e^{-2\bar\lambda})U_X^\top,\qquad
 L_X(Y)=\frac{\operatorname{tr}((Y-X)^\top F_X(Y-X))}{3N}.
\]

\[
 \nabla_h L_X(D(h,z_0))=
 J_{D,h}^{\top}\frac{2F_X(D(h,z_0)-X)}{3N}.
\]

[Inference] Bu denklemde h decoder girişindeki clean latenttir; noisy diffusion durumuna guidance için ayrıca denoiser Jacobian'ı zincirlenir. Bu formül bozuk referansa bağlı bir sadakat ölçüsüdür; temiz
geometriye veya doğru sınıfa yaklaşmayı garanti etmez. Decoder Jacobian'ı
nedeniyle latent bant filtresi ile fiziksel decoder filtresi de eşit değildir.
PM'de zaten decoder'a girilmesi, bu Jacobian üzerinden backward maliyetini
otomatik olarak ortadan kaldırmaz. Kullanıcının hardM100/200 ve SCD/2+GSD/2
önerisi, bu kampanyanın P100_C önkoşullandırıcısından farklı, ayrı agenta
devredilmiş deneydir; GPU sonucu bu kampanyada üretilmedi.

## Hipotezlerin kararı

| Soru | Gerçek kanıt | Karar / sınır |
|---|---|---|
| H0: native tekrar tabanı | Altı fresh-process çiftinin ilk farkı aynı giriş/RNG ile ilk local-prior `noise_pred` çıktısında; Gaussian3/3, Background0/3 tekrar kapısı geçer. | Fark SCD/backward/PM'den önce başlıyor. Hedefli altı forward tekrarı aynı process içinde farkı yeniden üretti; kapsanan ilk fark `sa_layers.0.0.voxelization` feature çıktısında. Tek kernel/atomik neden veya düzeltme kanıtı yok. |
| H1: bozuk referans yanlılığı | Önceki actual-F karşı örneği korunur; yeni ortak durumda temiz-CD/SCD+spektral sonuçları karışık. | Bir loss'u düşürmek temizliği/accuracy'yi garanti etmiyor; yeni koşulda evrensel nedensel zarar iddiası yok. |
| H2: graf eskimesi | X ve güncel Y heat-operator yönleri belirgin farklı; medyan göreli action farkları yaklaşık.946–1.077. | Betimsel eskime var. Yararlı refresh kanıtlanmadı; koşullu refresh/accuracy aşamasına geçilmedi. |
| H3: decoder bant karışması |16durum,128B2 perturbasyonu/256nesne gözlemi. Düşük-latent yönün fiziksel düşük-bant dışı enerji medyanı.506, identity-XYZ kontrolü.164. | Decoder frekansları koruyan bir izometri değil. Fiziksel "other", genişletilmiş düşük bandın tümleyenidir; yalnız en yüksek100mod değildir. |
| H4: eigenfree heat | Dört bozulmanın hepsinde K8/16/32 sabit1e-6 native loss+gradyan eşdeğerlik ölçütünü geçemedi; aynı graph/support doğrulandı. | Eşdeğer bütün-akış adayı ilerlemez. K16/K32 gradyan hatası1.619e-6–4.465e-6 düzeyinde kalır. B32 cachedK16 loss+VJP387–403ms, exactU11.72–12.36ms; kurulum tasarrufu bütün-akış hız kazancı değildir. |
| H5: referans güvenilirliği | Combined AUROC.271341,95%8nesne-clusterCI[.137627,.386096]; frekans/residual karşısında gerekli iki üstünlük yok; native tekrar tabanı da başarısız. | **NOT_ADVANCED**. Skor işareti ters çevrilmedi, yeni ağırlık/epsilon/oracle tuning'i yapılmadı. |
| H6: SCD yönünü düşük-bantta önkoşullandırma | Tam karşılaştırılabilir3durumda P100_C, C ve diğer yönlerin her biri iki adım oranında da4/6doğru. Her iki oranda tutarlı geometri yararı yok. | Yeni accuracy koşusuna geçilmedi.6gözlem4özgün nesne kümesine aittir; bağımsız6kişi denemesi değildir. |
| H7: latent / fiziksel yön | Aynı3durumda fiziksel heat temiz-CD'si C'den ortalamada iki oranda da kötü; tahmin sayımları aynı. | Fiziksel alanın bu koşullarda üstünlüğü gösterilmedi; geniş koşu ilerlemez. |
| H8: maliyeti azaltan Jacobian yaklaşımı | Önceki96/96joint-yön kontrolü ve32ID/all15dört-kol pilotu tekrar kullanılmalı; PM313/315/313/312of480. | Önceki accuracy farkları belirsiz; SCD'ye karşı eşzamanlı hız/accuracy kazanımı yok. Yeni geniş tekrar gerekmedi. |

[Run/CPU] Ortak-durum C yönü için16durumun15'inde FD değerlendirildi:
3geçti,12başarısız/çözümsüz; kalan1durum daha önce forward kapısında durdu.
Bu,12yanlış autograd kanıtı değildir: sonlu epsilon, NN/retained-set
değişimi ve native farklar olasıdır. Sabit adım/tolerans değiştirilmedi.
224olası B2 bulut anahtarından66'sı mevcut;30zero ve her nonzero kolda6.
Geniş zero18/30 sayımı ile C4/6 doğrudan karşılaştırılmadı. Ortak örneklerde
bütün nonzero kollar4/6'dır; tam trajectory/PM accuracy sonucu değildir.

## Kod düzeltmeleri ve yürütme kanıtı

[Code/CPU] İki dar doğrulama hatası onarıldı: native float32 özdeğerlerinde
mevcut unresolved blok içindeki küçük q tersliklerini reddeden kontrol ve
native-trace analyzer'ın meşru `repo/` kaynak alias'ını kabul etmemesi.
Ham q/U sırası, bantlar, skorlar, source/native kimlik denetimi veya toleranslar
değişmedi. Materyal ters sıralama, bilinmeyen/tehlikeli kaynak yolu ve hash
uyuşmazlığı hâlâ reddedilir. Başarısız ilk analizler saklandı.

[Run] H5 immutable kaynak b766e61/overlay99cd1b…ee57; tamamlayıcı kaynak
72e0eb4/overlay809573…b713. H5 doğrulama düzeltmesi CPU analyzer'da;
ortak-durum native validasyon düzeltmesi ayrı çalıştırılan operator'da
c6dc6308…bd535 olarak kaydedildi. Sonradan canonical kaynak onarıldı;
aktif runtime asset snapshot'ı yeniden yazılmadı.

## Tamamlanma ve arşiv kaydı

[Run/CPU] Son hedefli prior koşusu altı gerçek native forward ile47.277s'de tamamlandı. Tam giriş/RNG/model/native/source kontrolleri geçer. Gaussian ve Background tekrarlarında aynı process içinde yaklaşık1.4e-6–1.7e-6 prior çıktı farkı yeniden üretildi. Kapsanan ilk aynı-girişli numeric fark voxelization feature çıktısında; koordinat çıktısı eşittir. Hook'lar zamanlamayı etkiler,64MiB sayısal referans sınırı vardır. Bu bir modül düzeyinde yerelleştirmedir; tek CUDA kernel'i/atomic neden, tam amplification zinciri veya determinism onarımı kanıtı değildir. Kaynakta ortalama voxelization için FP32 atomic toplama bulunması olası bir açıklamadır; yüklü ELF'nin CUDA kaynak baytları eşleştirilmeden derleme nedeni kesinleştirilmez.

[Run/CPU] Önceden belirlenen zorunlu tanı aşamaları tamamlandı; koşullu uygulamalı H2/H4/H5/H6/H7 genişlemeleri karşılanmayan önkoşullarıyla **NOT_ADVANCED** olarak kapandı. Peer hard100/200 halfmix GPU deneyi bu kampanyaya taşınmadı. [18 gerçek Drive/yerel ham ve türetilmiş arşivin indeksinde](evidence/gsd_comprehensive_20261106/artifacts.json) toplam2,564,493,011byte boyut/SHA denetimi vardır; ham arşivler yeniden yazılmadı. Altı native ELF, model/dataset/checkpoint kimlikleri ve başarısız ilk analizler korunur.

| Aşama | Kayıtlı süre, saniye | Sınır / anlam |
|---|---:|---|
|H5 üç capture|416.558|45dk cap; worker/probe fazı wall|
|H0 altı trace + shared capture|229.731|35dk cap; yedi koşu faz wall|
|Ortak-durum mekanizma|172.351|45dk cap; child hesabı, sonraki arşivleme hariç|
|Dondurulmuş classifier okumaları|20.952|15dk cap; arşiv dahil|
|H4 dört teknik koşu|167.929|30dk cap; faz wall|
|Hedefli prior tekrarları|47.277|15dk cap; arşiv/Drive dahil|

Bu süreler GPU kernel utilization veya toplam VM ücret süresi değildir; kurulum/izin bekleme/yerel CPU analizleri ayrı süreçlerdir. Kullanıcı bütün GPU kampanyasını izin sorulmadan yürütme yetkisini önceden verdi.

[Code/CPU] Temiz integration checkout'unda807test geçti,2test atlandı (49.60s); [komut/logSHA/kapsam makbuzu](evidence/gsd_comprehensive_20261106/integration_validation.json). Native/model doğrulaması ayrı Colab ham kanıtıdır. Peer lowband kodu, control guard'ları ve kabul edilmiş notları korundu. Geçici wrapper'lar, ham arşivler, ağırlıklar ve CPU venv Git'e eklenmez.

[Run/Publication] **Ownr4 kapatıldı ve sunucuda bulunmadığı doğrulandı (2026-11-05 22:04UTC; İstanbul6Kasım01:04).** Sole monitor53240 önce exit0 ile durdu; ardından yalnız `colab stop -s tta-gsd-h5-reliability-a100-20261105-r4` çalıştırıldı. CLI `Session terminated`, bağımsız sunucu envanteri `No active sessions found on server` döndürdü. Öncesinde research worker kalmadığı,18ham/türetilmiş arşiv ve3,240,045byte nihai analiz paketi Drive/yerel SHA roundtrip'i doğrulandı; paketteki195manifest dosyası yeniden hash'lendi. Sonuçlar kapanmadan önce normalpush ile `origin/thesis-integration` b8783f3 olarak yayımlandı. [Kapanış makbuzu](evidence/gsd_comprehensive_20261106/closure.json), [curated yayın kapsamı](evidence/gsd_comprehensive_20261106/publication.json), [son yedek doğrulaması](evidence/gsd_comprehensive_20261106/final_backup.json). Hiçbir peer oturumuna veya ana kullanıcı checkout'undaki değişikliğe müdahale edilmedi.
