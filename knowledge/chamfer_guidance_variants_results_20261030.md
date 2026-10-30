# Chamfer-only guidance: doğruluk ve süre çalışması

2026-10-30. Durum: geliştirme ve140nesne/üç-seed doğrulaması kabul edildi;
fresh süre ölçümü kabul edildi; CurveNet accuracy kontrolü kabul edildi; son süre tekrarları çalışıyor. Geliştirme ve doğrulama tabloları ayrı tutulur.
[Çalıştırma günlüğü ve arşiv kimlikleri](chamfer_guidance_variants_20261030.md).
[Makale ve yazar kodu karşılaştırması](chamfer_author_code_consistency_audit_20261030.md).

## Sabit karşılaştırma

[Code/Run] Baseline, mevcut Chamfer-only probability mean akışıdır. Normal
bozulmalarda10, Background'da30 denoising adımı vardır. Üç ara decoded görünüm
ve terminal çıktının sınıf olasılıkları eşit ağırlıkla ortalanır. Normal ara
adımlar3/5/8, Background ara adımları8/15/23'tür. Eta=gamma=.01, p=.95;
scheduler ve decoder-style davranışı bütün kollarda korunur. Bu baseline,
makaledeki ayrı5/35-adımlı original3DD protokolüyle özdeş değildir.

[User report/Code/Run] Spectral/invariant loss ve graph/eigendecomposition
hesaplamaları araştırma kollarında kapalıdır; gerçek çağrı sayaçları doğrulanır.
Classifier ve LION ağırlıkları/buffer'ları donmuştur. BN/LN parametre güncellemesi,
CloudFixer'ın fiziksel rotasyon/yer değiştirme optimizasyonu ve birleşik yöntem
yoktur. Erken çıkış yalnızca gelecekteki bir deney olarak kayıtlıdır.

| Kol | Baseline'a eklenen bağımsız değişiklik | Makaleyle ilişki ve sınır |
|---|---|---|
| Sparse | Interval3; son adım dâhil4/11 gerçek guidance backward, bütün10/30 prior forward korunur. Eta/gamma artırılmaz. | Adaptive Guidance'dan esinlenen hesap bütçesi fikri; yazarın CFG/conditional-forward atlama algoritmasının aynısı değildir. |
| KL | Son iki mevcut ara görünümde, decoded Point-MAE olasılığı ile bir kez hesaplanıp detach edilen bozukX/FPS olasılığı arasında simetrikKL. | MAMVI tutarlılık fikrinin latent guidance'a uyarlaması; model/normalizasyon adaptasyonu yoktur. |
| H | Aynı iki görünümde nesne başına H((p_decoded+p_X)/2). | GDA/MAMVI ile ilişkili semantik düzenleme; GDA'nın AugMix veya entropy-weighted log-fusion kodunun reprodüksiyonu değildir. |
| Weighted | h0'ın ilk3 latent koordinatında nonself10-komşu ortalama karesel uzaklığının tersi, epsilon1e-6, ortalama ağırlık1. Aynı uzaklık temelli p=.95 trim; iki Chamfer yönünde doğru referans ağırlığı. | CloudFixer'ın yoğunluk-güven fikri; fiziksel noktadaki hareket cezası yerine latent fidelity residual'ı ağırlıklandırılır. |

[Code/Run] Semantik referans mean'e yeni oy eklemez. Mevcut decoded görünüm ve
classifier logits'i yeniden kullanılır; son iki görünümde gradyan graph'ı
korunur. Ek maliyet yalnızca birX-forward'dan ibaret değildir: decoder ve
classifier üzerinden local/style'a ek backward da vardır. Her loss nesneler
üzerinde sum kullanır; ağırlıklar B2 ve B70 arasında amaçlanan ölçeği korur.

[Run] Önceden sabitlenen, doğruluk kullanmayan kalibrasyon iki veri kümesinin
70'er geliştirme nesnesi/all15/seed0 kaydından4200oran/adayı birleştirir.
Local/style update normlarının birleşik normu kullanılır. Semantik katsayı
.1/median(candidate/C), weighted eta/gamma ölçeği1/median(weighted/C)'dir.

| Sabit katsayı | Değer |
|---|---:|
| KL loss ağırlığı | .6285361973358488 |
| H loss ağırlığı | 1.4715424107401223 |
| Weighted eta/gamma ölçeği | .9171509895707142 |

Bu hedef pooled median içindir; her nesne için yüzde10 üst sınır değildir.
[Dondurulmuş gerçek rapor](evidence/chamfer_author_code_audit_20261030/frozen_calibration_report_20261030.json).

## Tamamlanan geliştirme:70 nesne/veri kümesi, seed0

[Run] All15/severity5, dört classifier'ın eşit ortalaması; probability mean.
Sayılar yüzde; farklar yüzde puandır. CI, nesneyi cluster olarak yeniden
örnekleyen2000bootstrap/seed0 koşullu yüzde95 aralığıdır.

| Kol | ModelNet doğruluk | Fark [CI] | ScanObjectNN doğruluk | Fark [CI] |
|---|---:|---:|---:|---:|
| C baseline | 62.7381 | — | 40.5952 | — |
| Sparse | 61.2381 | −1.5000 [−2.2619, −.7613] | 39.4762 | −1.1190 [−1.7857, −.4762] |
| KL | 63.0000 | +.2619 [−.3095, +.8571] | 40.1429 | −.4524 [−.9762, +.0714] |
| H | 62.7857 | +.0476 [−.5714, +.6429] | 40.2857 | −.3095 [−.8095, +.2149] |
| Weighted | 62.4286 | −.3095 [−.9762, +.3571] | 39.6667 | −.9286 [−1.6905, −.1905] |

[Inference] Geliştirme net bir doğruluk kazancı göstermiyor. Sparse iki veri
kümesinde de düşüyor. Küçük ModelNet KL/H farkları sıfırı kapsayan aralıklara
sahip; Scan'e taşınmıyor. Bu tablodan parametre yeniden seçilmedi veya kol
elenmedi. Aynı beş koşul, ayrı140nesnede seed0/1/2 ile değerlendiriliyor.

## Süre ölçümü ve yorum sınırı

[Run/Open] Geliştirmede C fresh encode, diğer kollar eşleştirilmiş cache replay
kullanır. Bu sürelerden deployment hız kazancı çıkarılmaz. Planlanan süre
uç noktası70nesne/B70/seed0/Gaussian+Background üzerinde üç ayrı fresh-encode
tekrarıdır; kol sırası döndürülür. Shared reconstruction bir kez, terminal
classifier süreleri ayrıca sayılır. GPU kurulumu ve Drive arşivleme inference
süresine eklenmez; araştırma duvar saati bütçesinde ayrıca izlenir.

[Run/Code] Mevcut native C/C tekrarında aynı latent/RNG ile ilk prior çıktısı
farklıdır. Fresh encoder da aynı ham girdi/noise/RNG ile farklı z0/h0 üretebilir.
Bu nedenle doğruluk kolları gerçek ortak cache ve tam başlangıç hash'leriyle
eşleşir. Fresh süre kollarında ham girdi/noise/scheduler/RNG ve kaynak/model
kimlikleri tam eşleşir; encoded farklar ayrıca raporlanır. Süre tekrarlarından
accuracy sonucu çıkarılmaz. Noktasal0×semantik loss gradyanı KL/H için gerçek
native akışta local ve style yönünde tam sıfır doğrulanmıştır.

[Run]140nesne/üç-seed doğrulaması ve üç fresh süre tekrarı kabul edildi.
Tam veri kümesi veya dış veri üzerinde genelleme iddiası yoktur. Yerel CUDA
nondeterminism nedeniyle arşivlerden yeniden skor hesaplamak ile bitwise
yeniden inference üretmek farklı güvencelerdir.

## Kabul edilen doğrulama:140 nesne/veri kümesi, seed0/1/2

[Run]120ham classifier arşivi,3600batch ve214fiziksel cache bağımsız kabul
edildi. Dört modelin eşit ortalaması, all15/severity5, probability mean; her
modelde6300, dört model ortalamasında25200karar/veri kümesi/kol vardır.
Nesne-cluster2000bootstrap/seed0 CI, seçili üç seed ve model grubuna koşulludur.
Farklar yüzde puandır; aile çapında çoklu-karşılaştırma düzeltmesi yapılmadı.

| Kol | ModelNet doğruluk | Fark [CI] | ScanObjectNN doğruluk | Fark [CI] |
|---|---:|---:|---:|---:|
| C baseline | 66.9008 | — | 45.0833 | — |
| Sparse | 65.5437 | -1.3571 [-1.7818, -0.9524] | 44.1151 | -0.9683 [-1.2540, -0.6904] |
| KL | 67.0635 | +0.1627 [-0.0952, +0.4049] | 45.0317 | -0.0516 [-0.2660, +0.1707] |
| H | 67.2262 | +0.3254 [+0.0794, +0.5714] | 45.1667 | +0.0833 [-0.1071, +0.2778] |
| Weighted | 66.1746 | -0.7262 [-1.0357, -0.4325] | 44.5794 | -0.5040 [-0.7303, -0.2619] |

[Run/Inference] H dört-head ModelNet ortalamasında küçük pozitif işaret verir;
Scan ortalama farkının aralığı sıfırı kapsar. Sparse ve weighted her iki veri
kümesinde ve dört classifier’ın hepsinde düşer. KL için iki veri kümesinde de
aralık sıfırı kapsar. Bu sonuçlardan ağırlık, seed veya kol yeniden seçilmedi.

[Run] Point-MAE özelinde H farkı ModelNet +.1746pp CI[-.1587,+.5079],
Scan +.3333pp CI[+.0794,+.5873]; KL sırasıyla+.0952pp CI[-.3016,+.4766]
ve+.1270pp CI[-.2222,+.4607]. Geliştirmedeki ModelNet+1.0476pp örneği
doğrulamada aynı büyüklükte sürmedi. Bunlar ayrı nesne gruplarıdır.

[Run] Root ve bağımsız analiz tam JSON olarak eşleşir; düzeltilen/bozulan
nesne kimlikleri, per-corruption değerler ve bootstrap aralıkları dahildir.
[Kompakt doğrulama kanıtı](evidence/chamfer_author_code_audit_20261030/confirmation_acceptance_20261030.json).
Tam proof/analiz/ham arşivler `result/preparation/chamfer_guidance_variants_20261030/`
ve canonical result dizininde korunur. Cache süreleri deployment hızına
yorumlanmaz; üç fresh tekrar tamamlanınca süre tablosu eklenecek.

[User report/Decision/Open] Kullanıcının yetkilendirdiği sonraki kontrol,
Scan’de yalnız H ve daha güçlü tarihsel source model CurveNet’tir. Aynı140
nesnenin yeniden kullanımı açıkça post-hoc/keşifseldir; CurveNet’in kendi
probability mean C→H farkı birincildir. Yeni phase mevcut süre ölçümü ve veri
toplama bitmeden başlamaz. [Kaynak modeli kontrolü](scan_curvenet_entropy_followup_20261030.md).

## Kabul edilen süre: üç fresh tekrar, Gaussian + Background

[Run]120ham head arşivi/240batch bağımsız kabul edildi. Her sayı, iki
bozulmada70’er girişin shared reconstruction toplamı ve dört terminal head
süresini içerir; ortak reconstruction bir kez sayılır. Model/native başlatma,
toplu veri yükleme, config kontrolleri ve Drive arşivleme hariç; probe/hash
işi dahildir. Tek-head deployment veya all15 süre ölçümü değildir.

| Kol | ModelNet medyan saniye | Eşlenmiş süre farkı | Scan medyan saniye | Eşlenmiş süre farkı |
|---|---:|---:|---:|---:|
| C | 20.875 | — | 20.950 | — |
| Sparse | 16.479 | -21.15% | 16.529 | -21.07% |
| KL | 22.435 | +7.49% | 22.652 | +8.16% |
| H | 22.452 | +7.43% | 22.665 | +8.16% |
| Weighted | 21.011 | +0.67% | 21.048 | +0.45% |

[Run/Inference] Yüzdeler üç eşlenmiş arm/C oranının medyanıdır; ayrı süre
medyanlarının oranı değildir. Sparse Gaussian maliyetini yaklaşık%13.4,
Background maliyetini%25.5 azaltır. H normalde%10.7/%11.4 ve Background’da
%5.6/%6.3 ekler (MN/Scan). Kol sırası döndürülür; üç tekrar uzun dönem
sistem varyansının güven aralığı değildir.

[Run] Encoded drift224head satırı/56grup karşılaştırmasında kayıtlıdır; ham
girdi/noise/scheduler/RNG, kaynak/native/asset/model state/kimlikler tam
eşleşir. Süre koşularından accuracy çıkarılmaz; cached accuracy başlangıç
eşitliği gevşetilmedi. [Süre kanıtı](evidence/chamfer_author_code_audit_20261030/timing_acceptance_20261030.json).

[Inference/Decision] Sparse yaklaşık%21 G+B süre kazancı karşılığında
−1.3571pp MN ve−.9683pp Scan accuracy farkı verir; kabul edilebilir kayıp
kullanıcının tercihidir. Weighted accuracy’yi düşürür ve küçük ek maliyet
getirir. H’nin%7–8 maliyetli küçük olumlu işareti sonraki güçlü source-model
kontrolünde sınanacak; yeni ağırlık/seed taraması veya yöntem birleşimi yoktur.

[Run] Orijinal GPU/controller aşaması13:25:16.894UTC→18:07:18UTC, yaklaşık
16922s/4saat42dakika ile5saat sınırı içindedir. Ledger active_seconds14093.35s
fazlar arası bekleme/debug içermez; bütçede daha büyük sürekli süre kullanılır.
CurveNet aşamasının saati ayrıdır; eski saat sıfırlanmaz.

## 2026-10-30 — CurveNet C/H accuracy accepted

[Run]24head ZIP/720batch ve90yeni fiziksel cache bağımsız kabul edildi;
95a4997 kaynağı,658asset,8e8d...frozenH raporu, model/native state ve gerçek
C/H initial h0/noise/RNG, nesne kimlikleri/etiketler tam eşleşir. Hiçbir
eski0fd C arşivi veya cache yeni kontrolle karıştırılmadı. Ham arşivler ve
yeni Drive kökü `scan_curvenet_h_postcampaign_20261030` içinde korunur.

[Run] Birincil sonuç CurveNet’in kendi temporal probability mean’idir;
all15/severity5,140nesne, seed0/1/2. Her head6300karar. CI140nesne
cluster bootstrap2000/seed0; seçili seed/head koşulludur, multiplicity
düzeltmesi yoktur. Aynı140IDs’nin tekrar kullanımı posthoc/keşifseldir.

| Head | C accuracy% | C+CurveNet-H accuracy% | Fark pp [95%CI] |
|---|---:|---:|---:|
| CurveNet (birincil) | 55.6667 | 55.9048 | +.2381 [−.1587,+.6349] |
| DGCNN (transfer) | 38.3810 | 38.5714 | +.1905 [−.3810,+.7460] |
| PointNeXt (transfer) | 46.6984 | 47.1111 | +.4127 [+.0159,+.8258] |
| Point-MAE (transfer) | 39.0000 | 38.9365 | −.0635 [−.3175,+.1746] |
| Dört-head accuracy ortalaması (ikincil) | 44.9365 | 45.1310 | +.1944 [+.0040,+.4049] |

[Run/Inference] CurveNet seed0/1/2 C→H doğru sayıları1165→1166,1168→1176,
1174→1180 /2100; toplam+15doğru/6300. Yön üç seed’de pozitiftir fakat
birincil nesne-cluster aralık sıfırı kapsar; güçlü/genel bir katkı kanıtı
yoktur. PointNeXt ikincil olumlu gözlemi, birincil endpointin yerine
geçmez. Source-model gücü ve yeni Scan kalibrasyonunun saf etkileri izole
edilmedi; kendi matched C’sine karşı bir paket karşılaştırmasıdır.
Dört-head ortalama bir model ensemble oyu değildir: her classifier’ın
kendi dört-zamanlı probability-mean accuracy’sinin ortalamasıdır.

[Run/Code] Root ve bağımsız tam JSON raporları eşleşir; ham/source/ID/RNG
ve modellerin donmuş durumları kabul edildi.
[Kompakt proof](evidence/chamfer_author_code_audit_20261030/scan_curvenet_heldout_acceptance_20261030.json).
Bu tarihli accuracy kabulünde süre henüz bekleniyordu; aşağıdaki nihai
fresh timing kabulü bu durumu günceller. Varsayılan çalışma akışı
experimental flags dışında otomatik değişmedi; birleşik yöntem yoktur.

## 2026-10-30 — tüm deneyler ve fresh süre ölçümleri tamamlandı

[Run] CurveNet takip aşamasının 13 worker/49 head ZIP ve90 fiziksel cache'i
kabul edildi. Üç döndürülmüş fresh tekrar: dev70/B70/seed0,
Gaussian+Background, ortak rekonstrüksiyon bir kez ve dört terminal classifier.
C medyanı20.955446s, H medyanı22.759866s; birincil eşlenmiş H/C oranları
1.08557173,1.08981485,1.08468403; medyan ek maliyet **%8.5572**.
Gaussian ek maliyet medyanı%13.0753, Background%6.0390.
Model/native kurulum, toplu veri yükleme, paketleme ve Drive aktarımı bu
batch ölçümüne dahil değildir; tek classifier veya all15 deploy süresi değildir.

[Run/Code]24 timing head/48 batch bağımsız kabul edildi. Input/noise/scheduler/
RNG, kaynak95a4997,658asset, donmuş checkpoint/state ve katsayı tam eşleşir.
40 head-batch encoded-state farkı açıkça kayıtlıdır; fresh encode bitwise
paritesi iddia edilmez ve bu koşulardan accuracy çıkarılmaz. Root ve bağımsız
tam JSON, yalnız encoded-difference satırlarının sırası normalize edilince
birebir aynıdır. Guidance spectral/graph hesap sayıları sıfırdır; classifier
mimarisinin kendi KNN/Curve işlemleri aynen korunur.
[Süre kabulü](evidence/chamfer_author_code_audit_20261030/scan_curvenet_timing_acceptance_20261030.json).
Timing phase ZIP183436bytes SHA1e106a454c0df905ba444355f431ecd8ac9b0164bb2b14f9c6a368e0fb2e2cd9.

[Run] Yeni sürekli GPU saati18:23:23.664750→19:14:23.569UTC,
yaklaşık3059.90s/51dakika;4200s sınırının altında. Ledger active_seconds
2448.191s bekleme/aktarım aralarını içermez; bütçede sürekli süre kullanılır.
Orijinal4saat42dakika/5saat bütçesi değiştirilmedi.

[Inference/Decision] CurveNet kendi accuracy'sindeki +.2381pp farkın95%CI'si
[−.1587,+.6349] olduğundan, %8.56 batch maliyetine karşı güçlü bir fayda
kanıtlanmadı. PointNeXt/ortalama ikincil olumlu işaretleri birincil belirsizliği
kaldırmaz. Şimdilik hiçbir varyant varsayılan yapılmadı veya birleştirilmedi.
Tüm sonuçlar Chamfer-only ve mevcut dört-zamanlı probability mean üzerindedir.
Kullanıcının talimatıyla kalan iş yalnız nihai Drive özeti, dosya doğrulaması
ve sadece sahip olduğumuz runtime'ın kapatılmasıdır; yeni GPU işi yoktur.

## 2026-10-30 — nihai Drive kabulü ve yalnız sahipli runtime kapanışı

[Run/Artifact]407ham head ZIP+304NPZ'nin711yerel dosyası SHA/CRC/güvenli yol
kontrolünden geçti; eksik/çakışma yok.304physicalcache ayrıca Drive'da
`scan_curvenet_h_postcampaign_20261030/phase-evidence/paired_physical_caches_final_remote_20261030T1926Z.zip`:
1290931819bytes SHAea98e68d50330bd4b988594089564c35010a4e86e674c4920c65e510e85bbebd.
Geri indirilen ZIP'nin tam SHA/CRC'si ve304member SHA'sı ayrı doğrulandı.
Nihai260member bilimsel özet aynı Drive klasöründe
`final_research_summary_20261030T1930Z_verified.zip`:78981869bytes,
SHA19e7c6735f0b60438c3347d68d87183b1ee6d886aaca084cb1605746742077f5.
Geri indirme ve259member SHA/güvenli yol/CRC kabul edildi. Dört kurulum
history üyesi paylaşılan özet kopyasından çıkarıldı; orijinal kaynak ZIP'leri
ve ham araştırma kanıtları değiştirilmedi. Özet, kapanış öncesi bilimsel
snapshot'tır; gerçek kapanış makbuzu aşağıdaki yerel/Git kaydıdır.
Son roundtrip kabul makbuzları da Drive'a ayrıca başarıyla yüklendi.

[Run] CLI'nin büyük doğrudan upload'ları400/brokenpipe verdi; başarı olarak
raporlanmadı.304cache VM üzerinde CPU ile paketlenip SHA/CRC doğrulandı;
özet5×16MiBparça ile yüklenip aynı byte'lara birleştirildi. Bir CPU archive
CLI prelude bağlantı kaybı kullanıcı kodu çalışmadan önce kanıtlandı;
yalnız sahipli lokal CLI5577 sonlandırılıp aynı script yeniden çalıştı.
Bu arşiv işleri yeni GPU deneyi veya araştırma bütçesi uzatımı değildir.
Private URL/token içeren ham CLI history paylaşılmadı.

[User report/Run/Decision] Kullanıcının tamamlanınca yalnız kendi session'ımızı
onaysız kapatma talimatı uygulandı. `tta-chamfer-guidance-variants-a100-20261030-r1`,
endpoint`gpu-a100-hm-ass1c0-3nh8smksm50de`; izleyici36160/PID59645
19:34:38.530UTC'de stop protokolüyle exit0 kapandı. Ardından yalnız tam bu
adla `colab stop -s tta-chamfer-guidance-variants-a100-20261030-r1` exit0;
19:34:46UTC itibarıyla sunucu envanteri aktif assignment göstermiyor.
Peer zaten önceki envanterde yoktu; diğer agent'ın runtime/izleyicisine veya
bilinmeyen süreçlere stop/exec/reconnect işlemi yapılmadı. Yeni GPU işi yoktur.
[Cache yedeği](evidence/chamfer_author_code_audit_20261030/final_cache_backup_acceptance_20261030.json) ·
[Özet yedeği](evidence/chamfer_author_code_audit_20261030/final_summary_backup_acceptance_20261030.json) ·
[Kapanış kanıtı](evidence/chamfer_author_code_audit_20261030/owned_runtime_closure_20261030.json).
