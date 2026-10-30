# Chamfer-only guidance deneyleri: nihai teslim — 2026-10-30

[Run/Decision] İstenen üç bağımsız değişiklik ve ek Scan/CurveNet entropy
kontrolü tamamlandı. Mevcut temporal probability mean korunur; model/BN/LN
parametreleri güncellenmez. Guidance spectral loss hesaplanmadı. Yöntemler
birleştirilmedi; varsayılan yöntem otomatik değiştirilmedi.

## Accuracy ve süre

140 farklı nesne/veriseti,15 corruption,severity5,seed0/1/2;
her classifier6300karar. Eşlenmiş C bu çalışmanın B70/sabit10normal-30background
adımlı, üç clean-probe+terminal probability-mean host
baseline'ıdır; özgün3DD-TTA makalesinin5/35terminal protokolü değildir. Ana tablo dört classifier accuracy ortalamasıdır,
model ensemble değildir. Farklar kendi eşlenmiş Chamfer baseline'ına göre
**yüzde puan**; maliyet üç ayrı fresh G+B tekrarının eşlenmiş oran medyanıdır.

| Bağımsız ek | ModelNet Δpp | Scan Δpp | MN / Scan süre farkı |
|---|---:|---:|---:|
| Sparse backward (interval3) | −1.3571 | −.9683 | −%21.15 / −%21.07 |
| Semantik symmetric KL | +.1627 | −.0516 | +%7.49 / +%8.16 |
| Marginal entropy H | +.3254 | +.0833 | +%7.43 / +%8.16 |
| Density-confidence weighted Chamfer | −.7262 | −.5040 | +%0.67 / +%0.45 |

[Inference] H'nin dört-head ModelNet aralığı[+.0794,+.5714]pp, Scan aralığı
[−.1071,+.2778]pp. Point-MAE'nin kendi H farkı MN+.1746pp/Scan+.3333pp;
bunlar dört-head tabloyla karıştırılmamalı. KL aralıkları iki verisetinde
sıfırı kapsar. Sparse hız kazandırır fakat accuracy kaybı açıktır. Weighted
bu protokolde maliyet/accuracy katkısı sağlamadı. H küçük ve sınırlı olumlu
bir işaret verir; genel başarı iddiası desteklenmez.

[Run] Kullanıcının istediği daha güçlü kaynak kontrolü için yalnız Scan/H,
yeni CurveNet-only etiketsiz gradient kalibrasyonu ve kendi matched C'si:
CurveNet **%55.6667→%55.9048**, **+.2381pp**,95%CI[−.1587,+.6349].
Üç seed doğru sayıları1165→1166,1168→1176,1174→1180 /2100.
PointNeXt transfer+.4127pp ve dört-head ortalama+.1944pp ikincildir;
birincil CurveNet belirsizliğinin yerine geçmez. H ek G+B dört-head pipeline maliyeti
**%8.5572**; C20.9554s/H22.7599s medyan. Gaussian+%13.0753,
Background+%6.0390. Aynı140IDs tekrar kullanıldığı için keşifsel/posthoc
paket kontrolüdür; source-model kalitesi ile yeni katsayının etkisi izole edilmedi.

## Kapsam, kaynak ve kanıt

[Run/Code] Kabul edilen ana accuracy/timing deneylerinin GPU kaynağı
0fd258110abf48dc92d739b288010ebfbee475b2;
ek CurveNet GPU kaynak95a49977ef7c3124bde5faa57e3ef36b4393ebcc.
Erken başarısız smoke/native tanı arşivleri a1d298a kaynağına aittir;
metadata onarımı sonraki0fd kaynağını oluşturur. Orijinal86planlıworker,
ek13worker; native tanı ve başarısız sham/parite
kanıtları da saklandı. Toplam **407ham head ZIP+304fiziksel NPZ**,711dosya,
3918498290byte yeniden SHA/CRC/güvenli ZIP yolu kontrolünden geçti;
eksik/çakışma sıfır. Tam katalog ignored result/preparation altında,
küçük kanıt knowledge/evidence içindedir. Başarısız/negatif/null sonuçlar korunur.

[Run] Orijinal sürekli GPU aşaması yaklaşık4saat42dakika/5saat;
ek aşama yaklaşık51dakika/70dakika. Ledger active süresi fazlar arası
beklemeleri içermez; bütçede sürekli saat kullanıldı. Kurulum ve nihai
arşivleme süreleri ayrı CPU/I/O işleridir; yeni GPU deneyi kalmadı.

[Run/Inference] Süreler B70/dev70/seed0/Gaussian+Background içindir;
ortak rekonstrüksiyon bir kez ve dört terminal classifier hesaplanır.
Model/native başlatma, toplu veri yükleme, paketleme ve Drive teslimi hariçtir.
Fresh encoded-state drift açıkça kayıtlıdır; input/noise/RNG eşitliği korunur.
Cached accuracy eşleştirmesi gevşetilmedi. CI nesne-cluster bootstrap2000/seed0,
seçili üç seed/head'e koşulludur; çoklu karşılaştırma düzeltmesi yoktur.
Tam benchmark, tek classifier/B1 deploy süresi, birleşimler ve erken çıkış
ölçülmedi. Yeni çalışma için ayrıca protokol/bütçe belirlenmelidir.

## Saklama ve oturum

[Run/Open] Ham arşivler ve faz/source/kalibrasyon kanıtları Drive'da:
`MyDrive/thesis/result/chamfer_guidance_variants_20261030` ve
`MyDrive/thesis/result/scan_curvenet_h_postcampaign_20261030`.
Nihai özet ve304physicalcache yedeğinin kesin byte/SHA/roundtrip makbuzları
ayrı son arşiv kaydına eklenecek. Bu belge hazırlanırken Colab açık;
kullanıcı tüm gerekli çalışma ve doğrulamalar bittiğinde yalnız kendi
oturumumuzu kapatma talimatını açıkça verdi. Gerçek kapanış ayrıca kaydedilecek.

[Sonuç ve güven aralıkları](chamfer_guidance_variants_results_20261030.md) ·
[CurveNet ayrıntıları](scan_curvenet_entropy_followup_20261030.md) ·
[Katalog kabulü](evidence/chamfer_author_code_audit_20261030/final_raw_artifact_catalog_acceptance_20261030.json).

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

[Code/Test/Limit] Seçili entegrasyon ağacı ayrı CPU doğrulamasıdır:
479test/373subtest geçti; bir tarihsel rawsnapshot lookup testi skip,
altı mevcut warning. Peer depth kontrolleri korunur; NumPy1.26.4 ile69/69
depth testi geçer. GPU tabloları yalnız kaynak0fd2581 ve95a4997'ye aittir;
entegrasyondaki shared runner/guard yeni kaynak byte'larına sahiptir ve
GPU'da yeniden denenmedi. Kalibrasyon/kaynak doğrulaması gevşetilmedi;
entegrasyon üzerinde yeni GPU koşusu kaynak kontrolü reddederse yeni seal
ve etiketsiz kalibrasyon gerektirir. Bu teslim yeni GPU koşusu başlatmaz.
