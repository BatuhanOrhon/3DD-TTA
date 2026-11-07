# PM düşük frekanslı halfmix — 7 Kasım 2026 sonuçları

[Run/Inference] Sınırlı pilot tamamlandı; güvenilir ek GSD katkısı gösterilmedi.
M100 en yüksek betimleyici PM sayısını verdi, fakat yarım-SCD kontrolüne farkı
2/240 ve koşullu güven aralığı sıfırı içeriyor. M200 yarım-SCD kontrolünün gerisinde.
Native sayısal kararsızlık sürüyor; varsayılan yöntem değiştirilmedi.

[İlerleme ve eski allocation/bekleme geçmişi](gsd_lowband_halfmix_study_20261105.md),
[başlangıç handoff'u](gsd_lowband_halfmix_agent_handoff_20261105.md),
[makinece okunabilir sonuç](evidence/gsd_lowband_halfmix_20261107/summary.json),
[tüm bozulmalar/terminal sayımları](evidence/gsd_lowband_halfmix_20261107/per_corruption.csv).

## Kilitli protokol ve gerçek kapsam

[Code/Run] Kaynak d6761c895ef6fa38d882bb2d4ca3f3205c1a1fa3,
755,258byte overlay SHA256
`9dcf49ecb2e48d0d5e61331c70837a10978f90c9b0680d5610d9eed92acd86f8`.
Checkpoint, dataset ve native kimlikleriyle bağlanmış asset manifesti SHA256
`51847c0330191bdef2ef84b37df92d83a1b08915729dd2f0a290ff05daa93021`.
Gerçek setup checkout'u ayrı dev@a458cd46; bilimsel kaynak overlay'iyle karıştırılmaz.
Taze native baytlar Drive'a doğrulanarak yedeklendi, ardından önceden belirtilen
dört sabit native dosya yerleştirildi; bütün kollar aynı kimlikleri kullandı.

ModelNet40-C,15bozulma,severity5,seed0,B2; tek seferlik16nesnelik gelişim pilotu.
Nesneler:197,502,536,1008,1064,1240,1285,1422,1549,1670,1738,1824,1858,2100,2285,2310.
Kalibrasyon ayrı411,672,777,2247; Gaussian/Background. Önceden belirlenmiş
582dışlanmış ID havuzuyla çakışmayan seçim değiştirilmedi. Teknik kontroller
Gaussian44/91 ve Background495/512. [Önceden mühürlü plan](evidence/gsd_lowband_halfmix_20261107/study_plan.json).

Dondurulmuş PointMAE ve raw/eval/EMAoff LION; DDIM100,finalalpha1,
p=.95,gamma=.03,eta=.003. Normal8adımda clean-PM2/4/6, Background35adımda9/18/27.
Hazırlanmış encoder girdisi B×2048×3 üzerinde k10/delta.1/graph_gamma.6 sabit
fiziksel graf; decoder aynı hazırlanmış frame ve vertex sırasıyla karşılaştırıldı.
PM için zaten gereken decoder çıktısı tekrar kullanıldı; özgün z0 ile decode,
tam decoder türevi, classifier gradient'i yok. Yalnız clean-PM'de
`0.5*SCD + 0.5*alpha_M*GSD_M`; kalan adımlar SCD1, terminalde yeni update yok.
Smooth/all-active profil M100 yerine kullanılmadı.

## Kalibrasyon ve matematik kapıları

[Run] Her nesne/checkpoint için gerçek güncelleme vektörü
`(-.03*grad_local, -.003*grad_style)` üzerinden SCD/GSD norm oranı hesaplandı.
Her bantta4ID×2bozulma×3PM=24aktif,sonlu oran; float64 norm aritmetiği,
<=1e-12norm reddi, sessiz clamp/örnek düşürme yok. Alpha seçim kodu etiket veya
clean veri okumadı; standart değerlendirme dosyalarındaki accuracy alpha seçimine
girmedi. İki ek bileşen VJP yalnız kalibrasyonda, gerçek joint backward ayrıca korunur.

| İstenen bant | Dondurulan alpha | Gerçek pilot rank aralığı |
|---|---:|---:|
| M100 | 4433.612924010353 | 100–223 |
| M200 | 4321.537946627087 | 200–235 |

[Kalibrasyonun48satırı ve freeze kimliği](evidence/gsd_lowband_halfmix_20261107/calibration.json).
Freeze SHA256`fb121b127d0ac78ed8ac1ee8dc500d52c5f2a631c4dabaac2a932d6595eca9a1`
pilot/zamanlama öncesi ve sonrası aynı. Pilot sırasında norm VJP'leri kapalı.
Kalibrasyondaki48ek VJP toplam4.8701s; dört kalibrasyon sürecinin dış wall toplamı145.2969s.

[Run] Rank genişlemesi tam sayısal sınır eigenspace'ini korur; M tam tutulan rank
değil, istenen kesimdir. Full eigh ve dense F hâlâ hesaplanır. M100 filtreleri
live0/half0/M100 arasında tüm eşlenmiş durumlarda byte-identical. M100/M200
aynı aktif desteği120/120B2durumunda kullandı; aynı filtre0/240nesne-bozulma
operatörü: bu pilotta operatör alias'ı yok. Gerçek bazlar her batch için arşivlendi.
504bağımsız CPU float64 baz-loss kontrolü geçti; maksimum mutlak hata
8.574130638816513e-8, sabit kapı1e-6. M100/200 tam decoder FD kontrolleri
.001/.01ölçeklerinde orijinal1e-5+0.1relative eşikle geçti.
[Native FD kanıtı](evidence/gsd_lowband_halfmix_20261107/native_fd.json).
Bu decoder sertifikası denoiser'ı veya tüm yörüngeyi sertifikalandırmaz.

## Sayısal tekrar başarısızlığı

[Run] Gaussian44/91 teknik tekrarlarında control/control3/3,live/live3/3,
cross9/9geçti. Background495/512'de aynı karşılaştırmalar0/3,0/3,0/9;
en büyük logit farkı3.4207287. Önceki kararsızlık çözülmüş değil.

Pilot operational/live0 karşılaştırmasında30/120B2durumu başarısız; yalnız
Background değil,13bozulmaya yayılıyor. Etiketli PM sayısı guidance olmadan
165'ten164'e değişti. Eşikler logitmax.05, nesne/checkpointRMS.01,
ilkdecoderRMS1e-4 olarak kaldı. Aynı cache/RNG/source/native/graf kimlikleri
geçtiği halde bu drift var. [Tüm kapı ölçümleri](evidence/gsd_lowband_halfmix_20261107/numeric_gates.json).
Bu sonuç tek bir kernel nedenini veya deterministik onarımı kanıtlamaz.

## Beş kolun yeniden sayılmış sonucu

[Run] Her kol16nesne×15bozulma=240eşlenmiş nesne-bozulma çifti; bağımsız
istatistiksel örnek sayısı240değil,16nesnelik cluster'dır. PM birincil,
terminal ikincil metrik. İki8-IDshard'ın ikinci kol sırası ters çevrildi.

| Kol | PM doğru/240 | PM % | Terminal doğru/240 |
|---|---:|---:|---:|
| Özgün operasyonel SCD | 165 | 68.7500 | 165 |
| Canlı tam-SCD, GSD0 | 164 | 68.3333 | 164 |
| Canlı yarım-SCD, GSD0 | 169 | 70.4167 | 169 |
| M100 halfmix | 171 | 71.2500 | 172 |
| M200 halfmix | 166 | 69.1667 | 168 |

[Run/Inference] M100–operasyonel farkı+2.50puan, koşullu16nesne-bootstrap95%CI
[+0.4167,+5.0000]. Ancak bu CI native tekrar değişkenliğini içermez ve güvenilir
GSD etkisi diye yorumlanamaz. Esas ek-bileşen karşılaştırması M100–yarım-SCD
+0.8333puan,4düzelen/2bozulan çift,CI[-0.8333,+2.5000]. M200–yarım-SCD
-1.25puan,1düzelen/4bozulan çift,CI[-2.9167,+0.4167]. Kontrol ağırlığını
azaltmanın etkisi, spectral ekleme ve native drift ayrıştırılmadan kazanım iddiası yok.
10,000nesne-cluster bootstrap, seed20261101; bütün15bozulma cluster içinde birlikte
örneklenir. Tüm kontrol kontrastları ve bozulma bazlı farklar summary.json'da.

## Zamanlama ve bellek kapsamı

[Run] Önceden hazırlanan126cache oluşturma dış wall30.6898s; her measured kol
aynı actual cache'i tekrar oynattı. Aşağıdaki süreler16nesne/all15iki shard toplamıdır.
Rekonstrüksiyon senkronize PM/gradient/probe maliyetini içerir; graf kurulumu,
terminal head/geometri, başlangıç ve IO ayrı kapsamlardır. Worker wall init/IO'yu,
dış wall ayrıca mühürleme/Drive aktarımını içerir. Decoder+hash/spectral loss
ayrı kayıtlı, joint backward rekonstrüksiyonun içinde; izole backward zamanı iddia edilmez.

| Kol | Rekonstrüksiyon+PM s | Graf s | Worker wall s | Dış wall s | Rekonstrüksiyon peak MiB |
|---|---:|---:|---:|---:|---:|
| Operasyonel SCD | 169.002 | 0 | 221.603 | 235.750 | 3752.10 |
| Canlı tam-SCD/GSD0 | 171.793 | 12.575 | 269.745 | 296.885 | 4452.99 |
| Canlı yarım-SCD/GSD0 | 172.168 | 12.596 | 270.989 | 298.320 | 4452.99 |
| M100 | 195.253 | 12.604 | 293.052 | 320.241 | 4569.65 |
| M200 | 195.280 | 12.481 | 304.889 | 339.974 | 4570.83 |

Graf+rekonstrüksiyon M100'de operasyonelden%22.9914,M200'de%22.9342fazla.
Peak, daha önceki graf-build geçici peak'ini dışlar; resident graf/bazı içerir.
Yukarıdaki IO maliyetleri üretim/deployment throughput'u değildir.

[Run] Ayrı taze-önbellek ölçümü: her kolda B2,Gaussian+Background,tek tekrar,
10farklı actual cache. Bu dört nesne-bozulma çifti pilot accuracy'ye dahil edilmedi.

| Kol | Taze rekonstrüksiyon s | Graf s | Case wall s | Worker wall s | Dış wall s |
|---|---:|---:|---:|---:|---:|
| Operasyonel SCD | 6.137 | 0 | 6.247 | 24.569 | 30.902 |
| Canlı tam-SCD/GSD0 | 6.140 | .323 | 8.546 | 27.155 | 34.952 |
| Canlı yarım-SCD/GSD0 | 6.148 | .323 | 8.553 | 27.303 | 35.300 |
| M100 | 6.512 | .320 | 8.921 | 27.687 | 35.673 |
| M200 | 6.525 | .320 | 9.098 | 27.857 | 35.909 |

B32veya tam benchmark hızına genelleme yok. Aynı anda güvenilir doğruluk/hız
kazancı kurulmadı. Bütün34mantıksal iş, canonicalsetup hariç tek bilimsel saatte
2419.940985s=40.33235dakika; tahmin30–45,sınır55dakika. Kullanıcı ayrıca
bu görevin gerekli GPU testleri için yeniden izin sorulmamasını açıkça belirtti.

## Arşiv, doğrulama ve kalan sınırlar

[Run] 33sealed inference/diagnostic ZIP,7standart dosya + predictionNPZ;
632B2durumun source/native/model/cache/RNG/ID/CSV/PM-terminal kontrolleri geçti.
126ortak+10taze actual cache, her batch baz/rank/support, kaynak overlay+pinned
asset base/native kaynaklar,10gerçek native ELF ve fresh-before-restore baytları
korundu. Ham bozuk/yarım paket yok; başarısız sayısal kapılar ayrıca korundu.
CPU807test/2skip,500subtest geçti; kaynak/core kod önceden yayımlanmış opt-in
uygulamadır. [CPU kaydı](evidence/gsd_lowband_halfmix_20261107/cpu_verification.json).

Ham ZIP2,329,149,449byte,SHA256
`e2f2e4c4025fc4b7a64b25b08504760432a3ffdbcfa64ee1594101c1712d1898`.
Drive `MyDrive/thesis/result/gsd_lowband_halfmix_20261107/raw_package_20261107T105026Z.zip`;
yerel `result/modelnet40_c/lowband_halfmix_study/20261107-105300_raw/`.
70CLIparçası ve birleştirilen orijinal ZIP boyut/SHA eşleşti; path/type/CRC ve
383payload SHA doğrulandı;33orijinal ZIP ayrıca canonical result dizinine kopyalandı.
[Arşiv receipt'i](evidence/gsd_lowband_halfmix_20261107/raw_receipt.json),
[yerel kabul](evidence/gsd_lowband_halfmix_20261107/local_acceptance.json),
[analiz kodu/kimliği](evidence/gsd_lowband_halfmix_20261107/executed_analysis.json).
Tam analysis_v1/v2 ve raw byte'lar değişmeden tutulur; compact evidence repository'dedir.

[Inference/Open] Sınırlı tek-seed gelişim pilotu ve başarısız native tekrar kapısı
nedeniyle doğrulanmış GSD üstünlüğü, unseen/full-benchmark sonucu, tekrar-seed
sonucu veya varsayılan terfi yok. Yeni alpha/ID/bozulma seçimi ya da kapsam
genişletme yapılmadı. Sonraki bilimsel adım deterministik native onarım ve aynı
kilitli karşılaştırmanın bağımsız tekrarıdır; bu çalışma bunları gerçekleştirmiş sayılmaz.

[Run/Open] Kendi standardA10040GB oturumu
`tta-gsd-lowband-halfmix-a100-20261107-r1`, endpoint
`gpu-a100-s-usc1c0-gz8ghzlq66bg`; izleyici91825canlı kalır. Araştırma süreçleri
bitti, VM kapatılmadı; kullanıcı tamamlanma+kapatma izni vermedi. Peer runtime,
worktree ve izleyiciye dokunulmadı. Yayın/derived yedek son receipt'leri bu kayda eklenir.


[Run] Full analysis/source/test derived ZIP402,060bytes,
SHA256`61583a1d613837eb711cd51b10b759ef316ec923671cddfb8bf113ab3e4c2bc8`,
Drive `MyDrive/thesis/result/gsd_lowband_halfmix_20261107/derived/derived_20261107T110815Z.zip`.
Local CRC/member SHA, Drive size/SHA and actual Drive-path CLI roundtrip pass.
[Derived backup receipt](evidence/gsd_lowband_halfmix_20261107/derived_backup.json).
Clean integration807tests/2skip/500subtests passed61.61s; the33-run recount from
that checkout is exactly equal to accepted analysis_v2. Independent review found
no blocker. [Integration checks](evidence/gsd_lowband_halfmix_20261107/integration_verification.json).
