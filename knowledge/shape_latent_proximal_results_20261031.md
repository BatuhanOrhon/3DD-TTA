# Global shape proximal guidance: eşlenmiş küçük pilot sonucu

2026-10-31. **[Run/Inference] Kendi matched original referansına güvenilir accuracy üstünlüğü gösterilmedi.** Proxy3 +0.78125 puan verdi; koşullu %95 aralık [−3.125, +4.6875]. Temiz latent'e toplu MSE %10.4213 azaldı. Guidance'sız prior3'e göre hem mesafe hem accuracy nokta tahmini iyileşti; accuracy aralığı sıfıra değiyor. Doğrudan mean kolu accuracy'de proxy3'ten bir olay daha iyi. Operasyonel global diffusion kapalı kalır; hiçbir kol default yapılmadı.

Etiketler: [Paper], [Code], [Run], [User report], [Inference], [Open].
[Önceden sabitlenen tasarım](shape_latent_proximal_pilot_20261031.md), [plan](shape_latent_proximal_execution_plan_20261031.md), [kompakt kanıt](evidence/shape_latent_proximal_pilot_20261031/README.md).

## Sabit protokol ve accuracy

[Run] ModelNet40-C severity5; ilk 32 önceden exposed development nesnesi; Gaussian, Background, Rotation ve LiDAR; seed0, B32, frozen Point-MAE. Kol başına 128 nesne–bozulma olayı vardır; bunlar 128 bağımsız nesne değildir. Local N8/Background35, gamma=.03, eta=.003, p=.95, total100, SCD normalization off, raw/eval LION ve EMA off korunur. Primary readout dört adayın eşit probability mean'idir: normal clean2/4/6+terminal8, Background clean9/18/27+terminal35.

| Kol | Doğru/toplam | Accuracy, % |
|---|---:|---:|
| original | 58/128 | 45.31250 |
| mean | 60/128 | 46.87500 |
| prior3 | 54/128 | 42.18750 |
| proxy3 | 59/128 | 46.09375 |
| oracle3 | 64/128 | 50.00000 |

| Eşlenmiş karşılaştırma | Fark, puan | Düzelten/bozan | Koşullu %95 aralık, puan |
|---|---:|---:|---:|
| mean-original | +1.56250 | 4/2 | [-2.34375, +5.46875] |
| prior3-original | -3.12500 | 1/5 | [-7.03125, +0.00000] |
| proxy3-original | +0.78125 | 4/3 | [-3.12500, +4.68750] |
| proxy3-prior3 | +3.90625 | 6/1 | [+0.00000, +7.81250] |
| oracle3-original | +4.68750 | 7/1 | [+0.00000, +10.15625] |
| oracle3-proxy3 | +3.90625 | 7/2 | [-0.78125, +8.59375] |

[Run/Inference] Aralıklar 10,000 draw, seed20261031 ile hesaplandı. 32 orijinal nesne yeniden örneklenirken dört bozulma birlikte taşınır. Tek diffusion seed'i, sabit koşullar ve exposed cohort'a koşulludur; çoklu karşılaştırma düzeltmesi yoktur. Altı aralığın hepsi sıfırı içerir veya sıfıra değer. Sıfır sınırı pozitif significance diye yorumlanmaz. Proxy3−prior3 +3.90625 puan, 6 düzelen/1 bozulan olay verir: küçük keşifsel mekanizma desteğidir, güvenilir baseline kazancı değildir. Mean 60, proxy3 59 doğru verdi; ek global zincir doğrudan proxy müdahalesini accuracy'de aşmadı.

| Koşul, accuracy % | original | mean | prior3 | proxy3 | oracle3 |
|---|---:|---:|---:|---:|---:|
| gaussian | 65.625 | 62.500 | 65.625 | 62.500 | 75.000 |
| background | 56.250 | 59.375 | 50.000 | 59.375 | 56.250 |
| rotation | 28.125 | 34.375 | 28.125 | 31.250 | 31.250 |
| lidar | 31.250 | 31.250 | 25.000 | 31.250 | 37.500 |

[Run] Proxy3−prior3 farkı Background +9.375, Rotation +3.125, LiDAR +6.25, Gaussian −3.125 puandır. Bozulma adına göre yöntem seçimi veya tuning yapılmadı. Bütün per-corruption doğru/toplam ve düzelten/bozan farkları kompakt JSON/CSV'dedir.

## Müdahale ve temiz latent'e mesafe

[Code/Run] LION yapısı q(z|X)q(h|X,z), p(z)p(h|z)p(X|h,z); global z128, local h2048×4. Proxy `zR=mu_new+sigma0*aynı eps_z`, önceki tamamlanmış baseline-denoised LION-frame çıktının global encoder'ından gelir ve temiz veri kullanmaz. Original z0'ı, mean zR'yi seçer. Prior3/proxy3/oracle3 frozen global prior'ı üç kez çağırır; global backward yoktur.

DDIM: 1000 train/100 inference, linear beta .0001→.02, epsilon prediction, clip off, final alpha1, eta0; t20/10/0, model time21/11/1. Üç kol aynı private Gaussian'ı kullanır; ortak RNG ilerlemez. Proxy3/oracle3 her temiz tahminde `c_guided=(c_prior+target)/2`, `prev_guided=prev_DDIM+sqrt(alpha_prev)*(c_guided-c_prior)` uygular. Eski %5 küre, grid veya yeni katsayı yoktur. Bu closed-form clean-space guidance'dır; tam DPS/DDNM/DiffPIR veya calibrated posterior değildir.

[Run] Beş kol aynı immutable preparedX/z0/h0/local noise/postprepare RNG ve h0XYZ Chamfer anchor'ını kullanır. İkinci local zincir ilk h0/noise'dan başlar; h yeniden encode edilmez ve h* üzerinde ilave denoising yapılmaz. Clean probe/final decoder her kolun kendi seçilmiş başlangıç z'sini kullanır; mevcut style gradientleri sürer. Oracle hedefi ayrı dosyadadır; yalnız oracle3 inference'ı onu okur. Diğer kolların temiz hedef karşılaştırması yalnız posthoc analizdir.

[Run] Fiziksel global proof `final_z`, aynı ID ve eps ile eşlenmiş eski clean encoder hedefiyle karşılaştırıldı. Absolute MSE, olaylar ve 128 koordinat üzerindeki ortalamadır. Toplu oran `sum(arm squared error)/sum(original squared error)`; olay oranlarının ortalaması/medyanı ayrı istatistiklerdir. Cosine, `cos(final_z-z0, clean_z-z0)` olarak tanımlıdır.

| Kol | Absolute MSE | Toplu MSE oranı | Olay oranı ortalaması | Olay oranı medyanı |
|---|---:|---:|---:|---:|
| original | 0.12835938 | 1.000000 | 1.000000 | 1.000000 |
| mean | 0.11804054 | 0.919610 | 0.932199 | 0.884446 |
| prior3 | 0.13492245 | 1.051130 | 1.093273 | 1.047105 |
| proxy3 | 0.11498263 | 0.895787 | 0.896702 | 0.868910 |
| oracle3 | 0.00210692 | 0.016414 | 0.017092 | 0.016337 |

| Kol | Medyan ‖final_z−z0‖ | Medyan temiz-yön cosine |
|---|---:|---:|
| original | 0.000000 | tanımsız |
| mean | 1.778015 | 0.39004 |
| prior3 | 0.982989 | 0.021268 |
| proxy3 | 1.561492 | 0.398376 |
| oracle3 | 3.478889 | 0.999299 |

[Run/Inference] Proxy3 toplu MSE'si original'a göre %10.4213, prior3'e göre %14.7787 daha azdır. Mean'dan da daha yakındır, accuracy'si ise bir olay daha düşüktür; latent yakınlığı classifier faydası garantisi değildir. Oracle3 toplu oranı .016414 ile yaklaşık %98.3586 yaklaşır, accuracy %50 olur. Temiz bilgi kullanan oracle3 uygulanabilir TTA veya üst sınır değildir. Bu sonuç h−z uyumsuzluğunu tek neden olarak belirlemez.

[Run/Open] Sıfır original-error paydası yoktur. Original hareketi sıfır olduğu için 128 cosine tanımsızdır ve null saklandı; diğer kolların cosine'ları sonludur. Clean encoder hedefi önceki runtime'dandır; küçük cross-runtime native farklar ve normalizasyon/interpolasyon etkileri korunur. Bu mesafe saf fiziksel corruption hatası değildir.

## Tekrar davranışı ve yorum sınırı

[Run/Open] Mevcut smoke/pilotun aynı Gaussian/Background 32 girdisinde ID/label, initial tensor hash'leri, actual reconstruction RNG before/after, classifier/LION state ve executed source kimlikleri eşleşir. Buna rağmen original Gaussian 19/32→21/32; Background 18/32→18/32 ve iki koşulda ikişer tahmin değişir. Prior3 Gaussian 20→21, Background 18→16 verir; diğer kollarda da bazı tahminler değişir. [Fiziksel karşılaştırma kaydı](evidence/shape_latent_proximal_pilot_20261031/technical_pilot_repeatability.json).

[Inference/Open] Hash/RNG eşleşmesi bitwise native inference tekrarı sağlamamıştır. Bu, önceden bilinen native sayısal değişkenlikle uyumludur; kesin operator/root cause izole edilmedi. Object bootstrap bu tekrar varyansını veya yeni seed/unseen nesne genellemesini kapsamaz. Küçük accuracy farkları sağlam nedensel/global-prior üstünlük kanıtı değildir. Yeni GPU tekrarı başlatılmadı.

## Süre, bellek ve deployment maliyeti

| Kol | Worker wall/artifactIO, s | Reconstruction+terminal, s | Peak, MiB |
|---|---:|---:|---:|
| mean | 38.2963 | 16.1668 | 25223.078 |
| oracle3 | 38.5402 | 16.3777 | 25223.297 |
| original | 38.3437 | 16.2139 | 25223.078 |
| prior3 | 38.5989 | 16.4046 | 25223.281 |
| proxy3 | 38.5651 | 16.4041 | 25223.281 |

[Run/Inference] Üç global kol batch başına 3 global forward/0 global backward kullanır; pilotta toplam 3kol×4batch×3=36 batched global forward vardır. Global/local farklı ağlardır; 3/8 ve 3/35 çağrı oranları zaman veya noise-depth eşitliği değildir. Worker ölçümleri önceki baseline pass ve proxy encoder maliyetini dışlayan replay'dir; deployment hızlanması çıkarılmaz. Gerçek mean/proxy3 yöntemi ilk local pass+global encoder ve ikinci local pass gerektirir (normal8+8, Background35+35); proxy3 ayrıca üç global forward yapar. Matched end-to-end deployment timing ölçülmedi.

[Run] Bilimsel faz 18:50:29.446863 UTC başladı; smoke, yerel geçiş, pilot ve son uzak ham arşivleme 478.451071s=7.9742dk sürdü; tek 2400s sınırının altında. Ön tahmin 15–25dk idi. Canonical setup ve sonraki CPU analiz/Git bu bilimsel GPU ledger'ından ayrıdır. Aynı VM'de Cell1/2 tekrarlanmadan Cell3–10 tamamlandı. Yalnız kullanıcı tarafından ertelenmiş Cell2 fastrlock hatası korundu; sonraki requirements kurulumu fastrlock0.8 yükledi. İki inherited pip conflict uyarısı kaydedildi; CUDA/native/data/checkpoint kapıları geçti.

## Kanıt ve kaynak

[Run] Pilot 617 dosyalık manifest/20 prediction shard/20 selected cache/20 global proof; smoke 567 dosyalık manifest/10 shard içerir. SHA/CRC, güvenli path, duplicate/symlink ve bağımsız NumPy1 fiziksel DDIM/half-target/Gaussian/old-proof/imported h0/noise/anchor/RNG/private Gaussian/own-z/style/frozen-state/source/ID/label/PM kontrolleri geçti. Yerel acceptance ve tüm outcome matrisleri uzak değerlerle birebir eşleşir. Ham çıkarımlar değiştirilmedi; derived çıktılar ayrıdır.

[Code/Run] Worktree `7bf0`, branch `integration/chamfer-guidance-variants-20261030`, HEAD `a82200adbe117ce56a134dc3c22601cb73dc2654` + dirty 346 kaynak dosyası. Bu devamda tta/model/default değiştirilmedi; kullanıcı değişiklikleri korundu. Eski imported proof önceki `1bacabf` çalışmasına aittir; yeni kaynağın üretmiş olduğu iddia edilmez. Yerel bilimsel kod sealed snapshot ile byte-identiktir.

- Manifest: `690caa62d2a2603ebf9480cc86e3a66f5d13fca77ec1dbe34a810cd19fde15f5`.
- Source ZIP: `1b2fa5b423e407e5ce1339060af63751a46d23cb95a5dffc74b1524451523c82`.
- Actual executed source: `98148d90cb7fe7d5995669bd8ed5b6bfb3c9a47c30d7e78309afedd5b55542b9`.

19 yeni contract + 4 operator = 23 CPU test, 3 summary self-check ve dış wrapper session assertion RED→GREEN geçti. Yeni summary/wrapper bağımsız incelemesi blocker bırakmadı; son per-corruption contrast alanları ayrıca RED→GREEN ve gerçek dört-koşul hamıyla doğrulandı. CPU testleri GPU sonucu yerine sayılmadı.

| Ham arşiv | Byte | SHA256 |
|---|---:|---|
| pilot |86140365|f00225f0e404d8d7eb410030e5cb2abc2718122e9ee5d794168efccb64336293|
| smoke |63930417|10056573a4b40363e27644fc414449a2b478c15aa877e08971ff317673163b66|
| CPU deployment |1456046|8f73aa25619f21baf06c0956d86867bd982de765f066b96ae304d1cc9d1071a5|

Yerel kök: `result/preparation/shape_latent_proximal_pilot_20261031/operator/`; `pilot_raw.zip`, `pilot_extracted/`, `pilot_local_analysis/`, `pilot_summary/` ve karşılık gelen smoke kayıtları. Drive kökü: `MyDrive/thesis/result/shape_latent_proximal_pilot_20261031/phase-evidence/`; pilot `phase_pilot_20261031T185449177416Z`, smoke `phase_smoke_20261031T185029446885Z`. Ek source/operator/derived kapsül ve Git yayını bu metin ilk yazıldığında hazırlanıyordu; sonraki receipt/yayın kaydı bu durumu tamamlar.

[Run/Open] Kendi `tta-shape-proximal-a100-20261031-r1` oturumu, endpoint `gpu-a100-hm-ass1c0-zwl345mh6780`, son yerel analiz sonrası envanterde canlıydı. Kapatma izni veya komutu yok; yeni GPU işi kalmadı. Tam veri, ek seed/classifier veya guidance grid başlatılmadı. Falsifier: ayrılmış nesneler ve ek seed/native repeat kontrolleriyle matched baseline artışı gösterilmeden güvenilir method gain iddiası kurulamaz.

## Final archival acceptance — 2026-10-31

[Run] Source/assets/operator/derived copies were verified on Drive at
`MyDrive/thesis/result/shape_latent_proximal_pilot_20261031/final-archive-20261031T190545Z/`.
The 2,045,023-byte derived/source/operator capsule has SHA256
`322f3ff3ba16ed2f412b4b1e425be1d62cdd0d9fef62c00acceeec9dcc099094`;
Drive download roundtrip, all68 member identities and raw extraction recheck pass.
Both raw phases remain unchanged after analysis. The sealed capsule intentionally
retains its then-pending publication notes; these final receipts supersede them.
See compact `final_archive_receipt.json` and `derived_roundtrip_receipt.json`.
No GPU work remains and no runtime teardown was issued.

[Code/Run] Temiz latest-target yayın ağacında568CPUtest/412subtest ve ayrı60NumPy1test geçti. Kalan bir skip tarihsel CurveNet source fixture'ıdır. Birleşmiş ağacın GPU ile tekrar çalıştırıldığı iddia edilmez; opt-in dört kod/test dosyası yürütülmüş sealed kaynakla aynıdır. Entegrasyon kapsamı `thesis_integration_shape_proximal_20261031.md` kaydındadır.
