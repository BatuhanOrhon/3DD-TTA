# Shape latent refresh: üç-kol pilot sonucu

2026-10-31. **[Run] Tamamlandı; ham Drive+local SHA/CRC ve bağımsız NumPy1
recount kabul edildi. Yeni yöntem default yapılmadı.**
[Onaylı tasarım/geçmiş](shape_latent_refresh_pilot_20261031.md),
[plan](shape_latent_refresh_execution_plan_20261031.md),
[kompakt kanıt](evidence/shape_latent_refresh_pilot_20261031/README.md).
Etiketler: [Paper], [Code], [Run], [User report], [Inference], [Open].

## Sonuç ve karar

**[Run]** ModelNet40-C/severity5, aynı64 exposed development nesnesi,
15bozulma/seed0/Point-MAE/B32. Kol başına960nesne–bozulma olayı;960bağımsız
nesne değildir. N8/B35, gamma=.03,eta=.003,p=.95,total100,raw/eval/EMAoff,
DDIM final_alpha1, dört adayın eşit probability mean'i. Global diffusion
forward/backward **0**; frozen global encoder kullanıldı.

| Kol | Global z | Doğru /960 | All15 accuracy | Baseline farkı |
|---|---|---:|---:|---:|
| baseline | mu0+sigma0*eps |585|%60.9375|—|
| mean_refresh | mu_new+sigma0*aynı eps |593|%61.7708|+0.8333puan|
| posterior_refresh | mu_new+sigma_new*aynı eps |588|%61.2500|+0.3125puan|

| Eşlenmiş karşılaştırma | Fark, puan | Düzelten / bozan olay | Koşullu %95 aralık |
|---|---:|---:|---:|
| mean−baseline |+0.8333|30 /22|[-0.5208,+2.1875]|
| posterior−baseline |+0.3125|29 /26|[-1.0417,+1.6667]|
| posterior−mean |−0.5208|7 /12|[-1.3542,+0.3125]|

**[Run/Inference]**10,000bootstrap çekilişi(seed20261031),64orijinal nesne
üzerinde tüm15koşulu birlikte örnekler. Bütün aralıklar0içerir; tek diffusion
seed/fixed corruptions/exposed dev cohort koşullu aralıklardır. Çoklu
karşılaştırma düzeltmesi yok. Ortalama kayması nokta tahmininde daha iyi;
istatistiksel olarak güvenilir gain veya sigma yenilemenin zararlı olduğu
kanıtlanmadı. Yeni tam veri/ek seed/katsayı araması otomatik başlamaz.

**[Decision]** Operasyonel global diffusion kapalı ve mevcut N8 baseline
korunur. Mean refresh bu pilotta küçük, belirsiz bir pozitif sinyal verdi;
iki-pass maliyet karşılığında henüz güçlü bir sonuç değil. Full posterior
yenileme ek fayda göstermedi. İlerleme kararı sonuçlar incelendikten sonra.

## Kontrol edilen müdahale

**[Code/Run]** Her batch için tek baseline denoising çıktısı
`X*=D(h*,z0)` LION2048×3 frame'inde global encoder'a verildi. İki yeni z
aynı X*, aynı mu_new ve aynı eps_z kullandı. E/F yalnız sigma_old/new
seçiminde ayrılır. İlk h0/noise/postprepare RNG ve A-hXYZ Chamfer anchor
ortaktır. İkinci local zincir ilk h0'dan başlar; h* üzerinde ilave sekiz adım
çalıştırılması değildir. h yeniden encode edilmedi. SCD style gradientleri
baseline'daki gibi devam eder; her kolun clean probe ve final decoder'ı
**kendi ilk z'sini** kullanır. Model/classifier ağırlıkları ve buffer'ları sabit.
Clean input/latent/etiket öneri üretimine veya inference seçicisine girmez.

**[Run]**90gerçek batch shard,30common state,30posterior proof ve90selected
cache doğrulandı. Her batch'te actual reconstruction RNG before/after
hash'leri üç kolda aynı; clean-probe/final style hash'leri kendi ilk z'sine
eşit. Tüm kaynak/state/ID/label/PM/cache/posterior denklemleri ve ham
evidence_manifest hash'leri geçer. Baseline tamamlanma receipt'i son model
state/source guard ve beklenen shard sayısından sonra üretilir; adaylar
tamamlanmamış veya kaynak imzası değişmiş baseline'ı kullanamaz.

## Bozulma davranışı

**[Run]** Background A/E/F %56.25/%59.375/%56.25;
macro14 %61.2723/%61.9420/%61.6071. Mean refresh impulse'da+6.25puan,
Background/occlusion'da+3.125puan; cutout ve distortion'da−4.6875puan verir.
Full refresh cutout'ta−7.8125puan. Bunlar64nesnelik betimsel satırlardır;
bozulma adına göre yöntem seçimi/tuning yapılmadı.
[Tüm satırlar](evidence/shape_latent_refresh_pilot_20261031/per_corruption.csv).

## Global ortalama gerçekten temiz hedefe yaklaştı mı?

**[Run/Inference]** Önceki oracle'ın aynı-ID temiz posterior ortalamaları
yalnız **posthoc teşhis** hedefidir.960/960dirty prepared input ve eps_z
önceki oracle proof ile byte-identiktir; model state'leri de aynıdır.
Temiz hedef encoder çıktısında küçük cross-runtime native hesap farkları
bulunabileceği için bunu tam aynı-process counterfactual saymıyoruz.

`R=||mu_new−mu_clean||²/||mu0−mu_clean||²`:

| Teşhis | Denoising sonrası | Yalnız AE çevrimi |
|---|---:|---:|
| Medyan R |0.9535|1.0076|
| Nesne–bozulma başına R'lerin ortalaması |1.0454|1.1105|

**[Inference]** Medyan örnek biraz yaklaşır fakat bazı örnekler uzaklaşır;
ortalama oran1'in üzerindedir. Bu ortalama oran toplu MSE oranı değildir.
Temiz yöne cosine medyanı0.3475; pozitif cosine tek başına doğru adım boyu
veya accuracy garantisi değil. Mean z müdahalesi medyan%17.2385||z0||,
full posterior%17.4863; sigma değişiminin ek farkı medyan%1.7866||z0||.
AE-only kontrolü yalnız yön teşhisidir, ayrı accuracy kolu değildir.
Bu sonuç, basit geri beslemenin temiz-z oracle kazanımını güvenilir biçimde
geri kazandığını göstermiyor; z−h uyumsuzluğu veya global prior başarısızlığı
için yeni tek neden kanıtı da üretmiyor.

## Maliyet ve sınırlar

**[Run]** Actual worker wall A/E/F136.523/118.114/118.195s;
reconstruction+terminal105.141/86.974/87.058s. A ek encoder/AE teşhisi ve
ortak proof hazırlığını yapar; diğerleri replay eder. Bu sayılardan adayın
baseline'dan hızlı olduğu çıkarılamaz. Deployment ilk baseline geçişini
de gerektirir: normalde8+8, Background35+35local reverse çağrısı ve ek
global encoder vardır. Matched fresh end-to-end deployment timing ölçülmedi.
Whole-process peak A/E/F25222.77/25223.80/25223.80MiB.

**[Run/Open]** Önceki oracle32GB smoke %70.3125, yeni smoke%67.1875;
aynı girdiler/eps/noise/model state'lerine rağmen global mu maxabs~0.0007,
hmu~0.0019farkları vardır. Bilinen native numerical variation ile uyumludur;
accuracy farkına katkı/root cause izole edilmedi. Bu pilotun kendi fresh
baseline'ı kullanılır; eski %60.8333oracle baseline'ı transplant edilmez.
Koşullu bootstrap native tekrar varyansını ve seed genellemesini ölçmez.

## Kaynak, arşiv ve yürütme

**[Code/Run]** Kaynak worktree/branch korunur, HEAD a82200adbe117ce56a134dc3c22601cb73dc2654
plus dirty kaynak342dosya.4yeni opt-in kod/test dosyası; tta/model/default
bu test için değiştirilmedi. Manifest
`035eb71e3952a08ba88d9619ed1d2086a0a96df48cd9207825daf232d9861d3b`;
source ZIP`998a7918296a6f98c9a747c18c0a444f07a3b0a9506de334a48fa7a28c1bc659`.
Asset payload`6cdb02407569c185b4098d6333cfa957c9c49687c6f09d7e3dc2d7f3c7563bdb`;
operator ZIP`852e5ace4e16783d6c1774ce3a5f5812ddb55ec7d7004566fe1cb21b746b1137`.

**[Run]** Research16:49:13.681384UTC'de başladı; teknik+pilot+geçiş+arşiv
631.3478s=10.5225dakika, ortak3300s sınırının altında. Setup ayrı10canonical
hücreyle tamamlandı; cell2knownfastrlock dışında hata ertelenmedi.
Pip dask/importlib-metadata ve scikit-image/networkx uyarıları korundu;
CUDA/native/dataset/checkpoint kontrolleri geçti. Fresh native bytes pinned
4binding restore öncesi arşivlendi. CPUdeployment strict/no-forward/no-CUDA
proof geçti.80ilgiliCPUtest/23subtest,14yeni contract ve4operator test geçer;
review source-binding/completion gate'leri ile controller shadow düzeltildi.

| Ham arşiv | Byte | SHA256 |
|---|---:|---|
| pilot |410634711|d828536a5d7ab5482aa65a0ada9e913eccb969651c014c020163ea8823fc7c6e|
| smoke |35333496|1a419aa2c9c367222fb630912b9de5006332863d84b98b11964ad92d55df0bde|
| CPUdeployment |1454476|f4cdab7ed6634cf805c36ab3e3cbfb6c4d8db8e735670262ec719693cf91d703|

Yerel root `result/preparation/shape_latent_refresh_pilot_20261031/operator/`;
`pilot_extracted/`, `pilot_local_analysis/`, `pilot_direction_analysis/`.
Smoke'ın ilk CLIrecount çağrısı optional output arg'ını yoksayıp ilk extraction'a
türetilmiş çıktı yazdı; immutable ham ZIP değişmedi. Manifest doğrulanmış
`smoke_extracted_v2/` yeniden çıkarıldı; ayrı derived output ile exact recount
geçti. İlk extraction kabul edilmiş ham ağaç olarak kullanılmaz.
Drive root`MyDrive/thesis/result/shape_latent_refresh_pilot_20261031/phase-evidence/`;
pilot phase`phase_pilot_20261031T165244807861Z`, smoke`phase_smoke_20261031T164913681408Z`.

**[Run/Open]** Kendi oturumumuz `tta-shape-refresh-a100-20261031-r1`, endpoint
`gpu-a100-hm-euw4a0-3e5wlo5wf3or6`; final arşiv sonrası envanterde canlı.
Kapatma komutu verilmedi; explicit task-complete+close izni yok. Yeni GPU işi
kalmadı. Yeni türetilmiş/source/operator kapsülü2314123bytes SHA256
`60ce471b4a213ace2077fb19c07c1e8cd42a56aa629aea0eeefa392d0a3db332`
Drive'da read-back ile doğrulandı; kapsül o ana ait publication-pending
notları da korur. Son Git commit'i publication receipt/final yanıtta kaydedilir.
Önceki oracle ek Drive kopyası kullanıcı kararıyla ertelenmiş kalır.
