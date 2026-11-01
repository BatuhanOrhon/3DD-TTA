# Shape latent'e local/spectral guidance: geçmiş deney ve ayrı-loss değerlendirmesi

2026-11-01. [User report/Code/Run/Inference/Open] Kullanıcı, local latent
üzerinden z gradienti veya global diffusion + Graph Spectral Loss denenip
denenmediğini ve z'ye ayrı loss vermenin mantığını soruyor. Knowledge, eski
Git kaynakları ve mevcut kabul edilmiş analizler denetlendi. Yeni GPU/model
inference, kod/config değişikliği veya runtime müdahalesi yapılmadı.

## Sonuç: üç farklı işlem karıştırılmamalı

| İşlem | Gerçek durum | Sonuç kanıtı |
|---|---|---|
| Local predicted-clean h üzerinde spectral loss; local prior üzerinden z/style'a türev | GSD ve block-routing içinde uygulandı | Kabul edilmiş küçük pilot/replication ve sonraki C+S deneyleri var |
| Global z diffusion açık; local spectral loss ile global state güncelleme | Eski dual_sync kaynakta var | Belirli koşuya bağlı doğrulanmış accuracy/runtime arşivi kurtarılamadı |
| Global temiz-z tahmini üzerinden koşullu local denoising enerjisiyle guidance | Son conditional pilotta uygulandı | Gerçek native gradient testi geçti; güvenilir accuracy veya net clean-z yakınsaması yok |

Bu nedenle global diffusion + spectral guidance için "hiç yazılmadı" veya
"doğrulanmış biçimde başarısız oldu" denmez. Kod var; ilgili eski koşunun
sayısal sonucu mevcut kayıtlarla doğrulanamıyor. Ayrı local/style loss ise
yalnız bir öneri değil, geçmişte küçük ölçekte gerçekten denendi.

## 1. Local loss'tan z'ye gradient gerçekte nasıl gidiyor?

[Code/Derivation] Donmuş local prior'ın temiz local tahmini
`h_hat = F(h_t, z, t)` olsun. Spectral loss `L_spec(h_hat, h_ref)` için:

\[
\nabla_z L_{spec}
= J_{F,z}^{T}\nabla_{\hat h}L_{spec}.
\]

Local h gradientini z'ye kopyalamıyoruz: h8192 ve z128 farklı uzaylardır.
Türev local prior'ın z koşullaması üzerinden zincir kuralıyla alınır. Bu
işlem için global diffusion prior'ının çalışması gerekmez. Aynı all55
config'te global2style identity'dir; başka checkpoint'e genellenmez.

`tta_gsd_v2.py` canlı predicted-clean h'den spectral loss üretir,
`total_loss.backward()` ile local/style gradientlerini alır. Hedef, ilk
bozuk girdiden encode edilmiş h'nin spektral yapısıdır. Bu temiz-z hedefi
değildir; low-frequency local XYZ fidelity de semantik global-z yönü
garantisi değildir. [Önceki gradient analizi](gsd_scd_gradient_composition_review_20261004.md).

## 2. Ayrı local/style loss gerçekten test edildi

[Run] 2026-10-05 hard-v1 routing pilotu: frozen Point-MAE, ModelNet40-C,
severity5, Gaussian/Impulse/Background/Shear, seed0,64 örnek/bozulma,
local5/Background35, original-style final decode; global prior kullanılmıyor.

| Kol | Local h loss | z/style loss | Doğru/256 | Dört-bozulma accuracy % |
|---|---|---|---:|---:|
| R_S0 | SCD | kapalı |164|64.062500|
| R_SG | SCD | spectral |167|65.234375|
| R_G0 | spectral | kapalı |147|57.421875|
| R_GS | spectral | SCD |152|59.375000|

R_SG−R_S0 +1.171875puan:4düzelten/1bozan olay. Bozulma farkları
Gaussian+1.5625, Impulse+3.125, Background−1.5625, Shear+1.5625puan.
Per-corruption aralıklar0içerir veya0'a değiyor; ortak nesne eşlemesi
bozulmalar arasında doğrulanmadığından pooled CI üretilmedi.

[Run/Limit] Bu karşılaştırmanın referansı style-off'tur. Standart SCD'nin
hem h hem style güncellediği eşlenmiş C_SCD kontrolü bu routing fazında
yoktur. Eski diagnose kontrolünün input/native/runtime hash'leri farklıdır;
onun accuracy'sini buraya taşıyamayız. +1.17puan, standart SCD üstünlüğü
diye sunulmaz. Reverse route +1.953125puanı kendi zayıf style-off kontrolüne
göre verir; artış Background'da yoğunlaşır, Gaussian/Impulse geriler.
[Tarihli pilot değerlendirmesi](gsd_pilot_review_action_plan_20261005.md).

[Run] Sonraki style-gradient projection çalışması aynı routing deneyi
değildir. Dört bozulma/sonraki256indeks/üç seed development replication'da
C_SCD65.1042%, P_SUM65.3320%, P_PC65.9505%, P_NORM65.2669%;
P_PC−C_SCD+.8464puan, bütün seed'lerde pozitif; global prior yine kapalı.
Bu sınırlı pozitif sonuç, global diffusion + spectral guidance kanıtı olmaz.

[Run] Daha sonraki140nesne/veriseti,all15/üç seed eşlenmiş component
confirmation'da Point-MAE ModelNet C64.3492% → C+S64.3968% (+.0476puan);
Scan C34.3810% → C+S33.9048% (−.4762puan). İki dataset/dört head balanced
C53.013889% → C+S52.962302%; C−C+S+.051587puan CI[−.125,.234127].
Genel spectral katkı üstünlüğü veya eşdeğerliği gösterilmedi. Bu C+S, iki
blokta toplam-loss katkısıdır; yalnız z-loss routing ablation'ı değildir.
[Kapsam, ayarlar ve sınırlar](guidance_component_sensitivity_results_20261028.md).

## 3. Eski global/local dual kodunda ne var?

[Code] `a458cd46bba1288a3f41302548bc777700cbc7d4` kaynakları Git'ten
yeniden okundu; dosya SHA/byte kimlikleri [audit](evidence/shape_latent_spectral_audit_20261101/audit.json)
içinde. Kaynak mevcut checkout kökünde bulunmadığından Git nesnesi kullanıldı.

- `dual_seq`: önce koşulsuz global DDIM; sonra yerel spectral/SCD guidance.
  Global DDIM fazı spectral loss ile guide edilmez; sonraki style güncellenir.
- `dual_sync`: global `prev_sample` detach edilip yeni leaf yapılıyor;
  local prior'a bu ara z koşul olarak veriliyor. Spectral/SCD loss'tan
  `next_noisy_z.grad` alınıp global state gerçekten güncelleniyor.

[Code/Inference] Sync'deki detach bütün z gradientini yok etmez: sonraki
local prior üzerinden türev vardır. Kestiği yol global denoiser Jacobian'ıdır.
Üstelik local prior'a clean estimate yerine ara noisy z verilir. Bu,
temiz global koşul eğitiminden farklıdır; son conditional pilotun
clean-estimate/global-VJP yoluyla eşdeğer değildir. Eski launcher's p=.5
override, eval/RNG ve decoder seçenekleri de ayrı confound'lardır.

[Open/User report] Önceki kullanıcı gözlemi global diffusion'un daha yavaş
ve düşük accuracy verdiğidir. Erişilebilir ref/knowledge/notebook/yerel
result aramasında o denemeye bağlı komut/receipt/raw tablo bulunamadı.
Kesin accuracy, fark veya tek hata nedeni verilemez. [Önceki arama ve
sınırlar](shape_latent_tta_research_20261031.md),
[arama kapsamı](evidence/shape_latent_research_20261031/search_ledger.md).

## 4. Son conditional pilot Graph Spectral Loss kullanmıyordu

[Run] Sabit h*'ın noising'inden local epsilon-prediction MSE'sinin z
gradienti alındı; global clean-estimate Jacobian'ından global noisy state'e
taşındı.32nesne/Gaussian-Background-Rotation-LiDAR/seed0:
encoder off42.1875% → guided42.96875%; son-SCD-z off42.96875% →
guided41.40625%. İki başlangıçta clean-z MSE başlangıca göre4.55%/4.58%
artar. Native türev testleri geçer, bütün yedi accuracy CI0içerir.
[Sonuç ve farklı loss'un tanımı](shape_latent_conditional_results_20261031.md).

Ortak dalın daha yeni, ayrı tam-z+h yeniden encoding pilotu da clean-z
MSE−5.14% ile accuracy faydasının ayrılabildiğini gösterir: baseline
PM60.5208%, ikinciN3 PM58.0208%; global prior kapalı, h ve anchor da değişir.
Bu, önceki yalnız-ortalama refresh deneyi değildir ve burada yeniden
çalıştırılmadı. [Sabit6ff2e49 kaydı](https://github.com/BatuhanOrhon/3DD-TTA/blob/6ff2e4952a0fa4842aa32b14afaf1ae8fa123f27/knowledge/lion_reencode_results_20261101.md).

## 5. z güncellerken ayrı loss mantıklı mı?

[Inference] Evet, araştırma hipotezi olarak mantıklı. LION'un hiyerarşik
local/global temsilleri farklı rollerdedir; tek bir local geometrik
fidelity loss'unun her iki blok için en yararlı yön olduğu varsayılmamalı.
Ancak h ve z tam ayrışmış değildir. Spectral fidelity de SCD de aynı bozuk
h referansını koruyabilir; loss adını değiştirmek temiz bilgi eklemez.
[LION Sec.3](https://arxiv.org/html/2210.06978v1#S3).

[Derivation/Inference] `h <- h - alpha_h grad_h L_h` ve
`z <- z - alpha_z grad_z L_z` biçiminde farklı objective'ler kullanmak
mümkündür. Genel durumda bu vektör alanı tek bir ortak scalar loss'un
gradienti değildir; toplam-loss yakınsaması veya conflict çözümü garantisi
vermez. Optimize edilen objective ile raporlanan clean-z MSE/accuracy'yi
ayrı tutmak gerekir. z128 koordinatına nokta grafiğinin GFT'si doğrudan
uygulanmaz; graph loss local/decoded point yapısı üzerinden kurulmalıdır.

[Inference/Options] Mevcut kanıtların desteklediği üç ayrı soru:

1. Local SCD korunurken z'ye spectral loss: geçmiş R_SG sinyali var,
   fakat standart SCD ve style-off'la aynı hazırlıktan eşlenmiş kontrol şart.
2. z'yi frozen encoder'ın zR önerisine yaklaştırmak: mean/proximal pilotlar
   zaten küçük-belirsiz accuracy sinyali verdi. İki-pass maliyet ve proxy'nin
   kendi yanlışını yeniden üretmesi sınırdır; temiz hedef değildir.
3. Local conditional-energy gradientini global DDIM olmadan izole etmek:
   son deneydeki zayıf yönün global prior driftinden ayrılmasını test eder.
   Bu saf adım henüz çalıştırılmadı; yeni metric başarı iddiası değildir.

[Decision/Open] Önce aynı local SCD/aynı h0-noise/decoder altında yalnız
z objective'ini ayıran sınırlı teşhis daha açıklayıcıdır; yeni global
DDIM/noise/graph değişikliğini aynı anda eklemek hangi bileşenin etkili
olduğunu belirsizleştirir. Sonraki global-guided spectral varyantında clean
z tahmini kullanılmalı, global denoiser grafiği korunmalı ve gradient doğru
noisy z_t'ye taşınmalı. [DPS Sec.3](https://arxiv.org/html/2209.14687v4#S3)
bu zincir-kuralı ayrımını destekler; unknown-corruption SCD/spectral loss'u
kalibre likelihood yapmaz ve LION accuracy garantisi vermez.

Kabul ölçütü loss azalması tek başına değildir: frozen classifier paired
accuracy, kontrollü applied norm/maliyet ve bağımsız clean-z posthoc yön
teşhisi birlikte okunmalıdır. Clean hedef/etiket inference objective'i veya
örnek başına step seçicisi olmaz. Yeni GPU/coeff/grid/kod çalışması
başlatılmadı; bu not yöntem seçimi/onayı veya runtime yetkisi değildir.
