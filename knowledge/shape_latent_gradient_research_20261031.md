# Temiz z yönü için koşullu denoising gradienti — 2026-10-31

[User report] Kullanıcı, shape latent diffusionunu temiz z'ye yaklaştıracak
bir gradient bulmak için literatür taramasının yenilenmesini istedi. Accuracy
birincil amaç; latent mesafesinin azalması tek başına başarı ölçütü değil.
Bu çalışma literatür/kaynak denetimidir: yeni yöntem uygulanmadı, GPU deneyi
başlatılmadı ve modellerin ağırlıkları değiştirilmedi.

[Inference/Decision] İlk araştırma adayı **donmuş local denoiser'ın koşullama
girdisi z'ye göre denoising-enerji gradienti**. Önce bu yönün yararlı bilgi
taşıyıp taşımadığı ölçülmeli; sonra global prior'ın temiz tahmini üzerinden
noisy global latent'e taşınmalı. Bu aday [önceki raporun B seçeneğiydi](shape_latent_clean_free_research_20261031.md#b--local-denoiser-ile-z-için-koşullu-uyum-enerjisi);
yeni keşfedilmiş veya uygulanmış bir yöntem diye sunulmuyor. Mean-refresh ve
quadratic-proxy pilotları sonrasında önceliği yükseliyor. Diffusion-TTA,
Diffusion Classifier ve özellikle CWGF §3.2/Eq.21 bağlantısı artık açık.

## 1. Bildiğimiz sonuç ve hedefin sınırı

[Run] Aynı64 nesne/all15/seed0/Point-MAE [oracle pilotunda](shape_latent_oracle_results_20261031.md)
baseline %60.833333; **temiz z/eski h %64.895833**, +4.0625puan,
koşullu object-cluster %95 CI[1.979167,6.354167]. Temiz çift %77.708333
ayrıca local hazırlığı değiştirir. Bu iki sayı ne yeni yöntemin garantili
hedef accuracy'si ne de matematiksel üst sınırdır. Eski h çalışmalarındaki
başarı, global z'nin de aynı gradientle düzeleceğini göstermez.

[Run] [Refresh](shape_latent_refresh_results_20261031.md) mean yönü
+.833333puan CI[−.520833,2.1875] ile belirsizdi. [Proximal pilot](shape_latent_proximal_results_20261031.md)
32nesne/dört bozulmada original %45.3125, prior3 %42.1875, proxy3 %46.09375,
mean %46.875, oracle3 %50 verdi. Proxy3 temiz-z toplu MSE'yi %10.4213
azaltırken original'a accuracy farkı +.78125puan CI[−3.125,4.6875] kaldı.
Dolayısıyla hem **gradient yön kalitesi** hem **son classifier accuracy**
ayrı ölçülmeli. Cohort/baseline'lar birbirine karıştırılmamalı.

[Inference] Global prior tek başına p(z)'yi sağlar; gözlenen nesneye özgü
temiz z'yi tanımlamaz. Aradığımız eksik bilgi, yerel şekil ile global temsilin
koşullu ilişkisinden gelebilir. Ayrıca paired clean z, aynı posterior Gaussian
örneğiyle tanımlanmış bir tanıdır; tek mümkün doğru latent veya accuracy'nin
bire bir parametrizasyonu değildir.

## 2. Birincil literatür ve aktarılabilir parça

[Paper] Aşağıdaki sürümlerin tam metinleri incelendi; konferans bilgisi ile
okunan preprint sürümü ayrı tutulur. Tam kaynaklar Git'e kopyalanmaz;
[SHA/URL/sürüm makbuzları](evidence/shape_latent_gradient_research_20261031/primary_source_receipts.json)
ve [arama/kaynak denetimi](evidence/shape_latent_gradient_research_20261031/search_ledger.md)
kalıcıdır. Bu tarama sistematik literatür taraması veya dış replikasyon değildir.

| Kaynak / incelenen yer | Kaynağın gösterdiği | Bizim protokole aktarım ve sınır |
|---|---|---|
| [Diffusion-TTA, NeurIPS2023; v2 §3.2/Eq.2/Alg.1, §4.4/Tab.5](https://arxiv.org/html/2311.16102v2) | Denoising hatası, tahminlerden üretilen sürekli koşullama üzerinden discriminative modele gradient sağlar. Tek timestep/noise tahmini kararsız; çoklu örnekleme önemlidir. | Koşullama z'ye türev için güçlü analog. Makale model ağırlıklarını günceller; biz yalnız örnek başına z'yi optimize etmeyi öneriyoruz. Makalenin180örnek/55s maliyeti LION süresi değildir. |
| [Diffusion Classifier, ICCV2023; v2 §3.2/Eq.8, §3.3/Eq.9, §4.1](https://arxiv.org/html/2303.16203v2) | Koşullar denoising-error/ELBO yaklaşımıyla karşılaştırılır; aynı(t,ε) çiftleri fark tahmininin varyansını azaltır. | z_d/z_proxy/z_clean enerji karşılaştırmaları eşlenmeli. Imagenet/text sınıf ayırımı, LION'ın sürekli z'sinde temiz yön garantisi sağlamaz; t≈500 sonucu doğrudan kopyalanmaz. |
| [CWGF, 2026-05 preprint v1; §3/Eq.8–14, §3.2/Eq.20–21, §3.3/Eq.23–25](https://arxiv.org/pdf/2605.07907v1) | Koşul/prompt gradienti, noising-yolu boyunca denoising-score-matching integrali üzerinden yazılır; posterior parçacıkları ve koşul birlikte güncellenir. | **Eq.21 bizim koşullu enerji gradientine en doğrudan matematiksel dayanak.** LION local DDPM zaten ε/score verir; bu koşul-gradient fikri için yeni LCM gerekmez. Tam CWGF'nin LCM, bilinen lineer ölçüm/Gaussian model ve encoder yaklaşımı mevcut unknown-corruption TTA'ya hazır çözüm değildir. |
| [P2L, ICML2024 metadata](https://proceedings.mlr.press/v235/chung24b.html); [okunan v1 §3/Alg.1–4](https://arxiv.org/html/2310.01110v1) | Donmuş diffusion denoiser'ın text condition vektörü, temiz tahmin/decoder/ölçüm kaybından gradientle güncellenir. | LION z'si, P2L'nin **condition C** değişkenine karşılık gelir; P2L diffusion latenti ise LION local h'ye daha yakındır. Bilinen A ve gözlem likelihood'u bizde yok. |
| [LATINO-PRO, ICCV2025; okunan v1 §5/Eq.7–9, App.C](https://arxiv.org/html/2503.12615v1) | Prompt, ampirik Bayes/marginal likelihood ve koşullu prior gradientiyle ayarlanır. | Koşulu sabit kabul etmeme fikri yararlı; tam algoritma LCM ve ölçüm modeli kullanır. V1'de8NFE sabit-prompt LATINO'ya,68NFE prompt-optimizasyonlu PRO'ya aittir; LION için8adım iddiası yapılamaz. |
| [DPS, ICLR2023; v4 §3/Eq.4–7](https://arxiv.org/html/2209.14687v4) | Ölçüm likelihood'u, tahmini temiz örnek üzerinden noisy state'e türevlenerek prior'a eklenir. | Temiz-z enerjisinin global noisy z_t'ye zincir kuralıyla taşınmasını destekler. DPS, bilinmeyen veri terimini bizim yerimize bulmaz; SCD veya arbitrary latent L2 otomatik likelihood olmaz. |
| [DMPlug, NeurIPS2024; v2 §3.1/3.3](https://arxiv.org/html/2405.16749v2) | Sabit reverse diffusion zinciri bir fonksiyon kabul edilip girişi optimize edilir; gürültülü veriye aşırı uyum riski ayrıca ele alınır. | İkinci aşamada kısa global seed optimizasyonu düşünülebilir. Yararlı enerji hâlâ gerekir; çok adımlı backward maliyeti ölçülmeden verimli denemez. Önceki raporda zaten vardı. |
| [CloudFixer, NeurIPS2024; v1 §4.2/Eq.1](https://arxiv.org/html/2407.16193v1) | Diffusion'ın denoised çıktısı supervision olur; Chamfer ile geometrik dönüşüm güncellenirken denoiser'dan backward yapılmaz. | Nokta-kümesi mesafesi ve gürültü seviyesi önemli. Buradaki stop-gradient tekniği koşullama-gradientine doğrudan taşınamaz: fixed h girdisinde ε tahminini detach etmek z gradientini sıfırlar. CloudFixer-O model adaptasyonu ayrı protokoldür. |
| [ANaLOG, 2026-09 preprint v1; §3/Eq.4–7, App.F.4/G](https://arxiv.org/html/2609.31933v1) | Latent ölçüm likelihood'u için ortalama/anisotropic covariance başlıkları eğitilir; inference'ta precision-ağırlıklı gradient kullanılır. | Belirsizlik ve koordinat ölçeği değerli uyarı. **Ek görev eğitimi gerektirir**; şu anki frozen/training-free sınırında doğrudan aday değil. Basit ensemble varyansı, makalenin kalibre covariance'sı yerine konamaz. |
| [Cross & Ragni, ML-DE/ECAI2024; §4.2, §5/Limitations, PDFpp.7–9](https://proceedings.mlr.press/v255/cross24a.html) | Grad-TTS/SDXL koşullu likelihood deneyleri, semantik koşula duyarlılığın varsayıldığı gibi olmayabildiğini gösterir. | “Koşullu enerji düşükse temiz semantik kesin doğru” çıkarımını yapmayız. Makale yalnız iki modele dair; bizim LION surrogate'ının başarısızlığına kanıt değildir. |

## 3. Önerilen gradient: z'yi local denoiser'ın koşulu olarak çöz

[Code] Denetlenen LION factorization'ı p(z)p(h|z)p(X|h,z). Global z128,
local h2048×4; `global2style` bu config'te identity. Local prior
`condition_input`'ı `style` olarak PVCNN'e geçiriyor. Dolayısıyla donmuş
ağırlıklarla da **girdi z'ye autograd mümkündür**. Kaynak/pin/yerel hash'ler
[code_audit.json](evidence/shape_latent_gradient_research_20261031/code_audit.json)
içinde; gerçek CUDA gradienti bu araştırmada ölçülmedi.

[Inference/Proposal] Baseline local denoising'den elde edilmiş h* sabitlenip
detach edilsin. Corrupted local h_d ayrı kontrol hedefi olsun. Eğitimdeki
noising convention ile h_τ=a_τh*+b_τε, a_τ=√ᾱ_τ,
b_τ=√(1−ᾱ_τ). Tüm4kanalı koruyarak, d_h=8192 için:

\[
 E(z;h^*)=\mathbb E_{\tau,\epsilon}
 \frac{w_\tau}{d_h}
 \|\epsilon^h_\theta(a_\tau h^*+b_\tau\epsilon,\tau;z)-\epsilon\|_2^2.
\]

\[
 \boxed{\nabla_z E=
 \frac{2}{d_h}\mathbb E_{\tau,\epsilon}
 \left[w_\tau\,J_z\epsilon^h_\theta{}^\top
 (\epsilon^h_\theta-\epsilon)\right],\qquad d_z=-\nabla_z E.}
\]

[Inference] Bu residual'ı condition Jacobian'ından geçiren **gerçek gradient**.
SDS'nin girdi-latent için kullanılan Jacobian'sız pseudo-gradientiyle aynı
değil. h*, ε ve model ağırlıkları sabit; ε_h çıktısı veya z conditioning
yolu detach edilmez. Tüm çağrıyı `no_grad` içine almak da doğru yolu keser.
z'nin128boyut olması,8192boyutlu PVCNN forward/backward'ı ucuz yapmaz.

[Paper/Inference] CWGF Eq.20–21, terminal mismatch ihmal edilebildiğinde ve
tam noising-yolu/score-matching varsayımları altında condition-KL gradientine
bağlantı kurar. Bizim az sayıdaτ ve keyfî w ile kullanacağımız enerji ise
**surrogate** kalır. Dirac h* üzerinde sürekli-zaman KL'nin endpoint'leri
ayrıca sorunludur; integral/terminal terimler denetlenmeden exact
−log p(h*|z) veya kalibre posterior iddiası yapılmaz. Yerel config
`mixed_prediction=false`, `use_p2_weight=0`; mevcut noising helper'ı unit
loss weight döndürür. Bu, unit-weight MSE'nin exact likelihood olduğu
anlamına gelmez. Yerel model τ girdisi1..1000; schedule index'i0..999'dur.
Global DDIM20/10/0 ile conditional enerji örnekleme zamanlarını karıştırma.

[Open/Falsifier] h* eski z altında üretildiği için E(z) eski yanlışı
pekiştirebilir; corrupted h_d kullanmak corruption'ı ödüllendirebilir.
Local model koşul değişimine yeterince duyarlı olmayabilir. Bu nedenlerle
optimizable olması, **temiz z yönü verdiği anlamına gelmez**.

## 4. Bu gradient global shape diffusionunu nasıl guide eder?

[Inference/Proposal] Önce z üzerinde yön teşhisi yapıp bu sorunu global
prior'dan ayır. Başarılı olursa global noisy state u_t (kullanıcının z_t'si)
için clean prediction ve veri-gradienti:

\[
 \widehat z_0(u_t)=
 \frac{u_t-\sqrt{1-\bar\alpha_t}\,\epsilon^z_\phi(u_t,t)}
 {\sqrt{\bar\alpha_t}},\qquad
 g_t=-J_{u_t}\widehat z_0{}^\top
 \nabla_{\widehat z_0}E(\widehat z_0;h^*).
\]

\[
 u_{t^-}=\operatorname{DDIM}(u_t,\epsilon^z_\phi,t)
             +\rho_t g_t.
\]

[Inference] Bu, DPS-inspired bir ayrık guidance şablonudur; yeni exact
posterior sampler veya seçilmiş optimizer/schedule değildir. Doğrudan
−∇_zE'yi noisy state'e ölçek ve Jacobian olmadan eklemek farklı algoritmadır.
Global clean-estimate Jacobian'ında identity yaklaşımı kullanılırsa bu açık
bir approximation/ablation olarak adlandırılmalı. Her global adımda local
conditional backward ve tam yol için global-prior backward maliyeti vardır.
ρ_t/timestep sayısı/enerji örnek sayısı henüz seçilmedi; bir grid yetkisi yok.

[Code] Önceki proximal `propose` bütünü `@torch.no_grad()` altında,
`guided=(clean+target)/2` ve DDIM previous-state düzeltmesi kullanır. O deney
bu gradienti **ölçmedi**. Yeni enerji, aynı proxy'ye daha kuvvetli quadratic
çekim değil; p(h|z) bilgisinden başka bir yön arar. Bir ilerideki uygulama,
aynı seçilen z'yi local conditioning, clean probes ve final decoder'a
taşımalı; original h0/noise/Chamfer anchor'ını sessizce değiştirmemeli.

## 5. Alternatiflerin önceliği

[Inference/Proposal] **İkinci aday: P2L-inspired geometry gradient.**
Sabit bir local noisy h_τ ve değişken z için temiz local tahmin
ĥ0(z)=(h_τ−b_τε_h(h_τ,τ;z))/a_τ, sonra
E_geom(z)=SCD(D(ĥ0(z),z),reference). Local condition ve decoder üzerinden
z'ye geri türev alınabilir; gerekirse kısa local zincir de farklı bir
maliyetle unroll edilir. Fakat SCD/style gradienti mevcut TTA'da zaten var;
yalnız bunu yeniden adlandırmak yeni yöntem değildir. Hangi referansın
nesneyi koruduğu, global temiz tahmine taşınması ve corrupted anchor'a aşırı
uyum ayrı sorulardır. Bilinen corruption A olmadığı için P2L'nin gerçek
ölçüm likelihood'u varsayımı burada sağlanmaz. Decoder doğrudan z etkisi
`.01` residual ile ölçeklenir; bu Jacobian normu için üst sınır değildir.

[Inference/Decision] **Üçüncü aday: DMPlug-inspired seed optimizasyonu.**
Yararlı enerji gösterildikten sonra z=G_DDIM(seed) kısıtıyla düşünülür.
Şimdilik uzun zincir backward veya tekrar mean-refresh/proxy ağırlık grid'i
öncelikli değil. ANaLOG'un öğrenilmiş operator/covariance başlıkları da bu
training-free çalışmanın dışında. Classifier entropy/pseudo-label gradienti,
ayrı destek olmadan temiz-z gradienti sayılmaz: güven artışı semantik
doğruluk kanıtı değildir.

## 6. Sonraki deneyin sorusu ve karar ölçütü — henüz yürütülmedi

[Proposal/Open] İlk bağımsız soru: **Donmuş conditional ε-loss'un negatif
z-gradienti, temiz z yönüyle anlamlı bilgi taşıyor mu?** Yeni uygulama/GPU
fazından önce somut örnek/τ/ε/adım/normalizasyon/bütçe sabitlenmeli.

1. Aynı h0/local noise/anchor, model eval/frozen state, posterior noise,
   Point-MAE N8/B35 PM readout ve kendi-z decoder sözleşmesini koru.
   h* baseline terminal local çıktısı detached; h_d kontrolü ayrı.
2. Ortak(t,ε) ile z_d/proxy/clean-z posthoc enerji farkları; ayrıca
   örnekler arası shuffled z/h kontrolü. Optimizasyon ve held-out enerji
   değerlendirmesi için bağımsız gürültü bankaları; ortak RNG'yi tüketmeyen
   özel generator. Tekτ'daki düşüş üzerinden likelihood sonucu çıkarma.
3. Negatif gradientin posthoc cos(d_z,z_clean−z_d), norm ve clean-z
   MSE değişimini raporla. Temiz z/etiketler **objective, inference selector,
   nesne başına step, radius, yön veya checkpoint seçimi girdisi olamaz**.
   Diagnostic sonucu görüp ilerleme, sadece exposed development kararıdır;
   unseen confirmation değildir.
4. Önceden belirlenmiş küçük adımın tuttuğu held-out enerji düşüyor mu;
   temiz hedef enerjisi bozuk hedefe göre sıklıkla daha iyi mi; direction
   shuffled kontrollerden ayrılıyor mu? Ayrılmıyorsa global diffusiona
   bağlamak için gerekçe zayıflar. Bu nitel falsifier'lar formal kabul eşiği
   değildir; sayısal eşikler uygulamadan önce yazılmalı.
5. Model/state/RNG hash'leri, finite gradient ve autograd'ın koşullama yolu;
   actual CUDA path'te directional finite difference. PVCNN'in koordinat
   türev sınırlamalarıyla condition türevini aynı şey sayma; mevcut CPU
   sentetik testini gerçek native gradient doğrulaması diye sunma.
6. Yön yararlıysa matched fresh baseline/mean/conditional-energy/global-prior
   karşılaştırması ve **accuracy**. Enerji/MSE/cosine secondary; correct/harmed
   olaylar, corruption bazlı sonuçlar, original-object cluster belirsizliği,
   seed/native tekrar varyansı ayrıca raporlanmalı. Geçmiş smoke/pilot aynı
   state/RNG ile prediction farkı gösterdi; eski baseline'ı yeni runtime'a
   aynen taşımak doğrulanmış repeat değildir.
7. Bütün ilk local trajectory, hedef çıkarımı, τ/ε tekrarları, backward,
   global zincir ve ikinci local pass maliyetini ölç. Sadece replay worker
   süresinden deployment hızı sonucu çıkarma. Bu madde GPU launch değildir.

## 7. Gerçek çalışma durumu, kanıt ve yayın sınırı

[Code] Kaynak checkout `integration/chamfer-guidance-variants-20261030`,
HEAD `a82200adbe117ce56a134dc3c22601cb73dc2654`, önceki dirty değişiklikler
korundu. Karşılaştırılan shared integration `e8523e023f53650a057ddb1b2fa097bdf17078dd`.
Bu tur yalnız rapor/kompakt primary receipts/kaynak denetimi ve knowledge
ekleri içerir. Production/test/config/checkpoint değişmedi; GPU sonucu yok.
Sonraki uygulamanın parametrelerini veya bilimsel run bütçesini kilitlemez.

[Run/Open] Önceki kendi A100 `tta-shape-proximal-a100-20261031-r1`, son
yayın/arsiv doğrulamasında19:11–19:13UTC canlıydı. Bu araştırmada19:42UTC
envanteri boştu;19:47:53UTC `colab sessions` yeniden **No active sessions
found on server** döndürdü. Kapatma veya allocation komutu verilmedi.
Kesin kayıp zamanı/nedeni bilinmiyor. Önceki pilotların doğrulanmış yerel
ve Drive ham/derived/source arşivleri bu durumdan etkilenmez. Önceki dated
“runtime korunuyor/canlı” kayıtları bu yeni gözlemle superseded, bilimsel
sonuçlar geçerli kalır. [Envanter kaydı](evidence/shape_latent_gradient_research_20261031/runtime_inventory.json).

[Evidence] Raw primary HTML/PDF ve decoded pinned code:
`result/preparation/shape_latent_gradient_research_20261031/primary_sources/`.
On makale tam metni, bir konferans metadata sayfası ve bir pinned code
dosyası fiziksel SHA256 ile sabitlendi. Diffusion-TTA resmi kodu
`eedeb6e01f358e5645ecbcf42fdb4b3e7b1a9917`:
`context → _unet_pred_noise → mse_loss`; output detach yok. Bu yalnız code
inspection; repo çalıştırılmadı. Formal mathematical proof, external
replication veya yeni LION native backward ölçümü yapılmadı.

[Decision] Bir sonraki yararlı iş, conditional enerji yönünün küçük,
önceden kayıtlı teşhisi; ardından desteklenirse global diffusion guidance.
Kör biçimde daha çok global prior adımı veya oracle hedefiyle ayarlanmış
katsayı taraması için bu rapor bir gerekçe sağlamıyor.
