# Diffusion guidance: güncel literatür, kod karşılığı ve iki minimum deney

[Code/Run] Güncelleme31.10.2026: aşağıdaki inceleme **deney öncesi tarihsel
öneridir**. Sonraki kullanıcı seçimiyle fixed-X/decoded semantik loss,
Chamfer-only probability mean ve ayrı sparse/KL/H/weighted deneyleri tamamlandı.
Aşağıdaki “henüz yok/uygulanmadı” ifadeleri yazıldıkları aşamaya aittir.
Güncel [makale–kod–sonuç rehberi](chamfer_guidance_methods_results_20261031.md)
uygulananı, uygulanmayanı ve yeni agent görevini ayırır.

Tarama tarihi: **30 Ekim 2026**. Etiketler: **[Paper]** birincil makale,
**[Code]** incelenen kaynak, **[Run]** mevcut kabul edilmiş deney kaydı,
**[Inference]** aktarım/öneri, **[Open]** henüz bilinmeyen.

## Karar için kısa değerlendirme

[Inference] İlk iki adayım **aynı diffusion çizelgesinde seyrek guidance
backward** ve **iki kontrol noktasında frozen-head semantik tutarlılık
guidance'ı**. İlki süre, ikincisi doğruluk hipotezidir. İkisini ilk deneyde
birleştirmemek gerekir. Yeni bir spectral katsayı taramasını önceliklendirecek
kanıt bulmadım. Mevcut GSD-inspired yönteminin katkısız veya hatalı olduğu da
evrensel olarak gösterilmiş değildir.

[User report/Code] Başlangıç promptu uygulandı. Araştırma dalı
`research/diffusion-guidance-literature-20261030`, Agent B'nin temiz
`gsd-guidance-components@17dd87c355fd779460297fb23f3dabbb40902d2c` kaydından
türetildi. Bilimsel matematik denetimi `7143ede`; son deneyin çalıştırılmış
kaynağı `8169271` ve mühürlü ZIP'tir. Ana repo `f57b170` üzerindeki kullanıcı
değişiklikleri ve Agent B dalı değiştirilmedi. Bu dalda yalnız araştırma
belgeleri/kanıtları yazıldı; model forward/backward, Colab komutu, GPU deneyi
ve üretim kodu değişikliği yapılmadı.

## 1. Mevcut kanıtın izin verdiği başlangıç

[Run] Son yeni 140 nesne/dataset, all15, dört head, seed 0/1/2 doğrulamasında
original3DD terminal **53.763889%**, aynı GSD host'undaki C probability mean
**53.013889%**, seçilmiş C+S probability mean **52.962302%**. C−CS farkı
**+0.051587 pp**, nesne-kümeli koşullu %95 aralık **[-0.125000,+0.234127]**.
Original−CS **+0.801587 pp** bütün politika farkıdır; tek başına loss katkısı
değildir. 23 başarılı iş ve bir sayısal hata korunur. Kaynak ve arşiv SHA'ları:
[kabul edilmiş sonuç kaydı](guidance_component_sensitivity_results_20261028.md).
Bu literatür oturumunda ham deney arşivleri yeniden hesaplanmadı.

[Run] Önceki ayrı 70-ID doğrulamasındaki C **56.8929%** / CS **56.7222%**
başka nesne kohortudur. **%24.94** azalma taze-encode iki-bozulma smoke'unda
araçlandırılmış ortak reconstruction süresidir; tüm job+validation farkı
%9.86 idi. Cache-replay ve dört head paylaşımı deployment gecikmesi değildir.
[Önceki doğrulama](guidance_component_confirmation_results_20261025.md).

[Code] `tta.py:tta_reconstruct` ve `tta_gsd_v2.py:tta_gsd_reconstruct` kaybı
decoded fiziksel noktalar üzerinde değil, predicted-clean local latent'in
XYZ kanalları ile bozuk girdinin encode edilmiş XYZ'si arasında hesaplar.
GSD yolu B×8192×1×1 → B×2048×4 açar; dördüncü kanal da prior'a girer.
Beta2/raw-lambda/M800 fidelity aktiftir. Bu,
[denetlendiği gibi](guidance_mathematics_execution_audit_20261030.md),
normalize tüm-mod smooth ve tam GSDTTA'dan farklıdır. Guidance kopuk değildir;
bilinen self-only isolate kusurunun doğruluk kaybının nedeni olduğu gösterilmedi.

[Code] Original/GSD host farkları: B32/B70, normal-background 5/35 ile 10/30,
original/updated decoder style, `set_alpha_to_one=True/False`, terminal/readout.
GSD update'inde **eta local**, **gamma style**'dır. `total=100` sabitken
`steps_back_local` değiştirmek hem reverse güncelleme sayısını hem ilk
`alpha_bar`/başlangıç gürültüsünü değiştirir. Aynı başlangıçtan daha seyrek
guidance, daha kısa-noise bir koşulla aynı deney değildir.

## 2. Atıf taraması ve sınırları

[Paper] Hedef tek yayın olarak tekilleştirildi:
[arXiv:2411.14495v1](https://arxiv.org/abs/2411.14495), ilk preprint
**21 Kasım 2024**; [WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Dastmalchi_Test-Time_Adaptation_of_3D_Point_Clouds_via_Denoising_Diffusion_Models_WACV_2025_paper.html),
ss.1566–1576; DOI **10.1109/WACV61041.2025.00160**. Bunlar iki bağımsız çalışma
sayılmadı. OpenAlex `W4409262973` venue/arXiv kimliğini birleştiriyor.

[Paper/Open] 30 Ekim'de OpenAlex'in atıf filtresi **4** kayıt döndürdü:
MAMVI, GDSD, SA-P2SB ve robotics bin-picking. Semantic Scholar hedef isteği
**429**, Scholar cited-by isteği **403** döndürdü; tekrarlı baskı veya erişim
engeli aşımı yapılmadı. Web okuyucusu da bu API/cited-by adreslerini açamadı.
Bu dört kayıt aşağıdaki altı tam metinle veya önceki 15 yayınla toplanamaz.

Önceki [25 Ekim atıf denetimi](/Users/batuhan/Desktop/projects/3DD-TTA/knowledge/3dd_tta_citation_reproduction_audit_20261025.md)
15 doğrudan atıf veren yayını kaydetmişti. O envanter başlangıç kaynağı olarak
okundu; tüm 15 tam metnin bu oturumda yeniden incelendiği iddia edilmiyor.
Bu oturumdaki altı ilgili PDF'nin kaynakçasında doğrudan atıf yeniden doğrulandı.
Başlık/`3DD-TTA`/arXiv numarası ve 24 Ekim sonrası hedefli aramalar da yapıldı.
Daha yeni, doğrudan uygulanabilir başka bir makale doğrulanamadı; bu sonuç
eksiksiz tarama veya başka yayın bulunmadığı iddiası değildir.

### Doğrudan 3DD-TTA atfı doğrulanan ilgili çalışmalar

Tablodaki maliyetler yöntem yapısını anlatır; bizim donanımımızda ölçülmüş
süreler değildir. MN/Scan/Shape sırasıyla ModelNet40-C/ScanObjectNN-C/ShapeNet-C.

| Çalışma ve birincil bağlantı | İlk preprint / yayın-güncelleme durumu | Doğrudan atıf kanıtı | Mekanizma / kapsam ve maliyet | Kısıt uyumu ve repoda durumu |
|---|---|---|---|---|
| [SATOR](https://arxiv.org/abs/2610.21367) | 15.10.2026 v1; arXiv ve yazar kodu WACV2027 kabulünü bildiriyor; henüz 2027 proceedings'i olarak sunulmadı | PDF s.9 ref.[5]; §2.1 | Attention+geometri token enerjisi, Otsu partition, `H(pruned)/keep_ratio < H(full)` routing; MN/Scan/Shape; iki dalın forward maliyeti ve küçük token grafı | Ağırlıklar sabit; token akışı değişir. Yazar varsayılanı BN_reset da kullanır. Bizde uygulanmadı; PG/entropy selection öncülleri mevcut |
| [TTAD](https://arxiv.org/abs/2610.27410) | 20.10.2026 v1; arXiv journal reference CAD203:104182 (2027); ilk preprint tarihiyle karıştırılmadı | PDF s.15 ref.[33]; HTML ref.[7], §2.2.1 | Residual GMM, noisy–noisier pairing, reconstruction/identity/consistency ve EMA denoiser güncellemesi; sentetik/gerçek denoising, sınıflandırma benchmark'ı değil | Tam yöntemi frozen pretrained-model koşulunu değiştirir; artık/noise-model fikri ayrı bir aktarım olabilir. Bizde denenmedi |
| [MAMVI](https://arxiv.org/abs/2606.12939) | 11.06.2026 v1; ICPR2026 kabul kaydı; OpenAlex venue tarihi 03.08.2026, DOI 10.1007/978-3-032-31654-7_24 | PDF s.14 ref.5; §2.1 | 12 maskeli görünüm, entropi+consensus KL, tek toplu backward; confidence ile LR; Point-MAE, MN/Scan/Shape | BN/LN affine parametrelerini günceller. Salt girdiye semantik loss aktarımı yeni hipotezdir. Bizde readout mean var; bu backward loss yok |
| [LoTT-PC](https://openaccess.thecvf.com/content/CVPR2026/papers/Ye_Low-Rank_Test-Time_Training_for_Pre-Trained_Point_Cloud_Models_CVPR_2026_paper.pdf) | CVPR Haziran2026, ss.31472–31481; ayrı ilk preprint tarihi bu taramada doğrulanmadı | PDF s.9 ref.[5] | Maskeli feature alignment ve LoRA; MN/Scan/Shape; ek adapter optimizasyonu | Sınıflandırıcı davranışı/etkin parametreleri adapte edilir. Input-only protokolümüze doğrudan uymaz; denenmedi |
| [Uni-Adapter](https://arxiv.org/abs/2511.15311) | 19.11.2025 v1 → 20.11.2025 v2; AAAI2026 | PDF s.8 Dastmalchi et al.2025 kaynakçası | Online prototype cache, graph label smoothing, source/cache **logit** fusion; 3D VLFMs, üç dataset; feature/cache hesapları | Ağırlık eğitimi yok; kalıcı stream belleği ve yeni karar kuralı var. Dört klasik head'e doğrudan eşdeğer değil. Entropy-weighted trajectory readout daha önce önerildi, yeni buluş değil |
| [GSDTTA](https://arxiv.org/abs/2507.18225) | 24.07.2025 v1; ICCV2025 | PDF s.10 ref.[33] | Fiziksel low-frequency shift, eigenmap pseudo-label, alternatif classifier update; MN/Scan; eigendecomposition + input/model optimizasyonu | Tam yöntemi frozen-head koşulunu değiştirir. Bizde GSD-inspired latent fidelity ve birçok beta/M/loss deneyi zaten var |

[Paper/Open] Bir ek doğrudan atıf veren kaynak,
[GDSD / Geometry-Guided Diffusion SAR Point Cloud Denoising](https://www.mdpi.com/2072-4292/18/15/2458),
26.07.2026, §2.3/ref.[37] ile 3DD'yi anıyor. Bu kez indekslenmiş birincil
yayıncı metni yöntem açıklamasını sağladı; tam HTTP açılışı 429 verdi ve PDF
kaynakçası yeniden görsel denetlenmedi. LiDAR priors/prototype alignment ile
DiT-3D eğitiyor; SAR building denoising içindir. Yeni eğitim ve ek veri istediği
için mevcut LION'a hazır guidance loss'u sayılmadı. SA-P2SB ve bin-picking
OpenAlex kayıtları önceki denetimde vardır; burada tam yöntemleri yeniden
incelenmedi ve öneri kanıtı yapılmadı.

### Doğrudan atıf sınıfına eklenmeyen diğer ilgili çalışmalar

| Kaynak | Tarih/durum | Somut öncül ve maliyet | Bizim için durum |
|---|---|---|---|
| [CloudFixer](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04603.pdf) | İlk 23.07.2024 v1; ECCV2024, 3DD preprint'inden önce | §4.2/Eq.1: detached denoiser target ile fiziksel R/Δ optimizasyonu; inverse-neighborhood-distance displacement regularizer; diffusion backward yok | Temel input adaptation frozen-head uyumlu; CloudFixer-O classifier günceller. Bizde uygulanmadı. LION checkpoint'i/diffusion türü farklı |
| [GDA](https://openaccess.thecvf.com/content/CVPR2024/papers/Tsai_GDA_Generalized_Diffusion_for_Robust_Test-time_Adaptation_CVPR_2024_paper.pdf) | İlk 29.03.2024 → 02.04.2024 v2; CVPR2024 | §3, Eqs.6–8/Alg.1: augmentation-marginal entropy + style/content guidance; frozen classifier üzerinden input gradient, çoklu forward/backward | Semantik input guidance için doğrudan 2D öncül; CLIP/style parçaları aktarılmıyor. 3D faydası çıkarım; bizde entropy backward yok |
| [Adaptive Guidance](https://arxiv.org/html/2312.12487v1#S5) | İlk 19.12.2023 v1; AAAI2025 venue yayını 11.04.2025 | §5/Eq.7: conditional/unconditional score cosine yakınlaşınca CFG'nin ek forward'unu bırakır | Bizim SCD bir CFG branch'i değil; yalnız gereksiz guidance hesabını bırakma hipotezi aktarılır. Yayımlanan %25 bizim hız beklentimiz değildir |
| [SteeringTTA](https://openreview.net/pdf?id=iV0WBCDGcx) | ICML **PUT Workshop** 2025; yazar sayfası 18.07.2025; arXiv v1 16.10.2025 | §3.2–3.3: top-K confusion-set reward ve entropi zamanlaması, SMC/Feynman–Kac parçacıkları; ImageNet-C, ek trajectory/classifier forward | Parametreler sabit kalabilir; çoklu parçacık maliyeti nedeniyle ilk hız deneyi için seçilmedi. ICML ana konferans makalesi diye sunulmamalı |
| [Purge-Gate ve timestep-ensemble öncülleri](gsd_trajectory_attribution_20261025.md) | ICCV2025; Rosnati2023/Wang2025/Li2025 | Entropy selection, timestep prediction aggregation | Önceki atıf denetimi korunuyor. Probability mean, minimum entropy ve H+JS sampling planı zaten mevcut; yeni yöntem diye önerilmiyor |

## 3. Yazar kodunun değiştirdiği yorum

[Code] GitHub connector ile sabitlenmiş kaynaklar:

- **SATOR `e65983b60a83936e1e5241d339475436f1a1738d`**:
  [runner, satır173–189](https://github.com/iskong17/SATOR/blob/e65983b60a83936e1e5241d339475436f1a1738d/tools/tta_sator.py#L173),
  full ve pruned logits'i hesaplar. [BN helper](https://github.com/iskong17/SATOR/blob/e65983b60a83936e1e5241d339475436f1a1738d/utils/misc.py#L157)
  `track_running_stats=False`, `running_mean/var=None` yapar. Yayımlanan tam
  protokolü sabit BN/buffer'lı dört-head koşulumuzla özdeş sayamayız.
- **MAMVI `0b553973b7123da6198b79bba4ff704b8eec914c`**:
  [tools/tta.py, satır475–637](https://github.com/iskong17/MAMVI/blob/0b553973b7123da6198b79bba4ff704b8eec914c/tools/tta.py#L475)
  bütün loss'ları toplayıp backward yapar. Confidence modu düşük güvende
  daha büyük LR verir; [parser](https://github.com/iskong17/MAMVI/blob/0b553973b7123da6198b79bba4ff704b8eec914c/utils/parser.py#L277)
  eşikleri .80/.85 ve çarpanları 2/.50/.05'tir. Ayrıca başka modlar vardır;
  bunlar makalenin confidence kuralıyla karıştırılmamalı. Katsayılar batch
  confidence ortalamasından çıkarılabilir; otomatik olarak per-object input
  update yasası değildir. Bu eşik/rate'leri LION'a kopyalamayı önermiyorum.
- **CloudFixer `f043e24414fcf4165fdc892abff21215c9989141`**:
  [adapt.py, satır72–180](https://github.com/shimazing/CloudFixer/blob/f043e24414fcf4165fdc892abff21215c9989141/adapt.py#L72)
  hedef denoiser forward'unu `no_grad` içinde üretir; gradient R/Δ'ya gider.
  Bu, bizim `pred_original_sample.detach()` yapıp aynı loss'u çağırmamızla
  eşdeğer değildir: öyle bir değişiklik guidance gradyanını kaldırır.

MAMVI preprint'indeki eski `Inseok-kong/MAMVI` adresi 404 verdi; aramayla
yazarın `iskong17/MAMVI` deposu bulundu. Başarısız eski adresi “kod yok”
sonucuna çevirmedim. Kod kesitleri ve pin'ler research artifact'ında saklandı.

## 4. Dört aday: makale öncülü ile bizim önerimiz ayrı

### A — Aynı schedule üzerinde seyrek guidance backward: ilk hız adayı

[Paper] Adaptive Guidance §5/Eq.7 pahalı ek guidance hesaplarının daha sonra
gereksizleşebileceğini gösterir; bu CFG bulgusudur. CloudFixer §4.2, pahalı
denoiser Jacobian'ından kaçınan 3D input-adaptation öncülüdür. İkisi de
bizim seyrek SCD gradyanımızın doğruluğu koruyacağını kanıtlamaz.

[Inference] İlk test: T ve tüm timesteps/başlangıç alpha'sı aynı kalsın.
`G={1,4,7,...} ∪ {T}` loop sayımlarında normal C veya CS guidance uygula;
diğerlerinde `no_grad` prior+DDIM çalıştır, local/style guidance update yapma.
T10'da **4**, T30'da **11** prior backward kalır. Gradient katsayıları .01
ve p=.95 değişmez; sonradan ×3 ölçekleme eklenmez. Bu tek sabit schedule
adaydır. Adaptif gate, bu temel test yararlı olursa, yalnız geçmişte ölçülen
update/DDIM oranı veya güvenilirlik üzerinden ayrı aşamadır.

Maliyet: T prior forward **korunur**; backward T→|G|; C'de graph yok;
CS'de statik eigensystem bir kez hâlâ vardır. Loss/debug Chamfer çağrıları
yalnız G'de çalışmalı; mevcut GSD loop'u tanı ve gradient için iki kez
Chamfer çağırır. Aynı tanı politikasını tüm kollarda tutarak gerçek çağrı
sayısını raporlamak gerekir. Terminal politika için ek decode/classifier yok.
Probe-mean seçilirse mevcut üç probe maliyeti ayrıca korunur.

Etkilenecek yerler: `tta_gsd_v2.py` prior context/loss/backward/update blokları,
`tta.py` yalnız ayrı original-host aktarım deneyi seçilirse;
`scripts/eval_multibackbone_guidance.py` sayaç/config/provenance.
`gsd_composition.py` zorunlu değişiklik değildir. Hiçbiri bu oturumda değişmedi.

Risk: yararlı son guidance kaybolabilir; style da daha az güncellenir.
Full local + aynı seyrek style kontrolü bunu ayırır. Yalnız cosine yönü
küçük update veya doğru sınıf anlamına gelmez. Öneri yazar yöntemlerinden
esinlenen bir deneydir, AG/CloudFixer uygulaması veya yeni matematik iddiası değil.

### B — Seyrek semantik input guidance: ilk loss adayı

[Paper] MAMVI §3.2/Eq.4:
`mean_v [lambda_ent H(p_v) + lambda_cons KL(p_v || stopgrad(p_bar))]`;
§3.3/Eqs.5–7 confidence-LR. Yazar yöntemi normalizasyon parametrelerini
günceller. GDA §3/Eqs.6–8 ise sabit classifier ile input'a
`H(mean_v p_v)` gradyanı verir. Bunlar bizim salt readout mean'imizden farklıdır.

[Inference] Aynı predicted-clean h ve **aynı pre-update style** ile decode
edilen X üzerinde iki deterministik dış FPS görünümü kullan:
`p_v=softmax(f(A_v(X)))`, `q=stopgrad((p_1+p_2)/2)`.
İlk iki ve yalnız iki aday loss:

1. `L_sem = (KL(p_1||q)+KL(p_2||q))/2` — consensus-only;
2. `L_sem = H((p_1+p_2)/2)/log(C)` — augmentation-marginal entropy.

Her nesnenin loss'u batch boyunca toplamlanır; `L_total=L_C(+L_S)+lambda_sem
sum_b L_sem,b`. İki loss ayrı adaydır; entropy+KL+kapsamlı LR grid'i yok.
T10 için yalnız loop **5 ve 8**, T30 için **15 ve 23** kullanılır.
Bir label-free development kalibrasyonunda semantic'in local **update normu**
C update normunun medyan %10'una eşlenir; kalibre edilen tek katsayı dondurulur.
Style norm oranı ayrıca kaydedilir; local eşleme style eşlemesi değildir.

Maliyet: m=2 guidance checkpoint'inde **m decoder forward/backward + 2m
classifier forward/backward** ve sampling/normalization gradyan yolu.
Prior forward sayısı artmaz. Zaten C için kurulmuş prior grafına aynı toplam
loss ile tek backward uygulanabilir; graph retention ve ek aktivasyon belleği
vardır. “Tek backward” sabit maliyet demek değildir. Readout probe'u aynı
decode ile sayılacaksa gerçek reuse ve RNG paritesi doğrulanmalıdır; bugünkü
observer'da böyle bir paylaşım yoktur.

Etkilenecek yerler: `tta_gsd_v2.py` predicted-clean loss yolu;
`scripts/eval_multibackbone_guidance.py` frozen differentiable classifier ve
preprocessing callback'i. `shared_trajectory.py:clean` bugün detach/no-grad
observer'dır; içinden entropy backward alınamaz. `analyze_*` readout
modüllerinin loss'a dönüşmesi gerekmez. FPS/KNN indeks seçimi türevlenebilir
sayılmaz; seçilmiş noktalara gradient taşıyan gather yolu ayrıca doğrulanmalıdır.
LION PVCNN koordinat türevleri zaten kısmi: [mevcut native audit](code_audit_20260915.md).

Birincil guidance head önceden **Point-MAE** seçilsin; diğer üç head aynı
üretilmiş geometriyle yalnız transfer değerlendirmesi yapsın. Her head'in
kendi semantik trajectory'si istenirse dört ayrı reconstruction gerekir;
önceki dört-head amortizasyonu kullanılamaz. Frozen weights, BN/buffer ve
token algoritması aynen kalmalıdır.

Risk/falsifier: bütün görünümler aynı yanlış sınıfta uzlaşabilir;
entropy yanlış sınıfı keskinleştirebilir, KL yanlış consensus'u koruyabilir.
SCD korunur ama semantik doğruluk garantisi değildir. Hard pseudo-label yok;
label-free semantik hedef yine kendine referans verir. Confidence yükselirken
corrected−harmed veya all15 doğruluk düşüyorsa mekanizma başarılı sayılmaz.

### C — Yerel güvenilirlik ağırlıklı SCD: ikinci loss adayı

[Paper] CloudFixer §4.2/PDF s.7, inverse mean-kNN-distance ağırlıklı
displacement regularization ile izole noktaların daha fazla hareketine izin
verir. TTAD §3.3/Eqs.3–4, pseudo-clean residual'dan sample-specific GMM
çıkarır; ardından denoiser'ı günceller. GMM/noisy–noisier eğitimini doğrudan
LION'a taşımayı önermiyorum.

[Inference] Daha küçük aktarım: encode edilmiş Q XYZ'sinde nonself kNN ile
statik `w_j ∝ 1/(mean_k distance² + epsilon)` üret, toplamını N'e normalize
et ve detach et. Mevcut p=.95 trim'in mesafe sırasını koruyarak Q→Y tarafını
`w_j`, Y→Q tarafını nearest-reference `w_nn(i)` ile ağırlıklandır. İki yönde
de ağırlık kuralı açık olmalı; yalnız bir yönü değiştirmek ayrı ablation'dır.
Q yoğunluğu temiz/güvenilir geometriyle özdeş sayılmamalı.

Bu, spektral eigenbasis yerine residual güvenilirliğini değiştirir;
full latent-slot correspondence varsaymadan Chamfer eşleşmelerini kullanır.
Bir kNN hazırlığı, T adet gather/elementwise ağırlık ekler; ekstra decoder,
classifier, prior forward/backward veya dense eigendecomposition gerektirmez.
Yeni toplam loss ölçeği label-free update-norm kontrolüyle eşlenmeli.
Kod sınırı: `tta_gsd_v2.py` Chamfer nearest-indices/trim yolu; gerekirse paylaşılan
`tta.py:selective_chamfer_loss` uzantısı ve bütün çağrıları. Henüz uygulanmadı.

Risk: LiDAR/occlusion'da az sayıdaki doğru noktalar da izole olabilir; eşitsiz
yoğunluk sınıf parçasını gösteriyor olabilir. Bozuk referansı tümüyle düzeltmez.
Bir kNN sinyali ile güvenilirlik iddiası kurulamaz. All15/macro14, özellikle
LiDAR/Occlusion harmed artıyorsa bırakılır. Önce A/B denenmesi daha ayırt edicidir.

### D — Muhafazakâr nedensel erken çıkış: readout/politika adayı

[Inference] Mevcut 3/5/8 probe'ları üzerinde geçmiş prefix'ten çalışan bir
durma kuralı: iki ardışık probability argmax aynı, normalize entropi düşük,
iki dağılımın JS farkı küçük olduğunda yalnız **şimdiye kadarki** probability
mean ile çık; aksi halde tam horizon'a devam et. T30'da mevcut 8/15/23
checkpoint'leri kullanılır. Threshold'lar development'ta en fazla bir
önceden tanımlı seçimle dondurulmalıdır; future terminal kullanılmaz.

[Paper/Inference] SATOR §3.5/Eqs.14–15, salt düşük entropy'ye karşı
keep-ratio ile bir ceza/fallback kullanır. Bu güvenilirlik uyarısı aktarılabilir;
**diffusion'da adım oranı token keep-ratio değildir**. `H/(step/T)` kuralını
SATOR eşdeğeri diye üretmemek gerekir. H+JS ve erken çıkış daha önce
[yerel planda](gsd_trajectory_selection_20261011.md) vardı; yeni keşif değil.

Mevcut logits yalnız offline politika taramasını destekler. Yeni süre sonucu
için gerçek loop break gerekir. Çıkış k'da T−k prior forward/backward ve
sonraki loss'lar atlanır; gözlenen probe'ların decode/classifier maliyeti
kalır. B1'de doğrudan fayda ölçülebilir; B70'te maske koyup bütün batch'i
hesaplamak tasarruf değildir, compaction maliyeti ölçülmelidir. Statik graph
maliyeti çıkıştan önce ödenmiştir. Stabil yanlış sınıf en belirgin risk.
Kod: observer yalnız gözlemlesin; yeni politika sinyalini sampler loop'u
uygulamalı. CPU replay kazancı deployment hızı olarak raporlanamaz.

## 5. En iyi iki aday için minimum ayırt edici deney önerisi

**Aşağıdakiler yalnız öneridir. Yeni veri seçilmedi, çalıştırılabilir launcher
yazılmadı, deney kaydı veya GPU tahsisi oluşturulmadı.**

### Ortak protokol ve yeniden kullanma

[Inference] İki dataset, severity5; MN40 sınıf / Scan main_split15 sınıf;
Point-MAE, DGCNN, CurveNet, PointNeXt'in aynı mevcut checkpoint'leri. Her
karşılaştırmada tam original ID/label eşlemesi, preprocessing/interpolation,
encode h0/z0, başlangıç noise ve post-prepare RNG hash eşlemesi gerekir.
Frozen weights/buffers/BN/token akışı, native SHA ve decoder style sözleşmesi
korunur. CPU sentetik doğruluk, CUDA/model doğruluğu değildir.

Eski 70-development + 70-confirmation + yeni 140-confirmation ID birleşimi
dataset başına 280 nesnedir. Öneri: kalan havuzdan 70 yeni development ve
140 yeni confirmation nesnesi; ID seçimi label-free seed **20261030** ile
yapılıp hash'lensin. Scan581 içinde bu ayrım mümkündür (301 aday); gerçek
manifestlerin birleşimi önce doğrulanmalıdır. Bu oturum manifest üretmedi.
Önceki tam benchmark görülmüş olduğundan bunlara dışarıdan görülmemiş test
denmez. Development seed0; seçim/freeze ardından confirmation seed0/1/2,
nesneler bütün corruptions/heads/seeds boyunca aynı kalır.

Önceki koşulları tekrar “eksik kontrol” diye adlandırma. Eski kohortlarında
C/CS/O sonuçları tamamlanmıştır. Yeni ID'lerde gerekli kontroller yalnız bu
yeni koşullarla matched blok oluşturmak içindir. İki çalışma aynı ID ve
aynı config/h0/noise/hash sözleşmesini kullanırsa baseline ve geometri/logits
paylaşılır. Başka kohort/host sayıları yeniden matched gibi birleştirilmez.

Her host'ta aynı-corruption aynı-nesne paired accuracy kullanılır; farklı
başlangıç timestep'leri olan original ve GSD host'ları aynı trajectory diye
sunulmaz. Ortak temel Gaussian draw kullanılabilir; gürültü katsayısı yine
farklıdır. Full original kontrolünde batch/style/final-alpha/5–35 aynen
saklanır ve yalnız bütün-politika referansı olur.

### Deney A: seyrek backward, style ve host etkisini ayır

| Kol | Host / amaç |
|---|---|
| O | Değişmeyen actual original3DD, terminal: bütün-politika bağlamı |
| C | GSD host, C-only, her adım full local+style gradient: birincil matched kontrol |
| CS | Aynı host, dondurulmuş rho.1 C+S, her adım: spectral kontrol |
| C-sparse | Aynı C host, yalnız G'de local+style gradient: aday A |
| C-style-sparse | Her adım local gradient, yalnız G'de style update: style değişimi kontrolü |

`C-sparse − C` toplam aday etkisi; `C-style-sparse − C` style güncelleme
sıklığı; `C-sparse − C-style-sparse` seyrek local guidance katkısıdır.
CS-sparse, C adayı kabul edilirse aynı tek dondurulmuş G ile koşullu transfer
koludur; yeni schedule seçimi değildir. Hiçbir p/rate/beta/M grid'i eklenmez.

Birincil sonuç **terminal**; ikincil aynı üç probe+terminal probability mean.
Bu ayrım hızın readout'tan mı gradient'ten mi geldiğini gösterir. Tam trace
ölçümünün probe yükünü terminal-only deployment'a yükleme veya gizleme.
Süreyi kısa ve önceden seçilmiş taze-input profil alt kümesinde ayrıca ölç:
baseline/candidate sırasını sırala-rastgeleleştir, aynı warmup ve CUDA
senkronizasyonu kullan; model loading/hash/log backup sürelerini ayrı kaydet.
Cache yalnız accuracy eşlemesi içindir, deployment timing için encode dahil
gerçek akış gerekir. B1 p50/p95/ortalama gecikme ve B70 throughput ayrı.

İş hesabı: T10'da 10→4, T30'da 30→11 **prior backward**; forward ve
başlangıç noise aynı. Bu çağrı sayısı farkı duvar-saati kazancı değildir.
Önerilen ön-kayıt eşiği: all15 accuracy delta için nesne-kümeli alt CI
**>−0.25 pp** ve tam gerçek akışta hız artışı; CI yetmezse sonuç
**inconclusive**, eşdeğerlik kanıtı değil. Hız kazancı yoksa veya kabul
edilemeyecek accuracy kaybı varsa aday bu amaç için başarısızdır. Bu yeni
pratik marj eski sonuçlara geriye dönük uygulanmaz; çalıştırmadan önce kullanıcı
değerlendirmesine sunulacak öneridir.

### Deney B: semantik loss'u readout ve ekstra hesap bütçesinden ayır

Matched O/C/CS kontrollerini ortak protokol tam uyuşuyorsa A'dan yeniden kullan.
C'ye yalnız üç yeni kol ekle: **C+KL**, **C+marginal-H** ve **C+sham**.
Sham aynı iki checkpoint'te decoder/iki-view classifier forward'unu yapar,
semantic gradyanı update'e eklemez; C'nin geometri/logits paritesi doğrulanır.
Sham compute kontrolüdür; gerçek semantic backward maliyeti ayrıca ölçülür.

Development'ta yalnız iki semantik formdan biri seçilir; .1 update-ratio
kalibrasyonu, iki checkpoint, iki view, Point-MAE guidance head ve bütün
diğer ayarlar dondurulur. Aynı forward bütçesinin salt ek-vote readout'u da
kaydedilsin. Her kolda terminal ve aynı four-candidate probability mean olsun.
`semantic terminal − C terminal` loss etkisi; `mean − terminal` aynı-arm
readout etkisi; `semantic − sham` update etkisi; `CS+frozen semantic − CS`
koşullu ek doğrulaması spectral etkileşimidir. CS+semantic'de yeni loss/weight
seçilmez. Original karşılaştırması bütün politika olarak etiketlenir.

İlk teknik kabul: decoder/preprocess/classifier üzerinden local ve style'a
finite/nonzero input gradient, frozen state ve RNG doğrulaması. View FPS
indeksleri testte sabitken seçilmiş koordinatlara gradient kontrol edilir;
native operatörlerin tam koordinat türevi olduğu iddia edilmez. Uyumsuz
preprocess callback'ini sessizce başka yolla değiştirmek yerine başarısızlık
kaydedilir. Model ağırlığı/BN/buffer güncellemesi bu deneyde yasaktır.

Önerilen doğruluk kabulü: frozen adayın all15 terminal kazancı için paired
nesne-kümeli CI alt sınırı **>0**, üç seed farkları ve macro14 ayrıca görünür;
örnek bir sınırlı-maliyet hedefi, tek-head taze tam akışın **≤1.25×** baseline
süresidir. Bu maliyet toleransı prospective öneridir, ölçülmüş oran değildir.
Accuracy artışı yalnız daha pahalı readout'tan gelirse loss katkısı kurulmaz.
H artışı/azalışı tek başına kabul ölçütü değildir. Hız hedefi aşılırsa daha
doğru ama daha pahalı sonuç olarak kaydedilir; verimlilik hedefi başarısızdır.

### Her iki deneyin ortak raporu ve durma koşulları

- All15 eşit corruption makrosu **birincil**, macro14 ve Background ayrı;
  her dataset/head/seed/corruption satırı. All15 yerine iyi macro14 seçilmez.
  Önceki isim-temelli 10/30 horizon korunursa bunun oracle corruption bilgisi
  kullandığı açıklanır. Bilinmeyen-bozulma deployment iddiası ayrıca sabit,
  corruption-agnostic horizon/policy deneyi gerektirir.
- Aynı ID'de **corrected / harmed**, per-class değişim, confidence-wrong ve
  temporal/view stable-wrong sayıları, sınıf yoğunlaşması, prediction entropy
  ve geometri yer değiştirmesi; düşük loss doğru sınıf kanıtı değildir.
- En az 2.000 dataset-stratified **original-object cluster bootstrap**:
  o nesnenin bütün corruption/head/seed kayıtları birlikte taşınır. 140×2
  nesne 280 kümedir; 50.400 prediction event bağımsız örnek değildir.
  Seed SD ayrı; bu üç seed'e koşullu CI, seed-population kesinliği değildir.
- Tam encode/prior-forward/prior-backward/Chamfer/graph-eigh/decode/
  classifier forward-backward sayıları, senkronize bileşen ve toplam süre,
  peak memory; ortak reconstruction bir kez sayılır. Toplamı asenkron host
  dispatch span'larından türetme. Kendi-head semantic trajectory maliyetini
  dört-head readout paylaşımıyla bölme.
- NaN/nonfinite, değişmiş state/RNG, eksik ID/hash veya tamamlanmamış kol
  durdurma nedenidir. Başarısız slotu değiştirme, clipping ekleme veya iki
  seed'i tam endpoint yerine sunma yok. Yeni yöntemlerin sonucu olumsuz,
  null veya belirsiz çıkabilir.
- Bu aşama tahmini GPU dakika sayısı üretmez: canlı profil yok. Seçilen
  ileriki bütün GPU fazı teknik pilot/confirmation/timing dahil tek bütçe
  olarak hesaplanmalı; >60 dakika ise ayrıca açık approval alınmalı.
  Bu belge o fazı başlatmaz veya yetkilendirmez.

## 6. Şimdilik önerilmeyenler ve özgünlük sınırı

[Inference] Tam GSDTTA, MAMVI, LoTT-PC, TTAD/EMA-denoiser veya kaynak/prototype
eğitimi protokolü genişletir; ayrı araştırma sınıfında tutulmalı. SATOR'ın
frozen-BN token-only sürümü Point-MAE'ye özgü bir karşılaştırıcı olabilir;
input-only dört-head loss geliştirmesi değildir. GDSD'nin LiDAR/DiT eğitim
sonucunu LION test-time loss kazancı saymak geçersizdir.

CloudFixer tarzı tüm prior'u detach etmek style yolunu kaldırır. Kimlik-Jacobian
yaklaşımı `hhat=(h_t−sqrt(1−a)*stopgrad(eps))/sqrt(a)` local türev bırakabilir,
ama style türevini sıfırlar; ayrı surrogate yöntem ve yeni style kontrolü
gerektirir. A'nın seyrek gerçek backward testi önce daha kolay yorumlanır.

SMC/SteeringTTA çoklu trajectory maliyeti, her-step semantic guidance ek
decode/classifier maliyeti, dinamik dense graph/eigh ve büyük beta/M/rate
grid'leri ilk minimum çalışma için seçilmedi. Tamamen yeni logit-mean veya
entropy-weighted mean icat edildiği iddia edilmez; bunlar zaten kayıtlı
açık CPU-replay sorularıdır. Bu oturum yeni accuracy analizi çalıştırmadı.

[Open] Bu adayların hiçbiri için doğruluk veya hız kazancı kurulmuş değil.
İddia, mevcut bozuk girdiye bağlı fidelity'nin yanına yeni semantik bilgi veya
hesap bütçesi politikası eklemenin **test edilebilir** olmasıdır. Eşleşmiş
kontroller, bağımsız nesne doğrulaması ve gerçek süre ölçümü bu önceliği
değiştirebilir. Genel yöntem bileşenlerinin önceki literatürü vardır;
uygulama birleşimi otomatik olarak tez özgünlüğü veya yayın katkısı değildir.

## 7. Yeniden izlenebilir kanıt

Küçük, Git'te saklanan kanıtlar:
`knowledge/evidence/diffusion_guidance_literature_20261030/`.
Raw PDF/HTML, çıkarılmış metinler, seçili render'lar, yazar kodu kesitleri:
`result/preparation/diffusion_guidance_literature_20261030/`.
İndirme URL/tarih/byte/SHA manifestleri ve source kimlikleri kayıtlıdır.
Mevcut deney artifact'ları değiştirilmedi; yeni bir inference sonuç klasörü
uydurulmadı. Yerel literatür arşivi ve son kontrol kaydı evidence klasöründe
tanımlanır; Drive yüklemesi yapılmadı.
