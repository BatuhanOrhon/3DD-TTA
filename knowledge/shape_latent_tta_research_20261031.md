# Shape latent ile test-time adaptation: LION yapısı, eski deneme ve literatür

Tarih: 2026-10-31. Durum: **araştırma raporu; öneriler uygulanmadı**.
Worktree `/Users/batuhan/.codex/worktrees/7bf0/3DD-TTA`, branch
`integration/chamfer-guidance-variants-20261030`, HEAD
`a82200adbe117ce56a134dc3c22601cb73dc2654`. Branch değiştirilmedi;
önceki Chamfer audit değişiklikleri korundu. Yayın hedefi
`origin/thesis-integration`; bu çalışma commit/push yapmaz.
Üretim kodu, model ağırlıkları ve deney varsayılanları değiştirilmedi;
Colab/GPU veya pretrained model inference başlatılmadı.

Kanıt etiketleri: **[Paper]** makale; **[Code]** sabitlenmiş kaynak;
**[Run]** mevcut artifact veya bu rapordaki CPU hesabı;
**[User report]** kullanıcının bildirdiği deney; **[Inference]** yorum/öneri;
**[Open]** doğrulanmamış nokta. Kanıt dosyaları
[bu dizinde](evidence/shape_latent_research_20261031/).

## 1. Sonuç ve araştırma kararı

**[Inference]** Başlangıç shape latent'inden kontrollü uzaklıkta kalan global
diffusion araştırılmaya değer. En güçlü bağlantı LION'un kendi hiyerarşik
diffuse-denoise yapısı ile DPS ailesinin **temiz latent tahmini üzerinden
guidance** ilkesidir. İlk adayım, düşük gürültü seviyesinde **tek global
denoiser sorgusundan alınan temiz öneriyi**, başlangıç latent'i çevresindeki
bir küreye projekte etmek; sonraki style gradient güncellemelerini de aynı
sınırda tutmaktır. Daha uzun, guidance içeren global zincir ikinci aşamadır.
Bu bir accuracy iddiası veya mevcut en iyi yöntem seçimi değildir.

**[Code/Inference]** Global diffusion'u yalnızca açmak temiz bir deney olmaz:
LION'un local encoder'ı da global latent'e koşulludur; local prior temiz
global latent ile eğitilmiştir. Global z değiştiğinde h ile uyumu, kullanılan
gürültü seviyesi ve final decoder koşulu birlikte tanımlanmalıdır.

**[Code/Open]** 3DD-TTA'nın final decode'da ilk z'yi kullanması doğru bir kaynak
tespitidir. Yazarların global güncellemenin zararlı olduğunu deneyle görüp
bu tercihi yaptığına ilişkin açıklama, incelenen makale ve resmi kodda
bulunmadı. Bu niyet çıkarımı kanıtlanmış sonuç olarak kullanılamaz.

## 2. LION gerçekte neyi temsil ediyor?

**[Paper]** LION iki bağımsız latent kullanan bir decoder değildir:

\[
q(z,h\mid X)=q(z\mid X)q(h\mid X,z),\qquad
p(X,z,h)=p(z)p(h\mid z)p(X\mid h,z).
\]

Global prior önce z üretir; local prior bu **temiz z** altında h üretir.
LION Appendix C.1, diffuse-denoise sırasında da önce global latent'in
temizlenmesini, sonra local latent'in temizlenmiş global altında denoise
edilmesini söyler. Küçük başlangıç gürültüsü sadakati artırır; bu, kesin bir
uzaklık garantisi değildir. [LION, Sec. 3 ve App. C.1, s. 21](https://arxiv.org/abs/2210.06978v1).

**[Code]** Bu checkout/checkpoint için sözleşme:

| Parça | Gerçek yapı | TTA açısından anlamı |
|---|---|---|
| Global encoder | `q(z|X)`; 128 boyut; posterior örneklemesi | İlk z deterministik özellik değildir. |
| Local encoder | `q(h|X,g(z))`; 2048×4, toplam8192 boyut | h, onu üretirken kullanılan z'ye bağlıdır. |
| `global2style` | Bu config'te `style_mlp=''`, dolayısıyla identity | Raw z ve style sayısal olarak eşleşir; başka config için varsayılmaz. |
| Global prior `priors[0]` | 1×1 convolution/residual MLP, SE ve dropout | Geometriyi doğrudan komşuluklarla işlemez; 128 boyut olması süreyi tek başına kanıtlamaz. |
| Local prior `priors[1]` | PVCNN; AdaGN ile global style koşulu | z, local denoising boyunca çok sayıda özelliği etkiler. |
| Decoder | h'nin XYZ kısmı + `.01` ağırlıklı, style koşullu residual | h de şekil/kimlik taşır; z yalnız başına bütün şekil değildir. |

Kaynaklar: [VAE](../models/vae_adain.py),
[local encoder/decoder](../models/latent_points_ada.py),
[local prior](../models/latent_points_ada_localprior.py),
[global prior](../models/score_sde/resnet.py), [AdaGN](../models/adagn.py),
[config](../lion_ckpts/unconditional_all55_cfg.yml).
VAE `encode` s.55–95, `global2style` s.120–127; local encoder s.190–220;
decoder s.238–272. `.01` residual katsayısı decoder etkisinin sınırlı
olabileceğine işaret eder; bir Lipschitz veya küçük gradient garantisi değildir.

**[Code]** Resmi LION
[`train_2prior.py:281–302`](https://github.com/nv-tlabs/LION/blob/7711b3d185752eeb632d095494876e4de15f3195/trainers/train_2prior.py#L281)
her prior'ın kendi latent'ine gürültü ekler; local prior'a verilen
`decomposed_eps[0]` gürültüsüz global örnektir. Bu ayrım makale yorumundan
bağımsız olarak eğitim kodunda da doğrulandı.

**[Inference]** z'yi değiştirip eski h'yi tutmak, tutarlı bir posterior çifti
elde edildiği anlamına gelmez. Ancak otomatik olarak geçersiz de değildir:
local reverse zincir bu uyumsuzluğu kısmen düzeltebilir. Bu nedenle sabit-h
ve yeni-z altında yeniden encode edilen h ayrı kontroller olmalıdır.
CPU'daki h=x−z, D(h,z)=h+z örneği bunun mümkün olduğunu gösterir;
LION üzerinde nicel zarar göstermez.

**[Paper]** LION'un gösterdiği noisy/voxel input denoising uygulaması,
bozuk input ve temiz hedef çiftleriyle **encoder fine-tuning** içerir;
decoder ve latent diffusion modelleri korunur. Dolayısıyla bu sonuçlar,
bizim donmuş encoder ile source-free TTA başarısının doğrudan kanıtı
değildir. [App. C.2, s. 21–23, E.5](https://arxiv.org/abs/2210.06978v1).

## 3. İlk shape latent ile final decode neyi kanıtlıyor?

**[Code]** Resmi 3DD-TTA
[`tta.py:45–89`](https://github.com/hamidreza-dastmalchi/3DD-TTA/blob/107305fd7baf40b359f31c07d235599198be7324/tta.py#L45)
z'yi örnekler, style'ı SCD gradient'iyle günceller, fakat son decoder'a ilk
`shape_latent`'i verir. Güncellenmiş style sonraki local-prior çağrılarını
etkiler; yalnız son style update'inin bu yolda sonraki tüketicisi yoktur.
Dolayısıyla “shape update tamamen kullanılmıyor” yorumu yanlıştır.
Makalenin Eq.12'si global latent guidance'ını anlatır; bu kod tercihi için
bildirilmiş bir negatif ablation sonucu bulamadım.
[3DD-TTA, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Dastmalchi_Test-Time_Adaptation_of_3D_Point_Clouds_via_Denoising_Diffusion_Models_WACV_2025_paper.html).

**[Run]** Daha önceki kontrollü decoder deneyi de güçlü bir zarar göstermiyor:
ModelNet40-C severity5, tüm15 corruption, üç seed, aynı local trajectory:

| Decoder style | Üç-seed macro accuracy |
|---|---:|
| İlk z | %63.748424 |
| Gradient ile güncellenen style | %63.716910 |
| Güncellenen − ilk | −0.031514 yüzde puanı |

45 corruption×seed satırının18'i iyileşmiş,25'i kötüleşmiş,2'si eşit.
Üç orijinal all15 ZIP'in SHA'ları ve CSV correct-count'ları yeniden doğrulandı:
[yeniden hesap](evidence/shape_latent_research_20261031/decoder_control_recount.json),
[script](evidence/shape_latent_research_20261031/recount_decoder_control.py).
Per-example predictions bulunmadığı için bu raporda yeni eşleştirilmiş CI
üretilmedi. Bu kontrol global prior'ı çalıştırmaz; “global diffusion zararlı”
sonucu çıkarılamaz. Güncel C host'uyla da karıştırılmaz: güncel yöntem
10/30 local adım ve dört probability probe mean kullanır; eski orijinal
5/35 terminal decoder kontrolü ayrı protokoldür.

## 4. Önceki global diffusion çalışmasını ne kadar kurtarabildik?

**[User report]** Global diffusion'u açmak daha uzun sürdü ve accuracy'yi
düşürdü. Kullanıcı eski knowledge dosyalarının kaybolmuş olabileceğini
belirtti. Bu gözlem korunur; yeni bir yeniden üretim olarak etiketlenmez.

**[Code/Open]** Ulaşılabilir tüm ref'lerde knowledge'a dokunan249 commit,
tarihsel knowledge dosya adları, silinen Markdown yolları ve eski notebook
incelendi. İlk knowledge commit'i `b31fd23` (12 Eylül); dual kod değişiklikleri
9 Ağustos'ta yedi commit. Son dual kaynak
`a458cd46bba1288a3f41302548bc777700cbc7d4`. Silinmiş Markdown yolu çıkmadı;
iki tarihsel notebook sürümünde dual çalışma hücresi/çıktısı bulunmadı.
İki checkout ve korunmuş deney worktree'sindeki result dosya adlarında da
eşleşen dual/global sonuç arşivi bulunmadı. Bu, erişilemeyen ref/Drive/eski
makinede artifact olmadığı anlamına gelmez.
[Arama kapsamı](evidence/shape_latent_research_20261031/search_ledger.md).

**[Code]** Kurtarılan eski implementasyonda yorumlanabilirliği bozan noktalar:

| Yol | Kaynakta görülen davranış | Sonuç |
|---|---|---|
| `dual_sync:99–118` | Local prior'a global DDIM'in `prev_sample` çıktısı, yani ara gürültülü z veriliyor | Temiz z ile eğitim koşulundan sapma. |
| `dual_sync:107` | Global step çıktısı detach ediliyor | Sonraki guidance, global temiz tahminin Jacobian'ından geçmiyor; DPS ile aynı işlem değil. |
| `dual_seq:77–91` | Global diffuse-denoise koşulsuz; açık başlangıç anchor'ı yok | Temiz koşul türü doğru, fakat kimlik/sadakat sınırı yok. |
| İki yol | h, ilk z altında encode edilmiş | Yeni global koşulla posterior çifti uyumu ölçülmemiş. |
| Eski launcher | `--lambdaa=.95` tanımlı, iki çağrı da `p=.5` geçiriyor | CLI seçimi etkisiz; doğrulanmış statik launcher kusuru. |
| Eski launcher | Classifier eval, LION VAE/priors için eval çağrısı yok | Global dropout `.3`, local/VAE dropout `.1` etkin kalabilir. |
| Global step varsayılanı | Local horizon'dan miras alınabiliyor | Background için global gürültü de aşırı büyüyebilir. |
| RNG / static style | Global noise local noise'dan önce çekiliyor; static flag seq/sync'te farklı anlam taşıyor | Aynı seed/flag eşleştirilmiş aynı deney demek değil. |

Bunlar **kurtarılan kaynak** bulgularıdır; kullanıcının deneyini bu commit'e
bağlayan receipt yoktur. Accuracy kaybının kesin nedeni olarak sunulamaz.
Güncel ortak runner zaten VAE ve iki prior'ı eval/frozen kurar; eski eksiklik
güncel C'de yeniden varmış gibi raporlanmamalıdır.

**[Run/Inference]** Mevcut1000-beta /100-DDIM-grid sözleşmesinin analitik
forward noise seviyeleri:

| Reverse adım sayısı | Başlangıç scheduler t | ᾱ | √(1−ᾱ) |
|---|---:|---:|---:|
| 1 | 0 | .999900 | .010000 |
| 3 | 20 | .993735 | .079149 |
| 5 | 40 | .979767 | .142243 |
| 10 | 90 | .913264 | .294509 |
| 30 | 290 | .418394 | .762631 |

30 global adım, “biraz global düzeltme” sayılmaz. Bu tablo denoiser'ın
temizleme hatası veya gerçekleşen latent sapması değildir; forward process
hesabıdır. Tek sorguluk t20 önerisi, üç reverse sorgusu demek de değildir.
[CPU cebir/scheduler kontrolü](evidence/shape_latent_research_20261031/math_checks.json).

## 5. Gerçek cache'ler shape bozulmasını gösteriyor mu?

**[Run]** Confirmation cache'lerinden iki dataset×15 corruption×üç seed×iki
batch olmak üzere180 fiziksel NPZ'nin dosya SHA'sı ve global latent tensor
kimliği doğrulandı. Dataset başına aynı140 nesne için128-boyutlu örnekler
eşleştirildi. Her corruption'ın seed ortalamasının, diğer14 corruption'ın
ortalamasına RMS uzaklığı hesaplandı; ayrıca seed saçılması ölçüldü.

| Dataset/corruption | Nesneler üzerinde medyan RMS uzaklık | Medyan seed RMS saçılması |
|---|---:|---:|
| ModelNet / Occlusion | .554820 | .519448 |
| ModelNet / LiDAR | .511160 | .538175 |
| ModelNet / Gaussian | .488534 | .612331 |
| Scan / LiDAR | .500238 | .571674 |
| Scan / Gaussian | .498613 | .622611 |
| Scan / Occlusion | .497428 | .588123 |

**[Open]** Temiz input'un z'si ve encoder µ/logσ bu cache'lerde yok.
Diğer corruption'ların ortalaması temiz hedef değildir; seed saçılması
yalnız posterior belirsizliği olarak yorumlanamaz. Bu tablo corruption'ların
global temsili farklılaştırdığına ilişkin betimsel kanıttır, hangi latent'in
yanlış olduğunun sıralaması değildir. Rotation/shear için “global latent
bozuluyor, global prior düzeltir” iddiası bu veriden çıkmaz.
[Yöntem, tüm satırlar ve cache kimlikleri](evidence/shape_latent_research_20261031/cached_shape_variation.json),
[yeniden çalıştırılabilir analiz](evidence/shape_latent_research_20261031/analyze_cached_shapes.py).

## 6. Literatürden neyi aktarabiliriz?

Tarama kesiti31 Ekim2026. Birincil makale/proje/resmi kodlar kullanıldı;
bu kaynaklarda LION global latent'ini sınırlı guidance ile adapte eden,
bizim corruption/classifier protokolümüzde doğrulanmış doğrudan bir yöntem
bulamadım. Aşağıdakiler mekanizma bağlantılarıdır, yeni bir exhaustive
literatür yokluğu iddiası değildir. Altı primary PDF200 yanıtı ve SHA ile
[saklandı](evidence/shape_latent_research_20261031/primary_retrieval.json);
resmi repo sürümleri [sabitlendi](evidence/shape_latent_research_20261031/author_pins.json).

| Çalışma | Aktarılabilir fikir | Bizim sınırımız / karar |
|---|---|---|
| [SDEdit, ICLR2022](https://arxiv.org/abs/2108.01073v2) | Az gürültüden başlamak prior düzeltmesi ile girdiye sadakat arasında denge sağlar | Kimliği veya z uzaklığını kesin sınırlamaz. LION'un diffuse-denoise yaklaşımıyla uyumlu başlangıç noktası. |
| [DPS, ICLR2023](https://arxiv.org/abs/2209.14687v4) | Kayıp temiz tahminde hesaplanır, türev gürültülü state'e alınır | `noisy_z`'yi doğrudan temiz anchor'a yaklaştırmak aynı şey değildir. Başlangıç z anchor'ı gerçek observation likelihood sayılmaz. |
| [PSLD, NeurIPS2023](https://papers.nips.cc/paper_files/paper/2023/file/9c70cfa2e7d9328c649c94d50cbf8faf-Paper-Conference.pdf) | Sec.2.1 Eq.6–7: latent/decoder/encoder tutarlılığı ve measurement gluing | Lineer image operatörü ve teori, sırasız3D SCD'ye doğrudan taşınamaz. LION `encode()` no_grad; stok API ile differentiable cycle loss kurulamaz. Ek encoder/decoder backward ilk aday için pahalı. |
| [DAPS, CVPR2025](https://arxiv.org/abs/2407.01521v3) | Prior denoising ile veri uyumu adımlarını ayıran annealing | İç posterior örnekleme/optimizasyon bütçesi gerekir; süre şikâyeti varken ilk tercih değil. |
| [L-DPS, Wang–Tartakovsky, 25 Haziran2026 preprint](https://arxiv.org/abs/2606.26592v1) | VAE + unconditional latent prior; decoder ve differentiable surrogate üzerinden guidance; Sec.4.1 Eq.27, normalized update Eq.30 | LION global prior'ını fiziksel kayba bağlamak için yakın mimari örnek. PDE likelihood/noise bilgisi bizde yok; sabitleri veya posterior iddiasını kopyalayamayız. Resmi implementasyonu doğrulanmadı. |
| [CLAMP, ICML2026](https://arxiv.org/abs/2605.27990v1) | Denoiser pullback, curvature ve yönlere göre damping; matrix-free GMRES | İleride global guidance'ın kötü koşullanmasını ele alabilir. JVP/VJP ve latent decoder maliyeti ölçülmeden “128 boyutta ucuz” denemez; şimdilik ertele. |

**[Code]** Resmi DPS
[`condition_methods.py`](https://github.com/DPS2022/diffusion-posterior-sampling/blob/effbde7325b22ce8dc3e2c06c160c021e743a12d/guided_diffusion/condition_methods.py)
measurement farkını `x_0_hat` üzerinde hesaplar, `autograd.grad(..., x_prev)`
ile state gradient'ini alır. Bu, temiz tahmin/gürültülü değişken ayrımını
uygulama düzeyinde de doğrular. L-DPS yaklaşık posterior yöntemidir;
önerdiğimiz SCD/anchor işlemini exact Bayesian posterior sampling diye
sunmamalıyız.

**[Inference]** Önceki rapordaki EmbedOpt ile bağlantı koşullandırma
vektörünün gradient ile düzenlenmesidir. 3DD-TTA zaten style'ı günceller;
yalnız RMS-normalize style update eklemek global prior'ı açmak veya başlı
başına yeni bir shape yöntemi değildir.
[Önceki literatür denetimi](chamfer_guidance_implementation_literature_audit_20261031.md).

## 7. Öneri A: tek temiz global öneri ve açık latent sınırı

**[Inference; uygulanmadı]** Başlangıç z'sine `z_c` diyelim: bu corruption'dan
encode edilmiştir, gerçek temiz z değildir. Ayrı RNG ile tek ξ_z üret:

\[
z_\tau=\sqrt{\bar\alpha_\tau}z_c+
\sqrt{1-\bar\alpha_\tau}\xi_z,\qquad
z_{prop}=\frac{z_\tau-\sqrt{1-\bar\alpha_\tau}
\epsilon_{global}(z_\tau,\tau+1)}{\sqrt{\bar\alpha_\tau}}.
\]

Bu checkpoint'teki epsilon prediction/time sözleşmesi korunur;
`mixed_prediction=false`. İlk prospective tek nokta t20 olabilir; bu,
tek model sorgusudur. Güvenli düzeltme:

\[
\Pi_R(v)=z_c+(v-z_c)\min\{1,R/(\|v-z_c\|_2+\varepsilon)\},
\quad z^*=\Pi_R(z_{prop}).
\]

R her nesne için tanımlanır; örneğin tek küçük pilot değeri
`.05 max(||z_c||, ε)` olabilir. Bu değer veriden öğrenilmiş optimum
değildir. Sınırın normalizasyonu ve tek deneme noktası sonraki protokolde
dondurulmalı; target label veya confirmation skorlarına göre taranmamalıdır.
Sonraki style güncellemeleri de `z←Π_R(z−γ∇_z L_SCD)` biçiminde sınırlanır.
Sadece ilk proposal'ı sınırlamak, final latent'in yakın kalmasını sağlamaz.
Koşullandırma ve final decode'da aynı temiz z sürümü kullanılır.

Alternatif yumuşak anchor,
`min_z .5||z−z_prop||² + .5λ||z−z_c||²` için
`z=(z_prop+λz_c)/(1+λ)` verir. CPU stationarity kontrolü geçti; ilk
çalışmaya hem λ hem radius grid'i eklemek gerekli değil.

Karşılaştırma **C → C+anchor → C+anchor+global proposal** olmalı.
Böylece yalnız anchor'ın faydası global prior'a yazılmaz. Sınır kapalı ve
global proposal kapalı yol mevcut C ile aynı olmalı. **R=0 ise style
donar; mevcut gradient-updated C ile özdeş değildir.** Global noise,
local noise ve classifier RNG sırasını değiştirmeyen ayrı generator'dan
gelmelidir. Yeni h encoding'i ayrı kol olarak raporlanır.

**Yanlışlanma koşulu:** z sapması sınır içindeyken prior kolu anchor-only
kontrolünü iyileştirmiyorsa, prior eklemek için kanıt yoktur. İki kol da
yalnız C'yi iyileştiriyorsa kazanç anchor'a aittir. Düşük radius, yanlış z'yi
de koruyabilir; güçlü anchor'ın zarar gören latent'i onaramaması beklenebilir.

## 8. Öneri B: kısa global zincirde temiz z üzerinden guidance

**[Inference; uygulanmadı]** A'da prior katkısı görülürse, örneğin local
10/30'dan bağımsız üç global adım (t20/10/0) düşünülebilir:

\[
\hat z_0=G_\theta(z_t,t),\quad
L_{anchor}=\|\hat z_0-z_c\|_2^2/128,\quad
z_{t-1}=DDIM(z_t)-\rho_t\nabla_{z_t}L_{anchor}.
\]

Temiz tahmin ile başlangıç referansı karşılaştırılır; ham z_t ile z_c
karşılaştırılmaz. Kusursuz bir temiz tahmin varken bile z_t'de planlanan
forward noise vardır; ham-state anchor bunu gereksiz yere cezalandırır.
Global denoiser türev grafiği tutulmalıdır; eski sync'teki detach davranışı
bu guidance değildir. Adım normu nesne bazında sınırlandırılır; son temiz
çıktı ayrıca R sınırına alınabilir. Local prior'a ara noisy z verilmez;
ilk çalışma global prephase, sonra local phase şeklinde olmalıdır.

İleride `SCD(D(h_ref, zhat0), X)` gibi decoded veri uyumu eklenebilir.
Bu durumda h_ref açıkça tanımlanmalı; corruption'a bozulmuş X'e fazla
bağlanma ve ek decoder backward maliyeti ölçülmelidir. SCD likelihood
değildir. PSLD cycle/gluing veya CLAMP curvature ilk minimal adaya eklenmez.
Encoder posterior varyansıyla Mahalanobis anchor da şimdilik ertelenir:
`models/distributions.py` ikinci parametresi **log std**'dir; varyans
`exp(2 logσ)` olur ve bu sayı corruption güvenilirliği için kalibre edilmiş
değildir.

## 9. En verimli sonraki deney sorusu

1. **Global temsilde düzeltilecek headroom var mı?** Temiz/bozuk eşleşmelerde
   posterior µ'leri ve aynı Gaussian örneğiyle z'leri karşılaştır. Veri
   preprocessing/nesne kimliği eşleşmesini doğrula. Sadece cache seed
   varyansından corruption hassasiyeti sonucu çıkarma.
2. **Oracle tanısı:** temiz z swap'ını bir analiz kolu olarak sabit h ile
   ve `q(h|X,z_new)` yeniden encoding'iyle ayrı ölç. İkinci kolda aynı ξ_h
   kullan. Temiz z deployment'ta yoktur; bu kol yöntem seçen oracle değildir.
   Temiz z bile yardımcı olmuyorsa daha karmaşık global diffusion için
   gerekçe zayıftır. Başarılı oracle ise öğrenilmiş prior'ın bunu
   sağlayacağını kanıtlamaz.
3. **En küçük yöntem:** A'nın C/anchor-only/anchor+prior karşılaştırması.
   Başlangıç z'si, h, local noise, classifier/probe RNG ve dört probability
   mean protokolü eşleştirilsin. Yeni global encoder çalıştırmak gizli bir
   maliyet veya gizli yöntem farkı olmasın. Posterior-mean z kontrolü
   düşünülebilir; sampled-z üzerinde eğitilmiş prior ve koşullu h açısından
   ayrıca dağılım değişikliğidir, otomatik düzeltme değildir.
4. **Maliyet ve kabul:** aynı nesnelerde correct-count/paired prediction,
   z displacement, h–z uyumu göstergeleri, guidance normu, ek prior/encoder/
   decoder forward-backward sayısı, peak memory ve wall-clock kaydedilsin.
   Tüm15 macro birincil, macro14/Background ayrı ikincil sonuç olsun;
   yalnız beklenen corruption'ları seçip ana sonuç yapılmasın. Development
   ve confirmation ayrı kalsın, test label ile radius/horizon seçilmesin.

Bu sıra 4 Ekim'deki
[opsiyonel global dört-kol planını](colab_gsd_block_routing_20261004.md#optional-global-diffusion-study-separate-after-routing-evidence)
silmez. O plan style gradient'ini kapatıp local condition/final decode
yollarını ayırır; burada C üzerinde bounded style ve global prior katkısı
araştırılıyor. Deneyler aynı isim/sonuç altında birleştirilmemeli.

## 10. Yapılan doğrulama ve kalan belirsizlik

**[Run]** 180 cache kimliği/tensor kontrolü; üç tarihsel decoder ZIP'i;
sekiz CPU cebir/RNG/statik kontrolü geçti.19 kaynak dosyasının byte/SHA
manifest'i ve altı primary PDF receipt'i tutuldu. Bu kontroller modelin
accuracy'sini, gerçek global denoiser kalitesini, CUDA gradient doğruluğunu
veya ilave süreyi ölçmez. LION PDF extraction araç uyarısı bazı yoğun
form/şekil içeriğinin atlandığını belirtti; bu rapor okunabilen yöntem
metni ve doğrudan kaynak koduna dayanır, şekillerden nicel sonuç çıkarmaz.

**[Open]** Eski dual çalıştırmanın komutu, artifact'i ve kullanılan commit'i
hâlâ eşleşmiş değildir. Temiz/global oracle tanısı, proposal kalite testi,
native GPU smoke ve maliyet ölçümü yapılmadı. Hiçbir aday uygulanmadı,
Colab/GPU açılmadı. Yeni GPU fazı ancak ayrı ve toplam bütçesi belirlenmiş
bir yürütme görevi olabilir; bu rapor bir launch talimatı değildir.
