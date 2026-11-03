# GSD guidance: matematik, implementasyon ve hız–accuracy hipotezleri

Tarih: 2026-11-03. Taban: `origin/thesis-integration@daf37a4`.
Dal: `research/gsd-guidance-math-audit-20261103`.
Worktree: `/Users/batuhan/Desktop/projects/3DD-TTA-worktrees/gsd-guidance-math-audit-20261103`.
Kod tamiri commit’i: `dda342bef0728ad6ad06892a12fae7dd75e53ef8`.
Kanıt etiketleri: **[Paper] / [Code] / [Run] / [Inference] / [Open]**.
Bu çalışma literatür/kod denetimi, küçük tamirler ve sentetik CPU tanıklarıdır;
yeni accuracy veya Colab/GPU çalışması değildir.

## 1. Mevcut durum ve ana değerlendirme

[Code/Run] Tez, bozulmuş nokta bulutunu pretrained LION ile adapte edip source
classifier'ı donmuş tutuyor. Güncel operasyonel kontrol **Chamfer + Probability
Mean, N8/Background35**, B32, local `.03`, style `.003`, p`.95`, original-z
decode, DDIM100/final-alpha1, raw/eval LION/EMAoff'tur.
[Baseline sözleşmesi](current_baseline.md). Tarihsel `%64.53/%64.57` legacy
sonuçları başka host/kohort ayarlarıdır; güncel kohort skorlarıyla doğrudan
karşılaştırılamaz. Yeni branch güncel integration'dan oluşturuldu; ana çalışma
dizinindeki kullanıcı değişikliklerine ve diğer worktree'lere yazılmadı.

[Run] En yeni h:SCD/z:spectral çalışması: 128 aşama-ayrık ModelNet nesnesi,
15 bozulma, üç seed, frozen Point-MAE, PM:

| Kol | Accuracy | Spektral − kontrol |
|---|---:|---:|
| SCD | %65.5903 | −0.01736 puan, %95 CI[−0.39931,+0.36458] |
| h:SCD / z:spectral | %65.5729 | — |
| Style-off | %65.4861 | +0.08681 puan, CI[−0.31250,+0.46875] |

[Run] Aynı çalışmadaki fresh B32 reconstruction+head medyanları:
Gaussian SCD4.3341s / spectral14.4686s; Background7.0235s /19.7350s.
Spektral kol sırasıyla **3.338× /2.810× daha yavaş**. Bu kol SCD'yi de
korur; spektral-only versus SCD kıyası değildir. Önceki S-only/unguided
ve CS/C deneyleri de ayrıdır. Veri/kohort, ölçüm sınırı ve ham kanıtlar:
[son kayıt](scdh_spectralz_results_20261101.md),
[karışık tarihsel spektral sonuçlar](graph_spectral_retrospective_audit_20261101.md).
Bu tur bu ham arşivleri yeniden kabul etmiş gibi sunmaz; kabul edilmiş kayıtları kullanır.

[Inference] Öncelikli açıklama bir ters-GFT hatası değil, **hedefin yanlılığı**:
bozuk latent'e düşük-frekans sadakati, doğru düzeltmeyi de cezalandırabilir.
Sabit grafın eskimesi ve decoder'ın frekansları karıştırması ayrıca makul ve
henüz nedensel olarak doğrulanmamış açıklamalardır. **Hesaplamayı hızlandırmak
ve doğru hedef seçmek iki ayrı problem**. İkisini çözdüğümüz veya accuracy'nin
mutlaka artacağı söylenemez.

## 2. Birincil literatür taraması

[Paper] 3 Kasım'da başlık/konu araması ve aşağıdaki birincil tam metinlerin
ilgili yöntem/teorem bölümleri incelendi. Bu odaklı tarama sistematik tüm-yayın
sayımı değildir. İkincil özetler matematiksel kanıt olarak kullanılmadı.
GSDTTA için aramada doğrulanabilir resmi kod commit'i saptanmadı; resmi kodun
hiç var olmadığı sonucuna varılmıyor. Denklem/prose belirsizliklerinde makale
ile fork'un açık tercihleri ayrılır.

| Kaynak ve okunan bölüm | Bu çalışma için çıkarım ve sınır |
|---|---|
| [Wei et al., GSDTTA, ICCV2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Wei_3D_Test-time_Adaptation_via_Graph_Spectral_Driven_Point_Shift_ICCV_2025_paper.pdf), §2.2–3.2, Eqs3/5–18, Fig2; [supplement](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Wei_3D_Test-time_Adaptation_ICCV_2025_supplemental.pdf) | Fiziksel XYZ üzerinde `X_s=X+U_M ΔC`; self-training ve model adaptasyonu var. Bizim corrupted-reference fidelity'miz ve frozen-classifier kısıtımız farklı. Fig2'nin enerji yoğunluğu tek sandalye örneğidir. |
| [Dastmalchi et al., 3DD-TTA, WACV2025](https://openaccess.thecvf.com/content/WACV2025/papers/Dastmalchi_Test-Time_Adaptation_of_3D_Point_Clouds_via_Denoising_Diffusion_Models_WACV_2025_paper.pdf), §3, Eqs8–12/Algorithm1 | LION latent denoising + seçici geometrik sadakat; classifier donmuş. Spektral dönüşüm kendi başına yeni temiz bilgi üretmez. |
| [Zeng et al., LION, NeurIPS2022](https://papers.neurips.cc/paper_files/paper/2022/file/40e56dabe12095a5fc44a6e4c3835948-Paper-Conference.pdf), §3, Eqs5–7/Fig1 | Hiyerarşik `q(z|X),q(h|X,z),p(X|h,z)`; latent noktalar koordinat ve feature taşır. Latent ve fiziksel graf spektrumlarının eşitliği vaat edilmez. |
| [Shuman et al., The Emerging Field of Signal Processing on Graphs,2013](https://arxiv.org/pdf/1211.0053), §II–III, filtering/regularization Eqs12/15/18 | GFT bir koordinat değişimidir; graph filtering ile veri sadakati/smoothness farklı nesnelerdir. Vertex-domain polinom uygulaması mümkündür. |
| [Hammond et al., Wavelets on Graphs via Spectral Graph Theory,2011](https://arxiv.org/pdf/0912.3848), §6/Lemma6.1, polynomial recurrence | Tüm eigenbasis hesaplanmadan sparse polynomial filtering ve yaklaşım hata sınırları. Bu, accuracy teoremi değildir. |
| [Defferrard et al., Fast Localized Spectral Filtering, NeurIPS2016](https://arxiv.org/pdf/1606.09375), §2.1/Chebyshev recurrence | Seyrek graf üzerinde K dereceli filtre O(K·nnz(L)); matematik spectral kalırken explicit GFT gerekmeyebilir. Yeni GNN eğitmek gerekmiyor. |
| [Levie et al., On the Transferability of Spectral Graph Filters,2019](https://arxiv.org/pdf/1901.10524), §2.2–3/Theorem2 | Eigenvector'ların değişmesi filtreyi zorunlu olarak kararsız yapmaz; düzgün operator fonksiyonları incelenmeli. Hard rank cutoff ayrı risktir. |
| [Jin Zeng et al., 3D Point Cloud Denoising Using Graph Laplacian Regularization of a Low Dimensional Manifold Model](https://arxiv.org/pdf/1803.07252), §IV-C/D/Eqs35–37, Algorithm1 | Gürültüye dayanıklı patch grafı, graph update ve regularized linear solve; salt bozuk spektrumu korumakla aynı amaç değil. LION'a başarı garantisi taşımaz. |
| [Perraudin & Vandergheynst, Stationary Signal Processing on Graphs](https://arxiv.org/pdf/1601.02522v3), §5/Eqs10–11 | Wiener ağırlıkları sinyal/gürültü PSD'sine bağlıdır. `exp(-βλ)` genel olarak istatistiksel optimal ağırlık değildir; graph stationarity ve noise modeli gerekir. |
| [Zheng, Cheng & Sun, Wiener Filters on Graphs and Distributed Polynomial Approximation Algorithms](https://arxiv.org/pdf/2205.04019), §2–4/Theorems3.1/4.1 | Gürültü kovaryansı ve hesaplama yaklaşımı birlikte ele alınabilir. Varsayımlı MSE optimalitesi, sınıflandırma optimalitesi değildir. İncelenen metin arXiv sürümüdür. |
| [Isufi et al., Graph Filters for Signal Processing and Machine Learning on Graphs](https://arxiv.org/pdf/2211.08854), §IV–V/Property8, Proposition1, Box2 | Stability, operator matching, tekrar eden eigenvalue'larda eşit ağırlık ve polynomial/rational maliyetleri için güncel çerçeve. İncelenen sürüm arXiv PDF'sidir. |

[Inference] Literatürün ortak sonucu: ucuz ve kararlı bir spektral operator
seçilebilir; **hangi referansa ve hangi amaçla** uygulandığı faydayı belirler.
Bu kaynaklar bir LION-TTA accuracy artışının veya önerilerimizin yeniliğinin
kanıtı olarak kullanılamaz.

## 3. Kodun gerçekten optimize ettiği şey

[Code] Legacy/component yol: `graph_spectral_v2.py` → `tta_gsd_v2.py` →
`spectral_guidance_losses.py`. Modern alternatif `graph_spectral.py` /
`tta_gsd.py`; son z-routing `scdh_spectralz.py` / `tta.py` üzerinden yürür.
Aynı isimlerle anılan bu yolların graph/normalization/decoder ayarları farklıdır.

`R=h_ref[:,:,:3]`, `Y=hat_h_0(h_t,s_t)[:,:,:3]`, her biri B×2048×3;
`L=UΛUᵀ`, `E=Y−R`, `w_i=exp(−βλ_i)·1[i<M]`.
Legacy L2 loss:

\[
\mathcal L_s=\frac1{3N}\sum_b\sum_i w_{bi}\|u_{bi}^{\top}E_b\|^2
=\frac1{3N}\sum_b\operatorname{tr}(E_b^\top F_bE_b),\quad
F=U\operatorname{diag}(w)U^\top.
\]

\[
\nabla_Y\mathcal L_s=\frac{2}{3N}F(Y-R).
\]

[Code] `3N` sabit original nokta sayısıdır; batch-sum'dır. Modern hard-v1'in
`3·rank` ve normalize smooth-v2'nin all-active heat filtresiyle katsayıları
karıştırmamak gerekir. 4D seçeneğinde tarihsel `3N` sözleşmesi korunmuştur;
XYZ ve feature'u eş ölçekli saymak ayrıca bir modelleme varsayımıdır.

[Inference] Tam ortonormal bazda `U UᵀY=Y`, **Y artık eski grafın geometrisini
izlemese de** geçerlidir. M modla geri kurarsak `U_M U_MᵀY=P_MY`; kayıp,
truncation/projection'dan gelir, sabit U kullanımından otomatik gelmez.
U'yu M sütunla sınırlandırmak arbitrary Y için tam bir inverse vermez.
Çalışan fidelity yolunda her adımda explicit IGFT ile nokta kurulmuyor;
sonunda LION decoder çağrılıyor. Backward zaten U üzerinden vertex gradientini
veriyor. Eksik IGFT çağrısı bulgusu yoktur.

[Paper/Inference] GSDTTA ise `X_s=X+U_MΔC` öğrenir. Bir driving objective J
altında `∇_ΔC J=U_Mᵀ∇_X J`, dolayısıyla doğrudan physical gradient descent
adımı `−a P_M∇_X J` olur. Düşük-frekans hareketine **izin vermek** ile
aynı hareketi R'den uzaklaştığı için **cezalandırmak** farklıdır.

[Code/Inference] Bizim guidance'ımız denoiser Jacobian'ını da içerir:

\[
\hat h_0=(h_t-\sigma_t\epsilon_\theta(h_t,s_t,t))/a_t,\quad
J_h=(I-\sigma_t\partial_h\epsilon_\theta)/a_t,\quad
J_s=-\sigma_t\partial_s\epsilon_\theta/a_t.
\]

XYZ seçim operatorüyle `g_h=J_hᵀ P_xyzᵀ ∇_Y L`,
`g_s=J_sᵀ P_xyzᵀ ∇_Y L`.
Frozen ağırlıklar bu input Jacobian'larını kapatmaz. Doğrudan `F(Y−R)`yi
noisy h'ye uygulamak aynı gradient değildir. Low-mode projector ile J genel
olarak yer değiştirmez. `tta_gsd_v2`'de **eta local, gamma style**;
`tta.py`/güncel baseline'da **gamma local, eta style**. İsimlerden değil
çağrı/konfigürasyon sözleşmesinden hareket edilmelidir.

## 4. Loss azalırken temizliğin neden bozulabileceği

[Inference; yerel quadratic surrogate, LION/accuracy teoremi değil]
Tek modda denoiser'ın önerisi d, bozuk referans r olsun:

\[
\min_y \tfrac12 a_i(y-d_i)^2+\tfrac12\alpha w_i(y-r_i)^2
\Rightarrow y_i=(1-t_i)d_i+t_i r_i,\quad
t_i=\frac{\alpha w_i}{a_i+\alpha w_i}.
\]

`d=c+e_d`, `r=c+e_r`; merkezlenmiş hata ikinci momentleri v_d,v_r ve
cross-moment c_dr için:

\[
\mathrm{MSE}(t)=(1-t)^2v_d+t^2v_r+2t(1-t)c_{dr},\quad
t^*=\operatorname{clip}_{[0,1]}\frac{v_d-c_{dr}}{v_d+v_r-2c_{dr}}.
\]

Payda pozitif varsayılır; aynı hatalı iki tahmin gibi dejenere durumlarda
bu ifade ayrıca ele alınır. Hatalar bias'lıysa ikinci momentler bias'ı da
kapsar; variance diye yorumlamak ek merkezleme varsayımı ister.

[Run: sentetik CPU] Bağımsız hatalar/v_d=.04/v_r=1 için prior MSE=.04;
t=.5 anchoring MSE=.26; optimum t=.03846 ve MSE=.03846.
Güçlü fidelity aynı problemi kötüleştirebilir. Bu sayılar gerçek latent
varyansı değildir. Tam temiz tahminin corrupted-reference loss'u pozitif
olup salt fidelity adımı temiz hatayı artıran bir tanık da kaydedildi.

[Inference] Düşük frekans sinyalinin büyük enerji taşıması, o moddaki
**bozulmanın küçük** olduğu anlamına gelmez. Global kayma/shear/eksik parçadan
kaynaklanan şekil yanlılığı düşük modlara sızabilir. R'deki bu bileşenleri
korumak prior'ın faydalı değişimini bastırır. SCD de R'ye bağlar, fakat
büyük set-mesafelerini budar ve birebir latent slot eşitliğini zorlamaz.
Spektral L2 bunu aynı şekilde yapmaz. Salt `trace(YᵀLY)` ile değiştirmek de
constant/collapsed çözümler doğurabilir; otomatik düzeltme değildir.

[Run/Inference] Bu nedenle yalnız normu büyütme hipotezi zayıfladı:
önceden kalibre937.38 ölçekli CS deneyi ve son bağımsız z-step taraması da
güvenilir genel gain göstermedi. Daha fazla beta/M/alpha sweep ilk öneri değil.

## 5. Sabit/dinamik U: kullanıcının ilk hipotezi

[Inference] Sabit graph bir ölçüm uzayıdır; yanlış inverse değildir.
Fakat Q'daki komşulukların Y için geçerli kalmaması **metric mismatch** olabilir.
Düzgün heat profilinde aynı vertex kümesi/aktif destek için:

\[
F_j=e^{-\beta L_j},\qquad
\|F_1-F_0\|_2\le\beta\|L_1-L_0\|_2,
\]

\[
|\ell_{F_1}(E)-\ell_{F_0}(E)|\le
\beta\|L_1-L_0\|_2\|E\|_F^2/(3N).
\]

Bu, PSD L'ler için matrix-exponential integral identity ve contractivity'den
çıkar. Normalize L kullanılıyorsa fark **normalize operatorler arasında**
ölçülür. Aktif vertex maskesi değişiyorsa zero-extended filtre/maskeler de
karşılaştırılır; yukarıdaki aynı-destek sınırı doğrudan uygulanmaz.
Hard rank cutoff'un küçük eigengap sınırında aynı düzgünlük garantisi yoktur.

[Inference] Nokta başına hareket ≤ε ise pairwise distance değişimi ≤2ε;
`d_(k+1)−d_k>4ε`, ties yokken bir noktanın kNN kümesini korumak için yeterlidir.
RBF ağırlıkları yine değişebilir. Yalnız displacement normu veya eigenvector
cosine'ı yerine edge overlap, boundary gap, operator farkı ve aktif destek
ölçülmelidir. Ortak eigenvector işaret değişimleri L2 loss'u değiştirmez.

[Code] `dynamic_graph` iki sinyali de **aynı U_current**'a projekte eder;
başlangıç katsayısını eski bazda bırakma hatası yok. Fakat graph detached'dır:

\[
\ell(Y)=\operatorname{tr}(E^\top F(Y)E)/(3N),\quad
d\ell=\{2\langle FE,dY\rangle+
\operatorname{tr}(E^\top dF(Y)E)\}/(3N).
\]

Kod yalnız ilk terimi alır: **yenilenen/frozen metric ile alternating adım**.
Bu bilinçli surrogate olabilir; tam geometry-dependent loss gradienti değildir.
Daha sık eigh çalıştırmak bunu otomatik düzeltmez. Graph'a tam türev eklemek
kNN/pruning discontinuity'leri ve eigenvalue tekrarlarında ayrıca sorunludur.

[Run: sentetik CPU] Yumuşak, tiesiz üç-vertex RBF grafında full versus frozen
metric gradient farkı4.2283e−4; full yönün finite-difference kontrolü de geçer.
Bu gözlem gerçek LION'da dynamic modun zararlı/faydalı olduğunu göstermez.

## 6. Encoder/decoder sonrası spektral anlam: ikinci hipotez

[Code] `models/latent_points_ada.py`:
encoder mean XYZ=`X+skip·network(X,z)`; posterior ayrıca örneklenir.
Decoder `D(h,z)=h_xyz+skip·network(h_xyz,h_feature,z)` biçiminde residual'dır.
`view(B,2048,4)[:,:,:3]` düzeni doğru; transpoz/flatten kusuru bulunmadı.
Slot tabanlı residual yapı correspondence için bir dayanak sağlar, fakat
kNN, geodesic veya spectral isometry garantisi değildir.

[Inference] Fiziksel-space quadratic metric'in local latent'e pullback'i:

\[
G_{\rm phys}=J_D^\top(F_X\otimes I_3)J_D,
\quad
G_{\rm latent}=P_{xyz}^\top(F_h\otimes I_3)P_{xyz}.
\]

Genellikle eşit değildirler. Decoder feature ve style'a da bağlıdır;
latent XYZ loss'u bunların etkisini yalnız denoiser Jacobian'ı üzerinden görür.
Latent'te düşük-frequency değişim fiziksel çıktıda düşük-frequency kalmak
zorunda değildir. Bu **geçerli GFT'nin anlamsız/bozuk hesaplandığı** anlamına
gelemez; seçilen metric'in downstream amaçla uyuşmaması anlamına gelir.

[Run: sentetik CPU] Residual linear decoder Jacobian'ı
`J=I+.6·u_high u_lowᵀ` tanığında saf low-mode giriş değişiminin çıkış
enerjisinin%26.47'si high mode'a geçer. Bu gerçek LION ölçümü değildir;
isometry çıkarımına karşı örnektir.

[Code/Inference] Son routing'de style güncellemesi bir sonraki denoiser
çağrısını etkiler. Final decode original z0 kullandığı için **son adımın
style-spectral güncellemesi hiçbir sonraki prior/decode tarafından tüketilmez**.
Bu, önceki style adımlarını etkisiz yapmaz. Diagnostics gerekmeyen deployment
varyantında son spectral VJP atlanabilir; toplam hız kazancı ölçülmelidir.
Bu çalışmada route'un bilimsel diagnostics sözleşmesi değiştirilmedi.

## 7. En ucuz doğru hesap: hangi loss için?

[Code] Yeni küçük tamir `tta_gsd_v2.py`'de ölçüm için ikinci GFT/loss ve ikinci
Chamfer forward/sort çağrısını kaldırır; aynı gradient-enabled scalar'ın
`detach().item()` değeri raporlanır. Her aktif adımda spectral loss ve Chamfer
forward sayısı2→1. Guidance objective, gradientli işlem sırası ve katsayıları
korundu. Invariant raw metric eskiden M_max, uygulanan loss graph.M kullanıyordu;
artık metrik **uygulanan loss'un kendisidir**. Bu metrik kusuru desteklenen
invariant=0 deneylerinin accuracy'sini geçersiz kılmaz.

[Run: CPU] Regresyonlar önce beklenen nedenle başarısız oldu; tamir sonrası
pinned eski kaynakla aynı toy trajectory/local/style/RNG ve legacy loss
metrikleri korunuyor. Historical `spectral_loss_calls`/`chamfer_calls` sayıları
eski arşivlerde değiştirilmez. Yeni sayaçlar gerçek yeni çağrı sayısını bildirir.
Bu parity CUDA bitwise sonuç garantisi değildir.

[Code] Bu tamir **legacy/component yoluna** aittir. Son SCD-h/spectral-z route'u
`tta.py`/`scdh_spectralz.py` ile yürür; yukarıdaki3.338× maliyetini çözdüğümüz
iddia edilmiyor. Son route, SCD-local ve spectral-style için ayrı VJP'ler ve
ayrıntılı diagnostics taşır; bir scalar/backward ile aynı matematik değildir.

[Inference] Dense graph+full eigh yaklaşık O(N³) setup/O(N²) storage ister.
N2048/B32/fp32'de yalnız U **512MiB**; adjacency ve workspace bunun dışındadır.
Sadece ilk M loss'a girmek full eigh'ı ucuzlatmaz. Loss matmul'ünü M sütunla
hesaplamak faydalı olabilir, ama setup darboğazını kaldırmaz ve `3N` reduction
korunmalıdır. Elenen modlar önceden finite doğrulanmadan `0*NaN` davranışının
değişmesini veya farklı floating-point toplama sırasını exact parity saymamak gerekir.

[Paper/Inference] Gerçek hız adayı **eigenfree sparse heat loss**:

\[
\ell_{\rm heat}=\|e^{-\beta\bar L/2}E\|_F^2/(3N),\quad
\tilde\ell=\|p_K(\bar L)E\|_F^2/(3N).
\]

`p_K≈exp(−βλ/2)`, Chebyshev recurrence ile sparse matmul/scatter kullanır;
explicit U ve IGFT yok. Bir filtrenin uygulanması O(K·nnz(L)·3), graf kNN/setup
ayrıca hesaba katılır. Autograd'ın K adımlık activation maliyeti de ölçülür.
Tüm pipeline'ın O(KNk) veya daha hızlı olduğu sonucu çıkarılmaz.
**1/2 zorunludur**: exp(−βL) filtreleyip norm karesi almak β'yı iki katına çıkarır.
Square-root filtreyi yaklaştırıp norm karesi almak, sonlu polinomun olası
negatif değerlerine rağmen loss'un nonnegative olmasını sağlar.

`sup_spectrum |p_K−exp(−βλ/2)|≤ε` ise:

\[
|\tilde\ell-\ell|\le(2\epsilon+\epsilon^2)\|E\|_F^2/(3N),\quad
\|\nabla\tilde\ell-\nabla\ell\|_F\le
2(2\epsilon+\epsilon^2)\|E\|_F/(3N).
\]

Graph/reference frozen olmalı. İzole vertex'ler explicit maskelenir;
yoksa heat operatorü onların zero-mode hatasını korur. Üst spectral bound
`2·dmax` ile bulunabilir; isolate1000 penalty'yi doğrudan aralığa almak
yaklaşımı gereksiz zorlaştırır. Aktif grafı ayırmak farklı operator seçimi
olarak belgelenmelidir.

[Run: sentetik CPU] 64vertex/63active, iki component+isolate, β2,185nnz:

| Polinom derecesi | Mutlak loss farkı | Gradient L2 farkı |
|---|---:|---:|
| 8 | 2.63e−8 | 2.50e−8 |
| 16 | 3.33e−16 | 1.59e−16 |
| 32 | 4.44e−16 | 2.21e−16 |

İzole vertex gradienti0; hesaplanan spectral error'dan türetilen sınırlar geçer.
Bu tek grafın exact spectrum'una ilişkin kontroldür; tüm aralıkta sertifikalı
approximation garantisi, production implementasyonu veya GPU speedup değildir.

[Inference] Bu formül **normalize/all-active heat loss** için aynı hedefin
yaklaşımıdır. Raw-λ + rank-M hard-cutoff legacy loss ile eşdeğer değildir.
Hard cutoff'u atıp sonucu yalnız hız optimizasyonu diye sunmak yanlış olur.
Önce exact-heat versus polynomial-heat, sonra objective versus objective
karşılaştırılmalıdır. Önceki smooth null sonuçları silinmez; ucuz hesaplama
onları accuracy bakımından çürütmez.

## 8. Öncelik sıralı, yanlışlanabilir hipotezler

| Öncelik / hipotez | Ayırt edici deney | Ölçüm ve yanlışlayıcı |
|---|---|---|
| H1 — Bozuk anchor faydalı low-band değişimini bastırıyor | Aynı archived N8 durumunda C/S update yönlerini paired temiz counterpart'a ve decoder sonrası temiz geometriye göre incele; salt loss düşüşünü ayrı kaydet | Gerçekleşmiş küçük adım sonrası clean-set hata/karar margin değişimi. Yalnız cosine yetmez. S yönü tutarlı temizlik sağlıyor ve faydalı C değişimini bastırmıyorsa açıklama zayıflar. |
| H2 — Eski graf metric mismatch yaratıyor | Önce inference değiştirmeden başlangıç/ara clean-prediction grafı karşılaştır; sonra gerekirse **tek** graph refresh, aynı anchor/profile/step kalibrasyonu | Edge overlap, aktif destek, normalized operator/filter farkı, eigengap, yön farkı, wall time. Küçük operator değişimi ve eşlenmiş null refresh sonucu hipotezi zayıflatır. |
| H3 — Latent düşük frekansı decoder'da korunmuyor | Gerçek LION'da birkaç düşük/yüksek-band perturbasyonu veya JVP/VJP; fixed z, sonra feature/style etkisini ayrı ölç | Fiziksel graf üzerindeki output-energy leakage, clean Chamfer ve frozen-classifier margin. Düşük leakage ve latent/physical gradient hizası sorunu zayıflatır. |
| H4 — Aynı smooth hedef eigenfree hesaplanarak hızlanabilir | Aynı sabit sparse graf/destek/beta/normalization ile exact heat ve K=8/16/32 polynomial; önce teknik eşdeğerlik | Loss/gradient error, common-state trajectory, GPU B1/B32 latency/throughput/peak memory; encoder+kNN+decode+PM dahil. End-to-end maliyet azalmıyorsa hız hipotezi reddedilir; accuracy kazancı beklenen teknik kabul şartı değildir. |
| H5 — Güvenilirlik precision'ı mevcut exp(−βλ)'dan daha anlamlı | H1 desteklenirse oracle modewise risk'i **diagnostic-only** çıkar; sonra clean/label kullanmadan tahmin edilebilen güvenilirlik olup olmadığını kontrol et | Hata/covariance ve reliability sıralamasının ayrı kohortta doğruluğu. Test-time oracle yasak; güvenilirlik öngörülemiyorsa yeni loss taraması başlatma. |
| H6 — Fidelity yerine düşük-band hareketi parametrize etmek daha doğru aktarım | GSDPS-inspired `Δh_xyz=U_M a` veya smooth preconditioner'ı yalnız ortak driving objective ile karşılaştır | SCD kullanılıyorsa maliyeti korunur; semantik objective kullanılıyorsa yeni classifier/Jacobian maliyeti sayılır. Norm-matched güncelleme/accuracy başarısızsa aktarım desteklenmez. Bu full GSDTTA değildir. |

[Inference] H1/H3 önce, H4 paralel bir teknik eşdeğerlik sorusu olarak en
verimli sıradır. H2 için her adımda full eigh önerilmez. H5/H6'nın amaçlarını
ve clean-free yönlerini doğrulamadan geniş beta/M taraması verimli görünmüyor.
H4 başarılı olsa bile H1 çözülmediyse yalnız daha hızlı null bir yöntem elde edilir.

## 9. En küçük yeni deney paketi ve karar kuralları

[Proposal/Open; başlatılmadı] Önce mevcut cache/config kapsamı kontrol edilir;
kabul edilmiş accuracy koşulları yeniden çalıştırılmaz. Gereken Y/s/decoder
snapshot'ları eksikse yalnız yeni diagnostic capture tanımlanır. Temiz ve bozuk
latent slotları farklı interpolation/FPS/posterior draws taşıyorsa doğrudan
modewise fark alınmaz; ortak nokta kimliği/ortak sampling gerekir. Bu eşleşme
kurulamıyorsa clean **set** mesafeleri ve decoder üzerinden ölçülen değişim
kullanılır; latent modewise clean-error claim'i bırakılır.

1. **Teknik/teşhis:** mevcut paired state'ler, Gaussian/Background/shear/LiDAR
   gibi noise+structural bozulmalar; sabit az sayıda orijinal nesne, N8 host.
   H1/H3 directional small-step, H2 read-only operator ve H4 approximation
   kontrolleri. Temiz veri yalnız mekanizma teşhisi; deploy loss'a girmez.
2. **Hız:** aynı prepared inputs üzerinde compute-only ölçümün yanında fresh
   encoder+kNN+graph/probes/decode+frozen-head toplamı, B1 ve B32 ayrı; warmup,
   CUDA synchronization/events, ayrı tekrarlar ve peak allocated memory.
   Cache-replay süresi deployment hızı değildir. Diagnostic overhead dahil/hariç
   koşullar açıkça ayrılır. Route'ta son tüketilmeyen style-VJP atlama ayrıca
   teknik ablation olabilir; accuracy değişirse parity varsayımı yanlıştır.
3. **Koşullu küçük development:** yalnız yön ve maliyet teşhisi umut veriyorsa
   tek yeni objective adayı + exact aynı objective + mevcut SCD baseline.
   Smooth profilin geçmiş negatifleri korunur. Sadece hız soruluyorsa yeni
   objective veya accuracy hyperparameter sweep yapılmaz.
4. **Dondurulmuş doğrulama:** all15/severity5/0–1–2, original-ID bazlı development
   ayrımı; PM primary/terminal secondary, per-corruption tablosu, object-cluster
   paired CI, accuracy–latency Pareto. Aynı nesnenin bozulma/seed tekrarları
   bağımsız nesne sayılmaz. Önceden tüm benchmark görülmüştür; stage-disjoint
   olmak external/unseen validation değildir. Diğer başlık/dataset transferi
   ancak sabit adaydan sonra; farklı checkpoint/source aileleri karıştırılmaz.

Accuracy için mevcut kullanıcı hedefi **+1–2 mutlak puan** ve net maliyet kazancıdır;
bu sonuç garantisi veya sonradan seçilecek bir başarı eşiği değildir. Teknik
loss eşdeğerliği accuracy hedefinin yerine geçmez. Genel artış çıkmazsa
sonucu null/negative olarak raporlamak geçerli bilimsel kapanıştır.
GPU fazı bu raporda bütçelenip başlatılmamıştır; >60dk yeni whole-phase için
projedeki onay kuralı uygulanır. Başka çalışmanın süre muafiyeti bu öneriye
aktarılmaz.

## 10. Yeniden üretim, tamir ve sınırlar

[Run: CPU] [Assert içeren audit](evidence/gsd_guidance_math_20261103/audit.py),
[üretilen sayılar](evidence/gsd_guidance_math_20261103/audit.json),
[kaynak/test makbuzu](evidence/gsd_guidance_math_20261103/verification.json).
Seed20261103, CPU/float64/PyTorch2.15.0/Python3.12.10; bu Colab'ın pinned
PyTorch2.1.2 ortamı değildir. Fixed-U roundtrip max3.55e−15;
coefficient/operator loss farkı2.22e−16; analytic gradient farkı4.34e−18.

```bash
/tmp/tta-lion-reencode-venv/bin/python knowledge/evidence/gsd_guidance_math_20261103/audit.py
/tmp/tta-lion-reencode-venv/bin/python -m unittest discover -s tests -q
```

[Run] Son tüm-suite kontrolü: **465 test çalıştı,464 geçti,1 skip**;
26.557s. Skip: `test_default_pointmae_preview_matches_frozen_pre_followup_config`
(clean worktree'de tarihsel sealed source olmadığı için). Dar37-regression
kontrolü de geçti. Önceden var olan syntax/deprecation/tensor-scalar/native
unavailable uyarıları kayıtta tutuldu. Native CUDA/gerçek LION çalıştırılmadı.

[Code/Decision] Değişikliklerin tamamı küçük loss-computation tamiri, ona
bağlı testler, bu rapor/asserted evidence ve knowledge güncellemeleridir.
Yeni dependency, yeni production loss/graf, classifier eğitimi veya default
promosyonu yoktur. Bilinen legacy self-only isolation/eşik ve cutoff sorunları,
önceki offdiag/null deneyleriyle birlikte korunur; aynı graph'u sessizce başka
grafla değiştirmek bu accuracy tanısını çözmez.

[Open] Gerçek LION'da causal H1/H2/H3 ölçümü ve H4 end-to-end CUDA maliyeti
yoktur. Bu nedenle doğrulanmış tek accuracy-kaybı nedeni veya simultaneous
accuracy+speed çözümü sunduğumuz iddia edilmez. Bu rapor, hangi ölçümün hangi
alternatif açıklamayı eleyeceğini ve en ucuz sonraki deneyi belirler.
