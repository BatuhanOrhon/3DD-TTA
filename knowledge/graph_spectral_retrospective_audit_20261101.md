# Graph Spectral Loss: geçmiş kazanımlar, aktarım ve matematik denetimi

Tarih: 2026-11-01. Etiketler: **[Paper]**, **[Code]**, **[Run]**,
**[User report]**, **[Inference]**, **[Open]**.
Taban: `origin/thesis-integration@6ff2e4952a0fa4842aa32b14afaf1ae8fa123f27`.
Dal: `research/graph-spectral-audit-20261101`; ayrı managed worktree `51fb/3DD-TTA`.

## Sonuç ve kapsam

[User report] Kullanıcı, eski ModelNet40-C/Point-MAE deneylerindeki spektral
kazanımla sonradan SCD-only seçmemizin ilişkisini ve GSDTTA dönüşümlerinin
doğru uygulanıp uygulanmadığını soruyor.

[Inference/Decision] Eski pozitif gözlem gerçektir; **kalıcı, bağımsız spektral
katkı** olduğu gösterilmemiştir. Sonraki eşlenmiş doğrulamalar küçük ve değişken
artış/azalışlar verir. GSD'yi mevcut baseline'dan çıkarmanın dayanağı,
denenen varyantın ek maliyetine karşı tutarlı doğruluk katkısı göstermemesidir;
her spektral yaklaşımın gereksiz veya zararlı olduğunun kanıtı değildir.
Tek bir doğrulanmış accuracy-kaybı nedeni bulunmuş değildir.

[Code/Run] GFT yönü, latent yerleşimi ve uygulanan L2 gradienti tutarlı.
Legacy grafın makaleden ayrılan tercihleri ve gerçek bir self-only-isolate
kusuru vardır. Kusurun geçmiş düzeltme deneyi doğruluğu artırmamıştır.
Yeni CPU tanıkları, bozulmuş referansa sadakat ile temizliği optimize etmenin
aynı şey olmadığını ve sabit graf/filter ayarlarının ölçeğe duyarlılığını gösterir.
Bu çalışma üretim kodu/default/checkpoint/veri değiştirmez; GPU, Colab,
runtime sorgusu/kapatma, eğitim veya yeni accuracy deneyi yapmaz.

## 1. Hangi eski artış, hangi kontrol?

[Run] Aşağıdaki değerler **C+S eksi aynı-host C**, yüzde puanıdır. C, SCD;
S, spektral bileşendir. Her satır kendi nesne kohortu/readout'u içindeki
eşlenmiş karşılaştırmadır; farklı satırların mutlak accuracysi karşılaştırılamaz.

| ModelNet / Point-MAE aşaması | Kohort / seed | Terminal farkı | Probability Mean farkı |
|---|---|---:|---:|
| 25 Ekim geliştirme, sabit katsayı 1.17 | 70 nesne / 0 | +0.857143 | 0.000000 |
| 25 Ekim sabit doğrulama | ayrı 70 / 0,1,2 | -0.063492 | -0.412698 |
| 28 Ekim kalibre doğrulama | yeni 140 / 0,1,2 | -0.222222 | +0.047619 |

[Run] İlk doğrulamanın tek seed1 terminalinde +1.238095 puan vardır; diğer
iki seed negatif olduğu için üç-seed ortalaması negatiftir. Son kalibre
Probability Mean farkı yalnız 3 ek doğru/6300 tekrar tahmin olayıdır.
İlk terminal kazancının probability mean'de eşitliğe dönüşmesi o pilot için
doğrudur; her fazda aynı sonucun tekrarlanması veya PM'nin spektrali genel
olarak ikame etmesi gösterilmemiştir.
[17 ham ZIP bağımsız yeniden sayımı](pointmae_guidance_comparison_recheck_20261030.md)
bu tabloyu kaynak/başlangıç-state/kayıp ağırlığı/ID/logit kontrolleriyle doğrular.
Yeni çalışmada bu kabul edilmiş sayım ve ilgili mevcut CSV'ler kullanıldı;
17 ZIP'in tamamı yeniden sayılmış gibi raporlanmaz.

[Run/History] Daha eski, normalize smooth üç-seed all15 deneyinde S-only
%61.3596, unguided %61.2021'dir (+0.1576 puan); native binary ve ortak-state
kimlikleri eşlenmiş değildir. Aynı arşiv grubunda SCD %63.8799, SCD+smooth
%63.8520'dir (-0.0279 puan). S+U veya SCD+S versus U kontrastı, SCD üzerine
spektral eklemenin faydasını tek başına ölçmez.
[Eski ablation](gsd_guidance_ablation_results_20261004.md).

[Run/History] Tarihsel %64.5327 CSV ve geri yüklenen %64.5705 legacy sonucu
operasyonel yeniden üretimdir. B70/10-30/updated-style/final-alphaFalse host'u,
B32/5-35/original-style/final-alphaTrue kontrolünden birden çok faktörde
ayrılır; eşlenmiş spectral-off kontrolü olmadan farkın tamamı spectral'e
yazılamaz. İki/üç net doğruya dayanan eski Gaussian/Impulse pilot kazanımı
tam aynı iki bozulmaya genişletildiğinde de kaybolmuştur; yalnız Scan veya
yeni bozulmalar eklenmesiyle açıklanamaz.
[Tarihsel ölçek/host ayrımı](gsd_smoothing_diagnosis_20261001.md).

## 2. Karar neye dayanıyordu; neyi göstermiyordu?

[Run] Sekiz dataset/head hücresini dengeli ortalayan, sabit-katsayılı
70-nesne/üç-seed doğrulamada C-PM %56.8929, CS-PM %56.7222;
C-CS +0.170635 puan, koşullu %95 object-bootstrap aralığı
[-0.0397,+0.3651]. Fresh-encode Gaussian/Background smoke'ta C'nin paylaşılan
reconstruction süresi %24.94 daha düşüktür. Cache-replay karşılaştırma süresi
deployment hızlanması değildir; B1 gecikmesi ölçülmemiştir.
[Sabit doğrulama](guidance_component_confirmation_results_20261025.md).

[Run] Kalibre edilmiş yeni140/üç-seed doğrulamada C-PM %53.013889,
CS-PM %52.962302; C-CS +0.051587 puan, aralık [-0.125000,+0.234127].
Original3DD %53.763889'dur fakat bütün host/readout ayar paketi farklıdır.
Bu iki eşlenmiş all15 aralığı da sıfırı içerir: ne accuracy üstünlüğü ne
eşdeğerlik kanıtlanmıştır. Kalibre katsayı 937.376744'tür; eski1.17'nin
yerine yerel gradient ölçeği artırılmış olmasına rağmen tutarlı CS üstünlüğü
çıkmamıştır. Huber ve per-mode power geliştirmede denenmiş, yalnız seçilen
L2/rho.1 doğrulamaya alınmıştır. Beta/M/graf sabit tutulmuştur.
[Kalibre sonuç, 23 başarılı iş ve korunmuş bir sayısal failure](guidance_component_sensitivity_results_20261028.md).

[User report/Decision] Daha sonra Chamfer+PM, N8/Background35, eta=.003 style,
gamma=.03 local, B32/original-style politikası yeni baseline olarak seçildi.
Bu **bütün ayar paketinin** transfer sonucu ve kullanıcı kararıdır;
N8 üzerinde eşlenmiş spectral-on/off yapılmış olduğu anlamına gelmez.
[Güncel baseline sözleşmesi](current_baseline.md).

## 3. Veri kümesi tek başına açıklama mı?

[Run] Yeni CPU aritmetiği, kabul edilmiş case CSV'leriyle aynı byte/SHA'ya
sahip girdilerden aşağıdaki CS-C Probability Mean farklarını üretir.
Her hücre all15 ve üç seed ortalamasıdır; yeni inference veya hücre başına
anlamlılık iddiası yoktur. Sabit ve kalibre sütunları farklı nesneleri/katsayıları
kullanır; aralarındaki değişim saf katsayı etkisi değildir.

| Dataset / head | Sabit1.17 doğrulama | Kalibre L2 doğrulama |
|---|---:|---:|
| ModelNet / Point-MAE | -0.412698 | +0.047619 |
| ModelNet / DGCNN | -0.253968 | +0.158730 |
| ModelNet / CurveNet | -0.571429 | +0.031746 |
| ModelNet / PointNeXt | -0.126984 | -0.158730 |
| Scan / Point-MAE | -0.285714 | -0.476190 |
| Scan / DGCNN | -0.031746 | +0.142857 |
| Scan / CurveNet | +0.190476 | +0.142857 |
| Scan / PointNeXt | +0.126984 | -0.301587 |
| ModelNet / dört head ortalaması | -0.341270 | +0.019841 |
| Scan / dört head ortalaması | 0.000000 | -0.123016 |

[Inference] Dolayısıyla "ModelNet'te iyi, diğerinde kötü" ayrımı kanıta
uymaz. Aynı dataset'te head'e göre işaret değişir; Scan'de de pozitif hücreler
vardır. Rehberlik classifier-independent olduğu ve decoded cloud head'ler
arasında paylaşıldığı için aynı geometrik değişiklik farklı karar sınırlarını
farklı etkileyebilir. Bu, head'e özgü margin mekanizmasının makul olmasıdır;
margin üzerinden nedensel açıklama henüz ölçülmemiştir.

## 4. Makale ne yapıyor, biz neyi aktarmışız?

[Paper] GSDTTA'nın venue PDF pp.3-5, Eqs.3/5-13/17-18 ve Fig.2/3 görsel
olarak kontrol edildi; deney ve ablation bölümleri ile supplement de okundu.
[CVF ana kaynak](https://openaccess.thecvf.com/content/ICCV2025/papers/Wei_3D_Test-time_Adaptation_via_Graph_Spectral_Driven_Point_Shift_ICCV_2025_paper.pdf),
[erişilebilir primary tam metin](https://arxiv.org/html/2507.18225v1#S3).

| Bileşen | Makale | Yürütülen legacy GSD |
|---|---|---|
| Graf sinyali | fiziksel XYZ | corrupted LION local latent XYZ |
| GFT | C=U^T X | H_ref=U^T Q; H_pred=U^T Y |
| Adaptasyon | öğrenilen düşük-frekans DeltaC | Y-Q spektral farkına ceza |
| Geri dönüşüm | X_s=U(C+DeltaC)=X+U_M DeltaC | skaler loss'un gradienti; sonunda VAE decode |
| Yönü belirleyen hedef | pseudo-label CE, entropy/diversity, tek-yön Chamfer | SCD + corrupted-reference fidelity |
| Sınıflandırıcı | dönüşümlü model adaptasyonu | frozen |
| Modlar | M100, yaklaşık ilk %10 | exp(-2*raw lambda), ilk800/2048=%39.0625 |

[Inference] En önemli kavramsal fark: makale düşük frekansların
**değiştirilmesini öğreniyor**, bizim hedef o değişikliği **sınırlıyor**.
Makaledeki başarı, bu ters amaçlı regularizer'a otomatik olarak taşınmaz.
Makalede iki dataset'te de başarı raporlanır; bizim değişken transferimiz
full GSDTTA'nın dataset başarısızlığı değildir. Fig.2'deki fiziksel sandalye
enerji örneği her corrupted LION latentinde aynı spektral anlamı kanıtlamaz.

## 5. Dönüşüm ve türev denetimi

[Code] `models/latent_points_ada.py` per-point XYZ ve feature'u birleştirip
flatten eder; `view(B,2048,4)[:,:,:3]` doğru yerleşimdir. İncelenen graph
`graph_spectral_v2.py`, loss `spectral_guidance_losses.py`, consumer
`tta_gsd_v2.py`'dır. Aynı U her iki sinyale uygulanır. Gerçek amaç:

```text
E = Y - Q
w_i = exp(-2*lambda_i) if i < 800 else 0
L_spec = sum_b,i w_bi ||[U_b^T E_b]_i||^2 / (3*2048)
F = U diag(w) U^T
dL_spec/dY = 2 F (Y-Q) / (3*2048)
```

[Code/Run] Yeni float64 tanıkta GFT/IGFT roundtrip hatası 4.44e-15,
analitik-gradient farkı 2.78e-17; finite-difference kontrolü geçer.
Katsayı loss'u ile uzay-domain quadratic aynı çıkar. Skaler coefficient
loss'u için ayrıca IGFT çağrısı gerekmez; backward zaten U üzerinden geri
taşır. Batch-sum/3N her örneğin gradientini batch tekrarında korur.
Ortak-baz eigenvector işaret çevirmesi loss'u değiştirmez. Frozen parametreler
input/style türevini kapatmaz; gradient-enabled loss ayrı hesaplanır.
29 mevcut ilgili CPU testi de geçer. CUDA native türevlerinin tam doğruluğu
ve classifier faydası bu kontrollerden çıkarılamaz.

[Code/Run] Spectral fidelity, set eşdeğerliğinden daha güçlü slot eşleşmesi
ister: iki noktanın yerini değiştiren sentetik tanıkta Chamfer=0,
spectral>0'dır. Graf/sinyal/referansı birlikte permute etmek ayrı ve geçerli
bir equivariance koşuludur. Bu tanık LION'da gerçek slot hatası olduğunu
göstermez; latent slot sadakatinin ek varsayım olduğunu gösterir.
Koherent `.3` translation örneğinde spectral loss `.09` olur; corrupted Q
yerine daha temiz bir Y seçmek regularizer tarafından her zaman ödüllendirilmez.
Gerçek pipeline'ın centering'i nedeniyle bu tanık bir deployment translation
sonucu olarak yorumlanamaz.

## 6. Graf tarafındaki gerçek farklar ve kusurlar

[Paper/Code] Eq.10 `tau=gamma*sum(A)/(N*k)` kullanır; legacy
`tau=gamma*sum(A_symmetric)/N` kullanır. **Aynı A sabit tutulursa** k10 için
ikinci eşik10kattır. Uygulamadaki self-edge/symmetrization değişimleri nedeniyle
bütün algoritmalar arasında tam10kat equivalence iddiası yapılamaz.
Makalenin Eq.9 row-mask'i simetriyi kendi başına garanti etmez; max-symmetry
ve iki uç maskesi bizim açık seçimlerimizdir. Literal directed kNN/row-mask'i
`eigh`'a vermek doğru simetrik operator tanımlamaya yetmez.

[Code] Legacy self-query kendi noktasını da k komşuya dahil eder. Self-loop
L=D-A içinde iptal olur ama degree/threshold'da kalır. Sadece kendine bağlı
bir vertex degree>0 olduğu için `degree==0` testi onu isolate saymaz;
ama off-diagonal Laplacian satırı sıfırdır. Bu, tanımlanmış outlier-penalty
amacını bozan gerçek kusurdur. `+1000` isolate diagonal ve `1e-5 I` jitter
makalede yoktur; operator hâlâ simetrik/PSD olabilir, fakat çıplak graph
Laplacian değildir. Doğru eigen-decomposition yanlış graf varsayımını düzeltmez.

[Run] 10 Ekim full15/Point-MAE/seed0 legacy-offdiag kontrolü
%64.5705 -> %64.5543 (-0.0162puan) vermiştir; prepared draws tam eşlenmiş
değildir. Bu yüzden kusuru doğrulamak, accuracy kaybının nedeni olduğunu
doğrulamak değildir. Düzeltme otomatik olarak baseline'a uygulanmadı.

[Code/Run] Hard800 kesimi eşit eigenvalue altuzayını ortadan bölebilir;
iki geçerli tied bazlı yeni tanıkta loss 1/6 ve0 olur. Bu koşulda baz
bağımlılığı gerçektir, fakat gerçek örneklerde sınırda exact tie gösterilmiş
değildir. Salt ortak eigenvector sign farkı sorun değildir. Yeni smooth
formülasyon mean-degree normalization/all-mode ve active-isolate exclusion
kullanır; yürütülen legacy ile aynı yöntem değildir. Önceki smooth testlerinin
null/negatif sonuçları da korunur; hiç denenmemiş gibi sunulmaz.

[Paper/Open] Eq.7 d² yazarken prose d'yi squared Euclidean diye tanımlar;
metin/denklem belirsizliği vardır. Kod standart Gaussian RBF için Euclidean
kNN çıktısını karesine alır. Accessible KNN CUDA kaynaklarında sqrt dönüşü
vardır, fakat bu inceleme tarihsel kurulmuş wheel'i çalıştırmaz. Dördüncü
kuvvet distance hatası veya ters GFT hatası saptandığı iddia edilmez.

## 7. Yeni gerçek-cache CPU ölçümü: aynı ayar aynı filtre değil

[Run] Önceden kabul edilmiş development cache'lerinden dataset başına **bir
orijinal nesne** alındı: ModelNet ID38 ve Scan ID14. Her birinin15bozulması,
toplam30cell; bunlar30bağımsız nesne değildir. Aynı legacy graph forward'u,
açık Euclidean CPU kNN adaptörüyle çalıştırıldı. Native CUDA tie order/parity
iddiası yoktur. SHA'lar ve batch position0 audit JSON'dadır.

| CPU betimsel ölçüm, 15 bozulma | ModelNet ID38 | Scan ID14 |
|---|---:|---:|
| Mean sum(exp(-2*lambda)[:800]) | 442.7556 | 170.1105 |
| En küçük / en büyük weight mass | 112.5077 / 761.6275 | 44.8206 / 299.4208 |
| Kaçırılmış gerçek self-only düğüm olayı | 0 | 7 (5 bozulmada) |

[Run] Seçilen iki nesnede bütün15eş bozulmada ModelNet weight mass'i daha
yükselir. Gaussian referansını yeniden encode etmeden yalnız koordinatlarını
`.5/1/2` ile ölçekleyen kontrollü graf tanığı:

| Gaussian latent coordinate scale | .5 | 1 | 2 |
|---|---:|---:|---:|
| ModelNet ID38 weight mass | 144.9362 | 482.0787 | 799.7222 |
| Scan ID14 weight mass | 68.8715 | 223.2066 | 585.7674 |

[Inference] Sabit delta=.1 uzaklık ölçeği/topolojiyi, raw lambda üzerinde
sabit beta2 ise filter şeklini değiştirir; tek dış katsayı, frekanslara göre
dağılım farkını gideremez. Spectral weight mass bir gradient normu veya
temizlik ölçütü değildir. Bunlar graf ölçek duyarlılığının somut tanıklarıdır;
dataset geneli ortalamaları, accuracy etkisi veya Scan kaybının kanıtı değildir.
34 roundtrip'in tümü finite; float32 en büyük mutlak geri-kurma hatası
8.23e-5'ten küçüktür. Iso1000 nedeniyle operator conditioning'i bu hataya
katkı verebilir; accuracy'de neden olduğu gösterilmedi.

## 8. Neden fayda azalabilir? Kanıt gücüne göre açıklamalar

1. **[Run] Pilot/kohort/readout etkisi.** Önceki pozitifler gerçek ama
   tekrarların işaretleri karışık. Tek pozitif seed veya küçük seçilmiş
   subset kalıcı kazanımı göstermiyor; Point-MAE'de bile tersine dönebiliyor.
2. **[Code/Inference] Bozulmuş hedefe kilitlenme.** L_spec için Y=Q global
   minimumdur. Faydalı düşük-frekans değişimini de cezalandırır. SCD de Q'ya
   bağlar fakat büyük nearest-distance'ları budar ve slot kimliğini zorlamaz.
   İki loss'un toplamı ek temiz bilgi getirmek zorunda değildir. Semantik
   yararlı/yaramaz hareketin gerçek clean-reference/margin ayrımı açık kalır.
3. **[Run/Inference] Büyüklük ve yön farklı konular.** Eski küçük spektral
   normlar az etkiyi açıklayabilir; kalibre937.38 ile null sonuç sürmesi,
   yalnız "loss küçüktü" açıklamasını yeterli kılmaz. Eşit norm farklı
   projector/graf/hedefleri aynı yöne dönüştürmez.
4. **[Run/Inference] Blanket gradient çatışması gösterilmedi.** Eski
   Gaussian/Impulse smooth state'lerinde local negatif cosine yoktur;
   style'da vardır. Style projection'ın all15 artışı yalnız+.036917puandır.
   Bu gözlemler latest legacy için nedensel çatışma teşhisi veya tüm
   bozulmalarda cancellation iddiası sağlamaz.
5. **[Run/Inference] Graf ve domain aktarımı.** Yeni iki-nesne tanığı sabit
   parametrelerin eşit filtre vermediğini gösterir. Sentetik tam-şekil
   ModelNet ile gerçek tarama Scan'in geometri/görünürlük farkları latent
   referans kalitesini etkileyebilir; bunun nicel accuracy payı ölçülmemiştir.
6. **[Code/Run/Inference] Backbone ve ön işleme.** Aynı adapted cloud farklı
   frozen decision boundary/FPS/normalization sözleşmelerine gider. İşaretlerin
   aynı dataset içinde değişmesi dataset-only açıklamasına karşıdır.
   Makalelerin farklı source checkpoint/asset aileleri de çapraz makale
   skorlarını doğrudan hedef saymayı engeller.

## 9. Karar ve ayırt edici sonraki kanıt

[Decision] Mevcut Chamfer+PM N8 baseline'ı koru. Denenen **legacy corrupted-latent
fidelity** için tutarlı ek accuracy katkısı gösterilmediğini yaz; genel graph
spectral yaklaşımı çürüttüğümüzü yazma. Matris çarpımı/gradient hatası varmış
gibi sonuçları geçersiz sayma. Bilinen isolate kusurunu ve negatif düzeltme
deneyini birlikte raporla. Bu rapor yeni parametre araması başlatmaz.

[Open] Devam gerekirse önce saklanan **aynı** latent state'lerde spectral
yönün matched clean reference ve classifier margin'e etkisi ölçülmeli;
spektralin koruduğu modların temiz/corrupt içerik ayrımı test edilmeli.
Mevcut30graf probu, gerçek CUDA operator sözleşmesi veya bu semantik soruyu
yanıtlamaz. Yeni baseline üzerinde spectral katkı sorulacaksa yalnız
aynı N8 host/readout/rastgele girdilerde on/off karşılaştırması bunu ölçer.
Bu gelecekteki öneridir; şu anda bir GPU fazı veya geniş sweep planlanmadı.

Falsifiers: eşlenmiş on/off tekrarlarında tutarlı anlamlı CS üstünlüğü,
gerçek native-KNN distance-unit kusuru, source/state/ID uyuşmazlığı veya
clean-reference ölçümünün spectral yönü sürekli yararlı göstermesi ilgili
yorumları değiştirir. Mevcut bulgular bunları varsaymaz.

## Yeniden üretim ve yayın kapsamı

[Audit script](evidence/graph_spectral_retrospective_20261101/audit.py) ·
[tam CPU tanıkları ve kabul edilmiş CSV yeniden gruplamaları](evidence/graph_spectral_retrospective_20261101/audit.json) ·
[kaynak/primary PDF/komut/test makbuzu](evidence/graph_spectral_retrospective_20261101/verification.json).
Script `--repo`, `--archive-root`, `--output` alır. Archive-root mevcut
`gsd-guidance-components` worktree'sindeki immutable raw/cache kaynaklarıdır;
Git-only checkout cache dosyalarını içermez. Peer dosyalarına yazılmaz.
PDF/render/ham cache ve tekrar dump'ları entegrasyona alınmaz; yalnız rapor,
kompakt kanıt, assert içeren denetim kodu ve ilgili knowledge güncellemeleri
seçilir. Yeni dependency veya üretim/default değişikliği yoktur.
