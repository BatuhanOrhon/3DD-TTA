# Chamfer guidance: gerçek kaynak denetimi ve yeni adaylar — 31 Ekim 2026

[Decision] İncelenen sparse, KL/H ve weighted uygulamalarında kabul edilmiş
sonuçları geçersiz kılan **doğrulanmış bir implementasyon hatası bulmadım**.
Bu, bütün native türevlerin doğru olduğu veya hiçbir hata bulunamayacağı
iddiası değildir. En somut yeni teşhis, semantik kalibrasyonun nesne bazındaki
kuyruklarıdır: hedef medyan .1 iken kalibre edilmiş KL/C oranı en çok7.0093,
H/C3.7240 olabilir. Yanlış sabit-X referansı ve güncelleme öncesi oyların
ortalamada tutulması da katkıyı sınırlayabilecek mekanizmalardır; gerçek accuracy
kaybının nedenleri oldukları henüz gösterilmedi.

[Inference] Yeni aday olarak yalnız iki sınırlı aktarım öneriyorum:
**nesne başına semantik update tavanı** ve **seyrek gerçek backward arasında
gradyan tahmini**. İlki accuracy, ikincisi accuracy–süre dengesi hipotezidir.
Hiçbiri uygulanmadı, seçilmedi veya GPU'da çalıştırılmadı. Bunlar ilgili
makalelerin tam reprodüksiyonu ya da özgünlük iddiası değildir.

**1. Checkout ve kanıt sınırı**

[Code] Kullanıcının worktree'si
`/Users/batuhan/.codex/worktrees/7bf0/3DD-TTA`; yerel branch
`integration/chamfer-guidance-variants-20261030`. Başlangıç HEAD
`a82200adbe117ce56a134dc3c22601cb73dc2654`; çalışma ağacı temizdi. `git fetch
origin` sonrasında `origin/thesis-integration` aynı commit'teydi; fark yoktu.
Branch değiştirilmedi. Bu görev rapor/kanıt aşamasındadır; commit veya push
yapılmadı. Üretim kodu ve ham arşivler değiştirilmedi; Colab/oturum/GPU komutu
çalıştırılmadı, başka agent'ın runtime'ına dokunulmadı.

[Code/Run] Başlangıç rehberindeki bütün zorunlu notlar, tez kapsamı, protokol,
repository map, method synthesis ve ilgili findings/open-questions okundu.
Tarihsel kaynak, güncel kod ve yazar uyarlaması ayrı tutuldu:

- **P0**: `knowledge/evidence/chamfer_guidance_executed_code_20261031/original_0fd2581/`;
  kabul edilen ana GPU kaynağı `0fd258110abf48dc92d739b288010ebfbee475b2`.
- **P1**: aynı audit kökünün `curvenet_95a4997/` dizini;
  CurveNet kaynağı `95a49977ef7c3124bde5faa57e3ef36b4393ebcc`.
- İki gerçek mühürlü source ZIP'nin SHA/byte kimliği yeniden doğrulandı;
  P0'ın9 ve P1'in11 seçili üyesi ZIP üyeleriyle byte-identik. Audit manifestinin
  controller/planlar dahil46 dosyası da yeniden SHA kontrolünden geçti.
- Güncel sampler, `chamfer_guidance_variants.py`, ilk kalibratör ve source
  preprocessing P0 ile aynı. Sonraki farklar CurveNet teacher desteği, analiz
  desteği ve ayrı depth çalışmasının API/guard'larıdır; bunları geçmiş GPU
  koşularında çalışmış değişiklikler diye yorumlamadım.

[Artifact] Bu worktree'de eski `result/preparation/chamfer_guidance_variants_20261030`
dizininin bulunmaması deney kusuru değildir. Korunmuş raw kaynak salt okunur
kullanıldı:
`/Users/batuhan/Desktop/projects/3DD-TTA-worktrees/chamfer-guidance-variants-20261030/`.
Bu kökün altındaki `result/preparation/chamfer_guidance_variants_20261030/`
dizini aşağıda **PREP** diye kısaltılmıştır. Ham dosyalar taşınmadı veya üzerine
yazılmadı; yeni türetilmiş kanıtlar bu worktree'dedir.

Etiketler: **[Code]** kaynak gözlemi, **[Run]** arşivlenen koşu/yeniden sayım,
**[Paper]** birincil yayın, **[Inference]** yorum/aktarım, **[Open]** doğrulanmamış
nokta; kullanıcı talimatları **[User report/Decision]** olarak ayrılır.

**2. İstenen implementasyon denetimi**

[Code] Aşağıdaki satırlar, açıkça belirtilmedikçe P0'a aittir. P1 sampler ve
geometrik loss aynı byte'lardır; P1 semantik loss aynı formülleri korur.
[Gerçek kaynak indeksi](evidence/chamfer_guidance_executed_code_20261031/README.md)
dosyalara ve manifestlerine erişim sağlar.

| Denetim | Kaynak/satır ve izlenen davranış | Sonuç / etkilenen kaynak |
|---|---|---|
| Loss işareti ve reduction | `semantic_trajectory.py:33–48`, `tta_gsd_v2.py:175–181,253–265,282–291`: simetrik KL veya nesne içi H; sınıflar/nesneler üzerinde sum; toplam loss gradyanı çıkarılıyor | İşaret tersliği veya nesneler arası probability marginalization yok. CPU finite difference ve batch tekrarı kontrolü geçer. P0/P1 |
| Detach ve live input yolu | `semantic_trajectory.py:86–95`: local/style `clone`, detach yok; kaydedilen logits ayrı detach edilir. Runner `:432–460`: yalnız guidance head differentiable, X referansı detached | Readout detach'ı loss graph'ını kesmiyor. Bilerek sabitlenen q, yanlışlıkla detach edilmiş p değildir. P0; P1'de CurveNet'e açık seçim |
| FPS/normalizasyon | Runner `:328–369,415–460`; source `scripts/eval_multibackbone_source.py:248–271`; güncel runner `:661–663` | Decode çerçevesi probe/terminalde ortak. FPS indis seçimi ayrık, gather seçilen koordinatlara gradyan taşır. Scan CurveNet centroid/L-infinity yolu korunur. Yerel toy test native türevin tamamını doğrulamaz |
| Timestep ve sparse | `tta_gsd_v2.py:57–66,140–186`: aynı100'lük DDIM scheduler'ın son10/30 adımı; her prior forward var, yalnız1/4/7/10 veya1/4/…/28/30 backward | Gerçek4/11 backward;10/30 forward. Native arşiv sayacıyla uyumlu. Rate'i×3 yapmamak kayıtlı tercihtir; dose eşitliği iddia edilmez. P0/P1 |
| Style ve probe sırası | Sampler `:150–169,289–303`; semantik observer `:91–104` | Probe update öncesi p'yi kaydeder; sonra local eta/style gamma güncellenir. Updated final style decode'a gider. Son adım/scheduler/style güncellemesi kaybolmuyor. P0/P1 |
| Weighted NN ve trim | `chamfer_guidance_variants.py:42–61,64–87`; sampler `:105–110,253–265` | Q→Y kendi referans ağırlığını; Y→Q native nearest-reference gather'ını kullanır. Önce mesafeye göre trim, sonra weight. Nonself k10, squared distance, mean1/detach açık tercihler. Ters indis, weighted-sort veya ek batch mean kusuru bulunmadı |
| Katsayı ve kalibrasyon | `scripts/calibrate_chamfer_variants.py:27–42,64–91,121–150`; P1 `scripts/calibrate_scan_curvenet_entropy.py` ve ayrı validator | Combined norm gerçekten local eta/style gamma ile hesaplanıyor. `autograd.grad(...,retain_graph=True)` ölçümü .grad alanına yazmıyor. Etiketsiz katsayı kuralı doğru; nesne bazlı cap sağlamıyor |
| Cache/ID ve etiket | Runner `:372–412,657–710,745–756`; analyzer `scripts/analyze_chamfer_variants.py:187–234,287–350` | Cache girdi/RNG/tensor hash'lerine bağlı.144 arşivin paired başlangıç hash'leri,140-ID sırası, etiketleri ve frozen-state kayıtları yeniden eşleşti.304 fiziksel NPZ'nin byte/SHA, finite float32 ve tensor hash'leri geçti |
| Metrik/head eşlemesi | Analyzer `:287–350`; `scripts/analyze_legacy_trajectory.py:17–37` | Her head için softmax→dört-zaman ortalaması→argmax; sonra accuracy ortalaması. Head logits'leri karıştırılmıyor.2000 object-cluster bootstrap bağımsız yeniden üretildi |

[Code] Diagnostic `no_grad` Chamfer hesabı ayrı, gerçek guidance hesabı gradient
açıkken yeniden yapılıyor; ilk bloğu görmek “loss detached” sonucuna yetmez.
İki native Chamfer çağrısı maliyet taşır ama backward'ın eksik olduğu anlamına
gelmez. Bu ortak iş yükü eski süre tablolarında zaten vardır.

[Code] Tarihsel controller'lar da source ayrımını destekliyor:
`historical_controls/remote/metadata_recovery_v3/gpu_chamfer_metadata_recovery_v3.py:30–52,125–139`
yalnız source producer'ın `elif`→`if` metadata düzeltmesini ve yeni seal'i kabul
eder; başka loss/rate/data değişikliğine izin vermez. Ana controller
`remote/gpu_chamfer_phase.py:103–128,175–185,229–234,265–274` gerçek arm,
head, native-gradient ve kalibrasyon bağını denetler. CurveNet controller
`postcampaign_curvenet_h/scan_curvenet_h_controls.py:87` itibarıyla yeni
C/H job'larını ve ayrı katsayıyı üretir. Eski C sonuçlarını yeni teacher
baseline'ı yerine koyan bir metrik eşlemesi bulunmadı.

[Code/Paper] Yazar CloudFixer `f043e244.../adapt.py:97–178`, MAMVI
`0b553973.../tools/tta.py:368–637` ve GDA `4197ff4e.../main.py` yeniden
connector üzerinden indirildi; üç dosya önceki pinli indirmelerle aynı byte/SHA'dır.
[Yazar audit'i](chamfer_author_code_consistency_audit_20261030.md) korunur:
CloudFixer fiziksel displacement regularizer'ı, MAMVI masked-consensus+BN/LN
adaptasyonu, GDA AugMix marginalization'ı bizim loss'larımızla aynı değildir.
Bu kasıtlı farklar “yazar kodunu yanlış kopyalama” bulgusuna çevrilemez.

**3. Accuracy katkısını sınırlayabilecek, hata olmayan bulgular**

1. **Sabit yanlış X referansı semantik düzeltmeyi ters yöne çekebilir.**
   `[Code/Inference]` P0/P1 `semantic_trajectory.py:41–48` için
   `H((p+q)/2)=.5H(p)+.5H(q)+JS(p,q)`; q sabitken yalnız p'nin confidence'ı
   ve q'ya yaklaşımı optimize edilir. İki sınıflı CPU örneğinde q yanlış sınıfta
   keskin iken H'yi azaltan küçük adım doğru sınıf olasılığını
   `.59868766→.59376945` düşürdü. Loss minimizasyonu doğrudur; semantik
   doğruluk garantisi yoktur. Bu örnek P0/P1 loss'larında mümkündür;
   gerçek140-ID kohortunun hata nedenini veya görülme sıklığını ölçmez.
   X referansının güvenilirliğini doğrulamak ayrı tanıdır; aynı H/KL'yi yeni
   isimle yeniden önermiyorum.

2. **Medyan .1 hedefi, .1 güvenlik tavanı değildir.**
   `[Run/Code]` Dondurulmuş4200-oran raporu ve gerçek katsayılar yeniden
   çarpıldığında KL/C median/p90/max `.1/.741362/7.009261`, H/C
   `.1/.505746/3.723950`. Bunlar C trajectory'sindeki kalibrasyon noktalarıdır;
   H/KL ile değişen trajectory'nin gerçekleşmiş update dağılımı değildir.
   P0'ın bütün semantik kollarına ait ölçektir. P1'in yeni katsayısı3.15388558
   ayrı Scan/CurveNet raporudur; bu P0 kuyrukları P1'e taşınmaz.
   Global katsayı hesabında hata bulmadım; per-object baskın güncellemeler
   katkıyı heterojenleştirebilir. Negatif cosine tek başına zarar kanıtı değildir.

3. **Mevcut mean'in ilk iki oyu semantik update'den önce oluşuyor.**
   `[Code/Inference]` Normal probe'lar3/5/8/terminal10; ilk semantik update5
   numaralı probe kaydedildikten sonra yapılır. Dolayısıyla ilk iki oy bu
   semantik update'den etkilenemez; step8 oyu step5'in etkisini taşıyabilir,
   step8 update'i ancak sonraki state/terminali etkiler. Background8/15/23/30
   için aynı mantık geçerlidir. Bu, korunması istenen readout sözleşmesidir,
   off-by-one kusuru değildir. Aynı noktayı tekrar decode ederek oyu değiştirmek
   hem ek maliyet hem yeni karar kuralı getirir; bu görevde önerilmiyor.

4. **Sparse daha az fidelity düzeltmesi; weighted farklı bir fidelity yönü.**
   `[Code/Inference]` Sparse'da atlanan adımlarda hem local hem style update
   sıfırdır. Düşüş, yanlış timestep'ten kaynaklanmak zorunda değildir. Weighted
   ise bozuk h0 latent yoğunluğunu güven sinyali sayar; bu yoğunluk temiz
   fiziksel yüzeyin veya doğru sınıf parçasının güvenilirliği değildir.
   Mean1 ve norm kalibrasyonu yön değişimini geri almaz. Bu tasarım hipotezleri,
   gözlenen negatif sonucu hata diye silmek için gerekçe sağlamaz.

[Open] Native CUDA bu görevde tekrar yürütülmedi. FPS/komşuluk indisleri için
parçalı/surrogate türevler, PVCNN voxelization'daki coordinate-gradient sınırı
(`third_party/pvcnn/functional/voxelization.py:41–43`) ve Chamfer atomicAdd'leri
(`third_party/ChamferDistancePytorch/chamfer3D/chamfer3D.cu:166–171`) mevcut.
Atomik toplama nondeterminism için makul kaynak olabilir; tek başına mevcut
accuracy farklarının nedeni olduğunu kanıtlamaz. Önceki native zero/nonzero
kanıtları korunur; CPU toy doğrulamasını tam native Jacobian doğrulaması diye
sunmuyorum. Yeni hata düzeltmesi zorunlu olduğu sonucuna varmadım.

**4. Ham sonuçların bağımsız yeniden sayımı**

[Run] PREP `original_combined_physical_manifest.json` içindeki120 confirmation
head ZIP ve `postcampaign_curvenet_h/heldout_receipt_local_raw_manifest.json`
içindeki24 yeni head ZIP yeniden SHA/byte doğrulandı.4.320 batch'in logits'i
doğrudan okundu;15 corruption×140nesne×3seed kapsamı, source/head durumu,
başlangıç hash'leri ve kaydedilmiş terminal prediction'ı kontrol edildi.
Süre ZIP'lerinden accuracy çıkarılmadı. Her ana head/kol için6300 karar vardır;
aynı140nesne seed/corruption boyunca tekrarlanır. Dört-head satırı25200 olayı
ortalayan accuracy'dir; bir ensemble değildir.

| Bağımsız kol | MN dört-head delta pp [koşullu95%CI] | Scan dört-head delta pp [koşullu95%CI] |
|---|---:|---:|
| Sparse | −1.3571 [−1.7818,−.9524] | −.9683 [−1.2540,−.6904] |
| KL | +.1627 [−.0952,+.4049] | −.0516 [−.2660,+.1707] |
| H | +.3254 [+.0794,+.5714] | +.0833 [−.1071,+.2778] |
| Weighted | −.7262 [−1.0357,−.4325] | −.5040 [−.7303,−.2619] |

[Run] Yeni CurveNet matched C/H: birincil kendi mean'i
`55.6667→55.9048%`, `+.238095pp`, CI`[−.158730,+.634921]`;
94 düzeltilen/79 bozulan event, net15/6300. İkincil dört-head
`+.194444pp`, CI`[+.003968,+.404861]`. Aynı140ID'nin tekrar kullanımı
posthoc'tur; birincil CI0'ı kapsar. Sayımlar önceki raporla eşleşir.

[Run] Eski fresh B70 Gaussian+Background üç-tekrar süre sonucunu değiştiren
bir kanıt bulunmadı: sparse yaklaşık−%21; KL/H yaklaşık+%7–8; weighted
<+%1; CurveNet-H+%8.5572. Bu görev bunları yeniden zamanlamadı. Bunlar
startup/loading/archival hariç ortak reconstruction+dört terminal head
maliyetidir; full15/B1/tek-head deploy gecikmesi değildir.

[Code/CPU] İlgili yedi test dosyası: **101 passed,1 skipped,91 subtests passed**,
10.37s; Python3.13/Torch2.15/NumPy2.5, CPU. Skip yalnız tarihsel snapshot'ın
bu worktree'deki eski konumda bulunmamasıydı. Aynı default Point-MAE preview
karşılaştırması gerçek korunmuş source yoluyla ayrıca geçti. Yeni CPU kanıtı,
P0/P1 KL/H finite difference, sum/per-object batch scaling, descent işareti,
nonidentity-NN/unequal-N weighted trim türevi ve unit-weight kontrolünü içerir.
Yeni pretrained forward/backward veya CUDA inference yoktur.

**5. 31 Ekim birincil literatür ve forward-citation yenilemesi**

[Paper/Open] ArXiv/WACV3DD kimliği aynı yayın olarak tutuldu. OpenAlex
`cites:W4409262973`4, Semantic Scholar arXiv-target12 kayıt döndürdü
(ikisi de bu sefer HTTP200). Bunlar eksiksiz sayım değildir ve4+12 diye
toplanamaz. İki listenin birleşimi önceki envantere yeni doğrudan atıf eklemiyor.
Başlık,3DD-TTA ve24Ekim sonrası sorgularında daha yeni uygulanabilir bir
doğrudan atıf doğrulanamadı; “başka yayın yok” sonucu çıkmaz. Scholar'ın önceki
engellerini aşmaya yönelik işlem yapılmadı. Altı ilgili primary PDF yeniden
indirildi ve kaynakçaları kontrol edildi:

| Yayın / birincil kaynak | Doğrulanmış tarih/sürüm ve PDF atfı | Bu kısıtlar altında sonuç |
|---|---|---|
| [SATOR](https://arxiv.org/abs/2610.21367v1) |15.10.2026v1; arXiv'de WACV2027 kabul beyanı; PDFs9 ref[5] |Token/attention/geometri routing; resmi BN_reset yolu frozen-buffer koşulumuzla aynı değil. Eski early-exit fikrini yeni yöntem diye önermiyorum |
| [TTAD](https://arxiv.org/abs/2610.27410v1) |20.10.2026v1; CAD203:104182/2027 journal reference ayrı; PDFs15 ref[33] |Residual GMM ve denoiser-parametre adaptasyonu; tam yöntem frozen-model koşuluna uymaz |
| [MAMVI](https://arxiv.org/abs/2606.12939v1) |11.06.2026v1; ICPR2026 kabul kaydı; PDFs14 ref5 |Masked multi-view consensus+BN/LN affine update. Bizde denenmiş fixed-X KL/H onun reprodüksiyonu değildir; olumsuz sonuç MAMVI'yi çürütmez |
| [LoTT-PC](https://openaccess.thecvf.com/content/CVPR2026/papers/Ye_Low-Rank_Test-Time_Training_for_Pre-Trained_Point_Cloud_Models_CVPR_2026_paper.pdf) |CVPRHaziran2026 ss31472–31481; PDFs9 ref[5]; ayrı ilk preprint tarihi doğrulanmadı |LoRA/feature alignment modelin etkin parametrelerini değiştirir. Resmi kod bağlantısı bu taramada doğrulanamadı |
| [Uni-Adapter](https://arxiv.org/abs/2511.15311v2) |19.11.2025v1→20.11v2; AAAI2026; PDFs8 Dastmalchi2025 |VLFM prototype cache ve yeni fusion kararı; mevcut dört-zaman mean'i koruyan drop-in loss değil |
| [GSDTTA](https://arxiv.org/abs/2507.18225v1) |24.07.2025v1; ICCV2025; PDFs10 ref[33] |Tam yöntem classifier adaptasyonu/spectral shift içerir; kapsam dışı. Yeni spectral hesap önerilmedi/çalıştırılmadı. Resmi uygulama bu taramada doğrulanmadı |

[Code] Güncel resmi repo pinleri doğrulandı: SATOR
`iskong17/SATOR@e65983b60a83936e1e5241d339475436f1a1738d`, MAMVI
`iskong17/MAMVI@0b553973b7123da6198b79bba4ff704b8eec914c`, TTAD
`zuowenzhitiao/TTAD@0bf06d53b73b4324fa2dbf6f809c88ddf5505e4e`, Uni-Adapter
`Mehran-TAM/Uni-Adapter@cf0d8b17aefccbb8860643a24f3df050c7c2d844`.
İlk ikinin pinleri önceki audit'le aynı. Son ikide repository/source ağacı
varlığı doğrulandı; tüm uyarlama kodunun çalıştığı/reproduce edildiği iddia
edilmez. PDF metadata/kaynakça doğrulaması, yazar benchmark'ının bağımsız
reprodüksiyonu değildir. Web okuyucusunun LoTT403/Uni12.8MB sınırına karşı
normal doğrudan primary indirme200 ile tamamlandı; erişim kontrolü aşılmadı.

[Paper] Daha geniş güncel taramada şu iki2026 öncül yeni aktarım gerekçesi verdi:

- [EmbedOpt](https://arxiv.org/abs/2602.05285v2), ilk05.02.2026,
  v2=14.05.2026; §3/Alg1–2 ve Eq3, koşullama embedding'ine frozen denoiser
  üzerinden gradyan verir ve RMS normalize eder. Resmi
  [sampler](https://github.com/rs-station/embedopt/blob/fa27c519ff4f7de5d1cacf0125698a88006b6b0c/src/embedopt/sampler/embedopt.py#L298)
  `:298–314` bunu açıkça uygular. **3DD zaten style güncelliyor**; bunu yeni
  fikir saymıyorum. Alınabilecek yeni kontrol, örneğe göre update büyüklüğünü
  yönetmektir; protein likelihood başarısı3D sınıflandırma kazancı değildir.
- [RACER](https://arxiv.org/abs/2608.01740v1),03.08.2026v1,
  §Method/Eqs2–3: iki cache tahmininin disagreement'ına göre shrinkage ve
  refresh/repay; resmi
  [kod](https://github.com/LiZaiyuan0619/RACER/tree/eea38110687cd281d8fb9a0c97dff1f02058bf9f)
  ve `src/pipelines/sd3_forward.py`, `src/utils/__init__.py` incelendi.
  Makale feature forecasting yapar; bizim gradient forecasting aktarımı
  yeni bir hipotezdir, yazar algoritması veya teoreminin doğrudan uygulaması değil.

[Paper/Inference] [Trust Sampling](https://will-s-h.github.io/trust-sampling-website/)
(NeurIPS2024; yazar kodu `will-s-h/trust-sampling@04fc144...`) loss guidance'ın
güven bölgesini ayrıca gerekçelendirir; tam algoritması çoklu iç adım/model
kontrolü içerir, aşağıdaki basit cap ile eşdeğer değildir.
[CacheScope](https://arxiv.org/abs/2610.12530v1),08.10.2026, §3,
cache riskinde girdi hareketi ile network hassasiyetini ayırır. PVCNN'e DiT
cache kodunu doğrudan kopyalamak veya prior'u detach edip aynı style türevini
beklemek gerekçeli değildir. Resmi CacheScope kod bağlantısı doğrulanmadı.

[Paper/Inference] [Calibrated Test-Time Guidance](https://proceedings.mlr.press/v306/geyfman26a.html)
(ICML2026; preprint25.02.2026) gerçek posterior örneklemesini reward
maksimizasyonundan ayırıyor. SCD/fixed-X H'yi otomatik olarak doğru likelihood
sayamayız; consistent posterior estimator'ı bu taskta accuracy garantisi vermez.
[MMD Guidance](https://proceedings.mlr.press/v306/sani26a.html) hedef referans
dağılımı ister; mevcut bozuk X'i temiz referans sayarak yeni yöntem önermedim.
Bu iki çalışma seçilen adaylar değildir.

**6. Yeni aday A — örnek bazında sınırlanan semantik update**

[Inference] Mevcut C ve H loss'larını, aynı frozen guidance head'i ve yalnız
5/8 veya15/23 semantik noktalarını koru. Her nesne için aynı canlı state'te
ayrı C/H local/style gradyanları çıkar; gerçek rate ile
`uC_b=(eta*gC_local, gamma*gC_style)` ve `uH_b` birleşik update'lerini ölç.
Yeni prospective kural:

```text
a_b = min(a_frozen, 0.1 * ||uC_b|| / (||uH_b|| + eps))
update_b = uC_b + stopgrad(a_b) * uH_b
```

Sıfır/near-zero C için semantik ekleme0; nonfinite değer başarısızlıktır.
Bu cap küçük gradyanı büyütmez; eski sabit katsayının yalnız büyük kuyruklarını
sınırlar. Gradient conflict projeksiyonu, spectral loss, yeni teacher veya
entropi-ağırlıklı readout yoktur. Mevcut mean'in oyları/sırası aynı kalır.

[Code/Inference] Entegrasyon noktası sampler `tta_gsd_v2.py:267–291`, loss'ların
ayrı gradient extraction'ı ve kalibrasyonun mevcut
`update_gradient_rows` norm mantığıdır. Bu bir bug fix olarak sunulmamalıdır:
eski kalibrasyon tasarlandığı gibi çalışıyor. “Denenmiş H'ye yeni isim” yerine
bağımsız değişken açıkça **global sabit katsayı→per-object cap** olur.

[Inference/Open] Accuracy gerekçesi: kayıtlı ağır kuyruğu azaltırken küçük
semantik yönü tutmak. Yanlış q problemi devam eder; cap kazanım garantisi vermez.
Süre gerekçesi: yeni decode/classifier forward gerekmeyebilir ama C ve H'nin
ayrı vector-Jacobian pass'leri gerekir. Mevcut tek toplam-backward kadar ucuz
olduğunu iddia edemeyiz. Norm arithmetic maliyeti tek başına toplam overhead'i
anlatmaz; başlıca risk ek backward süresi ve yararlı büyük update'i kesmektir.

[Inference] Minimum ayırt edici çalışma: matched **C / sabit-H / capped-H**;
sabit-H ve capped-H aynı split-gradient altyapısıyla yürüsün. İlk teknik kontrolde
cap kapalıyken aynı loss/update sözleşmesi doğrulansın; native bitwise eşitlik
varsayılmasın. Yeni norm dağılımı, cap etkinleşme oranı, corrected/harmed,
Point-MAE'nin kendi probability mean'i birincil, dört-head transfer ikincil
raporlansın. AdayA, capped-H−sabit-H etkisi ve C'ye göre net accuracy/cost ile
değerlendirilir; yalnız H düşüşü başarı sayılmaz. `.1` tek önceden sabit öneri,
yeni katsayı grid'i veya CurveNet kaynak taraması yoktur.

**7. Yeni aday B — gerçek sparse backward arasında gradyan tahmini**

[Inference] Aynı10/30 forward ve aynı4/11 gerçek guidance noktasını koru;
seyrek yöntemin aradaki **sıfır update**'lerini geçmiş gradient/update cache'iyle
doldur. Her fresh noktada local ve style update vektörlerini sakla. Aradaki
adımlarda bir son-değer tahmini ile son iki fresh noktadan logSNR üzerinde
lineer tahmini karşılaştır; disagreement büyüdükçe lineer tahmini son-değere
yaklaştır. İlk iki fresh kayıt oluşana kadar yalnız son-değer kullanılır.
Basit ilk prospective sürümde refresh/repay veya eşik araması yoktur;
örneğin `kappa=1/(1+r)` ile shrinkage açıkça sabitlenebilir. Bu seçim yazar
RACER formülünün aynısı değildir ve kullanıcı tarafından seçilmiş sayılmaz.

[Code/Inference] Sampler `:140–186,289–291` noktalarında fresh/cache update
ayrımı gerekir. Nonfresh adımda prior forward `no_grad` kalır; scheduler prev
state'ine cache'deki local update ve style'a cache'deki style update uygulanır.
Terminal ve mevcut üç probe aynı kalır. Yeni semantik/weighted/spectral bileşen
eklenmez. Model parametreleri, BN/LN, native FPS ve decoder algoritması korunur.

[Inference/Open] Fark, zaten denenmiş sparse schedule'ı tekrar adlandırmak
değildir: o schedule arada düzeltme yapmıyordu; yeni hipotez **tahmin edilmiş
yoğun düzeltme**dir. Gerçek prior backward yine4/11; vector cache/aritmetik ve
her-adım update maliyeti eklenir. Full C'ye göre daha az backward tasarruf
potansiyeli taşır; sparse'ın−%21 ölçümünü bu adaya aktaramayız. NN/FPS sınırı
ve style değişimi gradient'i aniden döndürebilir; feature forecasting için
RACER teoremi gradient'e kendiliğinden geçmez. Bu yüzden A'dan daha risklidir.

[Inference] Minimum ayırt edici çalışma: **full-C / sparse-zero / sparse-hold /
sparse-forecast-shrink**. Son ikisi yeni; bütün kollar aynı yeni paired blokta
veya kanıtlı birebir uyumlu eski kontrolle karşılaştırılır. Önce küçük teknik
trace'te gradient norm/yön hatası ve native nonfinite kontrolü; sonra tek
forecast kuralı dondurulur. Hold vs forecast tahminin, zero vs hold eksik
düzeltmeyi doldurmanın etkisini ayırır. Başarı ölçütü full-C/sparse'a karşı
ölçülmüş accuracy–süre cephesidir; çağrı sayısı tek başına speedup değildir.

**8. Ortak deney sınırı, karar ve kanıt dosyaları**

[Decision/Open] Bu iki tasarım öneridir; şu an bir implementation veya GPU
fazı yetkilendirmez. Sonuçları görülen140ID tekrar bağımsız confirmation diye
kullanılmamalı. Önce bütün geçmiş ID manifestlerinin birleşimi çıkarılmalı;
gerçekten ayrılabilen yeni nesne sayısı kontrol edilmeli. Özellikle Scan581'de
kalan havuz varsayılmamalı; başka devam çalışmalarının ID'leri de hesaba
katılmalıdır. Harici görülmemiş veri iddiası için ayrı dış benchmark gerekir.

[Inference] Sonraki onaylı çalışmada severity5/all15 birincil, macro14 ve
Background ayrıca; sabit checkpoint'ler, frozen classifier/BN/LN, aynı
preprocessing, h0/z0/noise/RNG ve dört-zaman mean korunur. İlk teknik/development
seed0; kural dondurulduktan sonra nesne-disjoint confirmation seed0/1/2 ve
object-cluster CI gerekir. Birincil head/endpoint ve pratik maliyet/kayıp marjı
çalıştırmadan önce seçilmeli. Üç seed, üç kat bağımsız nesne sayısı değildir.
Mevcut corruption-name10/30 horizon'u unknown-corruption deployment diye
sunulamaz. Fresh B70 throughput ile B1/tek-head latency ayrı ölçülür.

[Decision] Önceliğim **A**: mevcut arşivde ölçülmüş bir ölçek sorununu tek
değişkenle sınar. **B** ikinci, daha yüksek riskli verimlilik adayıdır.
Weighted'ı yeniden yoğunluk etiketiyle, KL'yi yeni isimle, temporal mean'i,
H+JS sampling'i, entropy-weighted mean'i, logit-mean'i, early-exit'i veya
PixelAsParam projeksiyonunu yeni öneri diye sunmadım. Bilinen sonuçlar ve
olumsuz/null bulgular korunur. GPU süresi ölçülmeden dakika tahmini verilmedi;
olası sonraki fazın bütün GPU bütçesi ayrıca hazırlanmalıdır.

[Artifact] Yeni kanıt kökü:
`knowledge/evidence/chamfer_guidance_audit_20261031/`:
`cpu_checks.py/.json`, `recount_raw.py`, `raw_recount.json`,
`cache_verification.json`, `source_zip_verification.json`,
`raw_source_bindings.json`, `default_preview_parity.json`,
`author_retrieval.json`, `author_pins.json`, `citation_verification.json`,
`literature_retrieval.json`, `validation.json`.
Ham primary PDF/text ve indirilen yazar dosyaları, bu worktree'nin ignored
`result/preparation/chamfer_guidance_audit_20261031/` dizinindedir; orijinal
deney arşivleri PREP ve canonical result konumlarında korunur.

[Open] Kontrol edilmeyenler: yeni CUDA/pretrained inference, gerçek aday
accuracy/süresi, bütün native koordinat Jacobian'ları, yeni Drive roundtrip,
ham corrupt-array/checkpoint'lerin yeniden indirilmesi ve bütün forward-citation
evreni. Risk: doğru çalışan loss bile yanlış hedefi optimize edebilir; küçük
koşullu kazanım ve CI0'ı kapsayan CurveNet sonucu genellenebilir başarı değildir.
