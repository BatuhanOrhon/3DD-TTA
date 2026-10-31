# Denenen guidance yöntemleri: makale, uygulama ve sonuç rehberi

2026-10-31. [Paper/Code/Run/Inference/Open] Bu kayıt, 30 Ekim literatür
incelemesini tamamlanan Chamfer-only deneyleriyle birleştirir. Yeni deney,
hata tespiti veya kapsamlı yeni literatür taraması değildir. İlk öneri ile
son uygulama farklıysa **çalıştırılmış uygulama** aşağıda ayrıca belirtilir.

## Okuma sırası ve kanıt

1. [Nihai teslim, Drive ve kapanış](chamfer_guidance_final_handoff_20261030.md).
2. [Accuracy, güven aralıkları ve süre](chamfer_guidance_variants_results_20261030.md).
3. [CurveNet takip deneyi](scan_curvenet_entropy_followup_20261030.md).
4. [İlk literatür ve doğrudan 3DD-TTA atıfları](diffusion_guidance_literature_review_20261030.md).
5. [Yazar kodu denetimi](chamfer_author_code_consistency_audit_20261030.md).
6. [Çalıştırılmış kaynak kopyaları ve manifest](evidence/chamfer_guidance_executed_code_20261031/README.md).

[Run] Ortak kontrol C: B70, normal10/Background30 DDIM güncellemesi,
eta=gamma=.01, p=.95. Her classifier için üç decoded ara görünüm ve terminalin
**probability mean** sonucu; dört classifier accuracy'sinin ortalaması bir
model ensemble oyu değildir. LION/classifier/BN/LN değerleri ve buffer'lar
sabit; spectral/invariant/GraphSpectralDNA hesapları sıfırdır. Fiziksel R/Delta
optimizasyonu yoktur. Bu host, özgün 3DD-TTA'nın ayrı5/35 terminal protokolü
değildir. Her ekleme C üzerine **ayrı** uygulanmıştır; KL ile weighted birlikte
çalıştırılmamıştır.

## Makaleden alınan fikir ve gerçekten denenen uygulama

| Kol | Makale öncülü | Bizde uygulanan | Makalenin aynısı olmayan bölüm |
|---|---|---|---|
| Sparse backward | [Adaptive Guidance](https://arxiv.org/abs/2312.12487), §5; hız bağlamında [CloudFixer](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04603.pdf), §4.2 | Aynı10/30 timestep ve başlangıç noise; interval3, son adım dahil4/11 gerçek guidance backward. Bütün prior forward'lar ve mevcut mean probe'ları korunur. | AG, CFG'nin ek network forward'unu adaptif bırakır; bizim sabit SCD-backward çizelgemiz AG reprodüksiyonu değildir. CloudFixer'ın detached hedefli R/Delta optimizasyonu taşınmadı. |
| Symmetric KL | [MAMVI](https://arxiv.org/abs/2606.12939), §3.2/Eq.4 | Bozuk X'ten bir kez alınan detached deterministik FPS referansı q ile mevcut decoded logits p arasında .5*(KL(p||q)+KL(q||p)); nesne başına hesap, batch boyunca toplam. Son iki mevcut probe'da guidance. | MAMVI'nin masked12-view consensus'u, BN/LN affine update'i ve adaptif LR'si yok; gradyan yalnız decoded branch üzerinden local/style'a gider. |
| Marginal entropy H | [GDA](https://arxiv.org/abs/2404.00095), §3/Eqs.7–8 ve MAMVI'nin semantik fikri | Her nesnede H((p_decoded+p_X)/2), aynı iki probe, aynı sabit-X referansı; logits/decode mevcut mean akışından tekrar kullanılır. | GDA'nın16 AugMix görünümü, style/content loss'ları ve weighted-log-fusion branch'i yok. MAMVI'nin kendi loss'u değildir. İlk önerideki iki decoded-FPS görünümü yerine kullanıcıyla seçilen fixed-X/decoded çifti kullanıldı. |
| Density-confidence weighted SCD | CloudFixer §4.2, Eq.1 ve yazar `adapt.py` | Encoded h0'ın ilk3 **latent** koordinatında nonself k10 mean squared distance tersi; epsilon1e-6, mean1, detached ağırlık. p=.95 trim uzaklık sırasını korur; iki yönde doğru referans ağırlığı/NN gather uygulanır. | Fiziksel nokta yoğunluğu veya CloudFixer displacement regularizer'ı değildir. CloudFixer default self dahil k5/sum1 ile farklar kasıtlıdır; yoğunluk güvenilirliğin kanıtı değildir. |
| Scan CurveNet-H | Aynı GDA/MAMVI esinli H; bağımsız yeni makale yöntemi değil | Yalnız Scan'de frozen CurveNet, H gradyanını üretir; üç diğer head transfer readout'udur. Aynı fixed-X/decoded loss ve mevcut probability mean; ayrı etiketsiz Scan kalibrasyonu. | Yeni source model **ve** katsayı paketidir; yalnız teacher kalitesinin nedensel etkisi izole edilmedi. Aynı140ID tekrar kullanıldığı için posthoc/keşifseldir. |

[Code/Run] Semantik loss normalde5/8, Background'da15/23 loop sayımlarında
etkindir. Referans probability mean'e ek oy olmaz. Ek iş yalnız X-forward
ile sınırlı değildir: decoder/classifier üzerinden local/style'a backward da
vardır. Her adımda model parametresi değil giriş latentleri güncellenir.

[Code] Ana girişler: [sparse](../scripts/eval_chamfer_sparse.py),
[KL/H](../scripts/eval_chamfer_semantic.py), [weighted](../scripts/eval_chamfer_weighted.py).
Ortak uygulama: [sampler](../tta_gsd_v2.py), [geometrik loss/schedule](../chamfer_guidance_variants.py),
[semantik observer/loss](../semantic_trajectory.py), [paylaşılan runner](../scripts/eval_multibackbone_guidance.py),
[preprocessing/frozen head/cache](../scripts/eval_multibackbone_source.py),
[trajectory observer](../shared_trajectory.py).
[Asset/source hazırlama](../scripts/prepare_chamfer_variants_assets.py) ve
[CPU campaign planı/komut üretimi](../scripts/run_chamfer_variants_campaign.py)
GPU worker çalıştırıcısı değildir; gerçek controller/launcher kopyaları audit
paketindedir.

## Kabul edilmiş sonuçlar

[Run] All15/severity5; **140 farklı nesne/veriseti**, seed0/1/2; her head6300
karar. Aşağıdaki farklar dört head accuracy ortalamasında kendi C'sine göre
**yüzde puan**. %95 CI, nesne-cluster2000 bootstrap/seed0, seçili seed/head'e
koşulludur; çoklu karşılaştırma düzeltmesi yoktur. C accuracy MN66.9008%,
Scan45.0833%. Geliştirme70 tablosu ayrı tutulur; bu tabloya karıştırılmaz.

| Kol | MN delta pp [95%CI] | Scan delta pp [95%CI] | MN / Scan süre farkı |
|---|---|---|---|
| Sparse | −1.3571 [−1.7818,−.9524] | −.9683 [−1.2540,−.6904] | −21.15% / −21.07% |
| KL | +.1627 [−.0952,+.4049] | −.0516 [−.2660,+.1707] | +7.49% / +8.16% |
| H | +.3254 [+.0794,+.5714] | +.0833 [−.1071,+.2778] | +7.43% / +8.16% |
| Weighted | −.7262 [−1.0357,−.4325] | −.5040 [−.7303,−.2619] | +.67% / +.45% |

[Inference] Weighted sonucu bu kapsamda negatiftir; **belirsiz olan KL'nin
katkısıdır**, weighted ile KL'nin birleşimi değildir. KL'nin iki aralığı da
sıfırı kapsar. Sparse hız kazancı açık bir accuracy kaybıyla gelir. H'nin MN
ortalamasında küçük olumlu işareti vardır; genel ve güçlü katkı iddiası yoktur.
Point-MAE özelinde H MN+.1746pp/Scan+.3333pp; bunlar dört-head tablo değildir.

[Run] Ek CurveNet-H birincil endpoint: Scan CurveNet'in kendi mean accuracy'si
**55.6667→55.9048%, +.2381pp [−.1587,+.6349]**. Seed0/1/2 doğru sayıları
1165→1166,1168→1176,1174→1180 /2100. PointNeXt transfer+.4127pp ve dört-head
ortalama+.1944pp ikincildir; birincil CI sıfırı kapsadığı için güçlü kazanım
kanıtı sayılmaz. Yeni matched C kullanıldı; eski Point-MAE C arşivi taşınmadı.

[Run] Süreler dev70/B70/seed0, **Gaussian+Background**, üç döndürülmüş fresh
tekrarda eşlenmiş arm/C oranlarının medyanıdır; süre medyanlarının oranı
değildir. Ortak reconstruction bir kez ve dört terminal head sayılır;
startup, bulk loading, paketleme/Drive hariçtir. All15/B1/tek-head deploy
gecikmesi değildir. CurveNet-H ek G+B maliyeti8.5572%; C20.955446s,
H22.759866s medyan; Gaussian+13.0753%, Background+6.0390%.

[Run] Kalibrasyonlar accuracy'ye göre seçilmedi: ilk iki veri kümesinin dev70
pooled update normlarıyla KL=.6285361973358488, H=1.4715424107401223;
weighted eta/gamma ölçeği=.9171509895707142. CurveNet-only Scan H=3.153885581354952.
Semantikte hedef median update oranı.1, weighted'da C ile norm eşleme;
nesne başına güvenlik tavanı değildir. [İlk kalibrasyon](../scripts/calibrate_chamfer_variants.py),
[CurveNet kalibrasyonu](../scripts/calibrate_scan_curvenet_entropy.py),
[eski validator](../chamfer_calibration_report.py), [yeni validator](../scan_curvenet_entropy_report.py),
[rapor analizi](../scripts/analyze_chamfer_variants.py).

## İncelenen fakat bu kampanyada uygulanmayan çalışmalar

[Paper/Inference/Open] Bu envanter 30 Ekim incelemesinin durumudur; 31 Ekim'de
bu yayınlar için yeni tam metin taraması yapıldığı anlamına gelmez.

| Referans | İncelenen fikir / bu kampanyadaki durum |
|---|---|
| [SATOR](https://arxiv.org/abs/2610.21367) | Token/geometri partition, entropy routing; BN_reset ve token akışı koşulumuzdan farklı. Salt causal erken çıkış dördüncü **gelecek test** olarak kayıtlı, uygulanmadı. |
| [TTAD](https://arxiv.org/abs/2610.27410) | Residual GMM/noisy–noisier denoiser adaptasyonu; frozen-model koşulunu değiştirir, denenmedi. |
| [LoTT-PC](https://openaccess.thecvf.com/content/CVPR2026/papers/Ye_Low-Rank_Test-Time_Training_for_Pre-Trained_Point_Cloud_Models_CVPR_2026_paper.pdf) | LoRA/feature alignment; etkin model parametreleri güncellenir, denenmedi. |
| [Uni-Adapter](https://arxiv.org/abs/2511.15311) | Prototype cache/graph label smoothing/logit fusion; yeni stream belleği/karar kuralı, denenmedi. |
| [SteeringTTA](https://openreview.net/pdf?id=iV0WBCDGcx) | Confusion-set/SMC steering; ek trajectory maliyeti, denenmedi. ICML workshop kaydıdır. |
| [GSDTTA](https://arxiv.org/abs/2507.18225) | Tam yöntem fiziksel spectral shift ve classifier adaptasyonu içerir. Eski GSD-inspired deneyler ayrı kayıtlarda; mevcut kampanyada spectral hesap yok. |
| [GDSD](https://www.mdpi.com/2072-4292/18/15/2458) | SAR building denoising, ek eğitim/veri; hazır LION guidance transferi sayılmadı, denenmedi. |

[Open] SA-P2SB ve robotics bin-picking önceki atıf envanterindedir; burada
tekrar tam yöntem denetimi veya yeni sonuç iddiası yoktur. Eski probability
mean/Purge-Gate/sampling-consistency ve PixelAsParam/GSD kayıtları için
[attribution](gsd_trajectory_attribution_20261025.md),
[method synthesis](method_synthesis.md),
[önceki component sonuçları](guidance_component_sensitivity_results_20261028.md)
ve [matematik denetimi](guidance_mathematics_execution_audit_20261030.md).
Yeni agent bunları yeniden yeni yöntem diye sunmamalı.

## Kaynak, yazar kodu ve uygulama denetimi

[Paper] Temel yayın: Dastmalchi et al.,
[Test-Time Adaptation of 3D Point Clouds via Denoising Diffusion Models](https://arxiv.org/abs/2411.14495),
WACV2025. Ana öncüller: Castillo et al.,
[Adaptive Guidance: Training-free Acceleration of Conditional Diffusion Models](https://arxiv.org/abs/2312.12487);
Shim, Kim, Yang, [CloudFixer: Test-Time Adaptation for 3D Point Clouds via Diffusion-Guided Geometric Transformation](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04603.pdf), ECCV2024;
Kong, Jung, Jung, [MAMVI: 3D Test-Time Adaptation via Masked Multi-View Point Clouds](https://arxiv.org/abs/2606.12939);
Tsai et al., [GDA: Generalized Diffusion for Robust Test-time Adaptation](https://arxiv.org/abs/2404.00095), CVPR2024.
31 Ekim'de bu dört öncülün birincil başlık/özet veya PDF erişimi kontrol edildi;
GDA CVF sayfaları403 verdi, arXiv kaydı erişildi. Geniş yeni tarama sonraki
görevdir; yöntem/denklem/yazar-kod karşılaştırmaları önceki pinli denetimdendir.

[Code] Yazar pinleri: CloudFixer f043e24414fcf4165fdc892abff21215c9989141,
MAMVI 0b553973b7123da6198b79bba4ff704b8eec914c,
GDA 4197ff4e43b03b896830f5e70980c7e613be5126. Ayrıntılar:
[CloudFixer/MAMVI](evidence/chamfer_author_code_audit_20261030/cloudfixer_mamvi_audit.md),
[GDA](evidence/chamfer_author_code_audit_20261030/gda_audit.md),
[Adaptive Guidance](evidence/chamfer_author_code_audit_20261030/adaptive_guidance_audit.md).
AG için bulunan resmi repo website kaynağıdır; resmi sampler kodunun
incelendiği iddia edilmez. Yöntem farklılığı otomatik implementasyon hatası değildir.

[Code/Run] Ana kabul edilen GPU deney kaynağı0fd2581; CurveNet kaynağı95a4997.
Erken başarısız smoke/native tanı a1d298a'dır. Birleştirilmiş8dca857 ağacı
CPU'da479test/373subtest ve69pinned NumPy1 depth testiyle doğrulandı;
GPU tabloları bu birleştirilmiş ağacın koşusu değildir. Eski kalibrasyonun
kaynak/native bağını kaldırmak yerine gerektiğinde yeni seal/kalibrasyon gerekir.
[Çalıştırma controller'ları, launcher'lar ve kabul scriptleri](evidence/chamfer_guidance_executed_code_20261031/README.md)
artık SHA doğrulanan Git kopyalarıdır; üretim kodu yerine audit referansıdır.

[Run/Artifact]407raw ZIP+304NPZ, bilimsel özet ve Drive roundtrip kanıtları
[nihai teslimde](chamfer_guidance_final_handoff_20261030.md) bulunur. Raw/weights,
PDF, native binary veya credentials bu yeni Git paketine alınmadı. Sahipli
Colab ve monitor kapalıdır; bu dokümantasyon işi yeni GPU çalışması değildir.

[Open] Sonraki görev: [kısa agent promptu](chamfer_guidance_audit_literature_handoff_20261031.md).
Loss işareti/reduction, preprocessing/gradyan yolu, timestep/style güncellemesi,
weighted NN/trim, kalibrasyon ölçeği, cache/ID eşlemesi ve metrikleri gerçek
snapshot üzerinde tekrar denetle; hata bulma zorunluluğu yoktur. Onaylı kasıtlı
uyarlama, kanıtlanmış kusur, protokol farkı ve gerçek negatif sonucu ayır.
