# CloudFixer atıf adaylarının tarama kaydı

18 Kasım 2026. Bu liste, erişilen iki bağımsız atıf dizininin o gün döndürdüğü **23 farklı başlığı** ve aşağıda ayrı olarak verilen tam metin aramasını kaydeder. Sonuçlar Google Scholar dışa aktarımı değildir. Scholar'da başlık sorgusu doğrudan istekle oturum doğrulaması döndürdü. Normal tarayıcıyla erişim için site izni reddedildi. Scholar sayfa sayısı, küme numarası veya atıf toplamı saptanamadı. Yeniden denetim için [Google Scholar başlık sorgusu](https://scholar.google.com/scholar?q=%22CloudFixer%3A+Test-Time+Adaptation+for+3D+Point+Clouds%22).

## Yeniden üretim

- **OA-E**. OpenAlex'te ECCV DOI kaydı için `filter=cites:W4403770401` sorgusu 17 sonuç döndürdü. Kaydın arama ekranındaki sayaç 18 idi. Tarama için alınan gerçek satırlar kullanıldı. [Sorgu](https://api.openalex.org/works?filter=cites:W4403770401&per_page=100)
- **OA-A**. arXiv DOI kaydı için aynı sorgu bir sonuç döndürdü. Bu sonuç ECCV listesindeki gürültü giderme makalesinin aynısıydı. [Sorgu](https://api.openalex.org/works?filter=cites:W4406073732&per_page=100)
- **S2**. Semantic Scholar, `ARXIV:2407.16193` için 18 sonuç döndürdü. Yanıtta sonraki sayfa yoktu. ECCV DOI uç noktası 404 döndürdü. [Sorgu](https://api.semanticscholar.org/graph/v1/paper/ARXIV:2407.16193/citations?fields=title,externalIds,year,venue,url,openAccessPdf&limit=100)
- Başlıkların küçük harf, boşluk ve noktalama normalizasyonundan sonra iki dizin arasında 12 ortak kayıt kaldı. Ön baskı, bildiri ve aynı makaleyi yeniden anlatan tez bölümleri bağımsız deney sayılmadı. 23 adayın her biri aşağıda görünür. Kaynakça ve sayısal tablo ayrı kanıt türleridir.
- OpenAlex tam metin araması `search=CloudFixer` ile ek olarak 21 sonuç döndürdü. Genel web ve yayınevi aramalarında `"CloudFixer" "ModelNet40-C"`, `"CloudFixer" "ScanObjectNN-C"`, `"CloudFixer-O" test-time adaptation` ve bilinen adayların tam başlıkları kullanıldı. Yayımlanmış ana belge ve ekine öncelik verildi. [Tam metin sorgusu](https://api.openalex.org/works?search=CloudFixer&per_page=100)

| Aday ve kaynak | Dizin | Atıf kanıtı | Nesne top-1 taramasında karar |
|---|---|---|---|
| [Multi-level latent-diffusion adaptation for cross-region point cloud segmentation of ancient terraces](https://arxiv.org/abs/2611.18033) | S2 | Atıf, §1/6 | MELD. Hedef bölge verisini kullanan teras segmentasyonu. |
| [Recovering Black-Box Point Cloud Models from Pose Shifts via Derivative-Free Optimization](https://arxiv.org/abs/2610.34850) | S2 | Atıf [44] | CanDO. Poz değişimi altında skor sorguları. Top-1 için CloudFixer satırı yok. |
| [Test-time adaptation for point cloud denoising under unseen noise](https://arxiv.org/abs/2610.27410) | OA-A, OA-E, S2 | Atıf [27] | Gürültü giderme ve yüzey hatası. |
| [SATOR: Backpropagation-Free 3D Test-Time Adaptation via Sample-Adaptive Token Routing](https://arxiv.org/abs/2610.21367) | OA-E, S2 | Atıf [20] | SATOR. Sınıflandırma tablolarında CloudFixer yok. |
| [Test Time Adaptation Methods for Point Cloud Registration in Laparoscopic Surgery](https://arxiv.org/abs/2608.02883) | OA-E, S2 | Atıf [25] | 3B kayıt. Hizalama hatası ve klinik görüntü koşulları. |
| [MAMVI: 3D Test-Time Adaptation via Masked Multi-View Point Clouds](https://arxiv.org/abs/2606.12939) | OA-E, S2 | Atıf [20] | MAMVI. Sınıflandırma tablolarında CloudFixer yok. |
| [Adapting Point Cloud Analysis via Multimodal Bayesian Distribution Learning](https://arxiv.org/abs/2603.22070) | OA-E, S2 | Atıf [22] | BayesMM. Yedi bozulma ailesi ve görsel-dil modeli. CloudFixer satırı yok. |
| [Revisiting Layer Normalization for Point Cloud Test Time Adaptation](https://doi.org/10.1109/WACV61042.2026.00060) | OA-E, S2 | Atıf [24] | LN-TTA. Sınıflandırma tablolarında CloudFixer yok. |
| [Adapt-As-You-Walk Through the Clouds: Training-Free Online Test-Time Adaptation of 3D Vision-Language Foundation Models](https://arxiv.org/abs/2511.15311) | OA-E, S2 | Kaynakça ve T1–3 | Uni-Adapter. Yayımlanmış aynı-omurga kıyası, ayrıca aynı tablolarda Point-Cache. |
| [Low-Cost Test-Time Adaptation for Robust Video Editing](https://arxiv.org/abs/2507.21858) | S2 | Atıf [25] | Video düzenleme. |
| [3D Test-Time Adaptation via Graph Spectral Driven Point Shift](https://arxiv.org/abs/2507.18225) | OA-E, S2 | Atıf [32], T1–2 ve ek | GSDTTA. İki veri kümesinde aynı çalışmada karşılaştırma. |
| [SMART-PC: Skeletal Model Adaptation for Robust Test-Time Training in Point Clouds](https://arxiv.org/abs/2505.19546) | OA-E, S2 | Atıf [21], T1 | SMART-PC. Çevrimiçi sürümler girdi-temelli CloudFixer değerini geçiyor, eşleşen çevrimiçi CloudFixer daha yüksek. |
| [CodeMerge: Codebook-Guided Model Merging for Robust Test-Time Adaptation in Autonomous Driving](https://arxiv.org/abs/2505.16524) | S2 | Atıf [32] | Otonom sürüşte algılama, nesne sınıflandırma tablosu yok. |
| [APCoTTA: Continual Test-Time Adaptation for Semantic Segmentation of Airborne LiDAR Point Clouds](https://arxiv.org/abs/2505.09971) | OA-E, S2 | Dizin adayı | Havadan LiDAR segmentasyonu. Dizin bağı ile incelenen arXiv sürümündeki kaynakça eşleşmesi ayrı tutuldu. |
| [LiDAR Remote Sensing Meets Weak Supervision: Concepts, Methods, and Perspectives](https://arxiv.org/abs/2503.18384) | OA-E, S2 | Atıf [200] | LiDAR zayıf denetim taraması. |
| [Point-Cache: Test-time Dynamic and Hierarchical Cache for Robust and Generalizable Point Cloud Analysis](https://arxiv.org/abs/2503.12150) | OA-E, S2 | Atıf [46] | Point-Cache. Özgün sınıflandırma tablosunda CloudFixer yok. Uni-Adapter sonraki sayısal kaynak. |
| [AdapTable: Test-Time Adaptation for Tabular Data via Shift-Aware Uncertainty Calibrator and Label Distribution Handler](https://arxiv.org/abs/2407.10784) | S2 | Atıf [46] | Tablo verisi. |
| [Two Simple Principles for Diffusion-Based Test-Time Adaptation](https://arxiv.org/abs/2312.05274) | S2 | Atıf [32] | Görüntü difüzyonu. Güncellenmiş metin atfı, ilk ön baskının 2023 tarihi tek başına dışlama nedeni olmaz. |
| [A Survey on Self-Improving Test-Time Intelligence: Feedback-Driven Adapting, Learning, and Scaling at Inference](https://doi.org/10.48550/arxiv.2609.01679) | OA-E | Atıf [325] | Test zamanında öğrenme taraması. |
| [Robust and generalizable deep geometric representation learning for 3D point clouds](https://espace.etsmtl.ca/id/eprint/3869/1/BAHRI_Ali.pdf) | OA-E | PDF s. 141, 188 | Bahri tezi. SMART-PC tablosunu ve kaynakçasını içerir. Ayrı doğrulama sayılmadı. |
| [SMOTE3D: Geometry‐ and color‐aware volumetric data augmentation for 3D fashion asset segmentation](https://doi.org/10.4218/etrij.2025-0458) | OA-E | Yayıncı kaynakçası | SMOTE3D. Eğitim verisi artırımı ve moda varlıklarının segmentasyonu. |
| [Geometry-Guided Diffusion SAR Point Cloud Denoising](https://doi.org/10.3390/rs18152458) | OA-E | Atıf [38] | SAR binalarında yüzey gürültüsü, CD/P2M. |
| [Backpropagation-free adaptation of vision transformers for robust and efficient 3D and 2D perception](https://espace.etsmtl.ca/id/eprint/4122/1/YAZDANPANAH_Moslem.pdf) | OA-E | PDF s. 87 ve kaynakça | Yazdanpanah tezi. İlgili çalışma ve yazarın normalizasyon/token deneyleri. Yeni bir CloudFixer kıyası saptanmadı. |

## Dizin birliğine ek arama

- **LoTT-PC**. CVPR yayınevi makalesinin §2 ve kaynakça [26] kaydı açık bir atıf. Tablo 1'de CloudFixer satırı bulunmuyor. İki ileri atıf sorgusu bu çalışmayı döndürmedi. Bu örnek tek bir dizindeki sonuç sayısına güvenmenin sınırını gösterir. [Yayın](https://openaccess.thecvf.com/content/CVPR2026/papers/Ye_Low-Rank_Test-Time_Training_for_Pre-Trained_Point_Cloud_Models_CVPR_2026_paper.pdf)
- OpenAlex sözcük araması, başlangıç makalesinin iki sürümünü, makalenin yer aldığı ECCV cildini, TTA taksonomisini ve *4D Point Cloud Segmentation via Active Test-Time Adaptation* çalışmasını da döndürdü. Cilt ve sürümler yeni atıf değildir. Taksonomi özgün sınıflandırma deneyi vermez. 4D çalışması CloudFixer adını Zhang ve arkadaşlarıyla ilişkilendirir. Metin eşleşmesi, Shim ve arkadaşlarına yapılmış kaynakça atfı olarak kabul edilmedi. [4D çalışma](https://doi.org/10.1609/aaai.v40i26.39277) · [Taksonomi](https://doi.org/10.2139/ssrn.7456938)
- *Test-Time Adaptation With an Explicit Geometric Bridge* sahne segmentasyonu inceler. Tam metinde atıf kimliği ayrıca doğrulanmadı. *PTA*, *Purge-Gate*, *SVWA* ve özgün *3DD-TTA* yakın yöntemler olarak elle incelendi. Doğrulanmış ileri atıf sayısına eklenmediler. [TTA-EGB](https://pubmed.ncbi.nlm.nih.gov/42842505/) · [Geometri yöntemleri](../../papers/cloudfixer_geometric_comparisons_20261118.md) · [Vizyon-dil yöntemleri](../../papers/cloudfixer_foundation_comparisons_20261118.md)

Yeni bir sayısal çalışmanın dahil edilebilmesi için makale adı, CloudFixer atfı ve ölçüm yapılan tablonun veri kümesi, omurga, varyant ve metriği gerekir. Dizinler güncellendiğinde aynı sorgular farklı sonuç verebilir.
