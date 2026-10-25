# İncelenen çalışmalarda source-only ve 3DD-TTA accuracy — 2026-10-25

[Paper/Run] Hücre biçimi **source-only / 3DD-TTA terminal accuracy (%)**. `—`, incelenen eşleşen tabloda sayısal sonuç olmadığını veya bizim seçili envanterimizde tam koşulun bulunmadığını gösterir; sıfır değildir. Source ve3DD sayıları kendi çalışma/checkpoint koşullarıyla eşleşir. Aynı backbone adı altında farklı source eğitimleri, ağırlıklar, FPS/normalization, batch ve severity/ortalama politikaları vardır; bu bir ortak protokol sıralaması değildir.

Bu odaklı görünüm16 sayısal yayını ve bizim dört backbone/iki dataset ayarımızı kapsar. [147 koşulluk tam tablo](evidence/source_3dd_comparison_20261025/source_3dd_all_settings.md) ve [CSV](evidence/source_3dd_comparison_20261025/source_3dd_all_settings.csv), tekrar eden ana/ek tabloları ve tüm kaynak satır kimliklerini korur.128 eski source gözlemine11 ek materyal kaynağı eklendi;139 literatür source gözlemi ve8 yerel ayar vardır. Bir literatür satırı macro-recall olup accuracyye çevrilmez. Sayısal3DD değerleri özgün 3DD/GSDTTA/Uni-Adapter’daki 8 satır ve bizim 7 terminal sonucumuzla sınırlıdır.

| Çalışma | Backbone | Source eğitimi / koşul | ModelNet40-C SO /3DD | ScanObjectNN-C SO /3DD | ShapeNet-C SO /3DD |
|---|---|---|---:|---:|---:|
| Biz | CurveNet | seed0; original3DD terminal | 64.01 / 67.07 | 52.07 / 55.24 | — / — |
| Biz | DGCNN | seed0; original3DD terminal | 58.80 / 65.59 | 32.78 / 36.44 | — / — |
| Biz | Point-MAE | seed0; tarihsel eval/raw terminal | 53.69 / 63.71 | — / — | — / — |
| Biz | PointNeXt | seed0; original3DD terminal | 65.83 / 68.29 | 49.09 / 49.68 | — / — |
| 3DD-TTA | Point-MAE | makaledeki kaynak | 57.60 / 65.70 | 61.50 / 64.30 | 59.30 / 69.80 |
| GSDTTA | CurveNet | source-only | 71.38 / — | 53.06 / — | — / — |
| GSDTTA | DGCNN | source-only | 66.51 / 71.69 | 55.72 / 57.08 | — / — |
| GSDTTA | PointNeXt | source-only | 66.99 / — | 50.66 / — | — / — |
| GSDTTA | RSCNN | source-only | 63.54 / — | 49.66 / — | — / — |
| Uni-Adapter | OpenShape | source-only | 67.43 / — | 30.10 / — | 54.35 / — |
| Uni-Adapter | ULIP-2 | source-only | 45.00 / — | 33.05 / — | 42.22 / — |
| Uni-Adapter | Uni3D-Large | source-only | 59.15 / 56.06 | 38.07 / 30.78 | 57.92 / 55.30 |
| MATE | Point-MAE | Joint-Training | 57.60 / — | 45.60 / — | 59.30 / — |
| MATE | Point-MAE | Source-Only | 53.90 / — | 45.70 / — | 57.60 / — |
| MATE | Point-MAE | Source-Only; target batch128 | 53.90 / — | — / — | — / — |
| BFTT3D | CurveNet | 100−classification error | 71.53 / — | 35.23 / — | — / — |
| BFTT3D | DGCNN | 100−classification error | 65.35 / — | 53.91 / — | — / — |
| BFTT3D | PointNet | 100−classification error | 64.12 / — | 40.65 / — | — / — |
| BFTT3D | ek tabloda backbone belirtilmemiş | 100−classification error | — / — | — / — | 76.56 / — |
| SVWA | CurveNet | main/supp source | 64.00 / — | — / — | — / — |
| SVWA | DGCNN | main/supp source | 58.80 / — | 37.90 / — | 63.60 / — |
| SVWA | Point-MAE | main/supp source | 53.90 / — | 33.10 / — | 61.30 / — |
| SVWA | PointNet | main/supp source | 46.50 / — | 29.70 / — | 49.80 / — |
| CloudFixer | Point-MAE | tam 15; kaynak | 53.70 / — | — / — | — / — |
| CloudFixer | Point-MAE | yalnızca 6 corruption | 43.13 / — | — / — | — / — |
| CloudFixer | Point2Vec | tam 15; kaynak; tek/iid/ablation aynı source değeri | 62.09 / — | — / — | — / — |
| CloudFixer | Point2Vec | zamansal sıralı akış | 60.38 / — | — / — | — / — |
| CloudFixer | PointMLP | tam 15; kaynak | 53.53 / — | — / — | — / — |
| CloudFixer | PointNeXt | tam 15; kaynak | 67.03 / — | — / — | — / — |
| SMART-PC | Point-MAE | MATE-SO (tam eğitim); tam yeniden eğitim | 54.40 / — | 29.80 / — | — / — |
| SMART-PC | Point-MAE | MATE-SO (tek yıldız); ayrıntılı tablolar | 53.70 / — | 34.50 / — | 56.50 / — |
| SMART-PC | Point-MAE | MATE-SO (tek yıldız); kompakt Table1 | 54.40 / — | 34.50 / — | 56.50 / — |
| SMART-PC | Point-MAE | Org-SO* | 54.00 / — | 37.00 / — | 61.30 / — |
| SMART-PC | Point-MAE | SMART-PC-SO | 61.70 / — | 38.70 / — | 64.50 / — |
| Purge-Gate | Point-MAE | source-only | 54.16 / — | 36.96 / — | 61.35 / — |
| Purge-Gate | Point-Mamba | source-only | 58.32 / — | 29.11 / — | 51.92 / — |
| LN-TTA | Point-MAE | source-only | 54.16 / — | 36.96 / — | 61.35 / — |
| MAMVI | Point-MAE | source-only | 54.13 / — | 36.96 / — | 61.35 / — |
| SATOR | Point-MAE | source-only | 54.16 / — | 36.96 / — | 61.35 / — |
| LoTT-PC | Point-MAE family | LoTT-PC-SO; tüm severity ortalaması beyanı | 74.40 / — | 52.00 / — | 66.40 / — |
| LoTT-PC | Point-MAE family | MATE-SO [17]; tüm severity ortalaması beyanı | 57.60 / — | 45.60 / — | 59.30 / — |
| LoTT-PC | Point-MAE family | Org-SO [17]; tüm severity ortalaması beyanı | 53.90 / — | 45.70 / — | 57.60 / — |
| LoTT-PC | Point-MAE family | SMART-PC-SO [2]; tüm severity ortalaması beyanı | 61.70 / — | 38.70 / — | 64.50 / — |

**Koşul notları:** Bizim Point-MAE ModelNet satırı tarihsel eval/raw seed0 operasyonel 3DD referansıdır; diğer eski Point-MAE kontrol koşulları [mevcut deney matrisinde](modelnet40_c_all15_results_matrix_20261001.md) ayrı tutulur. Scan Point-MAE için bu seçili envanterde tam 15 source/original3DD çifti yoktur. GSD readout değerleri3DD sütununa konulmamıştır. SMART-PC kompakt tek-yıldız etiketleri ayrıntılı adaptation-only/tam-retraining etiketlerini sıkıştırır; varyantlar ayrıdır. LoTT captionı tüm severityleri ortalama der; baseline özelinde yeniden yürütme kanıtı değildir. GSD ek materyal RSCNN adı§1ile belirlenir; Table1captionında DGCNN yazması ayrı bir metin tutarsızlığıdır.

## Orijinal benchmark: beş severity ortalaması

[Paper/Code] Buradaki accuracyler ordinary error üzerinden100−error ile türetildi. Tek severity5 sonuçlarıyla birleştirilmedi. Hiçbirinde sayısal 3DD sonucu yok.

| Çalışma | Backbone | ModelNet40-C source-only | 3DD-TTA |
|---|---|---:|---:|
| ModelNet40-C benchmark | PointNet | 71.70 | — |
| ModelNet40-C benchmark | PointNet++ | 76.40 | — |
| ModelNet40-C benchmark | DGCNN | 74.10 | — |
| ModelNet40-C benchmark | RSCNN | 73.80 | — |
| ModelNet40-C benchmark | PCT | 74.50 | — |
| ModelNet40-C benchmark | SimpleView | 72.80 | — |

## Yedi atomik corruption: ayrı VLFM protokolü

[Paper] Bu satırlar15-corruption ModelNet40-C protokolüyle aynı değildir. Point-Cache basılı Avg yedi corruption üzerinden, BayesMM Avg genel olarak temiz+yedi üzerinden raporlanır. Basılı değerler aynen korunur; bazı aritmetik uyuşmazlıklar [geniş auditte](pointcloud_tta_reporting_audit_20261025.md) açıklanır. Bu çalışmalarda incelenen tablolarda3DD-TTA accuracy yoktur.

| Çalışma | Backbone | ModelNet-C source | Scan OBJ_ONLY-C source | Scan OBJ_BG-C source | Scan PB_T50_RS-C source |3DD-TTA|
|---|---|---:|---:|---:|---:|---|
| Point-Cache | ULIP / PointBERT | 47.52 | 40.82 | 34.99 | 23.97 | — |
| Point-Cache | ULIP-2 / PointBERT | 57.95 | 36.47 | 36.61 | 26.37 | — |
| Point-Cache | OpenShape / scaled PointBERT (32.1M point encoder) | 73.49 | 44.18 | 45.81 | 31.98 | — |
| Point-Cache | Uni3D-Giant (1016.5M point encoder) | 67.95 | 55.69 | 50.75 | 37.38 | — |
| BayesMM | ULIP / PointBERT | 48.60 | 41.85 | 36.36 | 24.64 | — |
| BayesMM | ULIP-2 / PointBERT | 59.61 | 37.16 | 38.04 | 27.25 | — |
| BayesMM | OpenShape / scaled PointBERT (32.1M point encoder) | 74.87 | 45.31 | 47.08 | 33.12 | — |
| BayesMM | Uni3D-Giant (1016.5M point encoder) | 69.69 | 56.93 | 51.94 | 38.46 | — |

## Accuracy olmayan source satırı ve sayısal kapsam dışı yayınlar

CloudFixer mainTable3 Point2Vec source **59.94 macro-recall** raporlar; sourceaccuracy gibi gösterilmez. Tam CSV bu satırı yerel metriğiyle korur. İncelenen diğer 11 yayın için [kapsam kaydı](evidence/source_3dd_comparison_20261025/nonnumerical_coverage.json) tutulur: sekizinde hedef corruption-classification source/3DD satırı yoktur; SA-P2SB,SARdenoising ve robotics-bin-picking tam metinleri incelenemediği için sonuçları bilinmiyor olarak kalır.

## Kaynaklar ve doğrulama

Her sayısal satırın primary URL, PDF sayfası, tablo, metric, source-evidenceID ve 3DD-evidenceID alanları tamCSVdedir. Başlıca önceki kaynaklar [823satırlık audit](pointcloud_tta_reporting_audit_20261025.md), [72satırlık ilk atıf matrisi](evidence/3dd_tta_citation_audit_20261025/evidence_matrix.csv) ve [kabul edilmiş yerel sonuçlar](gsd_multibackbone_final_findings_20261019.md). Uni-AdapterULIP-2/OpenShape kaynakları tamarXivPDFp13Table8den alınmıştır; p-valuetablosuaccuracy olarak kullanılmaz.

[Code/Run/Open] Bu bir kayıt birleştirmesidir; yeni GPU/Colab/inference veya yeni dünya-geneli literatür taraması yapılmadı. Önceki immutablearşivler değiştirilmedi. Primarytablo değerleri assetbit-parity veya bağımsız reprodüksiyon kanıtı değildir.
