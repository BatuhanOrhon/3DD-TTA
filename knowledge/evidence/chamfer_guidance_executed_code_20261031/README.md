# Çalıştırılmış guidance kaynaklarına ait audit kopyaları

[Code/Run] 2026-10-31. Bu dizin, ilgili source ZIP'lerden ve korunmuş kontrol
scriptlerinden **byte değiştirmeden** alınmış46dosyayı içerir:43Python+3JSON.
[Manifest](manifest.json) her dosyanın byte/SHA256 kimliğini, orijinal göreli
yolunu ve iki tam source ZIP'nin kimliğini kaydeder. Toplam kaynak590057byte.
Python3.8 syntax denetimi yapıldı; dosyalar çalıştırılmadı/import edilmedi.
İki `tta_gsd_v2.py` kopyasındaki tarihsel trailing whitespace SHA kimliğini
korumak için aynen bırakıldı; yeni belge metinlerinin whitespace kontrolü geçer.

- [original_0fd2581](original_0fd2581/tta_gsd_v2.py): ana kabul edilen GPU
  accuracy/timing kaynağının9ilgili üyesi.
- [curvenet_95a4997](curvenet_95a4997/tta_gsd_v2.py): CurveNet GPU kaynağının
  aynı9üyesi + yeni rapor validator'ı ve kalibratörü.
- [historical_controls](historical_controls/remote/gpu_chamfer_phase.py):
  house/common helper, ilk controller ve metadata-only devamı; gerçek
  launcher'lar; CurveNet controller/launcher/planı; native C/C ve0H tanıları;
  yerel confirmation/timing kabul scriptleri. Bazı helper/planlar ilk a1d298a
  aşamasına aittir; metadata-only controller sonraki0fd source'a bağlar.

## Güncel çalıştırılabilir girişler

Üç ayrı adayın ana girişleri repoda zaten vardır:
[sparse](../../../scripts/eval_chamfer_sparse.py),
[KL/H](../../../scripts/eval_chamfer_semantic.py),
[weighted](../../../scripts/eval_chamfer_weighted.py).
[Campaign hazırlayıcısı](../../../scripts/run_chamfer_variants_campaign.py)
CPU plan/komut üretir; GPU worker launcher'ı değildir. Bu ayrımı arşivlenen
controller'larla karşılaştır. Kalibrasyon ve analiz yollarının tamamı
[yöntem–sonuç rehberinde](../../chamfer_guidance_methods_results_20261031.md).

## Sınırlar ve yeniden kullanım

[Code/Open] Bunlar **okuma/audit referansıdır**, yeni bir runtime'da taşınabilir
tek-komut replay paketi değildir. Bazı yollar `/content/studies/...` altındaki
artık kapanmış VM'ye, eski plan/SHA/approval ve dış helper girdilerine sabittir.
Copy dizininin `__file__`/import yerleşimi üretim modül ağacı değildir;
buradan script çalıştırma. Eski onay yeni GPU koşusunu yetkilendirmez.

Tam source ZIP, asset manifest/checkpoint/native kimlikleri, ham inference
arşivleri ve304fiziksel cache Drive/yerel arşivlerde korunur;
[arşivler ve kapanış](../../chamfer_guidance_final_handoff_20261030.md).
Buraya binary/PDF/checkpoint/cache/credentials veya ham CLI auth history alınmadı.
Dosyanın sealed ZIP üyesi olması, içindeki her fonksiyonun GPU'da çağrıldığının
kanıtı değildir; gerçek çağrı/provenance raw config/diagnostics ile doğrulanır.

[Code] Birleştirilmiş8dca857 kodu ile bu iki dondurulmuş source farklı olabilir.
Yeni guard/depth API değişikliğini geçmiş deneyde çalışan hata diye yorumlama;
önce hangi revision'ın arşivi ürettiğini belirle. Bağlı kalibrasyon kimliklerini
gevşetmeden incele. Yeni belgeler/snapshot kopyaları matematik veya accuracy
sonuçlarını değiştirmez.
