# Tez ve yayın hazırlığı değerlendirmesi — 7 Kasım 2026

[Inference/Decision] Mevcut kanıt, GSD üstünlüğünü merkez alan yeni-yöntem makalesini desteklemiyor. Buna karşılık kontrollü deney, olumsuz sonuç, yeniden üretilebilirlik ve trajectory readout analizi üzerine bir yüksek lisans tezi için ciddi içerik var. Tez kabulü kurum/danışman/jüriye; makale kabulü hedef yayın ve hakemlere bağlıdır. Bu bir kabul garantisi veya kurum yeterlilik kararı değildir.

## Kaynak ve inceleme sınırı

[Code] Ana checkout gsd-multibackbone-study@5d0b89b eskiydi; kullanıcı değişiklikleri korunarak origin fetch edildi. Değerlendirme origin/thesis-integration@a9b807a üzerinden ayrı research/publication-review-20261107 checkout'unda yapıldı. Güncel indeks, aşağıdaki sonuç raporları ve probability readout/baseline kodu incelendi. Yalnız eski checkout kullanılsaydı, tamamlanmış kontroller yanlışlıkla eksik sayılacaktı.

[Run] Aşağıdaki sayılar kabul edilmiş arşiv analizlerinin knowledge kayıtlarından aktarılır. Bu inceleme tüm ham ZIP'leri yeniden hash'lememiş, inference tekrarlamamış veya yeni güven aralığı hesaplamamıştır. [Paper] TEDM birincil PDF ve tek-trajectory voting öncülü yeniden erişildi. [Inference] Yayın değerlendirmesi ve önerilen öncelik benim yorumumdur; [Open] dış genelleme ve özgünlük taraması sınırlıdır. Yeni GPU/Colab işlemi yoktur.

## Deneylerin şu an söylediği

| Kanıt | Sonuç | İddia sınırı |
|---|---|---|
| [11 Ekim Point-MAE trajectory](gsd_trajectory_implementation_20261011.md) | all15 terminal64.5921 → PM65.1567, +.5646pp; macro14 +.7959pp | Tam nesne kapsamı, seed0; macro14 all15 yerine geçmez. |
| [25 Ekim ModelNet üç başlık](modelnet_multibackbone_trajectory_handoff_20261025.md) | DGCNN/CurveNet/PointNeXt PM kazancı +.7347/+.9130/+.8860pp | 2468nesne/all15/seed0; tarihsel original3DD'den yaklaşık1pp aşağı, farklı bütün politika. |
| [19 Ekim Scan trajectory](gsd_multibackbone_final_findings_20261019.md) | Dört başlıkta PM−terminal +1.2507–1.5605pp | 581nesne/all15/seed0; her başlıkta original3DD üstünlüğü yok. |
| [25 Ekim eşlenmiş component doğrulama](guidance_component_confirmation_results_20261025.md) | C−CS PM +.1706pp,95%CI[-.0397,+.3651] | 70nesne/veri kümesi, iki veri kümesi/dört başlık/üç seed; üstünlük veya eşdeğerlik kanıtı değil. |
| [28 Ekim kalibre component doğrulama](guidance_component_sensitivity_results_20261028.md) | C−CS PM +.0516pp,CI[-.1250,+.2341] | Ayrı140nesne; kalibrasyon da tutarlı spectral üstünlük sağlamadı. Tuned-C kolunda sayısal failure korunuyor. |
| [30 Ekim depth doğrulama](chamfer_depth_tuning_results_20261030.md) | N5 PM66.5104; defaultN8 PM67.6997; tunedN8 PM68.7413 | 256nesne/Point-MAE/üç seed. TunedN8−N5 +2.2309pp; yalnız PM etkisi değil. N5 PM−terminal +.7205pp. |
| [31 Ekim N8 transfer](chamfer_depth_transfer_results_20261031.md) | Sekiz veri kümesi/başlıkta PM−terminal +.175–1.651pp;7/8koşulluCI sıfır üstünde | 140önceden görülmüş nesne/veri kümesi, üç seed; tam benchmark veya dış unseen doğrulama değil. |
| [1 Kasım SCD-h/spectral-z](scdh_spectralz_results_20261101.md) | Spectral65.5729 versus SCD65.5903; fark−.01736pp,CI[-.39931,+.36458] | 128nesne/üç seed; fresh maliyet Gaussian3.338×/Background2.810×. |
| [5–6 Kasım mekanizma kampanyası](gsd_comprehensive_results_20261106.md) | Referans yanlılığı, decoder bant karışması, graf değişimi ve native tekrar sorunu; ortak accuracy/hız kazancı yok | Küçük4/8nesne tanıları geniş benchmark değildir; koşullu genişlemeler ilerletilmedi. |

[Run/Code] [Güncel baseline](current_baseline.md) Chamfer+eşit PM, normal8/Background35, eta.003/gamma.03, frozen başlıklar; GSD kapalı. Kod `scripts/analyze_legacy_trajectory.py:policies` içinde softmax sonrası adaylar üzerinde aritmetik ortalama alır. `scripts/eval_chamfer_baseline.py` birincil CSV/prediction olarak PM'yi kaydeder. Terminal logits ayrı tanıdır.

[Run/Inference] Sparse guidance yaklaşık%21 Gaussian+Background süre kazancı karşılığında doğruluğu düşürdü; weighted da düştü, KL/H sonuçları küçük/karışık kaldı ([sonuçlar](chamfer_guidance_variants_results_20261030.md)). Global-z oracle fayda potansiyeli gösterdi; temiz-verisiz refresh/proximal/conditional ve reencode yolları güvenilir kazanca dönüşmedi ([oracle](shape_latent_oracle_results_20261031.md), [reencode](lion_reencode_depth_results_20261101.md)). Bunlar denenmemiş ilk seçenekler gibi önerilmemeli.

[Open] Hard100/200 halfmix [ayrı bekleyen çalışmadır](gsd_lowband_halfmix_study_20261105.md); GPU sonucu yoktur. Bu rapor onu başlatmaz veya iptal etmez. Tüm spectral yöntemlerin işe yaramadığı ya da tüm eski accuracy sonuçlarının native drift nedeniyle geçersiz olduğu sonucu çıkarılmaz.

## TEDM, intihal ve özgünlük

[Paper] Rosnati, Roschewitz ve Glocker, ML4H2023, *Robust semi-supervised segmentation with timestep ensembling diffusion models*: [birincil PDF](https://proceedings.mlr.press/v225/rosnati23a/rosnati23a.pdf), bölüm4, s516, Eş.8–9. Ortak pixel predictor farklı timestep temsilleriyle eğitilir; testte tahmin haritaları ortalanır. Dolayısıyla genel timestep-prediction averaging fikri önceden yayımlanmıştır. Eş.9 tek başına yazar kodunun bizim softmax sıramızla byte/algoritma özdeşliğini kanıtlamaz.

[Code/Inference] Bizde girdi guided reverse trajectory'den decode edilen temiz nokta bulutlarıdır; mevcut frozen sınıflandırıcı kullanılır, yeni predictor eğitilmez. Ortalama softmax sınıf olasılıkları üzerinde, üç ara aday+terminal ile alınır. Görev/temsil/eğitim farkları vardır; aritmetik ortalamayı yeni icat yapmazlar. [Wang et al., bölüm4.1](https://arxiv.org/html/2508.09138v3) tek trajectory ara cevap oylaması için ayrıca öncüldür. Diğer vision-logit ve point-cloud entropy öncülleri [önceki atıf denetimindedir](gsd_trajectory_attribution_20261025.md).

[Inference] Benzer yöntem kullanımı tek başına intihal değildir. Öncülü açıkça belirtmek, katkıyı doğru sınırlandırmak ve metin/kod/şekil ödünç alınmışsa bunu uygun şekilde atfetmek gerekir. Atıf, yöntem yeniliğini otomatik olarak sağlamaz. Bağımsız düşünülmüş yerel fikir de yayımlanmış önceliği ortadan kaldırmaz. Bu değerlendirme bir taslak-metin/kod benzerlik taraması değildir. ACM'nin [authorship ilkeleri](https://prod-www.acm.bloomreach.cloud/publications/policies/new-acm-policy-on-authorship) doğru credit/attribution yükümlülüğünü destekler; ana acm.org bağlantısı bu oturumda403 verdi.

Savunulabilir yazım: “Diffusion timestep tahminlerinin birleştirilmesine ilişkin önceki çalışmalar doğrultusunda, donmuş 3B sınıflandırıcılar için tek restorasyon yörüngesindeki seyrek temiz tahminlerin olasılık ortalamasını değerlendiriyoruz.” Bu cümle ilk buluş iddiası içermez. “Yeni probability mean yöntemi öneriyoruz” veya “GSD kazancını kanıtladık” mevcut kanıtı aşar.

## Yayın ve tez değerlendirmesi

[Inference] Güçlü yöntem konferansı/dergisi için mevcut yenilik iddiası zayıf: generic PM biliniyor, GSD üstünlüğü yok, güncel N8 politikasının tam veri/çok-seed confirmatory sonucu yok, corruption kimliğine bağlı Background35 var. 0.5–0.8pp tek başına bir kabul/red eşiği değildir; etki tekrarı, maliyet, karşılaştırmanın adaleti ve yeni bilgi belirleyicidir. Bir dergi otomatik olarak konferanstan daha düşük eşik anlamına gelmez.

[Inference] Kontrollü ampirik/reproducibility/negatif-sonuç çalışması veya uygun workshop için daha makul bir temel var; fakat sonuçların tek bilimsel soruya bağlanması, ilgili yayın kapsamına uyum ve aşağıdaki eksiklerin tamamlanması gerekir. Yüksek lisans tezi açısından sistematik hipotez sınama ve sınırları açıklanan negatif bulgu da araştırma katkısı olabilir. Kurumun yayın veya yöntem-yeniliği şartı bilinmiyor; “tez zaten kabul edilir” denemez.

[Inference] Önerilen ana soru: **Donmuş 3B sınıflandırıcıyla diffusion TTA'da, tek yörüngeden tahmin toplamak eşit hesap bütçesinde ne zaman ve neden yararlıdır?** Başlık örneği: “3B Nokta Bulutlarında Diffusion Tabanlı Test Zamanı Adaptasyonu: Yörünge Birleştirme, Spektral Kısıtlar ve Doğruluk–Maliyet Analizi”. GSD deneyleri bu sorunun kontrollü olumsuz bulgu bölümüdür.

## En yüksek öncelikli tamamlamalar — öneri, yeni koşu yetkisi değil

1. **Mevcut logits ile bilgi açığını ölç.** Terminal, her sabit tek checkpoint, eşit PM, logit mean ve minimum entropy aynı aday/ID üzerinde karşılaştırılabilir. Aday uyuşmazlığı, corrected/harmed örnekler ve oracle-any-candidate headroom yalnız teşhis için raporlanır. Terminalden fazla classifier geçişinin tek başına etkisini ayırmak için aynı terminal geometrisindeki FPS/augmentation voting kontrolü eksikse ayrıca gerekir; bunu mevcut olmayan logits'ten üretmiş gibi davranma. Logit mean daha iyi çıkarsa aynı testte seçilmiş exploratory sonuç olarak kalır.
2. **Dondurulmuş tek final politika ve adil bütçe.** Source-only, matched original3DD terminal/PM, currentN8 terminal/PM ve eşit hesaplı voting karşılaştırması. Prior, decode, classifier ve backward maliyetlerini dahil et; graph-free/shared-four-head/cache-replay ölçümünü tek-head deployment hızına dönüştürme. Yeni uygulama iddiası için frozen yöntemlere uygun güçlü güncel comparator ekle; model-adapting yöntemleri ayrı protokol olarak etiketle.
3. **Görülmemiş doğrulama ve dağıtım sözleşmesi.** İki mevcut veri kümesinin tam nesne kapsamı/üç seed ana tabloyu tamamlar, fakat tarihsel test maruziyetini silmez. Daha önce gerçekten kullanılmamış veri/severity/split ancak exposure envanteriyle unseen sayılır. En az iki farklı backbone ana doğrulama, dört başlık mevcut transfer desteği. Nesne-cluster CI, seed SD, bozulma bazlı zarar ve same-seed fresh-process repeat tabanı ayrı raporlanır. Background kimliği bilinmeyen kullanım için ortak sabit horizon veya development'ta dondurulmuş label-free politika gerekir;14bozulmayı raporlamak15bozulmanın yerine geçmez.
4. **Yeni algoritma gerekirse tek dar yön.** Ara tahmin çeşitliliği gerçekten faydalıysa, corruption adını kullanmayan bütçe sınırlı checkpoint/erken durma seçimi daha doğrudan adaydır. PM/entropy mevcut baseline; sadece farklı ağırlık vermek özgünlük sayılmaz. Prefix-only karar, kaynak/test etiketi sızıntısı kontrolü ve gerçek end-to-end maliyet şarttır. Offline erken durma simülasyonu gerçek hız kanıtı değildir. Bu fikir literatürce yeni veya başarılı ilan edilmez; uygulama öncesi ayrı tarama ve tasarım gerekir.

[Decision/Inference] Önceliği daha fazla GSD parametre taraması yerine mevcut PM bulgusunun bilgi kaynağı, adil maliyeti ve final doğrulamasına vermek daha verimli görünüyor. Oracle headroom yoksa selector tasarımını bırak; gain eşit maliyetli voting ile açıklanıyorsa katkıyı buna göre daralt; frozen dış doğrulama kazanımı kaybederse olumlu yöntem iddiasını bırakıp sınır/negatif sonuç çalışmasına dön. Sayısal tekrar kontrolü başarısızsa küçük accuracy farkını mekanizma kanıtı sayma.

## Bu incelemenin çıktısı

Yalnız bu rapor, README yönlendirmesi, findings kaydı ve open_questions önceliği güncellenir. Üretim kodu, baseline, ham deney, bekleyen peer çalışması ve runtime durumu değiştirilmez. Kaynakların/arşivlerin tamamının yeniden doğrulandığı iddia edilmez. Bu not danışmana sunulacak bilimsel değerlendirmeyi kaydeder; yeni deney uygulanmış veya yayın kabulü alınmış değildir.


[Run/Inference, later Nov7 update] The separate [hard-lowband halfmix pilot](gsd_lowband_halfmix_results_20261107.md) subsequently completed. M100171/240 vs half-SCD169; M200166. Numerical attribution gate fails, graph+reconstruction cost rises about23%, and no default promotion or publication-superiority claim follows. The earlier GPU-unrun statement records the review-time state.
