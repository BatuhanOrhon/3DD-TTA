# Yeni agent başlangıç promptu — literatür ve yöntem araştırması

Hazırlık tarihi: 30 Ekim 2026. Bu belge gelecek oturumun görevidir;
literatür taramasının yapıldığı veya yeni deney başlatıldığı anlamına gelmez.

```text
3D point-cloud test-time adaptation tezimiz için güncel literatür ve yöntem
araştırması yap. Özellikle 3DD-TTA'ya atıf veren çalışmalardan, diffusion
guidance loss'unu veya inference politikasını geliştirebilecek fikirler bul.
Amaç: doğruluğu koruyarak toplam süreyi azaltmak veya sınırlı ek maliyetle
doğruluğu artırmak. Bir yöntemin kazanacağını varsayma; mevcut sonuçları veya
kodda olmayan hataları seçtiğin fikri desteklemek için yeniden yorumlama.

Çalışma alanı ve başlangıç:
- Ana repo: /Users/batuhan/Desktop/projects/3DD-TTA
- Güncel Agent B kayıtları ve kodu:
  /Users/batuhan/Desktop/projects/3DD-TTA-worktrees/guidance-components
  Dal: gsd-guidance-components; son bilimsel denetim commit'i: 7143ede.
  Bu commit yerel araştırma kaydıdır; deneylerin çalıştırılmış kaynak commit'i
  ve sealed ZIP kimlikleri raporlarda ayrıca belirtilir.
- Önce AGENTS.md, RTK ve iterative-thesis-researcher skill'ini oku; gerçek
  HEAD/status'u denetle. Ana çalışma alanındaki kullanıcı değişikliklerini
  ve diğer agent'ın dalını değiştirme. Yazılı çıktı için gerekiyorsa güncel
  Agent B dalından ayrı bir araştırma worktree/dalı oluştur.

Araştırma bağlamı:
3DD-TTA, LION ile bozuk girdiyi latent uzayda denoise ediyor; Selective Chamfer
Distance (SCD) guidance uygulanıyor, kaynak sınıflandırıcı sabit kalıyor.
ModelNet40-C ve ScanObjectNN-C üzerinde DGCNN/CurveNet/PointNeXt/Point-MAE var.
Bizim GSD-inspired latent spectral fidelity yöntemimiz tam GSDTTA değildir.
Son denetim: smoothing açık, fakat legacy ham özdeğer/M800 filtresi kullanılmış;
normalize tüm-mod smooth farklı bir sürümdür. Guidance bağlantısı kopuk değil.
Bilinen self-only izolasyon kusurunun doğruluk farkına neden olduğu gösterilmedi.

Mevcut güçlü ipucu, aynı trajectory'nin üç predicted-clean probe'u ve terminal
çıktısından probability mean/minimum entropy okumalarıdır. Bunlar şu anda
loss veya entropy backward değildir; ek decode ve classifier maliyeti vardır.
İlk sabit doğrulamada C probability mean %56.8929, C+S %56.7222; farkın güven
aralığı sıfırı içeriyor. Son, farklı nesne kümesindeki doğrulamada original3DD
%53.7639, eşleşen C %53.0139, seçilen C+S %52.9623. Original–C+S farkı bütün
politika karşılaştırmasıdır; yalnız loss katkısı değildir. Son kampanya23başarı/
1sayısalhata olarak kabul edildi. GSD'nin doğruluk sabitken hız kazandırdığı
kanıtlanmış değil; ölçülen %24.94 süre azalması C'nin C+S'ye göre araçlandırılmış
iki-bozulmalı reconstruction ölçümüdür, deployment hızlanması değildir.

Okuma sırası (yollar güncel Agent B worktree'sine göre):
1. knowledge/README.md, thesis_scope.md, experiment_protocol.md.
2. knowledge/guidance_mathematics_execution_audit_20261030.md.
3. knowledge/guidance_component_sensitivity_results_20261028.md ve
   guidance_component_confirmation_results_20261025.md.
4. knowledge/gsd_trajectory_attribution_20261025.md: mevcut mean/entropy
   fikirlerinin öncülleri; bunları yeni buluş gibi sunma.
5. knowledge/gsd_accuracy_research_plan_20261011.md, method_synthesis.md,
   open_questions.md ve papers/3dd_tta.md, papers/gsdtta.md, papers/lion.md.
6. Gerekirse guidance_component_sensitivity_study_20261025.md ve önceki
   spectral/projection/step-count sonuçları: tamamlanmış deneyleri tekrar önerme.

Kod referansları:
- tta.py: tta_reconstruct, selective_chamfer_loss; original3DD kontrolü.
- tta_gsd_v2.py ve spectral_guidance_losses.py: uygulanmış loss, gradyan yolları.
- graph_spectral_v2.py / graph_spectral.py: legacy ve normalize smooth ayrımı.
- shared_trajectory.py: SharedTrajectoryObserver; legacy_trajectory.py:
  isolated_probe/RNG koruması.
- scripts/analyze_legacy_trajectory.py: probabilities/policies;
  scripts/analyze_scan_trajectory.py: aynı readout'ların Scan analizi.
- scripts/eval_multibackbone_guidance.py: veri/model/deney protokolü.
- gsd_composition.py ve ilgili knowledge kayıtları: mevcut gradient composition.

Tarama görevi:
- İnternette tarama yaptığın tarih itibarıyla en yeni point-cloud TTA işlerini
  araştır; yalnız eski listemize dayanma. Önce 3DD-TTA'yı
  arXiv:2411.14495 ve WACV2025 kaydıyla tekilleştir. Atıf dizinlerinden ileri
  atıf taraması yap; mümkün olduğunda PDF kaynakçasında doğrudan atfı doğrula.
  Erişilemeyen kayıtları/arama sınırlarını belirt; eksiksiz tarama iddiası kurma.
- Birincil makale, proceedings/OpenReview/arXiv ve yazar kodunu kullan.
  Yayın/ilk preprint/güncelleme tarihlerini, kabul durumunu ve deney kapsamını
  ayır. Arama özetiyle yöntem veya hız kazancı doğrulanmış sayılmaz.
- Öncelik: geometric/semantic/consistency guidance, confidence veya uncertainty
  ile loss ağırlığı/adım/zamanlama, trajectory ensembling/selection,
  güvenilir adaptif durdurma ve pahalı backward/graph hesaplarını azaltma.
  Gerekçeli 2D diffusion/TTA fikirlerini de incele, fakat 3D aktarımını çıkarım
  diye etiketle. Sadece daha büyük parametre grid'i önerme.
- Frozen-classifier/input-adaptation kısıtını koruyan fikirleri öncele.
  Self-training, classifier güncellemesi veya yeni eğitim gerektiren yöntemleri
  ayrı sınıfta, bu kısıtı değiştirdiklerini açıkça belirterek değerlendir.

Teslim:
1. Atıf veren çalışmalar ile diğer ilgili çalışmaların ayrı gösterildiği kısa
   tablo: tarih, birincil bağlantı, doğrudan atıf kanıtı, mekanizma, maliyet,
   kısıt uyumu ve repoda daha önce denenip denenmediği.
2. En güçlü3–5 aday; her biri için makaledeki somut denklem/algoritma bölümü,
   bizim loss/update/readout'a nasıl aktarılacağı, etkilenen kod ve tam maliyet.
   Probability mean'i gradyan loss'uyla, selection'ı entropy minimization'la
   karıştırma. Confident-wrong/pseudo-label collapse ve bozuk referans riskini
   somut değerlendir; doğruluk/süre kazancı uydurma.
3. En iyi1–2 aday için yeni ve ayırt edici minimum deney önerisi: matched
   original/C/GSD kontrolleri, development–confirmation ayrımı, seed ve nesne
   eşlemesi, all15/macro14, corrected/harmed, toplam decode/classifier/backward
   maliyeti. Cache-replay süresini deployment hızı diye kullanma. Loss, host ve
   readout etkilerini ayrı ablation'larla tanımla; başarısızlık ölçütünü yaz.
4. knowledge'da tarihli araştırma notu ve gerekli bulgu/açık soru güncellemeleri;
   kullanıcıya kısa önceliklendirme ve gerekirse önerilmeyen adayların nedeni.

Bu oturum literatür + kod uyarlanabilirliği + deney tasarımı içindir.
Üretim kodunu değiştirme, Colab açma veya GPU deneyi başlatma. Önce mevcut
kanıtlarla gerekçelendirilmiş önerileri sun; uygulama/deney kararı sonra verilecek.
```
