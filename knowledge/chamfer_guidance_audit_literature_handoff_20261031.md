# Yeni agent için kısa başlangıç promptu — 2026-10-31

[User report/Decision] Sonuç/kod kaydı ve `thesis-integration` push'u ardından
kullanılacak analiz görevi. Bu dosya agent'ı veya yeni GPU deneyi başlatmaz.

```text
Worktree: /Users/batuhan/.codex/worktrees/7bf0/3DD-TTA
Yerel branch: integration/chamfer-guidance-variants-20261030
Yayın hedefi: origin/thesis-integration

Önce bu worktree'ye geç; git rev-parse --show-toplevel ve git branch --show-current ile dizini/branch'i doğrula. Belirtilen checkout/branch'te kal; branch değiştirme, kullanıcı ve diğer agent değişikliklerini koru. AGENTS.md ve iterative-thesis-researcher skill'ini uygula; git status/branch/HEAD ve origin/thesis-integration durumunu kaydet. Checkout eskiyse eksik kayıtları git show origin/thesis-integration:<yol> ile oku; eksik dosyayı deney hatası sayma.

Önce knowledge/README.md; ardından knowledge/chamfer_guidance_methods_results_20261031.md, knowledge/chamfer_guidance_variants_results_20261030.md, knowledge/scan_curvenet_entropy_followup_20261030.md, knowledge/chamfer_author_code_consistency_audit_20261030.md ve knowledge/diffusion_guidance_literature_review_20261030.md dosyalarını oku.

1) knowledge/evidence/chamfer_guidance_executed_code_20261031/README.md ve manifest.json'daki gerçek 0fd2581/95a4997 kaynaklarını, mevcut kodu, kalibrasyon/analiz/controller scriptlerini ve pinli yazar repolarını karşılaştır. Sparse backward, ayrı KL/H ve weighted Chamfer'da accuracy katkısını engelleyebilecek hata ara: loss işareti/reduction, detach/FPS-normalizasyon/input gradyanı, timestep/style update, weighted NN/trim, katsayı ölçeği, cache/ID ve metrik eşlemesi. Her bulguya dosya:satır, etkilenen arşiv/kaynak ve uygunsa küçük CPU kanıtı ver; CUDA gerektiren şüpheyi doğrulanmış hata sayma. Kasıtlı uyarlamayı hata sayma, hata bulmayı zorunlu varsayma.

2) Güncel birincil literatürü ve özellikle 3DD-TTA'ya doğrudan atıf veren yeni makaleleri yeniden tara; yayın tarihi/sürümü, PDF atıf kanıtı ve varsa resmi GitHub kodunu doğrula. Önceden denenmiş fikri yeni isimle tekrar önerme. En fazla üç yeni aday için mekanizma, mevcut kodda entegrasyon noktası, accuracy–süre gerekçesi ve ayırt edici minimum deney öner.

Chamfer-only, mevcut temporal probability mean ve frozen classifier/BN/LN koşulunu koru; spectral hesap önerip çalıştırma. Sonuçlar140nesne/veriseti×3seed; KL belirsiz, weighted negatif, entropy küçük/sınırlı kazanım; CurveNet birincil CI sıfırı kapsıyor. Dört-head ortalama ensemble değildir. Önce bulgu ve öneri raporunu knowledge'a kaydet ve sun; üretim kodunu değiştirme, Colab/GPU başlatma. Başka agent'ın runtime'ına dokunma.
```
