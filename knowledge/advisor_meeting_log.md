# Hoca görüşmeleri: gündem ve görüşme kayıtları

Oluşturulma: **2026-10-11**. Amaç: hocayla konuşulacak konuları, gerçekten
gerçekleşmiş görüşmeleri ve alınan kararları birbirinden ayırarak saklamak.

Kanıt etiketleri: **[User report]** kullanıcı tarafından aktarılan bilgi/fikir;
**[Paper]** makale; **[Code]** kaynak kod; **[Run]** arşivlenmiş koşu;
**[Inference]** yorum/hipotez; **[Open]** açık soru.

Bir maddenin buraya eklenmesi, hocayla görüşüldüğü veya hoca tarafından
onaylandığı anlamına gelmez. Görüşme tarihi ve hocanın geri bildirimi ancak
kullanıcı aktardığında doldurulur. Önceki kararlar silinmez; yeni tarihli
kararlarla ilişkilendirilir.

## Konuşulacak konular

### G-001 — Purge-Gate ve tez kapsamıyla ilişkisi

- Eklenme: **2026-10-11**. Durum: **Konuşulacak**.
- Makale: [Purge-Gate: Backpropagation-Free Test-Time Adaptation for Point
  Clouds Classification via Token Purging](https://arxiv.org/abs/2509.09785),
  ICCV 2025.
- Kod: [Resmi GitHub deposu](https://github.com/MosyMosy/Purge-Gate).
- [Paper] Kısa özet: transformer içinde bozulmadan etkilendiği düşünülen yerel
  özellik token'larını attention öncesinde eler. PG-SP kaynak istatistiklerini
  kullanır; PG-SF önceden eğitilmiş modelin CLS temsiline dayanır ve adaptasyon
  sırasında kaynak veri gerektirmez. Test sırasında ağırlık optimizasyonu
  yapmaz. Yayımlanan yaklaşım BatchNorm'da mevcut test batch'inin istatistiklerini
  de kullanır. Temiz point cloud üretmek yerine sınıflandırıcı içindeki özellik
  akışını değiştirir. Ayrıntı: makale §3.2–3.3.
- [Inference/Open] Hocayla konuşulacak: yalnızca girdi adaptasyonu kapsamını mı
  koruyacağız, yoksa sınıflandırıcı içi token filtreleme/normalizasyon adaptasyonu
  da kabul edilebilir mi? PG-SF karşılaştırıcı veya GSD'ye ek modül olabilir mi?
- İlgili inceleme: [Accuracy araştırma planı](gsd_accuracy_research_plan_20261011.md).
- Görüşme tarihi / hocanın yorumu / karar: **Henüz kayıt yok**.

### G-002 — Diffusion adımları arasında temiz tahmin seçimi ve ortalama

- Eklenme: **2026-10-11**. Durum: **Konuşulacak**.
- [User report] Kullanıcı önerisi: farklı diffusion adımlarındaki clean point
  cloud tahminlerini sabit Point-MAE ile sınıflandırmak; düşük entropili adımdan
  karar vermek veya farklı adımların tahminlerinin ortalamasını almak.
- [Inference] İlk karşılaştırma: son çıktı, minimum entropili çıktı ve eşit
  ağırlıklı sınıf olasılığı ortalaması. Nokta koordinatları veya sınıf ID'leri
  ortalanmaz. Düşük entropi yüksek güveni gösterir; adımlar arası kararlılık ve
  doğruluk ayrıca ölçülür.
- [Open] Hocayla konuşulacak: bu yöntem sabit sınıflandırıcıyla girdi seçimi
  olarak kapsamımıza uyuyor mu? Yenilik için hangi karşılaştırıcılar ve dış
  değerlendirme gerekir? Önce seyrek kontrol noktalarıyla fizibilite uygun mu?
- Teknik kayıt: [Adım seçimi ve erken durdurma taslağı](gsd_trajectory_selection_20261011.md).
- Görüşme tarihi / hocanın yorumu / karar: **Henüz kayıt yok**.

### G-003 — Hesaplama maliyeti, erken durdurma ve rehberlik sadeleştirme

- Eklenme: **2026-10-11**. Durum: **Konuşulacak**.
- [User report] Kullanıcı önerisi: yöntem başarılı olursa GSD/SCD'nin süre ve
  accuracy katkılarını ölçmek; uygun olduğunda birini elemek. Yeterince güvenilir
  tahminde diffusion'ı erken bitirerek ek değerlendirme maliyetini dengelemek.
- [Inference/Open] Erken durdurma yalnızca o ana kadar görülen tahminleri
  kullanmalıdır. Gelecekteki en düşük entropili adımı bilmek erken durdurma
  değildir. Düşük entropiyle birlikte ardışık tahmin tutarlılığı araştırılmalı.
  Batch içindeki bazı örnekleri durdurmak tek başına GPU süresini azaltmayabilir.
- [Open] Hocayla konuşulacak: kabul edilebilir accuracy/süre dengesi nedir?
  Makale için hedef, sabit sürede daha yüksek accuracy mi, benzer accuracy'de
  daha kısa süre mi, yoksa iki sonuç da mı? Henüz sayısal eşik belirlenmedi.
- Görüşme tarihi / hocanın yorumu / karar: **Henüz kayıt yok**.

### G-003 eki — 2026-10-11 online kullanım açıklaması

[User report] Kullanıcı, gerçek online TTA'da tek örnek/küçük batch işleneceğini
ve erken durdurmanın bu senaryoda hız avantajı sağlayabileceğini vurguladı.
[Inference] B1'de kalan diffusion adımları doğrudan atlanabilir; büyük batch'te
diğer örnekleri bekleme uyarısı B1'e uygulanmaz. Net gecikme kazancı ek probe
maliyeti dahil aynı B1 baseline'a göre ölçülecek. B70 benchmark throughput'u ile
onlineB1 p50/p95 gecikmesi ayrı raporlanacak. Küçük batch>1'de farklı bitiş
zamanları ayrıca ele alınacak. Bu, hocanın geri bildirimi değil kullanıcı
yönlendirmesidir. [Session handoff](gsd_trajectory_session_handoff_20261011.md).

## Konuşulan konular ve alınan kararlar

Henüz bu dosyaya aktarılmış doğrulanmış bir hoca görüşmesi kaydı yok.
Yukarıdaki maddeler kullanıcı–asistan araştırma gündemidir.

## Yeni görüşme kayıt şablonu

- Görüşme tarihi:
- İlgili gündem ID'leri:
- Kullanıcının aktardığı konuşma özeti [User report]:
- Hocanın önerisi/itirazı (aktarıldığı biçimde):
- Alınan karar ve kapsam:
- Aksiyon, sorumlu ve varsa hedef tarih:
- Açık kalan sorular:
- Sonraki takip tarihi / ilgili kanıt bağlantıları:
