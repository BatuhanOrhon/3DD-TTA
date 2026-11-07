# LION guidance A/B/C pilot sonuçları — 7 Kasım 2026

[Run/CPU/Inference] Planlanan teknik kontroller ve üç mekanizmanın sınırlı pilotları tamamlandı. **Bu ayarlarda N8+PM üzerine doğrulanmış yeni accuracy katkısı veya accuracy korunarak hızlanma gösterilmedi.** Mekanizmaların tüm olası ayarları reddedilmiş değildir. Varsayılan sampler değiştirilmedi.

[Adım adım yürütme, izin ve başarısızlık kaydı](lion_guidance_pilot_20261107.md), [makinece okunabilir sonuç](evidence/lion_guidance_pilot_20261107/summary.json), [ham arşiv kimlikleri](evidence/lion_guidance_pilot_20261107/raw_receipts.json).

## Gerçek kapsam

[Code/Run] Dondurulmuş Point-MAE ve raw/eval/EMA-off LION; ModelNet40-C severity5, seed0, B2. DDIM100; normal8, Background35 adım; gamma.03, eta.003, p.95; özgün z0 ile decode. Eşit Probability Mean: N8 için clean2/4/6+terminal, Background için9/18/27+terminal. Aynı encoder girdisi/z0/h0/noise ve kayıtlı RNG tüm eşleşen kollarda korunur; model state hash'leri değişmez.

Önceden belirlenmiş4kalibrasyon ve16pilot nesne: [kimlik listesi](evidence/lion_guidance_pilot_20261107/selection.json). Aynı nesnenin farklı corruptions'ları bağımsız nesne sayılmaz. Bunlar tarihsel olarak görülmüş resmi test nesneleridir; held-out doğrulama değildir. Etiketler yalnız skorlamada kullanıldı.

İlk teknik faz Gaussian/Background/Density/Occlusion'da4nesne kullandı. Native eşdeğerlik/tekrar gate'i başarısız olunca A/B yalnız tüm kalibrasyon grupları geçen **Density+Occlusion** üzerinde16nesneye daraltıldı:32 corruption-object değerlendirmesi. Seçim accuracy'ye dayanmadı. C aynı ilk kayıtlı trajectory üzerinde conditional readout müdahalesi olduğundan dört corruption/64değerlendirme kullandı. Başarısız pilot nesneleri çıkarılmadı; iyi trajectory seçilmedi.

## Sonuçlar

| Faz | Denenen kollar | Doğru sayısı | Sayısal tekrar sonucu |
|---|---|---:|---|
| A: ucuz latent Chamfer | full, full-repeat, style-frozen, sparse, cheap, hybrid | Her kol22/32 (%68.75) | full-repeat14/16grup geçti |
| B: feature eşleştirme | full, full-repeat, geometric-beta0, feature, shuffled, direct4D | Her kol22/32 (%68.75) | full-repeat11/16grup geçti |
| C: geçmiş PM ile düzeltme | full, semantic-zero, zero-repeat, causal-feature, fixed-feature, causal-all | Her kol48/64 (%75) | zero-repeat ve within-readout32/32grup geçti |

[Run/Inference] A/B'de doğru/yanlış durumları aynı kaldı; bazı yanlış sınıf tahminleri başka yanlış sınıflara değişti. C'de semantic-zero'ya göre nihai sınıf tahmini de değişmedi. Küçük ve bütün farkların sıfır olduğu örneklemde bootstrap[0,0] **popülasyon eşdeğerliği veya0.3puan noninferiority kanıtı değildir**.

### A — hesap azalıyor, ortak accuracy/hız kazanımı doğrulanmadı

[Code] İki yöndeki retained Chamfer eşleşmelerini ve reverse-edge çokluğunu sayan kapalı biçimli proximal düzeltme, predicted-clean XYZ'ye uygulanır. Aynı epsilon ile gerçek önceki DDIM alpha üzerinden noisy state'e taşınır. Feature için doğrudan proximal güncelleme yoktur. Hibritte1/4/7... adımlar tam backward, aradakiler ucuz düzeltmedir; sparse kol aynı tam-backward takviminde diğer guidance'ı sıfırlar. Style-frozen ve cheap-only kontrolleri de vardır. τ=.1, kalibrasyon kümesinde gerçek güncelleme normlarına göre etiketsiz seçildi.

[Run] Bütün kollar22/32; hybrid, sparse'a göre düzeltilen örnek kazandırmadı. Kaydedilen B2 sampler+4decoder+4classifier toplamları full18.876s, full-repeat18.676s, style-frozen18.682s, sparse13.026s, cheap9.785s, hybrid13.143s. Bunlar sabit sıra/soğuk başlangıç içeren **parça süreleri**; ortak encoder, veri hazırlığı ve arşiv maliyetini içermez. Isınmış deployment benchmark'ı değildir.

[Run/Inference] İki Density grubunda native repeat gate'i başarısız: birinde latentRMS≈1.09e-7 iken logitRMS.034; diğerinde latentRMS.0002025/logitRMS.2398. Bu nedenle küçük örneklemde aynı accuracy ile daha düşük parça süresi, doğrulanmış ortak üstünlük sayılamaz. İlerleme ölçütü karşılanmadı.

### B — feature eşleşmeleri değiştiriyor, bu ayarda accuracy faydası yok

[Code] Ortak reference-derived XYZ/feature ölçekleri, k8XYZ shortlist ve beta.1 kullanıldı. Eşleşmeler detach edilir; loss XYZ üzerinde kalır. Geometric-beta0 aynı cdist/gather yolunu, shuffled kol reference feature permütasyonunu kullanır. Direct4D aynı ölçeklerle global4D fidelity loss uygular; shortlist/XYZ-only loss'tan farklı bir heuristiktir, tek başına feature-gradient ablation'ı değildir.

[CPU/Inference] Kayıtlı clean checkpoint'lerde CPUfloat64 yeniden hesaplaması, geometric'e göre eşleşmelerin baseline state'lerinde%28.181'ini, feature kolu state'lerinde%24.852'sini değiştiriyor. Gerçek/shuffled eşleşmeleri de ayrılıyor. Bu **exact CUDA assignment capture değildir**; feature mekanizmasının aktif olduğuna ilişkin bir tanısal sonuçtur, doğru correspondence kanıtı değildir. [Analiz](evidence/lion_guidance_pilot_20261107/matching_activity_summary.json).

[Run] Buna rağmen bütün kollar22/32. Feature geometric veya shuffled kontrolünü geçmedi. Native full-repeat5/16grupta başarısız olduğundan kapsam ayrıca sayısal olarak sınırlı kaldı. Beta accuracy'ye göre yeniden ayarlanmadı.

### C — feature-only düzeltme aktif, öğretmene uyum accuracy artışı değil

[Code] İlk candidate değişmez; sonraki candidate'larda yalnız daha önceki düzeltilmiş tahminlerin detached PM öğretmeni kullanılır. Gelecek bilgi yoktur. Bir RMS-relative.01 feature adımı; latent XYZ/z sabit, decoded relativeRMS hareket limiti.001. Tüm-kanal kontrolü aynı latent L2 sınırını kullanır. Her attempt'in before/after/chosen latent, decoded geometry, gradient ve logits'i saklanmıştır.

| Kol | Kabul edilen / denenmiş düzeltme | Yorum |
|---|---:|---|
| Causal-feature |191/192| Latent XYZ değişmedi;1attempt geometry limitiyle reddedildi |
| Fixed-feature |192/192| Sabit bozuk-input öğretmeni kontrolü |
| Causal-all |0/192| Bütün attempts geometry sınırını aştı; tamamı fallback |

[Run/CPU] Causal-feature median feature deltaRMS≈.01001, median logit deltaRMS≈.02231; kabul edilen155/191adımda pseudo-teacher CE azaldı. Düzeltme fiilen çalışıyor, fakat nihai sınıf/accuracy değişmedi. Causal-all tamamen fallback olduğundan **feature kanalının tüm-kanala üstünlüğü test edilmiş sayılmaz**. [Düzeltme analizi](evidence/lion_guidance_pilot_20261107/repair_activity_summary.json).

[Run/Inference] Full61.469s üzerine causal-feature ek readout21.418s harcadı; kayıtlı iş yükünde≈%34.84ek süre. Bu uygulanış optimize edilmiş deployment ölçümü değildir; feature değişkenlerinin dörtte bir olması compute'un dörtte bire düşmesini sağlamadı. full→semantic-zero çalışma-dalı etkisi ile zero→causal-feature müdahalesi ayrı tutuldu. Frozen-readout gate'i32/32geçtiği halde bu ayarda classification faydası görülmedi.

## Native kararsızlık ve karar

[Run] Teknik fazda sampler gate'i5/8geçti; Background'da original sampler'ın kendi tekrarları da büyük fark verdi. Full-repeat teknik doğrusu11→12/16değişti. Bir Gaussian grubunda da cross/full-repeat başarısız. Bu farklar yeni guidance'ın etkisi diye yorumlanmadı. Eşikler latent/geometryRMS1e-5 ve logitRMS1e-4 olarak korundu; örnekler elenmedi. Tek bir native kernel bu çalışmayla neden olarak ispatlanmadı.

[Decision] Mevcut ayarlar koşullu ilerleme ölçütlerini karşılamadı. Candidate-own-repeat, warmed deployment timing, all15/256nesne/üç-seed ve cross-head confirmation **NOT_ADVANCED**; bunlar tamamlanmış deney gibi sunulmaz. Sınırlı null/sonuçlandırılamayan pilotu raporlamak için zorunlu kalan GPU testi yok; bağımsız son inceleme bunu doğruladı. Mekanizma ailesinin tamamına olumsuz hüküm verilmez, mevcut sonuç yeni baseline/contribution iddiası sağlamaz.

[Inference/Open] Özellikle A için sayısal kararlılık ve deployment ölçümü çözülmeden küçük accuracy eşitliğinden hız katkısı çıkarılamaz. Yeni bir araştırma kapsamı açılırsa aynı-state güvenilirliği ve açıkça tanımlanmış tekrarlı/stokastik değerlendirme protokolü önce gelmelidir; bu pilotun başarısız gate'i sonradan geçirilmiş sayılmamalıdır.

## Kanıt ve tekrar üretim

[Run/CPU]4bilimsel arşiv toplam463,411,089byte; Drive ve yerel SHA-256eşleşiyor. Her arşivde yedi standart dosya, logits/PM/index/label, paired states/RNG, exact executed source ve case kontrolleri var. Teknik faz172, A284, B284, C700manifest dosyası. Faz runner duvar zamanları toplam562.164s (≈9.37dk); kurulum, aktarım ve fazlar arası analiz hariçtir. Son yerel CPU suite815pass/2skip/500subtest; independent review ham SHA/manifest/skor/gate/C-repair kararlarını yeniden doğruladı.

Teknik kaynak79445c0+v2overlay`ef423c01...`; pilot kaynak aynı base+v3overlay`f9405232...`, asset manifest`79bc848e...`. [Yürütme notunda tam hash'ler](lion_guidance_pilot_20261107.md) ve raw içindeki source receipts korunur. Setup checkout'u ayrıca dev@a458cd46dır; bilimsel kaynakla karıştırılmaz. Taze native binary'ler korunup önceden belirtilen dört pinned binary yerleştirildi. Son ek paket tam kaynak ve7native ELF içerir; PointNet++/setup EMD/Chamfer hash'lerinin ilk defa koşular sonrası yakalandığı sınırlaması açıkça kayıtlıdır.

[Run/Open] Own runtime `tta-lion-guidance-pilot-a100-20261107-r1` korunuyor; GPU işlerinin tamamlanması kapatma izni değildir. Ayrı halfmix çalışmasının kabul edilmiş bulguları ve runtime'ı bu çalışmadan bağımsızdır. Varsayılan sampler/model/checkpoint değiştirilmedi.
