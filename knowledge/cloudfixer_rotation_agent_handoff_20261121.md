# Yeni oturum için görev · CloudFixer XYZ zincirinde gradyansız rotasyon

3DD-TTA tez projesinde donmuş CloudFixer XYZ denoiser'a uyarladığımız proksimal SCD ve geç olasılık ortalaması (PM) üzerine iki rotasyon yöntemi uygula ve değerlendir. Birincisi SCD eşleşmelerinden kapalı formda rijit dönüş kestirimi, ikincisi donmuş sınıflandırıcıyla gradyansız poz seçimi. Hem Rotation kazanımını hem diğer bozulmalarda doğru sınıflandırmaların kaybını ölç. İki yöntemi ayrı kollar olarak çalıştır. Tek adımda doğru pozu bulmayı ve kademeli çözümün üstünlüğünü açık araştırma soruları olarak ele al.

## 1. Çalışmaya ayrı worktree'de başla

Kök repo `/Users/batuhan/Desktop/projects/3DD-TTA`. Önce `AGENTS.md`, ilgili tez becerisi ve `knowledge/README.md` talimatlarını oku. Uzak dalı güncelle. **Yeni ve ayrı bir worktree oluştur; yeni çalışma branch'ini o anda güncel olan `origin/thesis-integration` üzerinden türet.** Araştırma notlarının yayın commit'i `39d293b5018d444769338046e53377364ef11bf1`. Bu commit yeni tabanın geçmişinde bulunmalı. Başlangıç SHA'sını kayda geçir. Mevcut kullanıcı ve önceki araştırma worktree'lerini kendi çalışma alanın olarak kullanma.

```sh
repo=/Users/batuhan/Desktop/projects/3DD-TTA
rotation_branch=experiment/cloudfixer-rotation-pilot
rotation_tree=/Users/batuhan/.codex/research-checkouts/cloudfixer-rotation-pilot
git -C "$repo" fetch origin
git -C "$repo" merge-base --is-ancestor 39d293b5018d444769338046e53377364ef11bf1 origin/thesis-integration
git -C "$repo" worktree add --no-track -b "$rotation_branch" "$rotation_tree" origin/thesis-integration
cd "$rotation_tree"
git status --short --branch
git rev-parse HEAD
```

Komutları sırayla uygula. Ata kontrolü başarısız olursa fetch edilen uzak dalı doğrulayıp yayın kaydını bulmadan worktree oluşturma. Dal veya dizin kullanılıyorsa ikisine de yeni ve benzersiz bir ek seç. Önceki deney arşivleri başka worktree veya Drive'daysa doğrulanmış olanları salt okunur kullan.

## 2. Mevcut deneyi yeniden kur

- Önce [rotasyon notunu](cloudfixer_gradient_free_rotation_20261121.md), [temel yöntem kararını](cloudfixer_baseline_rotation_background_20261121.md), [tam Point2Vec sonuçlarını](cloudfixer_point2vec_full_results_20261120.md) ve [deney protokolünü](experiment_protocol.md) oku. [Sampler](../cloudfixer_diffusion.py), [proksimal çözüm](../pointcloud_proximal.py) ve [resmî karşılaştırmayı çalıştıran runner](../scripts/run_cloudfixer_point2vec.py) üzerinden gerçek çağrı sırasını takip et.
- Birincil sistem donmuş **Point2Vec + CloudFixer XYZ**, N8/K80, τ=0,1, %95 çift yönlü kırpma, t20/t10 ön-düzeltme ve gerçek t0 terminalinin eşit PM'si. Temel bütçe 9 denoiser ve 3 sınıflandırıcı çağrısı. Standart N8 ve N32, CloudFixer'ın optimize ettiği Δ veya R'yi kullanmıyor. Plain CloudFixer dış kıyas, rotasyonsuz N8 her adayın eşlenmiş iç kontrolü. Point-MAE ayrı bir aktarım sorusudur.
- Referans sonuçlar şiddet 5, 2.468 nesne, B1 ve seed 2'ye ait. N8 / plain CF sırasıyla Rotation'da %48,06 / %69,49, Background'da %65,32 / %76,09, tüm 15'te %74,95 / %77,24. Kalan 13'te fark 53 net doğru. N32 terminal Background'da %76,46. Aynı arşiv ölçümünü kullanırken protokole uyan seed ve nesneleri eşleştir.

## 3. Yöntem 1 · Tek ve kademeli Kabsch

1. [CPU geometri kontrolünü](evidence/cloudfixer_rotation_20261121/geometry_check.py) çalıştır. Satır vektörü konvansiyonu, dönüş yönü, determinant +1, maske/tekrarlı indeks, yetersiz geometrik destek ve birim dönüşte temel yolun eşdeğerliğini doğrula. Başlangıç gürültüsü ve prob başına FPS rastgeleliği eşleşsin.
2. Model çağrısından sonra, yerel SCD'den önce temiz tahmini q al. Ham X'i koru. Pozlanmış gözlem XP ile q arasında geçerli çift yönlü %95 karşılıkları kur. Artık dönüş D'yi 3×3 ağırlıklı Kabsch ile hesapla, **P ← PD** yap, sonra `correct(q, XP, mask, ...)` uygula. Yetersiz destek varsa artımı atla ve önceki P'yi koru. q'yu bozuk X'e geri hizalama. Ön-düzeltme PM okumaları ve gerçek t0 terminali mevcut anlamını korusun.
3. Dört kolu aynı koşullarda karşılaştır: birim dönüş, t80'de tek çözüm, t80'in **aynı** q hedefinde üç yeniden eşleme/çözüm ve t80/t50/t20'de **farklı** q hedefleriyle birer çözüm. Son iki kolun üçer çözüme sahip olması, daha fazla ICP iterasyonu ile denoising zamanına yaymanın etkisini ayırır. Aynı toplam açı bütçesini kullan. Adım başına açı, toplam açı, katedilen açı, eşleşme desteği, ret oranı, yeniden eşleme ve SVD sayısını tut.
4. Poz sinyalini, kimlikleri ayrılmış nesnelerde bilinen küçük/bileşik/büyük sentetik dönüşler ve temiz, Background, Rotation+Background koşullarında ölç. Gerçek ters dönüşü yalnız analizde kullan. Bilinen simetrilerde eşdeğer pozları dikkate al. Gerçek modelde yön sinyali yoksa veya artımlar sürükleniyorsa kayda geçir. Welsch ağırlığı ek bir seçenek; önce standart kırpma sonucunu al ve Welsch'i ayrı bir ablasyon olarak değerlendir.

## 4. Yöntem 2 · Sınıflandırıcıyla poz seçimi

- İlk geç prob t20'de birim dönüş ve üç eksende ±θ ile yedi aday dene. Aynı FPS çekilişiyle donmuş başlığın sınıf güvenini ve açı cezasını kullan. Kimlikten yeterli ilerleme sağlamayan girdide kimliği seç. θ, ceza ve kabul eşiğini bağımsız geliştirme aşamasında dondur.
- Aynı t20 kararıyla iki kol kur. İlkinde seçilen R yalnız t20/t10/t0 çıktılarına uygulanır. İkincisinde t20'de SCD sonrası bulutla gürültüyü birlikte döndür, sonraki gözlem referansını XR yap ve denoising'i sürdür. Daha sonra R'yi tekrar uygulama. t20 sınıflandırması geri besleme kolunda sampler içinde zamanında yapılmalı. Seçilen logiti yeniden kullanarak iki kolda da 9 denoiser + 9 sınıflandırıcı çağrısını doğrula. Eş sorgu bütçeli seçim kontrolüyle karşılaştır.
- Geometrik hatanın veya sınıflandırıcı güveninin iyileşmesini doğruluk kazanımı sayma. Aynı nesne ve bozulmalarda yüksek güvenle yanlış sınıfa geçen örnekleri de raporla.

## 5. Diğer bozulmalardaki kaybı ve maliyeti karar ölçütü yap

- Test sonuçları görülmeden nesne manifestini, seed planını, aday seçme kuralını, ortak açı/süre bütçesini ve izin verilecek clean / Rotation hariç 14 doğruluk düşüşünü yüzde puan cinsinden gerekçelendirip kaydet. Daha önce incelenmiş tam benchmark'ın geçmiş maruziyetini belirt. Her girdide aynı etiketsiz çıkarım kuralını kullan. Corruption adı veya gerçek açı ile yönlendirme yapma.
- İlk aşama iki yöntem için eşlenmiş ve yeniden başlatılabilir pilot olsun. Geliştirme ve doğrulama nesnelerini ayır. Seed 0/1/2 için eşlenmiş planı ve bütçeyi kaydet. Gerçek küçük pilotun süre/belleğini ölç, sonra seçilmiş ve dondurulmuş adayı tam 2.468 × 15 şiddet-5 değerlendirmesine taşımanın maliyetini hesapla. Devam ölçütleri karşılanmazsa o kolun negatif sonucunu bildir. Eksik kapsamı açık bırak.
- **Her bozulmayı ayrı bir satırda göster.** Aday ve N8 doğru/toplam, accuracy, eşlenmiş yüzde puan farkı, yanlış→doğru ve doğru→yanlış, seed farkı ve belirsizlik ver. Clean, Rotation, Background, Rotation hariç 14, Rotation ve Background hariç 13, en olumsuz bozulma ve all15 özetlerini yayımla. Nesne bootstrap'inde aynı nesnenin tüm bozulma ve seed'lerini birlikte çek. Çoklu karşılaştırmaları gözet. Herhangi bir kaybı makro ortalamayla örtme. Düşük örnek sayısında geniş aralıkları aynen raporla.
- B1'de aynı donanım, ısınma ve CUDA senkronizasyonu ile gecikme ve tepe belleği ölç. Toplam, mesafe/yeniden eşleme, SVD, denoiser ve sınıflandırıcı maliyetlerini ayır. Her yöntemi doğruluk, zarar ve süre ölçütlerinin üçüyle değerlendir. Olumlu Rotation sonucu tek başına varsayılan yöntemi değiştirmeye yetmez.

## 6. Atıf, Colab ve teslim

**Atıf zorunludur.** İki yöntemden biri tez veya makaleye girerse yöntem bölümünde kullanılan bileşeni ilham veren çalışmayla açıkça ilişkilendir. Her iki kolda **CloudFixer ve 3DD-TTA**, ilk kolda **Kabsch ve Trimmed ICP**, ikinci kolda **Park ve Park / CanDO** atıflarını ver. Welsch kullanılırsa ayrıca **Zhang, Yao ve Deng**'e atıf ver. Cebirsel türetim kullanılırsa Lawrence, Bernal ve Witzgall'ı kaynak göster. Özgün katkı iddiasını gerçekten geliştirdiğimiz entegrasyon ve kanıtlanan etkiyle sınırla. Birincil makale bağlantıları ve kullanım koşulları [atıf tablosunda](cloudfixer_gradient_free_rotation_20261121.md).

Colab gerekiyorsa `.codex/skills/colab-research-operator/SKILL.md`, `AGENTS.md` ve [Colab kurulum kaydını](colab_cli_setup_20261017.md) uygula. CLI kullan. Önce tüm GPU fazının beklenen süre/maliyetini, durma sınırını ve kurtarma planını kaydet. Görev kapsamındaki çalıştırma için ayrı GPU onayı isteme. Etkileşimli hesap onayı ve runtime kapatma için mevcut kuralları koru.

Teslimde çalışma branch'i, başlangıç ve sonuç commit'leri, yöntem bayrakları, kimlik/seed manifesti, koşu ve arşiv yolları, doğrulama kanıtı, bozulma tabloları, maliyet, atıf eşlemesi ve hangi adayın hangi gerekçeyle sürdürüleceği yer alsın. Knowledge kayıtlarını güncelle; uygulanmamış önerileri, CPU kontrollerini ve gerçek model sonuçlarını ayrı etiketle.
