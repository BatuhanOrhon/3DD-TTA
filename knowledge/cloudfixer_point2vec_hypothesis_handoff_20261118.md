[Run] **2026-11-20 completion update:** A full coverage and B confirmation are
scientifically accepted, downloaded and Drive-backed. A fixed proximal policies
did not beat plain CF. [Final A report](cloudfixer_point2vec_full_results_20261120.md)
and [B confirmation](cloudfixer_pm_hypotheses_confirmation_20261118.md) supersede
pending-result statements below. Owned-r13 flush/unmount and authorized close are verified; see [current execution](cloudfixer_two_colab_execution_20261118.md).
Original handoff/registration instructions below are retained as history.

# CloudFixer devam promptu — 18 Kasım 2026

Bu tez çalışmasına aşağıdaki mevcut worktree ve branch üzerinde devam et. Görevin yalnızca plan hazırlamak değil, gerekli dar kapsamlı kod değişikliklerini tamamlayıp iki ayrı Colab GPU oturumunda deneyleri çalıştırmak, sonuçları almak ve knowledge kayıtlarını güncellemektir. GPU çalışmaları, bu iki araştırma kapsamı içinde süre nedeniyle ek onay beklemeden yetkilidir. Drive hesabı onayı gerekiyorsa bunu kullanıcı yapar.

## Çalışma ortamı ve zorunlu okuma

- Worktree: `/Users/batuhan/.codex/worktrees/0e7f/3DD-TTA`
- Branch: `research/cloudfixer-transfer-20261114`
- Son doğrulanmış araştırma kodu/sonuç commit'i: `f3befcd85fb6fa68a9cd53edf152c7eb36a0a2fe`. Sonraki dokümantasyon commitlerini koru; başlangıçta gerçek HEAD/status kontrolü yap.
- Ana checkout `/Users/batuhan/Desktop/projects/3DD-TTA` ve diğer worktree/oturumların çalışmalarını değiştirme. Yeni worktree açmadan bu mevcut worktree'de ilerle.
- Resmî CloudFixer klonu: `/Users/batuhan/Desktop/projects/CloudFixer`, commit `f043e24414fcf4165fdc892abff21215c9989141`. Resmî kodu yeniden yazma; mevcut kopyalama/paketleme yolunu kullan, gerekli değişiklikleri bizim overlay/runner tarafında yap.
- Önce `AGENTS.md`, `/Users/batuhan/.codex/RTK.md`, `.agents/skills/iterative-thesis-researcher/SKILL.md`, `.codex/skills/colab-research-operator/SKILL.md`, `knowledge/README.md`, `knowledge/thesis_scope.md`, `knowledge/experiment_protocol.md` ve `knowledge/reproduction_gap.md` oku.
- Ardından `knowledge/cloudfixer_transfer_preparation_20261114.md`, `knowledge/cloudfixer_transfer_execution_20261114.md`, `knowledge/cloudfixer_transfer_results_20261115.md`, `knowledge/cloudfixer_paper_protocol_audit_20261115.md`, `knowledge/cloudfixer_nstep_entropy_20261115.md`, `knowledge/cloudfixer_entropy_weighted_pm_20261115.md`, özellikle `knowledge/cloudfixer_terminal_diagnosis_20261115.md` ve ilgili `knowledge/evidence/` kayıtlarını incele. Eski tarihli “live” satırlarını güncel oturum kanıtı sayma.

## Şimdiye kadarki doğrulanmış durum

LION bu CloudFixer deneylerinin hiçbir kolunda kullanılmıyor. CloudFixer'ın resmî ModelNet40 XYZ diffusion ağı ve donmuş sınıflandırıcı kullanılıyor. Bizim yöntemimiz deterministik DDIM-tipi ters zincirde gradient-free proximal SCD düzeltmesi ve ara temiz tahminlerin softmax olasılık ortalaması (PM). Orijinal CloudFixer ise resmî `adapt.cloudfixer(...)` içindeki rotation/delta optimizasyonu; yalnız diffusion çalıştırmak veya bizim tau=0 kolumuz değildir. CF-O ayrı bir yöntemdir ve burada istenen orijinal CF karşılaştırmasının yerine geçmez.

Önceki sonuçlar Point-MAE, ModelNet40-C, tüm15 corruption, severity5,128 geliştirme nesnesi, seed0, B32 içindir. Her ayar1920 tahmin içerir:

| Kayıt | Ayar | Accuracy |
|---|---|---:|
| Orijinal CloudFixer | Üç aynı-seed operasyonel tekrar ortalaması | %63.593750 |
| Eski48-ayar gridinin en iyi PM'i | N32/K80/tau.03, kanonik4 PM | %65.937500 |
| En iyi gerçek terminal | N32/K160/tau.03, yalnız son çıktı | %67.447917 |
| N8 kanonik PM / kendi terminali | K80/tau.1, t=[70,50,30,0] | %65.104167 / %66.093750 |
| N12 kanonik PM / kendi terminali | K80/tau.1, t=[67,47,27,0] | %65.468750 / %64.322917 |
| Şu an gözlenen en yüksek PM | N8/K80/tau.1, eşit PM t=[20,10,0] | %66.822917 |
| N12 geç PM | K80/tau.1, t=[27,0] | %66.250000 |

N=ters güncelleme sayısı; t0 terminal okumasıyla denoiser çağrısı N+1. K,500 seviyeli resmî çizelgedeki başlangıç indeksi; tau adım başına proximal güç; retained=.95. t=[20,10,0] native diffusion indeksleridir, loop numaraları değildir. Tek başına t20 veya t27 sınıflandırması PM ya da gerçek terminal değildir.

Geç-pencere sonuçları kayıtlı logits üzerinde sonradan yapılan keşiftir. N8 late3 kazancı +.729167 puan, nesne-kümeli95%CI[-.052083,+1.562500]; bağımsız doğrulama yok. Entropy ağırlığı N8'e küçük katkı yaptı, N12'yi düşürdü; genel üstünlük göstermedi. Bu sayılar Point2Vec sonuçları veya tam benchmark sonuçları değildir.

Ham kanıtlar: `result/modelnet40_c/cloudfixer_transfer/20261114-171811_development.zip`, `result/modelnet40_c/cloudfixer_exploration/20261115-110019_nstep_run.zip`, `result/modelnet40_c/cloudfixer_exploration/20261115-114749_entropy_run.zip`. SHA'ları ilgili notlarda sabit; arşivleri değiştirme. Son tanı Drive yedeği: `MyDrive/thesis/result/cloudfixer_terminal_diagnosis_20261115/`; eski sonuçların yedek yolları ilgili knowledge/receipt dosyalarında.

## Colab A — Point2Vec ile üç ana yöntemi karşılaştır

Ayrı ve sahipliği kayıtlı bir GPU oturumu oluştur. Aynı resmî diffusion checkpoint'i, aynı donmuş Point2Vec checkpoint'i ve resmî Point2Vec ön işleme/sınıf sırası ile şu üç ayarı test et:

1. **En yüksek mevcut PM adayı:** N8/K80/tau.1/retained.95, eşit olasılık ortalaması t=[20,10,0]. Entropy ağırlığı kullanma. Bunun sonradan seçilmiş aday olduğunu raporla.
2. **En yüksek mevcut gerçek terminal adayı:** N32/K160/tau.03/retained.95; yalnız native t0 terminal tahmini. PM uygulama. Üretim süre ölçümünde gereksiz ara classifier forward'larını çıkar; aynı seed/input/RNG ile terminal çıktısının değişmediğini doğrula.
3. **Orijinal CloudFixer:** Resmî rotation/delta/Adamax adaptasyonu ve Point2Vec için resmî ayarlar, donmuş classifier, plain CF/vote1. Parametreleri Point-MAE runner'ından varsaymak yerine resmî `/Users/batuhan/Desktop/projects/CloudFixer/scripts/run_cloudfixer.sh` ve `/Users/batuhan/Desktop/projects/CloudFixer/adapt.py` ile doğrula.

Source-only Point2Vec accuracy'yi ayrıca kontrol satırı olarak ölç; bu dördüncü bir TTA yöntemi değildir. Önce küçük CUDA/asset/sınıflandırıcı smoke'u ve kaynak tahminlerini doğrula, ardından ModelNet40-C'nin tam2468 nesnesi ×15 corruption ×severity5 üzerinde üç ana kolu çalıştır. Birincil seed olarak önceki paper audit'indeki resmî seed2 akışını doğrulayarak kullan; gerçek seed, sıra ve batch'i raporla. Pilot ayarları B32 ile seçilmişti: resmî akış B1 gerektiriyorsa tam karşılaştırmanın ortak B1 protokolünü önceden kilitle ve bu aktarım farkını belirt. Üç kol aynı ID/sıra/head/ön işleme protokolünde karşılaştırılsın. Tek seed'in sınırını yaz; tam test etiketleriyle yeniden tuning yapma.

Mevcut `scripts/run_cloudfixer_transfer.py` ve hazırlık manifest'i Point-MAE'ye bağlı: sadece checkpoint yolunu değiştirmek yetmez. Resmî `adapt.py` içindeki `Point2VecClassification`, `setup()`, `extract_model_checkpoint` ve classifier ön işlemesini kullanarak en küçük gerekli head desteğini ekle; yükleme anahtarlarını,40 sınıfı ve checkpoint SHA'sını doğrula. Point-MAE geçmiş deney yolunu bozma. Yeni checkpoint/head/probe seçimleri için yeni manifest, görev kimliği ve cache dizini üret; eski Point-MAE cache'lerini kullanma.

Bu üçlü karşılaştırma farklı N/K/tau ayarlarını da değiştirir: sonuçlarını bütün yöntem politikalarının karşılaştırması olarak raporla. PM etkisinin izole testi için Colab B'deki aynı-iz terminal kontrollerini kullan.

## Colab B — PM hangi koşullarda terminali geçiyor?

İkinci bağımsız GPU oturumunu oluştur. Kaydedilmiş dört hipotezi `knowledge/cloudfixer_terminal_diagnosis_20261115.md` üzerinden test et. İlk aşamada önceki Point-MAE head'ini koruyarak nedeni daralt; Point2Vec'e ilişkin genellemeyi ayrıca sınamadan yapma.

- **Probe zamanlaması:** N8/N12, K80/tau.1/retained.95 kontrolünde kanonik PM, tüm-adım PM ve önceden sabitlenen geç PM pencerelerini aynı rekonstrüksiyon izi ve kendi terminaliyle karşılaştır. N8=[20,10,0], N12=[27,0] adayları doğrulama etiketlerini görmeden kilitlensin. Her N için dense logits kaydını bir kez üretip okuma kurallarını yeniden diffusion çalıştırmadan değerlendir.
- **Entropy'nin koşullu güvenilirliği:** Pooled AUC yanında aynı nesnenin doğru/yanlış ara tahminleri arasındaki entropy sıralamasını, yüksek güvenli yanlış adımları ve PM repair/harm sayılarını ölç. Eşit PM ile önceden tanımlı lineer 1-H/log40 ve exp(-beta H) kurallarını karşılaştır; beta0/.5/1/2/4/8 kayıtlı aralıktır. Her corruption'a etiketle farklı ağırlık/pencere seçme.
- **Toplam düzeltme/son adım etkisi:** N değişirken sabit tau ile toplam güç değişiyor. Sabit adım gücünü, açıkça tanımlanmış N*tau eşleştirmeli tanısal kontrolle karşılaştır; bu eşleştirmeyi gerçek eşdeğer dinamik kanıtı sayma. Başlangıç, probe ve terminalde logSNR, düzeltme büyüklüğü, geometrik hareket ve sınıf marjlarını kaydet. Terminalin geç iyi tahminleri hangi nesne/corruption'larda bozduğunu incele; küçük ve sınırlı bir tau kontrolüyle hipotezi sınayarak kör büyük grid açma.
- **Gözlem konumu:** Gerekliyse aynı zincirde proximal düzeltme öncesi/sonrası temiz XYZ probe'larını yalnız gözlem olarak sınıflandır; zincire verilen tensörü veya RNG'yi değiştirme. Ek t0 çağrısının etkisini ayrı göster; ara görüntüyü sessizce yeni terminal diye adlandırma. LION decoder avantajı bu LION içermeyen deneyle doğrudan kanıtlanamaz.

Önce küçük teknik pilotla dense gözlemin trajektori/model state/RNG'yi değiştirmediğini ve eski N8 kayıtlarıyla pariteyi doğrula. Teknik parite kontrolünü tarihsel development128 ID'lerinde, yeni hipotez doğrulamasını ise `scripts/run_cloudfixer_transfer.py:registration()` içindeki ayrık256 confirmation ID'sinde, seed0/1/2 ile yap. Bu ID'lerin önceki kayıtlarla gerçekten ayrık ve bu hipotezler açısından kullanılmamış olduğunu kontrol et; önceden kullanılmışsa yeni ayrık manifest oluştur ve durumu kaydet. A ve B tasarımlarını A'nın tam benchmark sonuçlarını görmeden kilitle. B'de bağımsız doğrulama olarak ayırdığın etiketlerle pencere/ağırlık arama; sonraki aramayı ayrı development fazı olarak işaretle.

## Kod, çalışma ve teslim gereklilikleri

Başlıca kodlar: `cloudfixer_diffusion.py`, `pointcloud_proximal.py`, `cloudfixer_store.py`, `scripts/prepare_cloudfixer_transfer.py`, `cfgs/cloudfixer_transfer_assets.json`, `scripts/run_cloudfixer_transfer.py`, `scripts/run_cloudfixer_exploration.py`, `scripts/analyze_cloudfixer_exploration.py`, `scripts/analyze_cloudfixer_weighted_pm.py`, `scripts/diagnose_cloudfixer_terminal.py`; ilgili `tests/test_cloudfixer_*.py`. Geç3 PM'nin classifier çağrı sırası ve RNG izolasyonu GPU'da ayrıca doğrulanmalı; eski dense CPU analizini yeni sparse uygulamanın GPU paritesi sayma.

Colab CLI kullan. Her yeni VM için AGENTS.md'nin emrettiği `scripts/colab_environment_setup.ipynb`10 hazırlık hücresini ve `knowledge/colab_cli_setup_20261017.md` kayıtlarını izle: cell1 ayrı, restart sonrası aynı VM'ye bağlan, cells2–10 sıralı/doğrulanmış; yalnız bilinen cell2 fastrlock hatası ertelenebilir. Sonra CloudFixer/Point2Vec özel bağımlılık, CUDA ve asset kontrollerini yap. Hazırlıkta araştırma hücrelerini çalıştırma. Hesap değiştirme veya gcloud/API yetkisi akışı açma; yalnız Colab Drive mount ve gerektiğinde kullanıcı consent'i.

İki VM'nin adlarını görevleriyle ayır, sahiplik/endpoint/monitor kayıtlarını tut. Her faz için kaynak snapshot/hash, modeller/veri/IDs/seeds/batch/probes, tüm fazın süre/CU tahmini, toplam sınırı, hata durdurma ve kurtarma planını yazıp uygula. Uzun sürmesi ek GPU onayı isteme gerekçesi değildir. GPU kapasitesi paralelliği engellerse kayıt tut, tamamlananı koru; paralel gerçekleşmiş gibi raporlama.

Sonuçları batch/unit bazında Drive'a atomik receipt+SHA ile artımlı kaydet; kernel/VM kaybından sonra tamamlanmış birimleri doğrulayarak resume et, yanlış head/seed/config cache'lerini reddet. A/B'ye ayrı görev/cache/Drive dizinleri ver. Ham logits, olasılık, entropy, label/orijinalID, native indeks/logSNR ve gerekli tanısal veriyi sakla. Sonuçları CLI ile yerel canonical `result/` altına indir; SHA/CRC/coverage/label ve metrikleri yeniden say. Standart command/config/environment/stdout/summary/per_corruption/notes kayıtlarını, teknik başarısızlıkları ve negatif sonuçları koru.

Teslimde A için üç kol + source accuracy, her corruption, süre/bellek; B için aynı-iz PM−terminal farkı, repair/harm, seed dağılımı, orijinal nesneye göre kümeli güven aralıkları ve hipotezlerin destek/çürütme durumunu tabloyla ver. Macro15/macro14, severity ve pilot/tam veri sayılarını karıştırma. Knowledge/findings/open_questions/README güncellensin. Çalışma bitse veya VM boşta olsa bile kullanıcı görevin bittiğini ve oturumların kapatılabileceğini açıkça söylemeden runtime kapatma.
