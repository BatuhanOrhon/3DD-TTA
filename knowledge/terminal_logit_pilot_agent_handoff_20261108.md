# Yeni agent handoff: terminal hızlandırma ve Logit Mean küçük pilotu

Tarih: 2026-11-08. **[User decision/Open] Bu dosya görev devridir; yeni pilot
henüz uygulanmadı veya çalıştırılmadı.** Hazırlayan agent yalnızca bu belgeyi
yazar. Uygulama ve yeni Colab oturumu, bu belgeyi alan agent'ın görevidir.

## Agent'a doğrudan görev

İki yaklaşımı küçük ve kontrollü bir pilotla gözlemle:

1. **PM olmadan terminal 3DD-TTA'ya mevcut proximal hızlandırmayı eklemek.**
   Orijinal N5 terminal ve ayarlanmış N8 terminal için ayrı, aynı horizonlu
   classic/cheap çiftleri karşılaştır. Accuracy ve gerçek terminal-only süreyi ölç.
2. **Aynı N8 yörüngesinin class logitlerini ortalamak.** Hızlandırılmış mevcut
   baseline'ın dört adayında `softmax(mean(logits))` ile
   `mean(softmax(logits))` karşılaştır. İki reduction için aynı kayıtlı logits'i
   kullan; ikinci diffusion koşusu başlatma.

**Yeni bir worktree oluştur; en güncel `origin/thesis-integration` üzerinden
yeni bir experiment branch türet; bütün uygulama ve deney çalışmalarını bu
branch/worktree üzerinde yap.** Kullanıcının açık/dirty ana checkout'unu veya
başka agent'ın branch, dosya, runtime ve monitor süreçlerini değiştirme.
Yeni, görevine özel isimli bir High-RAM A100 Colab oturumu kullan.

## 1. Başlangıç ve izolasyon

Önce repo `AGENTS.md`, `.agents/skills/iterative-thesis-researcher/SKILL.md`,
`.codex/skills/colab-research-operator/SKILL.md` ve kod değişikliği için
`.codex/skills/development-agent/SKILL.md` oku. Bunlar gerçek checkout'ta ve
güncel integration branch'inde uzlaştırılmalı. Knowledge giriş noktası
`knowledge/README.md`; ayrıca `thesis_scope.md`, `experiment_protocol.md`,
`reproduction_gap.md` ve aşağıdaki görev belgelerini incele.

Örnek başlangıç komutları; isim/path mevcutsa silme veya resetleme, yeni bir
suffix seç. Komutlar ana checkout'taki kullanıcı dosyalarına dokunmaz:

```bash
git -C /Users/batuhan/Desktop/projects/3DD-TTA fetch origin
git -C /Users/batuhan/Desktop/projects/3DD-TTA worktree list
git -C /Users/batuhan/Desktop/projects/3DD-TTA worktree add \
  -b experiment/terminal-logit-pilot-20261108 \
  /Users/batuhan/Desktop/projects/3DD-TTA-worktrees/terminal-logit-pilot-20261108 \
  origin/thesis-integration
cd /Users/batuhan/Desktop/projects/3DD-TTA-worktrees/terminal-logit-pilot-20261108
git status --short --branch
git rev-parse HEAD
```

Bilinen integration referansı `87525568f6775da5f906f0ea66b8d5b159756d46`;
bu handoff daha sonraki bir commit ile yayımlanır. **Eski SHA'ya checkout yapma**:
fetch sonrası güncel remote tipinden başla; gerçek base SHA'yı kaydet.
Çalışmayı mevcut `thesis-integration` üzerinde doğrudan yapma. Sonuçlar
tamamlanınca gerekli kod/test/knowledge değişikliklerini son remote tipine
uzlaştırıp yalnız `origin/thesis-integration` hedefine normal push ile yayımla;
force-push, main/upstream push veya peer dosyalarını eski kopyayla ezme yok.

## 2. Okunacak kanıt ve mevcut durum

- [Aktif baseline](current_baseline.md).
- [PM-off terminal planı](plans/proximal_terminal_without_pm_20261108.md).
- [LM–PM planı](plans/logit_mean_readout_20261108.md).
- [15 bozulmanın common-horizon sonuçları](lion_guidance_uniform_horizon_results_20261108.md).
- [Süreç/setup hata ve kurtarma kaydı](lion_guidance_speed_20261107.md), özellikle
  Nov8 alternatif Conda bootstrap ve doğrulanmış cell1–10 devamı.
- [Colab setup kaydı](colab_cli_setup_20261017.md).
- [Önceki oturumların kapatılma kanıtı](evidence/lion_guidance_speed_20261107/colab_shutdown_20261108.json).
- [Baseline entegrasyon/test kaydı](evidence/lion_guidance_speed_20261107/baseline_promotion_20261108.json).

[Run] Önceki istenen tam-veri hız testleri tamamlandı; R5 ve R6 Nov8'de
arşivlenip kapatıldı. Bu, şu anki server inventory'sinin yeniden kontrolünün
yerine geçmez. Başka bir çalışma yeniden başlamış olabilir; oturum sahipliğini
isimden/idle durumdan varsayma. Yarım kalan peer617 GSD ekranını devam ettirme.

[Run] Kabul edilmiş Point-MAE/ModelNet40-C, 2468 nesne ×15 bozulma/s5/seed0,
B128/final36: N5 terminal61.445165%; tunedN8 terminal62.571583%; N8PM63.735818%;
cheapN8PM63.722312%. Cheap–N8PM−.013506pp, koşullu95%CI[−.180983,+.153971],
%27.673026 daha az pipeline süresi. Cheap hâlâ N5 terminalden %33.437234 daha
yavaş. **Bağımsız N8 terminal latency ölçümü yok.** PM edinme süresini terminal
süresi olarak kullanma. Eski BackgroundN35 sayıları ayrı tarihsel politikadır.

[Code/Open] `scripts/eval_chamfer_baseline.py` varsayılan profili
`lion_proximal_pm_n8_v1`: bütün bozulmalarN8, cheap tau=.1, retained.95,
PM2/4/6/terminal, original encoded decoder style, controlled backend. Kod
entegrasyonu875 CPU test/2skip/564subtest ile doğrulandı; yeni shared-artifact
CLI host'u GPU'da yeniden benchmark edilmedi. Audit/cache/RNG akışı ve NumPy
float64 PM, kabul edilmiş lightweight speed host'undan farklıdır. %27.67'yi
yeni wrapper'ın otomatik hız sonucu sayma. Diğer head/Scan transferi bu pilotta yok.

## 3. Küçük pilotun dondurulmuş kapsamı

Aşağıdaki pilot boyutu bu handoff'un önerisidir; kullanıcının verdiği bir
sayısal başarı garantisi değildir. Çalıştırmadan önce registration'a yaz:

- ModelNet40-C, bütün15 bozulma, severity5, yalnız frozen40-class Point-MAE.
- Pilot: **32 ortak original object ID**, inference seed0, **B32**; her
  corruption/arm aynı32 ID. B32 burada küçük kohort boyutudur; B128'den yeni
  bir optimal batch seçildiği veya aynı throughput sağlandığı iddia edilmez.
- Teknik smoke: ayrı4 ID, B2, Gaussian/Background/Density/LiDAR. Pilot+smoke
  listelerini accuracy görmeden `np.random.default_rng(20261108)` ile tek
  permütasyondan seç, JSON'da açıkça kaydet/hashle; sınıf/başarıya göre yeniden
  seçme. Tam benchmark zaten development'ta görüldü: yeni ID manifesti bu
  veriyi unseen confirmation'a dönüştürmez.
- Pilot seed0 için **3 fresh operational repeat**; primary accuracy repeat0,
  repeat1/2 ayrı stabilite/timing sonuçları. Birleştirip3 bağımsız veri kümesi
  veya ensemble accuracy üretme. Model inference seed'i ile repeat draw key'ini
  ayrı kaydet. Timing dağılımını ve tekrarlar arası accuracy'yi göster.
- Her timed arm/corruption için iki sabit warmup; aynı ölçüm sınırı, aynı
  preprocessing ve kaynak ağırlıkları. Sıra etkisini arm döngüsüyle dengele.
- Beklenen araştırma GPU fazı için **yaklaşık15–45 dakika** bir planlama
  tahminidir; yeni B32 host'u henüz ölçülmedi. Setup/native build dahil toplam
  Colab tahsis süresi daha uzun olabilir. Smoke sonrası gerçek süre tahminini
  güncelle; bilimsel faz için45dk yumuşak bütçe belirle. Sığmıyorsa kapsamı
  sessizce parçalama/tam benchmark'a genişletme; partial kapsamı kaydedip raporla.

Beş inference konfigürasyonu yeterli; LM altıncı GPU arm'ı değildir:

| ID | Guidance / horizon | Readout | İşlev |
|---|---|---|---|
| T5 | Gerçek original terminal SCD, N5, gamma=eta=.01/p=.95 | Yalnız terminal | Orijinal politika kontrolü |
| P5 | Cheap proximal tau=.1/retained.95, N5 | Yalnız terminal | T5'e matched hızlandırma |
| T8 | Gerçek terminal SCD, N8, gamma=.03/eta=.003/p=.95 | Yalnız terminal | Ayarlanmış politika kontrolü |
| P8 | Cheap proximal tau=.1/retained.95, N8 | Yalnız terminal | T8'e matched hızlandırma |
| P8-PM | Aynı cheapN8, tau=.1/retained.95 | clean2/4/6 + terminal8 | PM–LM için tek ortak logits kaynağı |

Her horizon bütün bozulmalara aynen uygulanır; Background35 veya label/corruption
kimliğine göre yöntem seçimi yok. P5/P8'de gamma/eta gradient uygulanmaz;
tau gerçek strength'tir, direct style update ve denoiser backward kapalıdır.
T5–P5 ve T8–P8 temel terminal karşılaştırmalarıdır. N5–N8 farkını yalnız
hızlandırmaya atfetme. P8-PM–P8, PM edinme maliyeti için ikincil karşılaştırmadır;
aynı son logit eşitliği native tekrar floor'u ölçülmeden varsayılmaz.

## 4. Sıralı uygulama batch'leri ve geçiş koşulları

### Batch A — CPU hazırlık, kayıt ve minimal uygulama

- Hedef/kapsam: yukarıdaki5 konfigürasyonu mevcut lightweight speed mantığında
  çalıştırabilecek en küçük değişiklik; güncel baseline varsayılanını değiştirme.
- İncelenecek alanlar: `scripts/run_lion_guidance_speed.py`,
  `lion_guidance_pilot.py`, `scripts/eval_chamfer_baseline.py`,
  `scripts/eval_chamfer_depth.py`, `scripts/eval_multibackbone_guidance.py`,
  `shared_trajectory.py`, `legacy_trajectory.py` ve
  `scripts/analyze_legacy_trajectory.py:policies`.
- Kritik eksik: cheap sampler şu anda horizon olarak8/35 kabul ediyor ve
  checkpoint/snapshot mantığı PM pilotuna ait. **N5cheap ve gerçek terminal-only
  yol henüz yok.** N5 desteği ile readout kapama davranışını açıkça uygula;
  sadece callback'i boş bırakıp gereksiz intermediate decode/head veya latent
  snapshot kopyalarını sürdürerek terminal hızını ölçme. N8/35 eski çağrılarının
  matematiğini ve defaultlarını koru. Sampler'ın `full` shim'i, gerçek legacy
  terminal kontrolünün yerine geçmiş parity başarısızlıkları gizlenerek konulamaz.
- Aynı küçük runner içinde aynı preprocessing/native/weights/backend ve timing
  sınırını kullan. Ağır audit wrapper süresiyle lightweight kontrol süresini
  karşılaştırma. Hazır olmayan komut/flag'i mevcut çalışır CLI gibi yazma.
- LM: logits **son class projection'ın softmax öncesi C=40 skorlarıdır**;
  penultimate hidden embedding veya LION feature kanalı değildir.
  `PM=mean(softmax(l_j))`, `LM=softmax(mean(l_j))`, eşit ağırlık/T1.
  Readout4'lü aynı tensörden hesaplanır; timestep/weight/temperature tuning yok.
- Doğrulama: N5/N8 gerçek scheduler/update sayısı; terminal arm'larda tam bir
  decode/classify, sıfır intermediate probe; orijinal decoder grad/style
  contract; eski PM davranışı; LM/PM aynı ve farklı karar verebilen sentetik
  örnekler; finite/shape/ID/policy guard'ları; source-freeze öncesi lazy-import
  gereksinimleri. Hedefli CPU testleri, sonra etkilenen ortak yolların suite'i.
- Knowledge hedefi: yeni görev notu
  `knowledge/terminal_logit_pilot_study_20261108.md`, source/registration/manifest
  hash'leri ve iki mevcut planın durum güncellemesi. Dondurulmuş pilot kaydı
  olmadan GPU'ya geçme. Routine implementation için yeni kullanıcı onayı gerekmez.

### Batch B — yeni Colab, ortam ve teknik smoke

- CLI kurulumunu güncelleme/başka hesap yaratma. `colab version`, `colab --help`
  ve ilgili altkomut help'ini kontrol et; mevcut kullanıcı auth'unu kullan.
  Kullanıcı GPU koşuları için tekrar izin gerektirmediğini önceden belirtti;
  scope küçük pilotla sınırlı. Google Drive interactive consent ayrı gereksinimdir.
- Göreve özgü isim örneği `tta-terminal-logit-pilot-a100-20261108-r1`;
  gerçek UTC tarihi ve boş suffix kullan. Server inventory'de çakışmayı kontrol
  et; High-RAM/A100 iste. Allocation başarısızsa farklı GPU'ya sessiz geçme.
  Session/endpoint/owner/branch/worktree/monitor/output yollarını görev notuna yaz.
- Kullanıcı hesabındaki kredili Drive'ın `MyDrive/thesis` kısayolu daha önce
  paylaşılan thesis klasörüne erişti. Gerçek mount/kısayol/dataset/checkpoint
  erişimini doğrula; adı görmek yeterli değil. Yetki gerektiğinde **yeni gerçek
  session adıyla** kullanıcıya `colab drivemount -s SESSION /content/drive`
  komutunu ver. Token/OAuth/session dosyalarının içeriğini okuma veya loglama.
- Her yeni VM: `scripts/colab_environment_setup.ipynb` içindeki10 kod hücresi.
  Cell1 tek başına; Conda restart sonrası **aynı VM'ye reconnect**; cell2–10
  sırayla. CLI çağrılarında working directory'yi açıkça `/content/3DD-TTA` yap.
  Sadece bilinen cell2 fastrlock pip hatası ertelenebilir; diğer shell/marker/
  native/data/checkpoint hatalarında setup tamamlandı deme. Cell5–8 `.complete`
  marker'ları ve cell10 gerçek GPU/data/weights kanıtı zorunlu; full notebook
  veya `Run all` yok. Native kurulum/research kaynak deployment'ını ayrı kaydet.
- Nov8 GitHub installer5xx geçmişini incele. Resmî Anaconda URL+SHA ile
  CondaColab `install_from_url` alternatif bootstrap daha önce Colab'da geçti;
  gerektiğinde kayıtlı reçeteyi kontrol edip sadece Colab üzerinde uygula,
  adaptation'ı kaydet. Mac'e Linux installer/model ortamı kurma. Bugünkü URL,
  kernel-Python ve SHA'yı doğrulamadan eski bootstrap'ı körlemesine yürütme.
- Eski bilimsel asset receipt yeni kodu kapsamaz: current source snapshot,
  dirty/untracked gerekli dosyalar, native/data/checkpoint hash'leri ve gerçek
  runtime/weights doğrulamasını yeni receipt ile bağla. Yeniden native build
  yaptıysan eski native kimliğini varmış gibi yazma veya sessiz restore etme.
- Tek CLI owner/monitor; eşzamanlı notebook hücresi veya duplicate job yok.
  Gerçek aşamalar heartbeat üretsin; sahte keep-alive/busy loop kullanma.
- Teknik4-ID smoke'ta5 arm'ı, 5/8 update ve1/4 readout sayısını, model frozen
  hash'lerini, finite sonuçları ve3 aynı-state/RNG repeat'i incele. Önceki native
  nondeterminism biliniyor; aynı-state tekrar floor'u ölç/raporla, toleransı
  sonuca göre gevşetme. Finite/source/weights/count hatası pilotu durdurur.
  Repeat eşitliği başarısızsa bunu saklama: matched causal/parity iddiası geçmez;
  descriptive fresh-repeat pilotunun sınırını açıkça kaydet.
- Her material setup/smoke sonucunu görev notu/findings'e ve Drive'a kaydet.
  Başarılı zorunlu setup/teknik gate'lerden sonra pilotu aynı VM'de sürdür.

### Batch C — pilot, arşiv ve analiz

- Her timed arm için raw CPU batch hazır sınırından itibaren fresh preprocessing,
  fresh native encode/noise, sampling, gereken decode/FPS/head ve reduction;
  CUDA synchronize ile gerçek süre ölç. Cache hit ile miss'i karşılaştırma.
  Aynı method-independent draw seed'ini kullan; arm ID seed'e girmesin.
  Native encode nondeterminism nedeniyle seed eşitliği state eşitliği değildir.
- Kaynak/dataset yükleme, warmup, RNG seeding, model hashing, D2H audit, ZIP ve
  Drive yazma sürelerini ölçümden eşit dışla ve ayrı kaydet. Peak allocated ve
  reserved memory'yi arm başına kaydet. B32 amortized ms/input, **B1 latency değildir**.
- P8-PM `B×4×40` logits/candidate order/ID/label/PMprobability/prediction kaydet.
  LM'yi bu tensörden hesapla; aynı precision kullan. Archived GPUfloat32 ile
  CPUfloat64 PM near-tie farkını ayrıca say; reduction farkıyla karıştırma.
- Accuracy ve exact correct counts: T5/P5/T8/P8/P8-PM/P8-LM; all15/macro14 ve
  her corruption. Terminal çiftleri, LM−PM ve ikincil PM−terminal için paired
  corrected/harmed/net ve pp farkı. Object-cluster bootstrap10000draw/seed20261108;
  repeat0 primary CI, repeat1/2 ayrı; corruption CI'ları descriptive/unadjusted.
- Terminal çiftlerinde `speedup=t_classic/t_cheap`,
  `%time_saved=100*(1-t_cheap/t_classic)`, saniye ve amortizedms ver.3timing
  repeat'in hepsini göster; yalnız en hızlı tekrar seçilmez. LM reduction
  overhead'i gerekirse ayrıca ölç; offline analiz süresini inference hızı sayma.
- LM için centered-logit norm/entropy/margin ve disagreement tanılarını raporla;
  centered/logit-normalized alternatifini ana yönteme sessiz ekleme. Target
  label yalnız değerlendirmeye girer. Öncelik olumlu sonuç bulmak değil, matched
  sonucu dürüstçe gözlemlemektir.32-ID CI ile−.3pp noninferiority ispatı bekleme.
- Yeni output kökü örneği
  `result/modelnet40_c/terminal_logit_pilot/<UTC_RUN_ID>/`; Drive karşılığı
  `MyDrive/thesis/result/terminal_logit_pilot_<UTC_RUN_ID>/`. Mevcut dosyaları
  overwrite etme. Her tamamlanan corruption/phase'i checkpointle; başarısız
  denemeleri de kendi klasöründe sakla. Required command/config/environment/
  stdout/summary/per_corruption/notes, logits/predictions/manifests, kaynak
  snapshot/native/weights/RNG bilgileri eksiksiz arşivlensin.
- ZIP path/CRC/member/hash, counts/coverage/config/log/model-state kontrolleri;
  Drive ve Mac kopyasının size/SHA eşitliği. Raw ZIP değişmez; derived analiz
  ayrı dizinde. Findings/open_questions/görev notu ve iki plan güncellensin;
  sonuçlar olumlu, negatif veya inconclusive olabilir.

## 5. Sınırlar, durma koşulları ve teslim

Bu görev **küçük pilot**; 2468-ID full run, yeni seed1/2, Scan/başka head,
hyperparameter rescue sweep, feature/GSD/global-z veya baseline'ı LM'ye
çevirme kapsamda değildir. İyi pilot sonucu otomatik genişleme yetkisi vermez.
İlgili eski full sonuçları yeniden üretmek için sıfırdan çalıştırma.

Kullanıcının önceki talimatı: runtime/bağlantı sorunu tekrarlanırsa yeni aksiyon
almadan kullanıcı komutunu bekle. Kayıp/disconnect, çalışan remote işin bittiği
anlamına gelmez: eldeki çıktıyı koru, durumu bildir; otomatik retry/replacement,
aynı işi tekrar launch veya başka oturumda recovery yapma. Yerel bilgi/status
ile elde edilemeyen remote durumu "unknown" bırak; belirsizken yeni hücre gönderme.

Bu yeni oturum için önceki R5/R6 kapatma iznini devralma. Pilot bittiğinde
doğrulanmış arşiv/sonuçları teslim et, session adını ve canlı/idle/unknown
durumunu açıkça bildir; kullanıcı bu yeni işin tamamlanıp kapatılmasına izin
vermeden runtime'ı sonlandırma. Peer/unknown oturumlara müdahale etme.

Son teslim: yeni branch/worktree ve actual base/source commit;5-arm config ve
manifest; teknik gate ve repeat floor'u; all15/per-corruption accuracy ve
terminal süre tablosu; LM−PM paired sonuçları/CI; raw Drive+Mac SHA receipt'leri;
tam/partial/failed durum; sınırlamalar; sonra istenirse yapılacak bir sonraki
dar doğrulama önerisi. Gerekli code/test/knowledge'i integration'a yayımla.

[Paper/Inference] LM atfı ve kapsam farkı
[kayıtlı planda](plans/logit_mean_readout_20261108.md) var: Li et al. arXivv3
B.4/D ayrı timestep-probe logits'lerini ortalıyor; biz aynı donmuş head'in
decoded point-cloud çıktılarındayız. Generic logit ortalamasını ilk buluş
veya pilot gain'ini genellenmiş accuracy katkısı olarak sunma.
