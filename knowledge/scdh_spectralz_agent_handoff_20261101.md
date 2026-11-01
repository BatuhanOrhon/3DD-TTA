# Yeni agent başlangıç promptu — h:SCD / z:smooth spectral routing taraması

Aşağıdaki metnin tamamını başlangıç promptu olarak uygula. Görev implementasyon,
yeni Colab setup, kayıtlı testler, bağımsız analiz ve knowledge teslimidir;
bu promptu hazırlayan agent hiçbir Colab/GPU çalışması başlatmadı.

## Kullanıcının isteği

Yeni bir Colab oturumu oluştur. Önceden denediğimiz **h için SCD, z/style için
Graph Spectral Loss** routing'ini smoothing içeren sürümle yeniden test et.
Spectral-z adımının SCD'nin style katsayısıyla aynı optimal değere sahip
olduğunu varsayma. Ayrı parametre taraması yap; M=800'ün yanına farklı M'ler
ekle. h'nin local step'ini de ikinci aşamada yeniden değerlendir. Amaç yüksek
accuracy; yalnız düşük spectral loss veya latent mesafesi yeterli değil.

## Çalışmaya başlama

1. Repo talimatları ve `.agents/skills/iterative-thesis-researcher/SKILL.md`,
   `.codex/skills/development-agent/SKILL.md`,
   `.codex/skills/colab-research-operator/SKILL.md` oku; knowledge README'den başla.
2. **`knowledge/scdh_spectralz_search_plan_20261101.md` dosyasının tamamını oku.**
   Batch kapsamları, aday grid'i, numerical/source/artifact kapıları ve
   süre/onay sınırları burada. Öneri olarak kayıtlı routine implementation
   kararlarını gerçek kodla uzlaştır; sonucu görünce kapsamı genişletme.
3. Güncel origin/thesis-integration'ı fetch et ve eski working tree'ye yeni
   branch checkout etmek yerine yeni izole implementation worktree oluştur.
   `/Users/batuhan/.codex/worktrees/7bf0/3DD-TTA` içindeki a82200a + dirty işleri
   olduğu gibi koru. Hazırlık notları son ortak taban7cec870'i inceler;
   daha yeni kabul edilmiş işleri oku. Kullanıcı dosyalarını stash/reset/clean etme.
4. `knowledge/current_baseline.md`, spectral-routing audit ve graph-spectral
   retrospective/math audit'lerini oku. Mevcut source/checkpoint/native/
   dataset ve önceki sonuç kapsamlarını doğrula. Tamamlanan eski işleri tekrar etme.

## Bilimsel sözleşme

- Global z diffusion **kapalı**; yalnız local denoiser'a koşul olan z/style
  spectral loss gradientiyle güncelleniyor. h update'inde yalnız SCD var.
  Bu, global prior + spectral guidance veya iki-pass encoder refresh değildir.
- Tek local trajectory: normal8/Background35, Point-MAE, ModelNet40-C severity5,
  B32, raw/eval LION EMAoff, frozen models, p=.95, DDIM100/finalalpha1.
  Başlangıç h0XYZ hem SCD hem static graph referansı. Original z0 decode;
  clean2/4/6+terminal8 ve Background9/18/27+35 eşit Probability Mean.
- Local ve style update parametrelerini semantik isimlerle logla:
  alpha_h_local ve kappa_z_spectral_effective. tta.py/composition'da gamma
  local, eta style; legacy tta_gsd_v2'de bunun tersi. Graph gamma=.6 ayrı.
- Primary smooth profil, mevcut truncated raw-eigenvalue
  `spectral_guidance_losses.py:legacy_l2`, beta2, batch-sum/(3*N),
  static `graph_spectral_v2_offdiag.py` grafıdır. M loss'taki M_max cutoff'u.
  Modern all-active normalized smooth path'te M cutoff değildir; onu yanlışlıkla
  M taraması diye kullanma. Bu profile yeni routing ekle; legacy/modern/graph
  varyantlarını sessizce karıştırma. Profile'ı source audit ile doğrula.
- Label-free development gradient ölçeğinden her M için b_M dondur;
  kappa_z=rho*b_M. Spectral katsayıyı1 tut: redundant weight×step grid yok.
  Clean z veya accuracy kalibrasyona/step seçimine girmez.

## Kayıtlı tarama ve kontroller

- A:64development ID, all15/seed0; local step.03;
  M={200,400,800,1200,1800}, rho={.1,.3,1,3};20aday + standart SCD ve style-off.
- B: en iyi2(M,rho) için local step{.01,.03,.05}. Tamamlanan .03 kollarını
  yeniden çalıştırma; yeni local step'lerde standart SCD/style-off kontrolleri
  ekle. A+B en çok30all15 development koşulu.
- Dev sonunda baseline'a pozitif aday varsa tek winner freeze; ayrık128ID,
  all15/seeds0/1/2 ile winner, operational C_base, aynı local-step C_match ve
  style-off Z0_match. Duplicate comparator'ı bir kez çalıştır; en çok12koşul.
  Confirmation'ı yeniden tuning için kullanma. Eski yüzdeleri baseline yapma.
- Aynı fiziksel input/z0/h0/noise/postprepare RNG ve frozen state bütün
  eşlenmiş kollarda korunur. Native bitwise prediction parity varsayma;
  teknik repeat farklarını gerçek kontrollerle raporla, guard'ları sessizce gevşetme.
- Her batch'te planın CPU/source/finite-difference/route-isolation/PM kapıları;
  her fazda raw+failure arşivleme, Drive doğrulama ve bağımsız NumPy1 recount.
  M/rho/local step seçimlerinin gerçek config ve seçim SHA'ları kaydedilir.
- Paired primary all15 accuracy, macro14/BG/per-corruption, üç seed,
  nesne-cluster CI ve matched end-to-end maliyet/peak memory raporla.
  Graph cache oluşturmayı aday maliyetinden düşüp yanlış hız iddiası kurma.

## Colab ve izin

Kendi yeni task-specific session'ını oluştur: öneri
`tta-scdh-spectralz-a100-20261101-r1`; tarih/collision'a göre suffix belirle.
Peer veya bilinmeyen runtime'a dokunma. Eski conditionalr1 kapatılmıştı;
başka canlı LION oturumunun varlığı bu göreve atanmış olduğu anlamına gelmez.

Önce canonical on hücreli setup notebook'unu skill'deki sırayla kullan:
cell1 ayrı/reconnect aynıVM, sonra2–10; cwd ve her native/data/weight kapısını
doğrula. Research notebook Run all kullanma. Anlamlı heartbeat ve read-only
monitoru çalıştır; fake keep-alive kullanma. Owner/endpoint/monitor ve faz
durumunu knowledge'a yaz. Drive etkileşimli consent kullanıcıya aittir.

**Bütün GPU fazı için taslak2–4saat tahmini var.** Bu prompt >60dk GPU
çalışma onayı değildir. Önce CPU implementasyon/test/review, gerçek sealed
source/asset/ID manifesti ve yürütülebilir phase komutlarını hazırla; bütün
planı ve güncel tahmini gösterip tek açık uzun-GPU onayını al. Aynı işi kısa
CLI çağrılarına bölerek sınırı aşma; scientific technical/calibration dahil
scope'u onayda kapsa. Routine setup/CPU/retrieval için tekrar izin döngüsü kurma.

İş bittiğinde runtime'ı kendiliğinden kapatma: ayrıca açık task-complete+
close talimatı gerekir. Son durum ve session canlılığını raporla.

## Teslim

Planın Batch1→2→3→4 kabul kapılarıyla ilerle; mevcut baseline/default'u
değiştirme. Sonuç pozitif/null/failure olsa da tam kayıt, kaynak/ID/metric
kimlikleri, seçim geçmişi, paired sonuçlar, maliyet ve sınırlamaları teslim et.
Knowledge README/findings/method/open_questions ve yeni run/results notlarını
güncelle. Gerekli tamamlanmış değişiklikleri research skill'e göre seçerek
origin/thesis-integration'a normal push ile yayınla; peer kabul edilmiş
kayıtlarını koru, raw/weights/credentials ekleme. Full benchmark veya yeni
arama turunu otomatik başlatma. İlk somut işin Batch1 ve durum doğrulamasıdır.
