Bu projede devam eden GSD beta-rho geliştirme deneyinin sonuçlarını devralmanı
istiyorum. Bu metin, önceki sohbeti görmeyen bir agent için hazırlanmıştır.
Önce mevcut dosyaları ve Git durumunu kontrol et; aşağıdaki bilgileri eski
belgelerdeki tarihsel ifadelerle karıştırma.

## Çalışma yeri ve yetki

- Repo: `D:\Akademik\Okul\Thesis\code\3DD-TTA`
- Branch: `gsd-smooth-spectrum`
- Bu devirde incelenen son runtime commit: `c1c7467` — `fix(gsd): rebuild interaction cells`.
  Sonraki dokümantasyon commit'leri bulunabilir; gerçek HEAD/status/diff'i kontrol et.
- Kullanıcı aynı branch'e ilgili, test edilmiş kod/knowledge değişikliklerinin
  commit/push edilmesine izin verdi. ZIP, PDF ve ilgisiz yerel dosyaları ekleme.
- Tüm LION/GPU/model koşularını kullanıcı Colab'de çalıştırıyor. Yerelde yalnızca
  CPU testleri, statik inceleme ve arşiv analizi yap.
- Colab repo `/content/3DD-TTA`, ortam `3dd_tta_env`, Python3.8. Standart çağrı:
  `conda run --no-capture-output -n 3dd_tta_env python <script> <args>`.
- Kullanıcı tekrarlanan kurulum ve koşu hatalarından yoruldu. Önce mevcut sonucu
  incele; çalışan/tamamlanan pahalı koşuları otomatik olarak yeniden başlatma.

## Önce oku

Varsa `AGENTS.md` ve `.agents/skills/iterative-thesis-researcher/SKILL.md`.
Ardından şu sırayla:

1. `knowledge/README.md` — en üstteki güncel durum; aşağıdaki tarihsel notlar daha eski olabilir.
2. `knowledge/thesis_scope.md`, `knowledge/experiment_protocol.md`,
   `knowledge/reproduction_gap.md` — kapsam ve kanıt sınırları.
3. `knowledge/gsd_interaction_review_20260928.md` — mevcut yeniden oluşturma
   koşusunun plan uygunluğu, sınırları ve sonuç kabul adımları.
4. `knowledge/gsd_calibration_audit_20260928.md` — hemen beta2'yi sabitleme
   önerisini düzelten audit; kalan geliştirme aşamaları.
5. `knowledge/gsd_calibration_review_20260927.md`,
   `knowledge/gsd_calibration_execution_20260927.md` — matematiksel gerekçe ve kayıtlı sıra.
6. `knowledge/colab_gsd_calibration.md` — en üstteki güncel restart notu;
   eski komutları mevcut koşuya otomatik uygulama.
7. `knowledge/open_questions.md` güncel GSD bölümleri ve
   `knowledge/findings_log.md` son kalibrasyon/weight-screen/beta-screen/audit/restart kayıtları.

Matematik veya host davranışı gerekiyorsa:
`knowledge/gsd_guidance_math_20260926.md`, `knowledge/gsd_smooth_review_20260926.md`,
`knowledge/repository_map.md`, `knowledge/method_synthesis.md`.

## Bilimsel kapsam

Bu çalışma GSD-inspired latent spectral guidance'dır; tam GSDTTA reprodüksiyonu
değildir. Classifier ve LION ağırlıkları sabit. Statik, detached referans
graph/baz ve smooth/hard fidelity loss kullanılır. Smooth loss:
`sum_i exp(-beta*q_i) * ||u_i^T(Y-R)||^2 / (3*N)`,
`q=lambda/mean_active_degree`; örnek loss'ları toplanır. Gradient denoiser
üzerinden local latent ve style'a gider. Diagnostic adaylar aynı SCD-only
durumlarında ölçülür, uygulanmaz. Alpha guidance boyunca sabittir:
`alpha = rho / median(per-example, per-probe local spectral/SCD norm ratio)`.
PxP, dinamik graph ve yeni holdout yöntemi bu aşamanın kapsamı dışındadır.

## Önceki bulgular: tarihsel sonuçlar

Ham ZIP'ler ve ayrı türetilmiş raporlar:
`result/modelnet40_c/gsd_latent_spectral_smooth_v2/`.
Yedi eski screen ZIP'i ve `report.json`, `calibration_summary.json`,
`screen_weight_summary.json`, `beta_screen_summary.json` yerelde mevcut.
Eski diagnostic referans ID:
`20260927-201522_gsd-cal-diagnose-reference-seed0-n64`.
Onun tam yedi dosyalı ham ZIP'i yerelde yok; Colab'de config'i de bulunamadı.
`report.json` analyzer girdisi olabilir, runner'ın ham referansının yerine geçmez.

Seed0, Gaussian/Impulse her biri128 örnek, rho=.001:

| Koşul | Gaussian | Impulse | Macro % |
|---|---:|---:|---:|
| SCD-only | 98/128 | 92/128 | 74.21875 |
| Smooth beta=.5 | 99/128 | 93/128 | 75.00000 |
| Smooth beta=2 | 100/128 | 93/128 | 75.390625 |
| Smooth beta=8 | 99/128 | 91/128 | 74.21875 |
| Hard-v2 | 98/128 | 92/128 | 74.21875 |

Beta2/rho=.01 eski koşusu 100/128 ve 92/128 (%75.0).
Beta2'nin rho=.001'de SCD-only'ye farkı üç toplam doğru tahmin (+1.171875 pp);
beta=.5'e farkı yalnızca bir tahmin. Optimum veya doğrulanmış kazanç değildir.
Eski rho=.001 alpha'ları beta .5/2/8/hard için sırasıyla
8.139104941932809 / 11.310243971397897 / 14.882176927730372 / 18.751460023983643.
Eski referansta beta=.5/rho=.01 alpha yaklaşık81.39104941932808 idi.
Bu sayıları yeni koşuya sabitleme; yeni ham referans belirleyicidir.

Audit'te gerçek guided local ratio medyanları rho=.001 çevresinde .000968–.001033
idi; style aynı anda eşleştirilmiş değildir. Hard requested M100, boundary
toleransı nedeniyle Gaussian/Impulse ortalama rank139/121, maksimum291/240
kullanır. Bunu tam ilk100 mod diye anlatma, graph/tolerans matematiğini değiştirme.

## Hata süreci ve bugünkü karar

1. Eksik eski config nedeniyle ilk launcher yeni diagnostic üretti ve eski
   dört karşılaştırma ZIP'iyle katsayı/manifest eşitliğini aradı.
2. Yerelde bulunan ZIP'ler Colab'de yoktu; dosya aktarma adımı gerekti.
3. Dosyalar geldikten sonra beta=.5/rho=.001 alpha eşitlik kontrolü başarısız
   oldu. Hata toleransı `rel_tol=1e-12, abs_tol=0`; yeni alpha ve farkın
   büyüklüğü henüz ham arşivden ölçülmedi. Nedeni bilinmiyor; büyük bilimsel
   fark, ortam veya eigensolver hatası diye kesinleştirme. Kontrolü bypass etme.
4. Kullanıcı gerekli ZIP'lerin baştan oluşturulmasını istedi. `c1c7467`,
   `--rebuild-prerequisites` seçeneğini ekledi. Artık tek yeni referans altında
   bütün gerekli karşılaştırmalar yeniden üretiliyor; eski ZIP'ler kullanılmıyor.

Ortam sorunu ayrıydı: yerel
`20260928-095332_gsd-cal-diagnose-reference-seed0-n64.zip` import sırasında
`torch.xpu` hatası içeriyor (Diffusers .36.0, Hub .36.2, Torch2.1.2+cu121).
Eski başarılı arşivlerde Diffusers .11.1 / Hub .11.1 vardı. Kullanıcı yalnızca
bu iki paketi mevcut Colab ortamında `--no-deps` ile düşürdü; çıktı
Torch2.1.2+cu121, Diffusers.11.1, Hub.11.1, CUDA=True,
DDPMScheduler import=OK oldu. Kurulumu yeniden yaptırma.
`env.yaml` ve `requirements.txt`, kullanıcı isteğiyle `dev` ile aynıdır ve
öyle kalmalıdır. Daha eski 092752 koşusundaki FPS `no kernel image` hatası
farklı bir kullanıcı raporudur; bu ham ZIP yerelde yok. Yalnızca tekrar
oluşursa o koşunun gerçek GPU/extension kanıtını incele; şimdi körlemesine
Torch/CUDA/extension değişikliği yapma.

## Şu anda hazırlanmış/başlatılmakta olan koşu

Kullanıcıya verilen komut:

```bash
%%bash
set -euo pipefail
cd /content/3DD-TTA
git switch gsd-smooth-spectrum
git pull --ff-only origin gsd-smooth-spectrum
conda run --no-capture-output -n 3dd_tta_env python \
  scripts/run_gsd_interaction_from_scratch.py --rebuild-prerequisites
```

Bu komutu otomatik tekrar çalıştırma. Devir sırasında yeni tamamlanmış
rebuild/interaction ZIP'leri yerelde yoktu; Colab'in canlı durumu buradan
doğrulanmadı. Önce kullanıcının son durumunu veya eklediği yeni arşivleri kontrol et.

Tek çağrının sırası:

1. Bir yeni diagnostic referans: corruption başına64 örnek, seed0; SCD-only
   durumlarında hard/.5/2/8 probe'ları, uygulanmış spectral guidance yok.
2. SCD-only, corruption başına128 örnek.
3. Smooth beta=.5/rho=.001, 128.
4. Smooth beta=2/rho=.001, 128.
5. Smooth beta=2/rho=.01, 128.
6. Smooth beta=.5/rho=.01, 128.
7. Beş development ZIP'i için otomatik kompakt interaction analizi.

Sabitler: ModelNet40-C severity5, Gaussian+Impulse, batch32, seed0,
split seed20260927, raw/eval LION, EMA kapalı, SCD weight1, lambda=.95,
gamma=eta=.01, mevcut decoder/preprocessing. Diagnostic64, development128'in
prefix'idir. Beta8/hard-guidance accuracy koşuları ve rho=.0001 tekrar edilmiyor.
Baseline'ın hard profile etiketi spectral weight0 olduğu için hard-guidance
deneyi değildir. Alpha her seferinde aynı yeni referanstan türetilir.

Beklenen çıktı: toplam6 ham ZIP (1 diagnostic + 5 development), her biri
command.txt, config.json, environment.txt, stdout.log, summary.csv,
per_corruption.csv, notes.md içerir. Yeni adlarda referans run ID bulunur.
Özet `beta_rho_interaction_summary_<CALIBRATION_RUN_ID>.json` olur.
Yeni sonuçların accuracy'si henüz bilinmiyor. Eski arşivlerle tek tabloya karıştırma.
Launcher'ın resume modu yoktur; yeniden çağırmak bütün diziyi başlatır.

## Devralınca yapılacak iş

1. Git status/diff ve yeni sonuç dosyalarını kontrol et. Sonuç yoksa mevcut
   koşunun durumunu al ve tamamlanınca altı ham ZIP ile kompakt özeti iste.
   Yalnızca hedef ZIP bütün yeniden kurulmuş karşılaştırmayı doğrulamaya yetmez.
   İşlem analiz/export aşamasında durmuşsa tamamlanan GPU koşularını koru;
   yalnızca başarısız kalan CPU adımını düzelt/çalıştır.
2. ZIP'leri Python ile güvenli biçimde oku: paths/CRC, tek root, yedi dosya,
   config/CSV/log uyumu, tamamlanma ve corruption başına64/128 sayıları.
   Başarılı subset coverage `partial`, execution_status `complete` olabilir.
   Büyük JSON/logları konuşmaya dökme, ham arşivleri değiştirme.
3. Beş development koşusunun bu yeni referansın aynı run ID ve config SHA'sını
   kullandığını ayrıca doğrula. Indices, tekrar/aralıklar, seed/batch/split,
   host/graph, source/data/checkpoint manifestleri ve environment kayıtlarını
   karşılaştır. Package/extension eşitliği yalnızca source hash'inden çıkmaz.
4. Yeni ham config'i `load_calibration` ile değerlendir; eski report'a göre
   beta .5/2 alpha ve median R farklarını mutlak ve göreli olarak raporla.
   Eksik denominator, SCD/state/DDIM oranları, kuyruklar ve style'ı incele.
   Launcher bu tanı incelemesini insan onayıyla duraklatmıyor; geçerli sayılar
   otomatik olarak bilimsel kararlılık kanıtı değildir. Yeni bulgular anormalse
   aday yükseltmek yerine kanıtı koruyarak nedenini araştır.
5. `scripts/analyze_gsd_calibration.py --interaction-screen` ile yalnızca bu
   beş yeni development ZIP'ini ve yeni ham calibration config'ini kullan.
   Analyzer gerekirse config'in güvenli, byte-identical çıkarılmış kopyasını
   okuyabilir; raw dosyaları düzenleme ve summary'yi raw run içine yazma.
   Eski/yeni karışabilecek geniş glob yerine config kimlikleriyle açık liste seç.
6. Gaussian/Impulse doğru sayıları, macro accuracy, yeni SCD-only'ye pp
   farkları, her beta için rho=.01 eksi rho=.001 etkisi ve bu etkilerin
   farkını (betaya bağlı etkileşim) ver. Bu betimsel bir karşılaştırmadır;
   seed0/128 örnekle anlamlılık, optimum veya bağımsız doğrulama iddia etme.
   Gerçek guided local spectral/SCD oranlarını hedef rho ile kontrol et;
   style'ı ayrı raporla. Eski sonuçları ancak ayrı bir tarihsel karşılaştırma
   olarak, kalibrasyon/ortam farklarını belirterek sun.
7. Sonra umut veren beta .5/2 koşulları ve eşleştirilmiş SCD-only kontrolünü
   daha fazla development örneği ve seed0/1/2 için açıkça kaydet. Aynı kabul
   edilmiş calibration'ı koru; her model seed'i için sessizce yeniden alpha
   fit etme. Sonuç görmeden kazanan veya nihai katsayı ilan etme.
   Ayarları bu aşamadan sonra sabitle. Object-disjoint holdout ve
   cross-corruption correspondence sonraki ayrı iştir; current guard'ları
   kaldırarak holdout taklidi yapma.
8. Knowledge kayıtlarını [Code]/[Run]/[User report]/[Inference]/[Open]
   ayrımıyla güncelle. Gerekirse mevcut `scripts/export_gsd_results.py`
   (`--stage all --name-contains <CALIBRATION_RUN_ID>`) ile altı ZIP'i Drive'a
   yedekle; exporter summary JSON'u taşımadığı için onu ayrıca koru.

## İncelenecek kod ve doğrulama sınırı

Önce `scripts/run_gsd_interaction_from_scratch.py`, `gsd_calibration.py`,
`eval_gsd_calibration.py`, `scripts/analyze_gsd_calibration.py`,
`run_baseline.py`, `gsd_protocol.py`, `scripts/export_gsd_results.py`.
Matematik gerektiğinde `graph_spectral.py`, `tta_gsd.py`.
Testler: `tests/test_gsd_interaction_launcher.py`,
`tests/test_analyze_gsd_calibration.py`, `tests/test_gsd_calibration.py`,
`tests/test_gsd_protocol.py`, `tests/test_gsd_trajectory.py`, `tests/test_gsd_worker.py`.

Son yerel kontrol: `python -m unittest discover -s tests -q` → 133 test geçti.
Launcher'ın iki yeni testi koşul/alpha komutlarını kontrol eder; tam CUDA
akışı/parity kanıtı değildir. Native original/instrumented SCD prediction/RNG
parity, per-example paired prediction evidence, cross-corruption object
correspondence ve bağımsız holdout hâlâ açıktır. Yeni graf/host/ortam değişikliği
yapmadan mevcut koşunun kanıtını tamamlamak ilk önceliktir.
