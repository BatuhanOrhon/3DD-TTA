# Yeni oturum başlangıç promptu: global shape latent proximal guidance

Bu dosyanın tamamı yeni Codex oturumuna başlangıç promptu olarak verilebilir.
2026-10-31 devir kaydıdır; canlı Colab envanteri ve dosyalar yeniden doğrulanmalıdır.
Kanıt etiketleri: [Paper], [Code], [Run], [User report], [Inference], [Open].

## Görev ve çalışma alanı

3DD-TTA tez araştırmasını kaldığı yerden sürdür. Amaç, LION global shape
latent diffusion sırasında guidance uygulamanın temiz encoder latent'ine
yakınlığı ve frozen Point-MAE doğruluğunu iyileştirip iyileştirmediğini küçük,
eşlenmiş bir pilotla araştırmaktır. Kullanıcı bu ek hipotezin test edilmesini
yetkilendirdi. Önceki çalışmanın kaydı ve integration yayını önkoşulu yerine getirildi.

**Yalnız şu worktree'de çalış:**

```text
/Users/batuhan/.codex/worktrees/7bf0/3DD-TTA
branch: integration/chamfer-guidance-variants-20261030
HEAD: a82200adbe117ce56a134dc3c22601cb73dc2654
yayın hedefi: origin/thesis-integration
```

Önce `pwd`, `git rev-parse --show-toplevel`, `git branch --show-current`,
`git rev-parse HEAD`, `git status --short` ile doğrula. Ortamın varsayılan
`/Users/batuhan/Desktop/projects/3DD-TTA` dizinine güvenme. Branch değiştirme;
reset/clean/stash yapma; mevcut dirty `tta.py`, knowledge ve untracked kodları
koru. Bu çalışma için tta/model/default değiştirmek gerekmiyor. Yerel HEAD'in
eski olması beklenen durumdur; tamamlanmış sonuçlar ayrı checkout'tan yayımlandı.

Önce `/Users/batuhan/.codex/RTK.md`, repo `AGENTS.md`,
`.agents/skills/iterative-thesis-researcher/SKILL.md`,
`.codex/skills/colab-research-operator/SKILL.md` ve `knowledge/README.md` oku.
Tez scope/protocol/reproduction kayıtları ile güncel checkout'u uzlaştır.
Gerekli implementasyon için proje development ve execution becerilerini uygula.
Bu belgede verilen eski durumun yerine yeni doğrulanmış kanıt geçer.

## Okuma sırası ve kod haritası

1. `knowledge/shape_latent_proximal_pilot_20261031.md`: yeni hipotez ve sabit protokol.
2. `knowledge/shape_latent_proximal_execution_plan_20261031.md`: uygulama/kabul planı.
3. `knowledge/shape_latent_refresh_results_20261031.md`: son tamamlanan çalışmanın sonucu.
4. `knowledge/shape_latent_clean_free_research_20261031.md`: hedef seçiminin gerekçesi ve literatür.
5. `knowledge/shape_latent_oracle_results_20261031.md`: temiz-z oracle headroom'u ve sınırları.
6. `knowledge/shape_latent_tta_results_20261031.md`: önceki bounded global pilotun null sonuçları.
7. `knowledge/shape_latent_tta_research_20261031.md`, `knowledge/papers/lion.md`: LION yapısı ve kaynak denetimi.
8. `knowledge/colab_cli_setup_20261017.md`, `scripts/colab_environment_setup.ipynb`: kurulum sözleşmesi.
9. `knowledge/findings_log.md`, `knowledge/open_questions.md`, `knowledge/method_synthesis.md`.

Yeni opt-in bilimsel kod ve testler worktree'de fiziksel olarak mevcut,
henüz tamamlanmış deney sonucu olarak yayımlanmış değildir:

```text
shape_latent_proximal.py                 # propose(), load_reference()
scripts/eval_shape_proximal.py           # sabit 5 kol, scope, state replay, DDIM
tests/test_shape_latent_proximal.py      # cebir, RNG, oracle ayrımı, state
tests/test_eval_shape_proximal.py        # scope, yönlendirme, cache/proof bağları
```

Mevcut akış referansları: `shape_latent_refresh.py`,
`scripts/eval_shape_refresh.py`, `shape_latent_oracle.py`,
`scripts/eval_shape_oracle.py`, `scripts/eval_chamfer_depth.py`,
`scripts/eval_multibackbone_source.py`, `legacy_trajectory.py`, `tta.py`.
LION encoder/decoder ve global/local prior çağrılarını gerçek kaynakta izle;
eski açıklamaları yürütülen kod yerine kullanma.

## Önceden tamamlanan sonuçlar: yeniden çalıştırma

[Run] Son üç-kol refresh pilotu: aynı64 exposed development nesnesi,
ModelNet40-C/all15/severity5/seed0/Point-MAE/B32, N8/Background35.

| Kol | Doğru /960 | Accuracy | Baseline farkı |
|---|---:|---:|---:|
| original baseline |585|%60.9375|—|
| mean_refresh |593|%61.7708|+0.8333 puan|
| posterior_refresh |588|%61.2500|+0.3125 puan|

64-orijinal-nesne cluster bootstrap95% aralıkları sırasıyla
mean−baseline `[-0.5208,+2.1875]`, posterior−baseline `[-1.0417,+1.6667]`
puan: güvenilir üstünlük gösterilmedi. Global diffusion bu çalışmada **kapalıydı**.
Baseline/default korunur. Yeni pilotta global denoising ağı gerçekten açılacak.

Önceki clean-z oracle'da kendi matched baseline'ına karşı temiz-z/eski-h
+4.0625 puan verdi; temiz-z ile h'yi yeniden encode etmek bu kolu aşmadı.
Bu, temiz z bilgisinde fayda olabileceğini gösterir; deployable yöntem sonucu
değildir. Farklı pilotların baseline yüzdelerini birbirine taşıma.

Son refresh kod/knowledge/kompakt kanıt yayını:
`1bacabfe20a70bf94e9a3c8a5988084d0ae48d3f`.
Devir kontrolünde `git ls-remote origin refs/heads/thesis-integration` bu SHA'yı verdi.
Remote sonraki oturumda ilerlemiş olabilir; fetch ederek yeni hedefi koru.

Eski hamlar yerel:
`result/preparation/shape_latent_refresh_pilot_20261031/operator/`.
`pilot_raw.zip`:410634711byte,
SHA256 `d828536a5d7ab5482aa65a0ada9e913eccb969651c014c020163ea8823fc7c6e`.
Kabul edilmiş `pilot_extracted/`, `pilot_local_analysis/`, `pilot_direction_analysis/`.
Smoke için yalnız `smoke_extracted_v2/` kabul edilmiş ham ağaçtır.
Drive ham/source/derived yedekleri doğrulandı. Eski oracle'ın ek derived
Drive kopyasını kullanıcı erteledi; sırf o kopya için yeniden bağlantı isteme.

## Hangi Colab oturumuna bağlanılacak?

**Devralınacak mevcut oturum:**

```text
CLI session: tta-shape-proximal-a100-20261031-r1
endpoint: gpu-a100-hm-ass1c0-zwl345mh6780
hardware: A100, High-RAM
kurulum repo/cwd: /content/3DD-TTA
deney root: /content/studies/shape-proximal
operator root: /content/proximal_operator
Drive root: /content/drive/MyDrive/thesis/result/shape_latent_proximal_pilot_20261031
```

Bu kullanıcının aynı görevi yeni Codex oturumunda sürdürme talebiyle
devredilen kendi araştırma runtime'ımızdır. Son `colab sessions` envanterinde canlıydı.
Eski `tta-shape-refresh-a100-20261031-r1` / `gpu-a100-hm-euw4a0-3e5wlo5wf3or6`
17:50UTC'de listeden kayboldu. Biz stop/release/delete çağırmadık; nedeni bilinmiyor.
Eski runtime'a bağlanma veya başka ajanın oturumunu kullanma.

CLI0.7.4 kullan; güncelleme/account/auth/config değişikliği yapma.
Her çağrıda `REQUEST_TIMEOUT=60` ve açık `-s` kullan.
Önce `colab sessions` ile sunucu envanterini doğrula. `colab status` tek başına
uzak kernel busy/completion kanıtı değildir. Bir anda yalnız bir notebook cell
çalıştır; busy kernel'e inspection hücresi kuyruğu ekleme. `colab run` kullanma;
iş bitiminde runtime'ı otomatik serbest bırakır. Kullanıcı runtime'ı kapatma
yetkisi vermedi: stop/release/delete/kapatma yapma.

## Devir anındaki tam ilerleme durumu

[Run] Yeni VM'de canonical Cell1 tamamlandı; Conda restart sonrası aynı VM'ye
bağlanıldı. Cell2 CLI süreci de tamamlandı ve logda exact
`SETUP_WRAPPER_CELL_2_COMPLETED` satırı var. Pip yalnız bilinen `fastrlock`
wheel hatasını verdi; kullanıcı yalnız bu Cell2 hatasının ertelenmesini kabul ediyor.
**Cell2 Python doğrulaması, Drive mount, Cell3–10 ve yeni strictCPU gate henüz yapılmadı.**
Yeni proximal GPU pilotu başlamadı; bilimsel2400s ledger henüz başlamadı.

İki upload tamamlandı; yerel loglarında başarılı CLI bildirimi var:

| Paket | Uzak yol | Byte | SHA256 |
|---|---|---:|---|
| scientific assets v1 | `/content/proximal_assets_v1.zip` |29970678|`347ef9c3c5304f2a168046cced7298b828fc387e93093b8de29d076c7276f6fa`|
| operator v2 | `/content/proximal_operator_v2.zip` |18507|`1f8a7da3fce912f694fe912233503e92217e2d09b7de4c54f7d60eaf09203220`|

Uzak fiziksel SHA/size, setup tamamlandıktan sonra deployment'ta ayrıca doğrulanmalı.
346bilimsel kaynak dosyası manifest SHA256
`690caa62d2a2603ebf9480cc86e3a66f5d13fca77ec1dbe34a810cd19fde15f5`;
source ZIP SHA256
`1b2fa5b423e407e5ce1339060af63751a46d23cb95a5dffc74b1524451523c82`.
Operatorv1 kaybolan VM'ye aittir; kullanma. v2 registration/run_phase yeni VM'ye bağlıdır.

**Bulunan ve henüz düzeltilmeyen preflight engeli:** yerel
`operator/install_and_deploy_wrapper.py` v2 ZIP hash'ini doğru kontrol ederken
`registration['session']` için hâlâ eski `tta-shape-refresh-a100-20261031-r1`
değerini assert ediyor. Çalıştırmadan önce bunu mevcut
`tta-shape-proximal-a100-20261031-r1` ile düzelt ve gerçek v2 ZIP registration'ı
ile eşitliğini kontrol et. Bu dış kurulum wrapper'ı ZIP içindeki sealed operator
dosyalarından değildir; sırf bu tek düzeltme için bilimsel snapshot/ZIP'i yenileme.
v2 `run_phase.py` içindeki kontrol zaten doğru yeni adı bekliyor.

Yerel bütün yeni çalışma kökü:
`result/preparation/shape_latent_proximal_pilot_20261031/`.

```text
setup/                          # canonical cell_01.py ... cell_10.py
  cell_02.log                   # tamamlandı; bilinen fastrlock hatası
  verify_cell2.py
  cell_03_mounted_checked.py
  cells_04_10_driver.py
  manifest.json
imported_v1/                    # eski çalışmadan immutable girdiler
prepared_v1/assets/             # mühürlü scientific assets
operator/
  progress.json                 # bu handoff'ta güncellendi; yeniden doğrula
  registration.json             # v2/yeni oturum
  operator_identity_v2.json
  asset_identity_v1.json
  proximal_assets_v1.zip
  proximal_operator_v2.zip
  install_and_deploy_wrapper.py  # yukarıdaki session assertion düzeltmesi gerekli
  deploy_runtime.py
  deploy_wrapper.py
  run_phase.py
  run_smoke_wrapper.py
  run_pilot_wrapper.py
  analyze_phase.py
  test_operator.py
  check_acceptance.py
  prepare_inputs.py
  prepare_assets_v1.py
  upload_assets_r1.log
  upload_operator_r1.log
```

[Code/CPU]19yeni contract test,69ilgili CPUtest/16subtest ve4operator kontrolü
önceki adımda geçti; loglar bu kökte. Fresh whole-change review substantive
RNG/style/source kontrolleri sonrası blocker bırakmadı. Bu, gerçek GPU sonucu
veya henüz yapılmamış deployment wrapper kontrolü demek değildir.
Yeni wrapper engeli devir doğrulamasında bulundu ve açıkça kaydedildi.

## Hipotezi ve beş kolu değiştirmeden uygula

LION: `q(z|X) q(h|X,z)`, `p(z) p(h|z) p(X|h,z)`.
Global z128; local h2048×4. Global/local ağlar ve classifier frozen.
`z0=mu0+sigma0*eps_z`. Baseline-denoised LION-frame
`X*=D(h*,z0)` yeniden global encoder'a girerek `mu_new` üretir.
Temiz veri gerektirmeyen proxy: **`zR=mu_new+sigma0*aynı eps_z`**.
Proxy eski tamamlanmış çalışmanın fiziksel proof'undan replay edilir.

| Kol | Global z işlemi | Global prior çağrısı | Temiz bilgi |
|---|---|---:|---|
| original | z0 aynen |0|Yok|
| mean | doğrudan zR |0|Yok|
| prior3 | z0'dan3 DDIM adımı |3|Yok|
| proxy3 | aynı3 adımda zR hedefli proximal guidance |3|Yok|
| oracle3 | aynı3 adımda gerçek temiz encoder z hedefi |3|Yalnız teşhis|

Global DDIM scheduler1000train/100inference, linear beta .0001→.02,
epsilon prediction, clip kapalı, final_alpha1, DDIMeta0.
Seçilen timestep'ler20/10/0; ağın t girişi21/11/1.
İlk forward-noising std yaklaşık.079; eski %5z0 küresi uygulanmaz.
Her adım `c_prior=pred_original_sample` için sabit λ1 proximal çözümü:

```text
c_guided = (c_prior + z_target) / 2
prev_guided = prev_DDIM + sqrt(alpha_prev) * (c_guided - c_prior)
```

Prior3 kolunda `c_guided=c_prior`. Üç global kol aynı private-generator
Gaussian noise kullanır; shared local/classifier RNG ilerlemez.
Bu closed-form clean-space guidance'dır; global backward/Jacobian yoktur.
DPS/DDNM/DiffPIR'ın tam uygulaması veya calibrated posterior diye adlandırma.
Grid/ek katsayı/radius/schedule/seed/başka hipotez ekleme.

Global3 adım, normal local8 adımın%37.5'i; Background35'in yaklaşık%8.57'si
kadar ağ çağrısıdır. Bunlar farklı ağlardır; zaman veya diffusion-noise-depth
eşitliği anlamına gelmez. Mevcut local8/35 zinciri ayrıca aynen çalışır.

## Pilotun sabit kapsamı ve state kontrolü

İlk32 **önceden exposed development** orijinal nesne, seed0,B32,severity5,
Gaussian/Background/Rotation/LiDAR, frozen Point-MAE. Teknik smoke yalnız
Gaussian+Background, aynı32IDs. Tam veri/all15/ek seed otomatik çalıştırılmaz.

Local N8/Background35; gamma.03,eta.003,p.95,total100,SCD normalizationoff,
raw/eval/EMAoff. Dört candidate eşit probability mean:
normal2/4/6+terminal8; Background9/18/27+terminal35.

Eski all15 çalışma preparedX/z0/h0/localnoise/postprepareRNG/refresh proof'ları,
ilk32ID/dört koşul için fiziksel ve hash-bound immutable input'tur.
Yeni dört-koşul sırası eski rawX interpolation RNG'sini yeniden üretmeyeceği
için eski **preparedX ve açık RNG boundary** replay edilir. Eski source-bound
proof'u yeni kodun ürettiği şeklinde etiketleme. Yeni proposal/proof current
executed-source kimliğine bağlanır; dataset/labels/checkpoint kimlikleri eşleşir.

Her kol ikinci local zincire **aynı original h0 ve local noise** ile başlar;
zR'yi üreten h* üzerinden ek denoising değildir. h yeniden encode edilmez.
SCD anchor aynı h0XYZ; mevcut style gradientleri sürer; her clean probe/final
decode kolun **kendi seçilmiş başlangıç z'sini** kullanır. Oracle target ayrı
dosya/CLI alanındadır; yalnız oracle3 inference'ı onu okuyabilir. Posthoc
analiz diğer kolları temiz hedefle karşılaştırabilir.

## Adım adım yürütme

### 1. Devral ve mevcut dosyaları doğrula

Git ve canlı envanteri kontrol et; bu kullanıcı aynı görev runtime'ını devrediyor.
Log/receipt/ledger mevcutsa önce oku; tamamlanmış aşamayı yeniden çalıştırma.
Cell2/upload yerel CLI süreçleri16663/19745/50463 tamamlandı; yeni session'da
bu process ID'lerine güvenme. Zaman sınırlı caffeinate kalıcı uptime garantisi değildir.
`colab --help`, `version`, `exec --help`, `drivemount --help` üzerinden arayüzü doğrula.
API/token dosyalarının içeriğini okuma veya loglama.

### 2. Aynı yeni VM'de canonical kurulumu tamamla

Cell1/2'yi körlemesine yeniden başlatma. Cell2 logunu gerçek hata ve exact
marker açısından incele; sonra şu **doğrulama** hücresini çalıştır:

```bash
REQUEST_TIMEOUT=60 colab exec -s tta-shape-proximal-a100-20261031-r1 -f result/preparation/shape_latent_proximal_pilot_20261031/setup/verify_cell2.py --timeout 120
```

`ORACLE_CELL2_ENV_VERIFIED` exact marker'ı ve beklenen Python3.8 ortamını kontrol et.
Ardından `colab drivemount -s tta-shape-proximal-a100-20261031-r1` ile gerçek
CLI mount yap. Gerekirse CLI'nin **o anda ürettiği** Drive yetki linkini kullanıcıya
ver; yeni VM'nin data/weights/native assets ihtiyacını açıkla. Önceki OAuth
linkini yeniden kullanma; onayı kullanıcı gerçekleştirir. Browser ile izin
atlatma yok. Bu yeni çalışmanın asset mount'u, ertelenmiş eski oracle yedeğinden ayrıdır.

Mount fiziksel doğrulanınca sırayla `setup/cell_03_mounted_checked.py` ve
`setup/cells_04_10_driver.py` çalıştır. Sonuncu için CLI timeout3300 kullanılabilir;
bu bilimsel GPU2400s ledger'ı değildir. Tek aktif cell ve aynı cwd kuralını koru.
Wrappers canonical notebook gövdelerini taşır; full notebook research cell'lerini çalıştırma.
Her fazın gerçek `.complete` dosyası ve wrapper marker'ı gereklidir.
Final `SETUP_ENVIRONMENT_VERIFIED`, `SETUP_WRAPPER_CELL_10_COMPLETED`,
`SHAPE_SETUP_CELLS_4_10_COMPLETE` ve `/content/cli_setup_verification.json`
`status=passed` koşullarını doğrula/indir. CUDA/native/data/checkpoint hatasında dur.
PointNet++ burada LION/native dependency'dir; classifier Point-MAE kalır.

### 3. Preflight engelini düzelt ve sealed deployment'ı doğrula

Yukarıda belirtilen dış install wrapper session assertion'ını yeni adla düzelt;
v2 ZIP'in gerçek registration'ına karşı küçük assert kontrolü ekle/çalıştır.
Başka sealed dosya değişirse yeni operator sürümü/hash/registration gerekir;
eski paketi sessizce yeniden yazma. Mevcut iki uzak ZIP'i SHA/size ile doğrula.
Sonra `operator/install_and_deploy_wrapper.py` çalıştır. Çıkarma güvenli/fresh
olmalı; mevcut remote deployment varsa önce durumunu incele, üstüne yazma.

Deployment yeni native build byte'larını koruyup4accepted pinned binding'i
restore eder; source/assets/raw+label/model kimliklerini doğrular. Actual
diffusers CPU20/10/0 proximal algebra, strict Point-MAE checkpoint load ve
no-forward/no-CUDA CPU gate geçmelidir. Preflight, pretrained GPU inference değildir.
Exact `SHAPE_CPU_PREFLIGHT_COMPLETE`, `PROXIMAL_DEPLOY_WRAPPER_COMPLETE`,
`PROXIMAL_INSTALL_DEPLOY_COMPLETE`, root `cpu_preflight_receipt.json`
`status=complete` gerekir. Receipt/ham archive/Drive readback kimliğini indir ve doğrula.

### 4. Teknik smoke:32ID×2koşul×5kol

15–25dk tahminli **bütün yeni bilimsel faz** için tek2400s cap vardır;
smoke, pilot, geçiş ve arşiv birlikte değerlendirilir. İlk smoke bu ledger'ı
başlatır; aradaki yerel analiz zamanı da sürekli UTC sayaçta sayılır.
Önce analiz araçlarını hazırla; smoke sonrası uzun geliştirme nedeniyle budget tüketme.
60dk üstü beklenen faz dışında yeni araştırma onayı isteme; scope'u genişletme.

`operator/run_smoke_wrapper.py` dosyasını CLI exec ile çalıştır; gerçek subcommand'ları
`run_phase.py:commands()` üretir. Beşkol/G+B/32ID=10prediction batch shard beklenir.
Nonzero/timeout/nonfinite/source/state/proof/RNG mismatch'da dur ve failure hamını koru.
Exact `PROXIMAL_PHASE_RECEIPT`, `PROXIMAL_SMOKE_COMPLETE`,
`PROXIMAL_SMOKE_WRAPPER_COMPLETE`, smoke receipt/status/ledger kabulü gerekir.
CLI exit0 tek başına başarı sayılmaz.

### 5. Hamı indir ve bağımsız smoke kabulü yap

`/content/studies/shape-proximal/smoke_receipt.json` dosyasını indir; archive
`local_path`/size/SHA değerlerini **gerçek receipt'ten** al. ZIP'i fresh yerel
yola indir; path/symlink/duplicate/CRC/evidence_manifest doğrulaması yap.
Hamı değiştirme; türetilmiş dosyaları ayrı dizine yaz.

`operator/analyze_phase.py` ile bağımsız CPU recount:
`analyze(phase_extracted, phase_extracted/'paired', fresh_analysis_output)`.
Bu yeni script CLI'de de üçüncü output argümanını destekler; yine fiziksel
ayrı çıktıyı doğrula. Beş outcome matrisi smoke için(2,32), pilot için(4,32).
Config scope, checkpoint/source/state, IDs/labels/PM, imported file/proof bağları,
actual3DDIM adımı/half-target cebiri, aynı private Gaussian, selected cache/RNG
ve own-z probe/final kontrollerinin tümü geçmeli. Remote acceptance ve local
recount/outcomes birebir eşleşmeli. Sırf hash eşleşmesine dayanıp fiziksel
vektör cebirini atlama.

### 6. Yalnız smoke kabulü ve kalan bütçe uygunsa pilotu çalıştır

`operator/run_pilot_wrapper.py`: aynı32ID/dörtkoşul/beşkol=20prediction shard,
128nesne–koşul olayı/kol. Aynı hyperparameter/target tanımı kalır.
`PROXIMAL_PILOT_COMPLETE` ve `PROXIMAL_PILOT_WRAPPER_COMPLETE`, receipt,
ledger/status/actual source guard ve frozen model state gereklidir.
Kapsamlı failure sonrası ledger'ı silip sıfırdan yeni budget üretme.
Ham download/SHA/CRC/fresh extraction/bağımsız recount'u tekrar tamamla.

### 7. Mekanizma ve accuracy analizini ayrı raporla

Mevcut analyzer kabul/recount üretir. **Yeni4×32/fivearm summary, clean-distance
ve cluster bootstrap analizi henüz hazırlanmadı; bunlar kalan iştir.** Eski
refresh analizini körlemesine kopyalama; eski15×64 varsayımlarını kaldır.

Her kol için paired temiz encoder z'ye karesel mesafeyi ölç. Global final_z
fiziksel proof'u ve phase/imported/oracle_targets içindeki pairedclean hedefi
kullan. Absolute MSE, toplu MSE oranı, nesne-başına oranların medyan/ortalaması,
z0'dan hareket normu ve temiz yöne cosine'ı açık tanımla; sıfır denominator
durumunu gizleme. Ortalama oran ile toplu MSE oranı farklıdır. Eski temiz
encoder hedefinde küçük cross-runtime native sayısal farklar olabileceğini koru.

Accuracy için own-fresh-original farkları; mean−original, prior3−original,
proxy3−original, **proxy3−prior3**, oracle3−original ve oracle3−proxy3 ver.
Doğru/toplam, mutlakpuan farkı, düzelten/bozan sayısı ve per-corruption satırları yaz.
32orijinal nesne üzerinden dört koşulu birlikte yeniden örnekleyen10,000draw
cluster bootstrap(seed20261031) kullan:128olay128bağımsız nesne değildir.
Tekseed/exposedcohort/sabitkoşul/exploratory/multiple-comparison sınırlarını yaz.

Karar ölçütleri:

- proxy3 prior3'e göre latent-distance ve accuracy iyileştirirse hipoteze küçük pilot desteği vardır.
- Yalnız mesafe düşerse classifier faydası kanıtlanmaz; accuracy düşüşünü negatif sonuç olarak koru.
- Oracle3 çalışır, proxy3 çalışmazsa target kalitesi sınırlaması desteklenir.
- Oracle3 bile işe yaramazsa önce cebir/scheduler/state kontrollerini yeniden incele;
  temiz-z ile h uyumsuzluğu veya tüm global diffusion için kesin neden iddiası kurma.
- Oracle3 deployable TTA veya üst sınır değildir. Hiçbir kol otomatik default olmaz.

Runtime/memory ve global call sayısını kaydet. Replay önceki baseline-denoising
ve proxy encoder maliyetini dışarıda bırakır; measured worker süresinden
deployment hızlanması çıkarmak hatalıdır. Gerçek yöntem o ilk geçişe de ihtiyaç duyar.

### 8. Arşivle, knowledge güncelle ve tamamlanan kapsamı yayımla

Başarılı/null/negatif/failed sonuçları immutable raw/source/operator/derived ve
receipt/hash'leriyle local+Drive koru. Credentials/OAuth URL'lerini yayımlama.
`knowledge/shape_latent_proximal_results_20261031.md` ile kompakt evidence
kaydı oluştur; README/findings/open_questions/method_synthesis/ilgili planları
güncelle. Gerçek canlı runtime durumunu kaydet; kapatma yetkisi yok.

Publication için önce `origin` fetch et, güncel integration skill/knowledge'i oku.
İzole temiz checkout referansı:
`/Users/batuhan/.codex/research-checkouts/shape-latent-publication-20261031`.
Devirde `publication/shape-refresh-pilot-20261031` branch'i1bacabf üzerindeydi;
yeni yayın öncesi temizliğini/latesttarget'i tekrar doğrula. Sadece gerekli
tamamlanmış kod/test/knowledge/kompakt kanıtı seç. Root'un eski README/findings
dosyasıyla remote'u topluca ezme; diğer ajanın accepted kayıtlarını koru.
Hamlar/weights/temporary wrappers Git'e girmez. Normal fast-forward ile yalnız
`origin/thesis-integration` push; force/main/upstream yok. Son remote SHA'yı doğrula.

## Son teslim beklentisi

Kullanıcıya önce "kendi matched baseline'ından üstün mü?" sorusunu net yanıtla;
sonra temiz latent'e yakınlık ile accuracy farkını ve oracle/deployment sınırını
ver. Sonuç yoksa hazırlık/kurulum durumunu söyle; CPU testini GPU sonucu sayma.
Ham/knowledge/yayın referanslarını ve runtime'ın korunup korunmadığını bildir.
Belirsizliği başarı gibi sunma; yeni scope/full-data/seed taramasına otomatik geçme.
