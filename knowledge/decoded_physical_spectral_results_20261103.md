# Decoded fiziksel spektral guidance: sınırlı teknik sonuçlar

Tarih: 2026-11-03. Dal: `research/gsd-guidance-math-audit-20261103`.
Kanıt etiketleri: **[Paper] / [Code] / [Run] / [Inference] / [Open]**.
Bu kayıt V3/V4'ün kabul edilmiş dokuz koşusunu özetler; V5 native sonucu bekleniyor.
Hazırlık ayrıntıları: [çalışma kaydı](decoded_physical_spectral_study_20261103.md).

## Sonuç ve kapsam

[Run] Gaussian/s5, ModelNet40-C, original ID **44 ve91**, B2, seed0,
frozen Point-MAE ve raw/eval/EMAoff LION üzerinde teknik kontroller yapıldı.
Dokuz koşulun tümünde PM **2/2**; aynı iki nesnenin tekrarları bağımsız örnek değildir.
Bu sonuç accuracy kazancı, tam benchmark veya genel surrogate doğruluğu göstermez.
[Inference] Yerel decoder türevinin küçük-adım FD uyumu ve H8'in ölçülen
denoiser-sonrası yön hizası destekleniyor; hedefin doğru temizleme yönü olduğu açık.
[Code/Decision] [Yayınlanmış operasyonel N8 Chamfer+PM baseline](current_baseline.md)
ve varsayılanları korunur; bu deneyler explicit opt-in dallardır.

[Run] Sabit politika: normal8 adım, local gamma`.03`, conditioning-style eta`.003`,
SCD p`.95`, toplam DDIM100/final-alpha1; clean2/4/6 ve terminal8 PM adayları.
Gerçekte yalnız Gaussian çalıştı; Background35 veya diğer bozulmalar ölçülmedi.
ReferansX: normalize → upsample2048 → scale3.3885 → rotate sonrası **encoder'a
giren aynı B2×2048×3 tensor**; classifier-normalized/FPS1024 verisi değildir.
Decoder original encoded z0'ı kullanır; classifier gradienti kapalıdır.
Yerel/style guidance, predicted-clean h üzerinden mevcut denoiser VJP'sine ulaşır.

## Fiziksel loss ve H8'in kesin ayrımı

[Code] `graph_spectral.build_smooth_spectral_target` aynıX'ten donmuş graf kurar:
k10, delta`.1`, graph_gamma`.6`, smooth beta2, mean-active-degree ölçeği.
Smooth profilde tüm aktif modlar ağırlıklanır; modes100 hard cutoff değildir.
İzole vertex'lerde filtre sıfırdır. Batch-sum ve original nokta sayısıN2048 korunur:

\[
E=D(h,z_0)-X,\qquad L=\sum_b\frac{\operatorname{tr}(E_b^\top F_bE_b)}{3N},
\quad F_b=U_b\operatorname{diag}(w_b)U_b^\top.
\]

[Code] Gerçek resolved LION konfigürasyonunda `latent_pts.skip_weight=.01`;
`models/latent_points_ada.py` decoder'ı **D(h,z0)=Πh+.01R(h,z0)** döndürür.
Π her local noktanın ilk üç kanalını seçer; dördüncü feature kanalı ayrı kalır.
H8 checkpointte `b_t=stopgrad(D(h_t,z0)−Πh_t)` artık terimini dondurur:

\[
\widetilde L_t(h)=\frac{\langle\Pi h+b_t-X,F(\Pi h+b_t-X)\rangle}{3N},
\quad g_X=\frac{2FE}{3N},\quad g_{sur}=\Pi^\top g_X,
\quad g_{full}=\Pi^\top g_X+.01J_R^\top g_X.
\]

[Inference] Aynı checkpointte fiziksel scalar değer eşittir; gradient aynı değildir.
Atılan terim `.01 J_Rᵀg_X`'tir. H8 tam decoder gradientinin hatasını gideren bir
tamir veya yeni exact yöntem diye adlandırılmaz; **frozen-residual surrogate**'tır.
Gradient hızı/benzerliği, corruptedX'e sadakatin accuracy açısından doğru olduğunu kanıtlamaz.
[Code] `decoded_spectral.py` default `freeze_decoder_residual=False` ile
`full_decoder_gradient`; true ile `frozen_residual_surrogate` kaydeder.
Kararlı ifade `decoded.detach() + (latent_xyz−latent_xyz.detach())`, aynı noktadaki
decoded değerini bit düzeyinde korurken yalnız identity XYZ türevini bırakır.
`tta.py` default `decoded_spectral_weight=0`; yalnız açık pozitif ağırlıkta
physical scalar mevcut SCD'ye eklenip aynı joint backward/update'te tüketilir.

## Native türev ve H8 yön kontrolleri

[Run] V3 first clean2/sample0 audit'inde her FD forward gradient-enabled branch'te,
aynı RNG/buffer başlangıcı ve requires-grad clone'larla çalıştı; scale`.001/.01`
ilk local latent normuyla çarpıldı. Kabul: `|FD−AD|≤1e−5+.1|AD|`, resolved
türevler, repeat-noise slope sınırı ve RNG/buffer/live-input korunması.
Bağımsız NumPy yeniden hesabı:

| Scale | AD yön türevi | Central FD | Göreli hata |
|---|---:|---:|---:|
| .001 | .002917239708 | .002851540491 | %2.25210 |
| .01 | .002917239708 | .002742544326 | %5.98838 |

[Run] Baseline loss`.030217683686`; repeat loss farkı`2.58438e−8`.
V3 yerel decoder full/surrogate cosine `[.999960004,.999847415]`, norm oranı
`[1.000152913,1.001122633]`; atılan terimin full-gradient normuna oranı
`[.008945777,.017514906]` (yaklaşık%0.895/%1.751).
[Run] V4 bağımsız native tekrarında FD hataları%2.20192/%6.00480 ile kabul geçti;
yerel omitted-term oranları`.008888007/.018045249` idi.
[Open] Bu, frozen graph ve seçilmiş yerel yönlerde empirical uyumdur.
PVCNN voxel-coordinate detach/eksik coordinate-adjoint yollarını veya bütün
decoder/denoiser Jacobian'ını matematiksel olarak sertifikalandırmaz.

[Run] V4'te zero-guidance clean2/4/6'da iki nesnenin **altı denoiser-sonrası**
local/style update vektörü karşılaştırıldı; sabit`.03/.003` adım ölçekleri içerilir.
Önceden tanımlı cosine≥`.99`, norm oranı`[.9,1.1]` kapısı geçti:
**minimum cosine`.999839139`; surrogate/full norm oranı1.000145594–1.001106079**.
[Inference] Bu iki Gaussian nesnesinde H8'in upstream yön yaklaşımı güçlüdür;
başka başlangıç, timestep, corruption veya uygulanan trajectory için garanti değildir.

[Run] Katsayı accuracy'ye bakılmadan önceden ilan edilen rho`.1` kuralıyla donduruldu:
`weight=.1×median(first-checkpoint per-object SCD/physical update-norm ratio)`.
Oranlar`8482.283250709/7567.718431619`; **weight802.5000841164167**.
Bu tek katsayı küçük applied-step smoke içindir; accuracy seçimi/grid sonucu değildir.

## Dokuz kabul edilmiş ham koşul ve maliyet

[Run] Aşağıdaki ZIP'ler ortak `result/modelnet40_c/decoded_physical_spectral_diagnostic/`
altındadır. V3 kaynak`3e8b9be3d9c913d6757ea09026478b925da77fe0`, V4
`b0bdbfbbfaa331690796c7b56546db8f3d7ae088`; her arşiv ayrı ham kayıttır.
Tablodaki saniyeler `per_corruption.csv.runtime_seconds` ölçümüdür:

| UTC run | Kaynak / kol | Ağırlık | Süre(s) |
|---|---|---:|---:|
| [20:17:14](../result/modelnet40_c/decoded_physical_spectral_diagnostic/20261103-201714_decoded-control-8aed642b.zip) | V3 no-grad control | 0 | 2.069573 |
| [20:22:20](../result/modelnet40_c/decoded_physical_spectral_diagnostic/20261103-202220_decoded-diagnostic-8aed642b.zip) | V3 FD+diagnostic | 0 | 10.190765 |
| [20:31:58](../result/modelnet40_c/decoded_physical_spectral_diagnostic/20261103-203158_decoded-diagnostic-8aed642b.zip) | V3 diagnostic tekrar1 | 0 | 2.270284 |
| [20:32:35](../result/modelnet40_c/decoded_physical_spectral_diagnostic/20261103-203235_decoded-diagnostic-8aed642b.zip) | V3 diagnostic tekrar2 | 0 | 2.291791 |
| [20:33:13](../result/modelnet40_c/decoded_physical_spectral_diagnostic/20261103-203313_decoded-physical-c08bc43d.zip) | V3 applied full | 802.500084 | 2.321140 |
| [20:38:50](../result/modelnet40_c/decoded_physical_spectral_diagnostic/20261103-203850_decoded-diagnostic-8aed642b.zip) | V4 FD+H8 diagnostic | 0 | 10.311131 |
| [20:39:37](../result/modelnet40_c/decoded_physical_spectral_diagnostic/20261103-203937_decoded-live_control-8aed642b.zip) | V4 hafif live0 | 0 | 1.869002 |
| [20:40:14](../result/modelnet40_c/decoded_physical_spectral_diagnostic/20261103-204014_decoded-physical-c08bc43d.zip) | V4 hafif full | 802.500084 | 2.017366 |
| [20:40:51](../result/modelnet40_c/decoded_physical_spectral_diagnostic/20261103-204051_decoded-physical-c08bc43d.zip) | V4 hafif surrogate | 802.500084 | 1.895920 |

[Run] Son üç kolun ölçülen peak'i aynı: **21756.4458MiB**.
Bu süreler **graph setup'u dışlar**; whole-worker wall süreleri sırasıyla
**27.904132/28.110213/27.906620s**. Tek B2 ölçümü dağılım/medyan değildir.
FD/diagnostic ekstra decoder/VJP/SCD işleri taşıdığı için deployment hızına eşitlenmez.
V4 surrogate decoder backward'ı atlıyordu, fakat forward autograd grafını hâlâ kuruyordu;
bu üç ölçüm decoder graph belleğinin kaldırıldığını göstermez. V5 bunun ayrı kontrolüdür.

[Run] Aynı zero-weight V3 tekrarlarının trajectory-logit maksimum farkı
**2.1808619499**; ID/seed/paired başlangıç ortaklığı native drift'i kaldırmadı.
[Inference] Bu nedenle farklı process koşularının final logit/geometry/loss farkını
tek başına physical guidance'a nedensel olarak yükleyemeyiz. Aynı-state yön testi
ayrı ve daha dar kanıttır; dokuz koşulun2/2 sonucu bu confound'u çözmez.

## Kanıt dosyaları ve literatür

[Run] [Dokuz koşulun türetilmiş metriği](../result/preparation/gsd_physical_spectral_20261103/all_nine_case_metrics.json),
[V3 bağımsız FD/H8 hesabı](../result/preparation/gsd_physical_spectral_20261103/operator_v3/independent_fd_and_surrogate.json),
[accuracy öncesi katsayı](../result/preparation/gsd_physical_spectral_20261103/operator_v3/calibration_rho01.json),
[V4 upstream kapısı](../result/preparation/gsd_physical_spectral_20261103/operator_v4/h8_upstream_gate.json),
[V4 bağımsız recount](../result/preparation/gsd_physical_spectral_20261103/operator_v4/independent_native_recount.json).
Ham ZIP'lerde command/config/environment/stdout/CSV/notes, bundle SHA manifesti,
IDs/labels/PM logits, scheduler/RNG/model-state kayıtları ve probe tanıkları tutulur.
İlkB2 physical filtreleri, preparedX/h0XYZ/ilk decode ve FD gradient/direction/±decode
vektörleri bağımsız denetime açıktır; kompakt kanıtlar `knowledge/evidence/decoded_physical_spectral_20261103/`.

[Paper] [İlk11 birincil kaynağın yöntem/denklem incelemesi](gsd_guidance_math_efficiency_20261103.md)
GSDTTA,3DD-TTA,LION ve graph filtering literatürünü ayırır. Ek
[DPS §3.1–3.2](https://arxiv.org/html/2209.14687v2) ve
[Piecewise Guidance §III](https://arxiv.org/html/2507.18654v1) okumalarının sınırları
[H8 çalışma notunda](decoded_physical_spectral_study_20261103.md) kayıtlıdır.
Bu görüntü/ölçüm-modeli yöntemleri H8 ile aynı algoritma veya point-cloud başarı kanıtı değildir;
bu rapor yeni sistematik literatür taraması, tam GSDTTA reprodüksiyonu veya novelty iddiası içermez.

## V5 — forward grafını kaldırma denemesi reddedildi

[Run/Decision] Kaynak`69318f5e44f82388b12c1f81f6ce0643e3dd61e1` için
[V5 kapısı](../result/preparation/gsd_physical_spectral_20261103/operator_v5/h8_upstream_gate.json)
**false**. Üç checkpoint'in max farkları[1.59353e-4,1.12236e-4,0], RMS farkları
[1.24230728e-5,9.37622735e-6,0]. İlk RMS, önceden sabitlenen1e-5 sınırını
geçti; max sınırı5e-4 ve bütün upstream yön/norm kapıları geçti. Native FD de geçti.
Aynı ilk decoded-pair NPZ'sinden bağımsız RMSE1.24230727e-5 olarak tekrar hesaplandı.
Bu, `.detach` değişikliğinin kesin hata/accuracy kaybı kanıtı değildir; sayısal
forward eşdeğerliği bu protokolde **kabul edilmedi**. Tolerans sonradan değiştirilmedi.

[Run] [V5 tanı ZIP'i](../result/modelnet40_c/decoded_physical_spectral_diagnostic/20261103-205445_decoded-diagnostic-8aed642b.zip),
42038416bytes/SHA256`5b2dbf7e2b94e5a49b65759cf4fca05fc59fdfa15f727209d63261a092c2f120`,
108sealed dosyası/source/state/cache/FD/gate bakımından bağımsız kabul geçti.
Runner'ın seçili iki nesnelik tanısı tamamlandı; dış operator kapısı başarısızdır.
**Üç V5 applied/timing koşulu başlatılmadı.** Toplam10runner arşivi vardır;
9ilk V3/V4 koşulu ile1V5 negatif tanısıdır, yalnız2benzersiz nesne içerir.

[Code/Decision] Decoder girişlerini ayıran son optimizasyon6025ed5, `07f5637`
ile geri alındı. Yayın için tutulan H8 decoder forward grafını korur, fakat
frozen-residual loss decoder backward'ından geçmez; V4native olarak doğrulanan
observer'a döner. Tanıdaki detached-forward tanığı ve başarısız kanıt korunur.
DefaultN8Chamfer+PM hâlâ korunur.487CPU/1skip,25.608s deneysel source kontrolüydü;
geri alınmış son ağaçta486test/1skip,26.222s geçti.
[Nihai kaynak/test/ham arşiv makbuzu](evidence/decoded_physical_spectral_20261103/final_receipt.json).

[Run] İlk bilimsel saat20:17:13.051620UTC, son GPU/operator bitişi20:55:31.395468UTC:
yaklaşık2298.34s/**38.31dk**, hazırlık ayrı. Değişmez3300s sınırı sıfırlanmadı
ve aşılmadı. Sonraki işler CPU kabul/arşiv/knowledge/integration'dır; yeni GPU
araştırması planlanmadı. Oturum ve monitor97932 açık tutuluyor; kapatma izni yok.

## Sonraki yanlışlanabilir öncelikler

[Decision/Open] P0: V5 graph-removal hız iddiası reddedildi. Önce aynı-state native
tekrar gürültüsünü ve backend protokolünü sabitle; yeni bir kabul ölçütü ancak
yeni ön-kayıtlı deneyde kullanılabilir, bu başarısızlığı sonradan başarıya çevirme.
[Open] P1: Aynı-state upstream H8 hizasını ayrı development nesneleri/bozulmalarında
ölç; cosine düşmesi veya norm oranının kapı dışına çıkması genellemeyi yanlışlar.
[Open] P2: Slot correspondence ve corrupted-anchor yanlılığını ayrı kontrol et;
matched frame/correspondence altında temiz referans oracle'ı yalnız açıklama için kullan.
[Inference/Open] Tam ortonormal sabitU için `UUᵀY=Y` geçerlidir; truncated baz projection'dır.
SabitU otomatik inverse hatası değildir. Değişen geometriye uygun olmayan graf/metric ve
bozukX'e sadakat hâlâ açık hedef sorunlarıdır; H8 hizası bunları çözmez.
[Open] Teknik ve hedef kontrolleri geçilirse katsayıyı kilitleyip aşama-ayrık all15/çok-seed
accuracy ve toplam maliyet ölç; bu iki nesnede yeni sweep veya accuracy seçimi yapma.


## Taşınabilir son kanıt paketi

[Run] Setup/başarılı ve başarısız operator logları, üç scientific source overlay,
son yayın kaynakları/raporlar ve gerçek native/cache kapsülü:
`result/preparation/gsd_physical_spectral_20261103/final_research_package.zip`,
3698932bytes/SHA256`44901d5160cbc191a430066cb47f46614b2ab86a7b8c1db6ca97beaac2766650`.
Drive task`gsd_physical_spectral_20261103/v5/deployment/final_research_package.zip`;
on ham runner ZIP'i task`/v3`/`/v4`/`/v5` koşul dizinlerinde ayrıca korunur.
Ham/source/ara ağırlıklar Git'e eklenmez; gerekli kod, testler ve kompakt kanıtlar
`origin/thesis-integration` için seçilir. Runtime kapatılmadı.
