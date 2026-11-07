# LION için GSD: dört hipotezin GPU mekanizma testleri — 2026-11-05

[Scope/Code] 2026-11-07 integration clarification: this is the standalone Nov4–5 anchor/surface/proximal/lift study from research/gsd-physical-followup-20261104@acf78fb. Its local H1–H4 labels differ from the later comprehensive H0–H8 campaign. Runtime/peer and unstarted-work statements describe this study at its Nov5 closure. See [dated inventory and remaining tests](gsd_hypotheses_test_inventory_20261107.md) and the [later Nov6 campaign](gsd_comprehensive_results_20261106.md).

[Run/Code/Inference/Open] Bu çalışma, [33 makalelik yeni literatür taramasından](gsd_lion_spectral_literature_20261104.md) çıkan dört önerinin uygulanabilirliğini sınar. Tam sınıflandırma veya SCD hız karşılaştırması değildir. Tamamlanan testlerin ham dosyaları, başarısız denemeleri ve olumsuz sonuçları saklanır. [Önceden yazılmış protokol ve yürütme geçmişi](gsd_hypotheses_gpu_20261105.md).


**Karar:** H1 yeniden oluşturulmuş anchor için üstünlük gösterilmedi; H2 robust surface yönü umut verici fakat belirsiz; H3 sabit lambda1, lambda0'a göre bu kohortta ortalama daha kötü; H4'ün identity-lift varsayımı desteklenmedi. Çözücü ve scheduler denklemleri doğru çalışıyor. Mevcut SCD/Chamfer+PM baseline korunuyor.

## Kapsam ve karşılaştırmanın anlamı

ModelNet40-C severity5, 15 bozulma, orijinal nesneler 44/91/103/137/158/223/236/265, üç seed 0/1/2, B2. Her hipotez grubu için 360 ana durum (8×15×3),32 ek timestep durumu: toplam 392 durum. Kabul edilmiş8 teknik durum yeniden kullanılınca18 yeni koşul/grup gerekir; teknik koşuyla19 ham paket/grup. Sekiz nesne daha önce açığa çıkmıştır;392bağımsız nesne veya görülmemiş doğrulama kümesi değildir.

Normal başlangıç t70, Background t340; ek kontroller t10/t70/t340. Tek bir noisy-to-clean tahmin ve donmuş z koşulu kullanılır. Çok adımlı reverse diffusion, global/style güncellemesi, sınıflandırıcı ve Probability Mean çalıştırılmaz. X aynı normalize/upsample2048/3.3885/rotasyon işleminden geçer; temiz veri yalnız tanı için, bozuk X'in merkez/ölçeğiyle dönüştürülür. Yönler aynı Y üzerinde, aynı global norm sınırı rho=.001 ile değerlendirilir. H4 ayrıca rho=.01 kullanır. Hiçbir yöntem temiz veriye veya sınıf etiketine bakarak seçilmez.

Temel tanı metriği simetrik **squared Chamfer**: iki yönlü en yakın komşu kare mesafelerinin ortalamalarının toplamı. Tablolardaki farklar LION'un hazırlanmış koordinat sistemindeki CD birimindedir; accuracy yüzdesi değildir. Aralıklar8 orijinal nesne üzerinden 10000 eşlenmiş cluster bootstrap (seed20261105) ile hesaplanır, yalnız bu kohorta koşulludur. Bozulma bazındaki aralıklar keşifseldir, çoklu karşılaştırma düzeltmesi yapılmaz.

H1 aynı bozuk-X heat operatörü F_X'i ve bütçeyi korur: −F_X(Y−X) ile −F_X(Y−R), R=encode/decode(X). H2 aynı durum ve bütçede robust unordered point-to-plane RHS'yi indexed spectral residual ile karşılaştırır; hem eşleşme hem amaç değiştiğinden farkı yalnız birine atfetmez. H3 aynı frozen surface operatörüyle lambda0/1'i karşılaştırır. H4 fiziksel delta'yı latent'in ilk üç koordinat kanalına ekler; dördüncü kanal ve style sabittir.

## Ana sonuçlar: eşlenmiş temiz geometri farkları

[Run/Inference] Negatif fark soldaki yöntemin lehinedir. Her satır aynı Y içindeki karşılaştırmadır; A ve B gruplarının farklı taze model durumları üzerinden yöntemler sıralanmaz. Her ana satır360 durum/8 nesne içerir.

| Hipotez / karşılaştırma | Ortalama CD farkı | Koşullu %95 aralık | Daha iyi durum |
|---|---:|---:|---:|
| H1: reconstructed − raw | +2.71092e-06 | [-1.45779e-05, +1.90364e-05] | 40.00% |
| H2: robust − raw | -5.33246e-05 | [-0.000170359, +4.68543e-05] | 60.28% |
| H3: graph1 − graph0 | +3.85913e-05 | [+1.00242e-05, +7.01087e-05] | 45.83% |

H1 ve H2 aralıkları sıfırı içerir. H2 ana ortalaması daha iyi olsa da LiDAR/occlusion açık başarısızlık bölgeleridir. Ek 32timestep durumunda H2 farkının işareti değişir (+3.446e-5; aralık[−1.034e-4,+1.713e-4]); H1/H3 probe aralıkları da sıfırı içerir. Farklı gürültü düzeylerinde evrensel üstünlük gösterilmedi.

| Ana durumlarda CD azalması | A: raw | A: reconstructed | A: robust | B: lambda0 | B: lambda1 |
|---|---:|---:|---:|---:|---:|
| Sayı /360 | 243 | 247 | 252 | 249 | 234 |

### Bozulma bazında eşlenmiş ortalamalar

Her hücre24 ana durum (8 nesne×3 seed). Ayrıntılı koşullu aralıklar compact JSON içinde; buradaki tablo keşifseldir.

| Bozulma | H1 reconstructed−raw | H2 robust−raw | H3 graph1−graph0 |
|---|---:|---:|---:|
| background | -4.8321e-07 | -0.000972581 | -0.000109308 |
| cutout | +9.40266e-06 | +4.67012e-05 | -5.07458e-05 |
| density | +1.00063e-05 | +4.4145e-05 | -4.86905e-05 |
| density_inc | +1.02918e-05 | -7.14621e-05 | -5.99e-05 |
| distortion | -1.28777e-06 | -0.000208083 | +7.39158e-05 |
| distortion_rbf | +7.91041e-06 | -3.12895e-05 | +8.21799e-05 |
| distortion_rbf_inv | -1.24329e-06 | -7.15636e-05 | +8.52305e-05 |
| gaussian | +2.05876e-06 | -2.34701e-06 | -2.88453e-05 |
| impulse | +1.71144e-07 | -6.66285e-05 | +2.10089e-05 |
| lidar | +7.9966e-05 | +0.000809089 | +0.00020479 |
| occlusion | -6.3525e-05 | +0.000246197 | +0.000118896 |
| rotation | -2.5852e-05 | -0.000113967 | +0.000124851 |
| shear | +8.37215e-06 | -0.000148528 | +6.80984e-05 |
| uniform | +1.82761e-06 | -0.000130566 | +4.76453e-05 |
| upsampling | +3.04823e-06 | -0.000128986 | +4.97418e-05 |

### H4 transfer doğruluğu

| Global norm sınırı rho | n | Relative hata min / median / max | Cosine min / median |
|---|---:|---:|---:|
| 0.001 | 360 | 0.7142 / 1.4978 / 6.5088 | -0.1045 / 0.4063 |
| 0.01 | 360 | 0.3256 / 0.6141 / 1.7333 | -0.1489 / 0.7945 |

rho=.001 için median hata %149.8, rho=.01 için %61.4; bunlar kabul sınırı değil ölçülen transfer hatalarıdır. En büyük zero-repeat decoder RMS4.24e-8, prior RMS2.44e-7; ana scheduler maxhata7.22e-7.784 cosine ayrıca gerçek dizilerden CPU ile yeniden hesaplandı, maxfark1.30e-7. Küçük repeat farkı, genel finite-lift uyumsuzluğunu açıklamaya yetmez.

**Posthoc tanı:** yüksek transfer hatası tek başına gerçek düzeltmenin her örnekte zararlı olduğunu göstermez. Arşivlenmiş changed NPZ üzerinde yeni CPU clean-CD hesabında rho=.001 gerçek decode226/360 durumda iyileşir (ideal hedef234/360); gerçek−ideal ortalama+7.955e-5, koşullu aralık[+3.052e-5,+1.486e-4]. rho=.01 gerçek233/360, ideal261/360; gerçek−ideal+.002516, aralık[+.001739,+.003553]. Her iki rho için gerçek−baseline aralığı sıfırı içerir. Dolayısıyla identity varsayımı başarısızdır ve ideal düzeltmenin faydası önemli ölçüde kaybolur; tüm olası backward-free guidance yöntemlerinin başarısız olduğu kanıtlanmış değildir. Bu ek hesap yeni GPU koşusu veya katsayı seçimi değildir.

## H3'ün matematiği: çözülebilirlik ile geometrik faydayı ayırmak

[Code/Run] Y üzerindeki k10 graph, PCA normalleri n, planarity ağırlığı q ve yerel median komşu mesafesi sigma sabitlenir. q kalibre doğruluk olasılığı değildir; sigma gerçek bozulma varyansı değildir. Gözlemden yüzeye nearest-neighbor eşleşmesiyle r_j=n_j^T(X_j−Y_i), a_j yalnız eşleşen düğümün normalini seçer. Huber c=1.345 için:

\[
J(\delta)=\frac\kappa2\|\delta\|^2+
\frac\lambda2\delta^T(L\otimes I_3)\delta+
\sum_jq_j\rho_c((a_j^T\delta-r_j)/\sigma),\qquad \kappa=1.
\]

Sıfır düzeltmede tek IRLS majorizer'ın ağırlıkları
\(w_j=q_j\sigma^{-2}\min(1,c\sigma/|r_j|)\) olur. Çözülen sistem:

\[
H\delta=b,\qquad H=\kappa I+\lambda(L\otimes I_3)+A^TWA,
\qquad b=A^TWr.
\]

kappa>0, lambda ve ağırlıklar negatif değilken H SPD'dir. 3×3 block-Jacobi preconditioner ve matrix-free PCG kullanılır. Çözücüde yalnız sparse kenar/node action vardır; graph oluşturma hâlâ dense cdist ile O(N²)'dir. Yöntemin tamamına doğrusal maliyet denemez. Genel A^TWA ile L komütatif olmadığından frekans başına skaler bölme genel çözüm değildir.

Tam çözüme delta* denirse Q(tau delta*)−Q(0)=(tau²/2−tau)delta*^T H delta*≤0, tau∈[0,1]. Bu yüzden norm sınırı altında frozen-Huber amaç azalımı anlamlıdır. Bu, temiz geometri veya accuracy iyileşme garantisi vermez. Eşleşmeler/normaller yeniden kurulunca aynı sabit-quadratic kanıt doğrudan tüm algoritmaya taşınmaz. GPARA prior art nedeniyle genel graph-posterior residual solve için yenilik iddiası yapılmaz.

[Run] GPU'daki784 lambda0/1 çözümün tamamı yakınsar; en fazla9 iterasyon, gerçek relative residual≤9.98e-7. Her capped düzeltmenin asıl frozen-Huber amaç farkı CPU'da yeniden hesaplanmış ve negatiftir. Bağımsız SciPy CSR/CG16 teknik durumda GPU delta'sıyla max5.84e-7, optimum amaçla relative3.55e-11 fark verir. Bu kontroller optimizasyonun çalıştığını gösterir; graph'ın temiz-CD faydasını ayrıca ölçmek gerekir.

## H4: DDIM aktarımı doğru, identity lift varsayımı başarısız

[Code/Run] Gerçek checkpoint/config decoder'ı \(D(h,z)=\Pi h+.01R(h,z)\) ve latent düzeni (B,N,4)'tür. İlk üç kanala delta eklemek için tam hata:

\[
D(h+\Pi^T\delta,z)-D(h,z)-\delta
=.01[R(h+\Pi^T\delta,z)-R(h,z)].
\]

.01 katsayısı tek başına sağ tarafın delta'ya göre küçük olduğunu söylemez. Yeni testlerin relative error metriği bu farkın normunu ||delta||'ya böler; önceki H8 gradient uyumu bunu sınamamıştı.

Sabit noisy h_t için yeni clean tahmininden epsilon yeniden hesaplanırsa, eta0 DDIM'in komşu s adımındaki hareketi:

\[
\Delta h_s=(a_s-\sigma_s a_t/\sigma_t)\Delta\widehat h_0.
\]

Eski epsilon'u sabit tutmak farklı algoritmadır. Gerçek scheduler dizileri CPU'da yeniden hesaplanır; ek t340/70/10/**0 terminal** kontrollerinin en büyük hatası3.27e-7'dir. Dolayısıyla scheduler cebrinin doğrulanması decoder lift'inin doğrulandığı anlamına gelmez.

Ek8 durumda translation/scale/random/robust-graph alanları rho1e-4/1e-3/1e-2 ile incelendi. rho1e-3'te ötelemenin median relative hatası.003665 iken robust graph alanının1.3946, random alanın1.9257; scale.4138. rho1e-4'e küçültmek robust alanı kurtarmadı (median4.5524). Bu, yön bağımlı bir aktarım sorunudur. Native komşuluk/voxel değişimlerinin nedensel açıklama olduğu kanıtlanmadı. Translation başarısından genel local deformation başarısı çıkarılamaz.

İlk ek VJP denemesi hazırlama checkout'ının eval PVCNN devoxelization backward tensörlerini saklamaması nedeniyle başarısızdır; kaynak/log/partial scheduler sonuçları korunur. Modeli train moduna almak veya native forward'ı değiştirmek yerine ikinci deneme aynı no-grad forward'da central finite difference kullanır (relative adım1e-5). Bu kesin VJP değildir. Finite lift ve scheduler NPZ'leri bağımsız CPU hesabından geçer; central plus/minus çıktıları ve permuted sparse diziler arşivlenmediğinden central-slope/graph-permutation sayıları GPU kaydı olarak etiketlenir, bağımsız yeniden hesaplanmış sayılmaz.

## Negatif kontroller ve yorum sınırı

[Run] Yeniden kurulan Y-graph + PCG için 8örneğin permütasyon relative çözüm farkı≤1.78e-14, tüm çözümler yakınsar. Gözlem sırası kontrolü H2'de ayrıca uygulanır. Bu, keyfi duplicate/tie durumlarında genel eşdeğişim kanıtı değildir. Normal-sign invariance ve bounded Huber influence için küçük bağımsız CPU tanıkları korunur.

[Run/Inference] Düzlemde tutarlı fakat yanlış gözlemler için GPU çözümü kendi amacını−.714475 azaltırken temiz-MSE'yi+.000786519 artırır. Graph sabit düzeltmeyi cezalandırmaz; gözlemlerin birlikte yanılması robust kayıpla otomatik çözülmez. Amaç azalımı ve düşük residual tek başına başarılı denoising değildir.

[Open] Buradan sınıflandırma artışı, bütün ModelNet40-C'ye genelleme, ScanObjectNN, optimum graph katsayısı, kalibre Bayesian posterior veya SCD karşısında end-to-end hız kazancı çıkarılamaz. Sabit lambda'nın bozulmaya göre sonradan seçilmesi de bağımsız validation olmadan bir yöntem doğrulaması olmaz. Üretim sampler'ı ve operasyonel Chamfer+PM N8/Background35 baseline değiştirilmedi.

## Yürütme, kaynak ve arşivler

[Code/Run] Yerel araştırma branch'i research/gsd-physical-followup-20261104, kaynak512b5aa9869673004d9b9b3724bb38dbdada5b76; ayrı gsd-lion-literature-20261104 worktree. Asıl istenen worktree başka eşzamanlı göreve geçtiğinden müdahale edilmedi. Hazırlama notebook'ı uzakta dev@a458cd46bba1288a3f41302548bc777700cbc7d4 klonlar; gerçek kaynak/native payload her ana pakette saklanır. Bu kaynak farkı örtülmez.

İki standard A100-SXM4-40GB, yaklaşık83GiB host RAM; OOM yok. Canonical on hazırlama hücresi sıralı çalıştırıldı; cell1restart sonrası aynı VM,2yalnız izin verilen fastrlock hatası,4pinned deps/CUDA12.1/GCC11.4,5–8ayrı native build/smoke,9veri,10tam verifier. CLI result-channel stall'ları gerçek completion marker/process denetimiyle çözüldü; aynı native build tekrar başlatılmadı. Konsol üzerinden exact saved source çalıştırma adaptasyonu ve eksik eski wrapper-output kanıtı yürütme notunda korunur. Pip resolver warning'leri saklıdır; global pip tutarlılığı iddia edilmez.

Torch2.1.2+cu121/Python3.8.20/NumPy1.21.2; raw LION, EMA kapalı. Checkpoint1508051672 bytes/SHA807f6732ad087a1ffdaeeaa456e32b4130c9a8a1708446cabc0059654a0a86c2. Config7860 bytes/SHA9d4a826f35fe9b3053210105eb4857811c8bfd53cd8c6a998f8ea7953e975816. Her veri dosyasının SHA'sı canonical verifier'a bağlanır. Frozen payloadv3 SHA66117f60c0f9e9acf63cca321913e165afd34c6b464edb3f400b7ad8e286c388.

İlk teknik payloadv2, gerçek load_model API'sinin use_ema argümanı olmaması nedeniyle hiçbir örnek hesaplamadan başarısızdır; ham paket korunur. V3 gerçek raw-loader çağrısını kullanır. İlk kurtarma launcher'ının yalnız archival-source yol hatası da hesaplama öncesinde korunmuştur; sonraki launcher __file__ ile gerçek çalıştırılan kaynağı arşivler. Eski wide wrapper'lar eski launch_wrapper kopyalamıştı; gerçek wide kaynakları final operator backup'ta ayrıca korunur.

A=tta-gsd-anchor-correspondence-a100-20261105-r3, endpoint gpu-a100-s-usc1f0-dym8osz58g99. B=tta-gsd-correction-transfer-a100-20261105-r8, endpoint gpu-a100-s-usc1f0-18odtdwk8cy9z. A19:32:11'de listede,19:33:12'de beklenmedik biçimde yok; kapatma komutu verilmedi. Tamamlanmış0–10 koşulların değişmez Drive ZIP'leri B üzerinden hash ile kurtarıldı. Yalnız eksik11–17, H3/H4 tamamlandıktan sonra B'de çalıştırılır; önceki başarılı koşullar tekrarlanmaz. Unsealed A condition11 partial cache ve konsol-only A setup logları kayıptır; varmış gibi raporlanmaz. A/B native ortamları ve kaynakları her koşulda ayrı kaydedilir; karşılaştırmalar aynı durumun içindedir.

Ham ve türetilmiş dosyalar: result/preparation/gsd_hypotheses_20261105/. Ana19 paket/grup için parts/package/member SHA ve ZIP CRC, komut/config/count/source bağları; geometry için bağımsız SciPy cKDTree hesabı, spectral yön için gerçek arşivlenmiş F, H3 için gerçek Huber amaçı, H4 için gerçek decode/scheduler dizileri kontrol edilir. CPU/GPU geometry toleransı önceden3e-6, yön1e-6; float32 Y+delta işlemi iki tarafta aynı tutulur. Aynı düzeltmeyi double Y'ye ekleyen önceki analizv1 korunup superseded edilmiştir.

Drive kökü: MyDrive/thesis/result/gsd_hypotheses_20261105/; anchors/,transfer/,supplement_failed/,supplement/,setup_B/,operator_timing/ ve son analysis yedeği. Tam source/log/config/NPZ/actual operator/filter/latent/RNG girdileri ana raw ZIP'lerdedir; ağırlıklar Git'e konmaz. Kompakt özetler ve tekrar hesaplama betikleri knowledge/evidence/gsd_hypotheses_20261105/ altında tutulur.

## Gerçek GPU bileşen süreleri

[Run] Aynı8cached teknik durumda B1,1warmup+3senkronize tekrar; tablo8 nesne medyanlarının medyanı ve aralığıdır. Model forward/backward, classifier/PM, disk I/O ve tam diffusion akışı yoktur. Farklı amaçların maliyetleridir.

| Bileşen | Medyan ms | Nesne medyanı min–max ms |
|---|---:|---:|
| PCG_lambda0_given_operator | 2.3395 | 2.2945–2.7266 |
| PCG_lambda1_given_operator | 6.1063 | 6.0392–7.5833 |
| dense_spectral_target_X | 51.8944 | 49.5673–57.5609 |
| surface_graph_Y | 1.6378 | 1.6165–1.6826 |
| surface_observations_given_normals | 0.4703 | 0.4611–0.4896 |
| robust_rhs_given_observations | 0.0846 | 0.0774–0.0960 |
| spectral_apply_given_filter | 0.0950 | 0.0812–0.1061 |

Graph PCG1 yaklaşık6.11ms, graph0 yaklaşık2.34ms. Dense spectral hedef kurmak51.89ms sürse de hedef önceden hesaplandığında uygulaması.095ms’dir; robust hazır-RHS.085ms. Bu nedenle hedef kurma maliyetini her adımda PCG ile karşılaştırıp bir speedup oranı ilan etmek yanlış olur. Graph kurulumu1.64ms ve observation.47ms ayrıca sayılır. Native/model backward maliyeti ve graph/F reuse stratejisi olmadan SCD hız kazancı söylenemez. Küçük tekrar sayısı ve tek GPU nedeniyle bu bir sistem benchmark’ı değildir.

Ana diagnostic loop wall toplamı A1568.65s / B144.22s; bu süreler veri hash/NPZ-filter compression/yazmayı içerir, model başlangıcını dışlar. A/B hız oranı olarak kullanılmaz. En büyük Torch allocated1.820GiB; CUDA context/cache ve host kullanımının tamamı değildir.

## Bu sonuçlardan çıkan yöntem kararı

[Inference/Open] En küçük savunulabilir yeni aday, decoder çıktısına güven sınırı altında tek physical surface düzeltmesini uygulayıp aynı frozen-head/PM okumasıyla sınamaktır. Böylece finite latent lift varsayımı gerekmez. lambda0 mutlaka kontrol kolu olur; graph1 bu çalışmada ortalama daha kötü olduğundan varsayılan seçilmez. Bu, henüz uygulanmış/tam diffusion-guidance SCD alternatifi değildir. Nesne-ayrık validation, matched SCD/off kontrolleri, tüm trajectory/classifier çıktısı ve gerçek B1/B32 toplam süre doğrulaması gerekir. Bu görevin sonuçları için baseline değiştirilmez ve yeni doğrulama kampanyası otomatik başlatılmaz.

Diffusion içinde düzeltme zorunluysa daha anlamlı sonraki hipotez decoder-aware düşük boyutlu lift'tir: sabit fiziksel basis Q için central finite difference ile T_j≈[D(h+epsilon Pi^T q_j)−D(h−epsilon Pi^T q_j)]/(2epsilon), sonra c=(T^T T+mu I)^−1 T^T delta ve Delta h=Pi^T Qc. Bu, identity varsayımını ölçülen decoder cevabıyla değiştirir. 2m ek decode, native nonsmooth davranış ve trust-region/actual decode kontrolü maliyeti vardır; hız avantajı garantisi yoktur. Bu yalnız türetilmiş yeni hipotezdir, çalıştırılmış yöntem veya literatür yeniliği değildir.

## Doğrulama ve devir

Toplam 38başarılı ana ham paketin manifestleri, frozen-source/count denetimi ve bağımsız CPU gate'leri geçti. Max CPU/GPU geometry farkı A2.205e-7/B2.628e-7, max spectral/robust yön farkı9.095e-9; önceden belirlenen toleransların altında. H2 observation-order maxrelative RHS5.18e-16/392 durum. Ek 109 diagnostic ve56 bileşen timing kaydı doğrulandı. Bunlar bağımsız nesne veya sınıflandırma test sayısı değildir.

[Makine okunabilir ana sonuçlar](evidence/gsd_hypotheses_20261105/anchors_final_summary.json), [H3/H4](evidence/gsd_hypotheses_20261105/transfer_final_summary.json), [kapsam/kaynak denetimi](evidence/gsd_hypotheses_20261105/final_audit.json), [ham arşiv konum/hash listesi](evidence/gsd_hypotheses_20261105/raw_acceptance_manifest.json), [H4 posthoc geometri](evidence/gsd_hypotheses_20261105/h4_geometry_posthoc.json), [ek kontroller](evidence/gsd_hypotheses_20261105/supplement_cpu_recount.json).

Kullanıcı bütün gerekli GPU testlerini ve tamamlandıktan sonra kendi oturumlarımızın kapanışını açıkça yetkilendirdi. Bilimsel testler, final analysis/source Drive roundtrip doğrulaması ve yetkilendirilmiş kapanış tamamlandı. B-r8 CLI ile kapatıldı; A-r3 önceki beklenmedik kayıp sonrası envanterde yok; kendi monitor93787 durdu. Peer H5-r4 korunuyor. [Kapanış kanıtı](evidence/gsd_hypotheses_20261105/closure.json). Drive snapshot kapanış öncesidir; son lifecycle metadatası repository/local kayıttadır.
