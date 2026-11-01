# SCD-h / smooth spectral-z — tamamlanan kayıtlı çalışma (2026-11-01 UTC)

[Run/Decision] SCD katsayısı sabit tutulup spektral-z adımı ayrı kalibre ve tarandığında, seçilen aday için güvenilir genel accuracy artışı gösterilmedi. Mevcut SCD baseline korunur; bu spektral profil varsayılan hâle getirilmez. Bu sonuç, tüm spektral yöntemlerin zararlı olduğunu veya gerçek etkinin tam sıfır olduğunu kanıtlamaz.

## Eşlenmiş doğrulama accuracy’si

ModelNet40-C severity5, frozen Point-MAE; 128 aşama-ayrık nesne, 15 bozulma ve seeds0/1/2. Her kolda5760 değerlendirme olayı var; bağımsız nesne sayısı128. Bütün kollarda alpha_h_local=.03. C: z/style için SCD eta=.003; Z0: style güncellemesi kapalı; aday: M400/rho1, kappa_z=19.769744597096295. C_base ve C_match aynı koşul olduğundan bir kez çalıştırıldı.

| Kol | Doğru/olay | All15 PM % | Macro14 % | Background % |
|---|---:|---:|---:|---:|
| Standart SCD (C) | 3778/5760 | 65.5903 | 65.7366 | 63.5417 |
| h:SCD / z:smooth spectral | 3777/5760 | 65.5729 | 65.6064 | 65.1042 |
| Style-off (Z0) | 3772/5760 | 65.4861 | 65.6808 | 62.7604 |

| Adayın farkı | Puan | %95 nesne-cluster CI | Düzeltilen / bozulan olay |
|---|---:|---:|---:|
| Spektral − C | -0.01736 | [-0.39931, +0.36458] | 69 / 70 |
| Spektral − Z0 | +0.08681 | [-0.31250, +0.46875] | 70 / 65 |

10000 bootstrap tekrarı, seed20261101: her yeniden örneklenen özgün nesneyle birlikte bütün bozulmalar ve üç seed tutuldu. Her iki güven aralığı0 içeriyor. Aralıklar dondurulmuş development seçimine koşullu; full-benchmark veya harici genelleme iddiası değildir.

| Seed | SCD % | Spektral % | Z0 % | Spektral−C puan | Spektral−Z0 puan |
|---|---:|---:|---:|---:|---:|
| 0 | 65.0521 | 65.1042 | 64.7917 | +0.0521 | +0.3125 |
| 1 | 65.4688 | 65.6771 | 65.9375 | +0.2083 | -0.2604 |
| 2 | 66.2500 | 65.9375 | 65.7292 | -0.3125 | +0.2083 |

Background’da aday C’ye göre +6/384, Z0’a göre +9/384 doğru olay sağladı. Diğer14 bozulmanın toplamında C’ye göre7, Z0’a göre4 olay kaybetti. Background avantajı betimseldir; ayrı bir doğrulanmış altgrup başarısı olarak sunulmaz.

| Bozulma | SCD % | Spektral % | Z0 % | Spektral−C puan |
|---|---:|---:|---:|---:|
| uniform | 77.3438 | 78.3854 | 78.6458 | +1.0417 |
| gaussian | 78.9062 | 79.1667 | 79.4271 | +0.2604 |
| background | 63.5417 | 65.1042 | 62.7604 | +1.5625 |
| impulse | 80.2083 | 79.4271 | 79.4271 | -0.7812 |
| upsampling | 84.1146 | 84.3750 | 83.8542 | +0.2604 |
| distortion_rbf | 63.2812 | 63.0208 | 63.5417 | -0.2604 |
| distortion_rbf_inv | 67.7083 | 66.1458 | 66.6667 | -1.5625 |
| density | 73.4375 | 72.6562 | 72.3958 | -0.7812 |
| density_inc | 86.1979 | 86.1979 | 86.4583 | +0.0000 |
| shear | 66.9271 | 66.9271 | 67.1875 | +0.0000 |
| rotation | 35.9375 | 35.9375 | 35.4167 | +0.0000 |
| cutout | 71.0938 | 71.8750 | 71.0938 | +0.7812 |
| distortion | 64.0625 | 63.8021 | 64.3229 | -0.2604 |
| occlusion | 39.0625 | 38.5417 | 38.5417 | -0.5208 |
| lidar | 32.0312 | 32.0312 | 32.5521 | +0.0000 |

## Tarama ve sabit protokol

Global z diffusion kapalıdır. Spektral adayda h yalnız SCD, z/style yalnız spektral gradient ile güncellenir. Normal8/Background35, B32,2048 LION noktası, raw/eval LION EMAoff, p=.95, DDIM100/finalalpha1; başlangıç h0XYZ hem SCD hem statik graf referansı. Decode bütün kollarda özgün z0 ile yapılır; son style güncellemesi final decode tarafından tüketilmez. PM normal2/4/6+8 ve BG9/18/27+35 eşit olasılık ortalamasıdır.

Profil: offdiag statik graf k10/delta.1/gamma.6, ham eigenvalue üzerinde exp(-2λ), gerçek M_max kesmesi, legacy_l2 batch-sum/(3N). Modern normalize/all-active profil ile karıştırılmadı. Label-free32development/4bozulma/3durum kalibrasyonunda her M için384 geçerli gözlem,0 dışlama; b_M değişmeden donduruldu. M200/400/800/1200/1800 ve rho.1/.3/1/3 tarandı; spektral ağırlık1, kappa=rho*b_M.

A22 ve B8 toplam30 development koşulu tamamlandı:29 kabul, M800/rho3 için1 nonfinite-gradient unavailable; accuracy’si0 yapılmadı ve tekrar denenmedi. B’de tamamlanmış .01/.05 local-step ablasyonları geçmiş olarak korundu. Kullanıcı talimatından sonra SCD adımı.03 sabit; yeni alpha taraması yok. Development winner667/960=%69.4792, C661/960=%68.8542, Z0666/960=%69.3750 idi. Development ve confirmation farklı nesneler içerir; ham accuracy seviyeleri doğrudan yöntem kazancı olarak karşılaştırılmaz. Development’teki +.625puan eşlenmiş katkı, confirmation’da güvenilir bir artışa dönüşmedi.

## Taze hazırlıkla maliyet

A10080GB/B32; her kol için ayrı ısınma ve üç taze tekrar, Gaussian/Background32nesne. Süre encoder, gerçek graf/basis I/O ve PM probe/head hesapları dahil reconstruction+head toplamıdır; model yükleme ve Drive arşivleme hariç. Bu B32 ölçümüdür, B1 online gecikmesi değildir. Süre sütunları medyan; bellek sütunları üç tekrarın maksimum kayıtlı PyTorch allocated peak MiB değeridir, toplam fiziksel GPU belleği değildir.

| Bozulma | SCD s/batch | Spektral s/batch | Süre oranı | SCD peak MiB | Spektral peak MiB |
|---|---:|---:|---:|---:|---:|
| gaussian | 4.3341 | 14.4686 | 3.338× | 25221.4 | 25900.2 |
| background | 7.0235 | 19.7350 | 2.810× | 15254.2 | 15933.4 |

Graf hazırlığı aday maliyetinden çıkarılmadı. Bu uygulamadaki ek süre karşılığında genel accuracy kazancı gösterilmedi; baseline’ı koruma kararının deneysel gerekçesi budur.

## Graph Spectral hesabı doğru mu?

[Paper/Code] [Önceki makale/GFT denetimi](graph_spectral_retrospective_audit_20261101.md) ve [style-routing ayrımı](shape_latent_spectral_guidance_audit_20261101.md): U sütun eigenvector düzeninde H=UᵀX, ters dönüşüm X=UH ve sabit bazda weighted-L2 gradienti doğru. Bu fork profili GSD-inspired latent spectral guidance’dır; tam GSDTTA replikasyonu değildir.

[Run] Aynı donmuş kaynak ve no-build native yükleyiciyle A100’de iki sentetik B2/N32/k10 durum (tiesiz ve duplicate-pair) test edildi. Euclidean kNN mesafeleri/top-k/self-edge CPU oracle ile; gerçekleşmiş tie-valid indekslerin RBF/pruning/isolate1000+jitter1e-5 Laplasyeni; GFT roundtrip ve M10 kesmeli spektral-coordinate gradient analitik hesapla uyuştu. M10 yalnız küçük sentetik operatör tanığıdır, yeni accuracy adayı değildir. Bütün vektörler ayrıca yerelde NumPy1 ile doğrulandı.

Maksimum GPU tanık hataları: mesafe 8.96e-08; Laplasyen 0.00013; GFT roundtrip 5.07e-07; coordinate-gradient 1.82e-12. [Tam çalıştırılmış assert kaynağı](evidence/scdh_spectralz_native_contract_20261101.py), [native/NumPy kanıtı](evidence/scdh_spectralz_native_20261101.json).

İlk native kabul sırasında ~1000 isolate-penalty kuyruğundaki sıralama ihlali eigenpair’leri birlikte sıralayarak giderildi. Bu modlarda exp(-2λ)=0 olduğundan bu düzeltmenin accuracy düşüşünün nedeni olduğu gösterilmedi. FD karşılaştırmalarında PVCNN’in grad-enabled forward dalı ve kayıtlı gradient-yönü kullanıldı;8/8 küçük native tanık geçti. Önceki başarısız arbitrary-direction FD kayıtları saklanır. Bu, bütün LION Jacobian’ının veya her CUDA tie sırasının doğruluk kanıtı değildir; native bitwise prediction parity varsayılmaz.

## Eski kazançlar ve veri setleri arasındaki farkın yorumu

[Run/Inference] Eski Point-MAE/ModelNet olumlu sonuçlar geçerli tarihsel gözlemlerdir; sonraki eşlenmiş sonuçlar aynı veri setinde bile head/profil/readout’a göre karışıktı. “Bir veri setinde çalışır, diğerinde çalışmaz” tek nedenli açıklama desteklenmiyor. Bu kontrollü çalışma, SCD katsayısını sabit ve spektral adımı ayrı tutarak da genel kazanç göstermedi.

[Inference] Olası etkenler: ham-λ filtresinin graf yoğunluğu/latent ölçeğiyle değişmesi; pruning/izole noktaların etkin operatörü değiştirmesi; bozulmuş h0 referansını koruyan ek kısıtın yararlı prior değişimlerini de sınırlaması; SCD ile örtüşen geometri hedefi ve backbone karar sınırlarının bu düzenlemeye farklı tepki vermesi. Bunlar doğrulanmış tekil accuracy kaybı nedenleri değildir. Negatif gradient cosine tek başına zarar kanıtı değildir.

[Run] Teknik32development nesnesi/Gaussian-BG üzerinde M800, M1800 filtre ağırlık toplamının medyan %99.37/%99.43’ünü; M400 %92.02/%93.61’ini taşıdı. Büyük nominal M her zaman anlamlı ek ağırlık getirmiyor. Bu filtre-kütlesi gözlemi gradient veya operatör eşitliği değildir; [ham eigenvalue’den bağımsız yeniden hesap](evidence/scdh_spectralz_filter_mass_20261101.json) yalnız iki bozulmayı kapsar.

## Kabul, kaynaklar ve sınırlar

Bütün6faz ve51başarılı raw koşul yerelde NumPy1 ile tekrar kabul edildi;1 unavailable aday failure olarak doğrulandı.52kayıtlı koşulun yanında eski yarım C0 arşivi de korunarak toplam53slot ZIP’i SHA/CRC kontrolünden geçti.232 fiziksel paired-state NPZ, kaynak/config/etiket/indeks/RNG/PM/faz zinciri, calibration transport ve10000-bootstrap yeniden hesapları kontrol edildi. Temiz integration checkout’unda163CPU ve37NumPy1 fixture testi geçti. Çalıştırılmış scientific worker1f631d4; sourceZIP7fedf23303660af51e74fb91860a5cd78d1a2a44db27b3f4f56d06ab9cd629d9; asset manifesteb85a9dae4df89b2fdb64432291d1413add4680ae9d0c3539ffbaa98bb5f3dda; plan231bb89e91815cfb0a2e93658c519fa7a30c2c9f02af28faaa7d4d0e679875b2.

r1 kaybından sonra bütün kabul edilmiş A/B sonuçları korundu. Yalnız900s sınırında58/60batch’te durmuş, kabul edilmemiş C0 yeni retry1 dizininde tamamlandı. r2’de iki generic native binary yeniden derlenmiş ve hash’leri farklıdır; eski byte eşitliği iddia edilmez. Orijinal calibration gözlemi/binding’i ve sayısal b_M aynen tutuldu; ayrı isimli transport raporu bu farkı açıkça kaydeder. Bütün C/timing kolları aynı yeni build’i kullandı. Diğer dört pinned native, model, veri ve worker kimlikleri eşlendi.

128ID development’ten ayrık olsa da geçmiş tam2468benchmark maruziyeti vardır; harici/unseen test sertifikası yok. Bu faz ScanObjectNN veya başka backbone deneyi değildir. Non-FD2048² bazların tamamı portable değildir; tam paired-state ve baz hash’leri saklandı. Tüm failed/setup/transport kayıtları [yürütme notunda](scdh_spectralz_execution_20261101.md) korunur. Yeni full benchmark, ek grid veya default değişikliği yapılmadı.

Yerel ham/türetilmiş kök: `/Users/batuhan/.codex/worktrees/cb07/3DD-TTA/result/preparation/scdh_spectralz_search_20261101/`. Başarılı/failed slot ZIP’leri `numerical_r3/retrieved/`; full bağımsız çıktı `runtime_r2/local_acceptance/independent_summary.json`. [Development CSV](evidence/scdh_spectralz_development_20261101.csv), [confirmation CSV](evidence/scdh_spectralz_confirmation_20261101.csv), [kompakt kabul/CI/maliyet/provenance](evidence/scdh_spectralz_progress_20261101.json).

Drive taşınabilir arşiv: `/content/drive/MyDrive/thesis/result/scdh_spectralz_search_20261101/recovery-evidence/portable_campaign_1793571746636615671.zip`; 465379670byte, SHA256`c6babbae390eac519c9c1a4ea6e18f83586077356bdd1e0959b8dcc44570f25b`. Aynı byte’lar Drive→local indirilip CRC, güvenli yollar ve üye hash’leri doğrulandı.

Yerel analiz kaynağı, ortamı, ham analiz stdout’u ve bağımsız özet ayrıca `/content/drive/MyDrive/thesis/result/scdh_spectralz_search_20261101/recovery-evidence/derived_analysis_20261101T223435Z.zip` içinde saklandı; 16059byte, SHA256`b7ebda385d56583ad190c3471547123b51a1606e1041b2109ad9680c97d88b58`. Drive upload→yeni local download SHA/CRC roundtrip geçti.

## Yayın ve oturum

Gerekli opt-in kod, testler, ID kayıtları ve research knowledge `origin/thesis-integration` için temiz ayrı checkout’ta hazırlandı. Geçici setup/transfer wrapper’ları, ham ZIP/cache ve ağırlıklar Git’e alınmaz; kaynak/kanıt arşivlerinde tutulur. Yayın ve yetkilendirilmiş r2 kapanışı henüz bekliyor; gerçekleşen durum aşağıya kaydedilecek.
