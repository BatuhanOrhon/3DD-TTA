# H5 temizsiz güvenilirlik kaydı — implementasyon

Tarih: 2026-11-05. Dal: `research/gsd-h5-reliability-20261105`.
Etiketler: **[Code] / [CPU] / [Open]**.
Kapsam: [H5 çalışma planındaki](h5_reliability_study_20261105.md) kayıt desteği;
bu not yeni GPU sonucu veya accuracy kazancı bildirmez.

[Code] `scripts/diagnose_decoded_spectral.py --capture-h5` yalnız `--arm control`
ve weight0 ile kabul edilir. Default kapalıdır; worker da bu sınırı yeniden denetler.
Mevcut clean-PM callback'lerinden üç fiziksel decoder çıktısının immutable,
detached CPU kopyası alınır. Dördüncü çıktı mevcut terminal decode'dur.
Ek decoder forward, model gradienti, filter uygulaması veya guidance eklenmez.
Classifier/PM/readout ve original-z decoder sözleşmesi korunur.

[Code] Her `batches/<corruption>/batch####/` dizinine şunlar eklenir:

- `h5_geometry.npz`: `index`, `candidate_names`, `candidate_loop_counts`,
  `decoded_xyz` (B×4×N×3), `prepared_x` (B×N×3),
  `normalization_center` (B×1×3), `normalization_scale` (B×1×1).
- `h5_spectra.npz`: `index`; her object i için `active_indices_i`, `basis_i`,
  `q_i`, `rank_i`. Index/rank int64; basis/q native çalışan dtype'ı korur.
- `h5_capture.json`: candidate'lar, tensor SHA'ları, dosya SHA/bytes kimlikleri,
  SpectralConfig ve object başına graph diagnostics, kayıt süreleri/bellek sınırları.

[Code] Clean loop sayıları observer'ın gerçekleşen probe kayıtlarından gelir;
terminal sayısı gerçek scheduler `active_timesteps` uzunluğudur. Mevcut N8
2/4/6/8 ve Background35 9/18/27/35 adayları aynı sırayla saklanır.
Snapshot SHA'ları mevcut observer decoded-geometry SHA'larıyla eşleşmelidir;
terminal de mevcut decoder tensoruyla bağlanır. Şekil/hash uyuşmazlığı durdurur.

[Code] Baseline terminal observer/classification ve mevcut elapsed/peak ölçümü
tamamlandıktan sonra, bir kez `build_probe_spectral_target(prepared, SpectralConfig())`
çalışır. HazırlanmışX kopyalanır; sampling öncesinde graph kurulmaz.
Eigensistem native GPU'da hangi aktif destek/basis/q/rank üretildiyse onu saklar.
Basis aktif altgrafın tam eigensistemidir; q mean-active-degree ile ölçeklenip
sıfırdan aşağısı clamp edilmiş Laplacian eigenvalue'larıdır; H5 reliability
ağırlıkları değildir. Rank, mevcut diagnostic hard-boundary seçimidir;
kayıt aşamasında hard/smooth filtre veya input güncellemesi uygulanmaz.

[Code] Probe/RNG izolasyonu mevcut `isolated_probe([])` ile yapılır; global RNG
öncesi/sonrası SHA'ları eşleşir. Helper GPU tensorunu caller'a döndürmez;
eigensistem sonraki baseline batch'e tutulmaz. Geometriler graph kurulumundan
önce yazılır; eigensolve hatasında dört snapshot'ın ham kanıtı korunur.
Yeni dosyalar mevcut batch Drive kopyasına ve son sealed ZIP'e otomatik girer.

[Code] Üç clean CPU-copy süresi baseline reconstruction ölçümünün **içindedir**
ve `clean_snapshot_copy_seconds_within_baseline` olarak ayrıca kaydedilir.
Terminal copy, graph build ve archive serialization baseline elapsed/peak
snapshot'ından sonradır; ayrı süreler raporlanır. Posttrajectory graph peak'i
toplam allocated GPU belleğidir; baseline'dan fark veya deployment peak'i değildir.
Worker wall bu ek tanı maliyetlerini içerir; ölçüm maliyeti sıfır sayılmaz.
Her B2 sonrasındaki eigensolve sonraki B2 case'lerin allocator/solver geçmişini
etkileyebilir; RNG/model-state koruması whole-process sayısal etkisizliği kanıtlamaz.
Repeat kıyasında her exactX için ilk repeat'in arşivlenmiş GPU basis'i üç
geometriye ortak uygulanmalıdır; diğer repeat basis'leri de korunur, fakat
farklı basis'lerden doğrudan özellik kıyası yapılmaz.

[CPU] Synthetic küçük gerçek probe grafıyla dört snapshot'ın şekli, değişmez
CPU kopyası, NPZ yeniden okuması, bütün spectrum tensor hash'ları, tek builder
çağrısı ve RNG korunması kontrol edilir. Simüle probe hatası geometry NPZ'yi
korur. Native GPU inference bu implementasyon batch'inde çalıştırılmadı.

[Open] Planlanan8 eski mechanism ID/four-corruption/üç fresh repeat kapsamı,
repeat başına16 B2 case üretir. Temiz veri yalnız sonradan CPU set-oracle'ıdır;
feature/weight/checkpoint seçimine veya guidance'a verilmez.
[Open] Bilinen Background tekrar kapısı başarısızlığı/native floor korunmalıdır:
[önceki accuracy kaydı](decoded_physical_accuracy_20261104.md).
Yeni capture flag'i sayısal tekrar edilebilirlik garantisi vermez.
