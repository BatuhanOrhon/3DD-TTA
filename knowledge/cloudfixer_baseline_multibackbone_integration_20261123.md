# CF-XYZ N8 üç-source-model sonucunun seçili entegrasyonu

2026-11-23 [Code/Run/Open] Bu kayıt hazırlık aşamasındadır; yayın/kapatma henüz tamamlanmadı.

Çalışma worktree'si `/Users/batuhan/.codex/worktrees/0e7f/3DD-TTA`, branch `research/cloudfixer-transfer-20261114`; bilimsel sonuç kaynak HEAD'i `0d36c91`. GPU'da gerçekten çalıştırılan sabit kaynak `a979bcec11c24963ff31f0574d66decb85d680d3`, kampanya `e8df74eb3976517f74452c49363df953ac5d103e4ff0cc6fb265d327769b0427`.

İzole entegrasyon checkout'u `/private/tmp/cloudfixer-baseline-heads-integration-20261123`, branch `integration/cloudfixer-baseline-heads-20261123`. Ortak hedef en son `origin/thesis-integration@6f4abdd`; bu hedefin PCA, selective-rotation, background ve diğer kabul edilmiş kayıtları korunuyor. İlk code hazırlığı `6ae0338`, knowledge-only target ilerlemesinden sonra temiz rebase code commit'i `8ade93c` oldu; henüz remote push yok.

## Seçilen kapsam

- `scripts/run_cloudfixer_multibackbone.py`: isteğe bağlı CF-XYZ N8 baseline profili; aynı trajectory PM/terminal, üç donmuş head, ham ve recovery doğrulaması.
- `scripts/prepare_cloudfixer_multibackbone.py`: yeni profile/source/asset manifest'i ve yalnız baseline için B64/B96/B128 kayıtları.
- `scripts/analyze_cloudfixer_multibackbone.py`: bağımsız ham probability/terminal sayımı, all15 ve ID/etiket eşleşmeli CF karşılaştırması.
- `scripts/setup_cloudfixer_multibackbone_colab.py`: seçilen Drive namespace/profile desteği.
- `tests/test_cloudfixer_multibackbone.py`: yeni profil, bozuk payload/manifest/RNG/cursor, kampanya izolasyonu, dynamic batch ve eski CF uyumluluk kontrolleri.
- Bu görevdeki operasyon/sonuç notları, plan, son durum/gate'ler, gerekli kaynak/batch/ham sayıma ait seçilmiş kompakt kanıtlar. README/open_questions başına yeni güncel durum eklenir; findings_log'a sadece bu görevin yeni tarihçesi eklenir.

## Korunan ve dışarıda tutulan kapsam

Paylaşılan `cloudfixer_diffusion.py` dosyası daha yeni peer pose/ölçüm callback'lerini içerir ve eski tam dosya ile değiştirilmedi. Deney kaynak sampler'ı ile bu ortak dosyanın default yolunda sekiz synthetic CPU durumda cloud/metadata/RNG birebir eşit; gerçek CUDA/checkpoint replay iddiası değildir. Diğer peer kaynak ve notları korunur. Tam source branch tarihçesi körlemesine merge edilmez.

Ağırlıklar, ham ZIP/NPZ'ler, cache'ler, geçici CLI/setup/taşıma betikleri, superseded allocation ve hiç GPU'da kullanılmamış alternatif paket receipt'leri shared branch'e taşınmaz. Bunlar kaynak branch'te ve belirtilen yerel/Drive artifact alanında korunur. Başarısız B128 Upsampling OOM ve kesilen tek-parça final upload, negatif/operasyonel kanıt olarak korunur.

## Doğrulama ve son kapılar

Kaynak implementasyonunda 272 ilgili ve 1110 genel CPU kontrolü geçti (2 skip); yeni source batch kaydı bağımsız review PASS. Entegrasyon checkout'unda 159 ilgili ve 15 ortak rotation CPU testi geçti. Hedef24be2e9→6f4abdd ilerlemesi sadece knowledge dosyalarındaydı; test edilen kod değişmedi. Yeni sampler default-path synthetic karşılaştırmasının8/8 durumu geçti. Deneyin actual native/smoke/golden/B96 kapasite ve tam390blok/15ZIP sonuçları ayrıca GPU artifact'lerinde doğrulandı. Entegre ortak branch için ek bir GPU yeniden koşusu yapılmadı.

Son raw/source bundle `final_20261123T112717Z.zip`,406927870B,SHA `c392dce804c213a21e97769590e3a036ee80deefb9b0d6379689958530646265`; host'ta ikinci strict analizden geçti. Tek-parça CLI upload500/remote hedef yok;49 adet8MiB parça ile kurtarma sürüyor. Bu nihai bundle'ın Drive→freshhost doğrulaması, bütün-branch son review, normal fast-forward push, official cloudflush ve yalnız ownedr25 kapatma tamamlanmadan bu kayıt complete olmaz. Kullanıcı bu kapsamda yeniden onay sorulmadan yayın ve tamamlandığında kapatmayı açıkça yetkilendirdi.

Ayrıntılı bilimsel sonuçlar: [tam karşılaştırma](cloudfixer_baseline_multibackbone_results_20261123.md). Yürütme/kimlik/kurtarma: [operasyon kaydı](cloudfixer_baseline_multibackbone_20261123.md).
