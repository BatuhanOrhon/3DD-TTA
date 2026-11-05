# Yeni agenta handoff — PM adımlarında düşük bant SCD/GSD karışımı

Bu metni yeni agenta doğrudan görev olarak ver. Kullanıcı 5 Kasım'da bu ikinci
fikrin ayrı branch'te hazırlanmasını istedi. H5 güvenilirlik çalışması başka bir
agenta/runtime'a aittir; bu görev düşük bant ve yarı yarıya karışımdır.

## Görev ve sahiplik

`research/gsd-lowband-halfmix-handoff-20261105` dalını başlangıç al. Bu dal,
`origin/thesis-integration`9e18bc7'den, mevcut
`/Users/batuhan/Desktop/projects/3DD-TTA-worktrees/gsd-guidance-math-audit-20261103`
worktree'sinde hazırlandı. Başka agent bu worktree'de H5 çalışacağı için onun
branch'ini değiştirme ve dosyalarına yazma. Önce kendine ayrı worktree ve bu
handoff dalından türeyen bir çalışma branch'i oluştur. Handoff'un yerel branch'i
repo ortak Git object store'unda erişilebilir; yayımlanmış olduğunu varsayma.

Kendi Colab runtime/monitor/Drive alanını kullan:
`tta-gsd-lowband-halfmix-a100-20261105-r1` önerilen addır. H5'in
`tta-gsd-h5-reliability-a100-20261105-r1` oturumu ve görünür diğer runtime'lar
korunur. Önceki `tta-gsd-physical-spectral-a100-20261103-r1` 4 Kasım'da kullanıcı
isteğiyle kapandı; onu yeniden kullanılamaz. Sonlandırma izni yeni runtime'a
taşınmaz. Serviste kota sorunu varsa peer runtime kapatma veya GPU türünü izinsiz
değiştirme.

## Önce okunacak kayıtlar

AGENTS.md ve RTK.md; `.agents/skills/iterative-thesis-researcher/SKILL.md`,
`.codex/skills/colab-research-operator/SKILL.md`, `knowledge/README.md` tamamı;
ardından `thesis_scope.md`, `experiment_protocol.md`, `current_baseline.md`,
`reproduction_gap.md`, `gsd_guidance_math_efficiency_20261103.md`,
`decoded_physical_spectral_results_20261103.md`,
`decoded_physical_followup_20261104.md`,
`decoded_physical_accuracy_20261104.md`, `gsd_zero_repeat_20261104.md` ve ilgili
compact evidence. Tarihli planları sonuçlarla uzlaştır; eski açık-runtime
ifadelerini güncel durum sanma. H5 çalışmasının yeni notlarını da kontrol et.

Mevcut kanıt: 32ID/all15/seed0 pilotunda PM baseline313/480, live0 315/480,
full fiziksel313/480, H8 312/480. Fiziksel hedef smooth beta2/tüm aktif modlardı;
`modes=100` burada 100-mod kesimi değildi. H8 yön kontrolleri96/96 geçti, fakat
accuracy kazancı oluşmadı. Filtreler240/240 batch'te aynıydı; sıfır-fiziksel-loss
kontrolü45/240batch'te sayısal sınırı geçemedi. Sonraki fresh-process testinde
Background495/512 için aynı SCD kontrolünün kendi tekrarları0/3, aynı live0
tekrarları0/3; Gaussian44/91 için3/3 geçti. Native/SCD/prior kararsızlığı
çözülmedi. Bu iki koşuldaki PM sınıfları değişmemişti. Bunlar GPU/counterexample
kanıtlarıdır; tüm yöntemlerin zararlı olduğunun kanıtı değildir.

## Test edilecek hipotez

Encoder'a gerçekten giren hazırlanmış X üzerinde sabit ortak grafik kur.
Decoder'ın PM clean checkpoint çıktısı Y için gerçek düşük-bant sadakatini
hesapla: `L_G,M = ||U_M.T @ (Y-X)||_F^2/(3N)`, nesneler üzerinde sum. X ve Y
aynı Bx2048x3 fiziksel frame ve vertex düzeninde olmalı; ham1024 input veya
classifier/FPS frame'i kullanılmaz. Her iki sinyal aynı U'ya projekte edilir.

M adayları100 ve200. Hard cutoff kullan; heat beta2/tüm-mod profiline geri
dönme. Tekrarlı/ayırt edilemeyen özdeğer bloğunu yarıda kesme. İstenen ve gerçek
rank, sıfır mod sayısı, aktif destek ve eigengap kaydedilir. M100/200 aynı bloğa
genişleyip aynı filtreyi oluşturursa bunları iki bağımsız yöntem sanma; alias
kanıtını arşivle. İzole vertex semantiğini mevcut sabit operator ile açıkça tut.

PM için mevcut decoder forward'ını tekrar kullan. Normal clean2/4/6,
Background9/18/27'de loss guidance'a girer; diğer sampling adımlarında özgün
SCD devam eder. Terminal8/35'te yeni final-update ekleme. Decoder original z0,
DDIM100/final-alpha1, gamma.03/eta.003/p.95, EMAoff/eval/raw ve frozen PointMAE
korunur. İlk uygulama full decoder türevli olsun; H8 ayrı sonraki ablation'dır.
Classifier'a gradient veya test-time clean/label/reference-class erişimi yok.

## Norm dengesi ve kontrol koşulları

Ölçek loss değerlerinden türetilmez. Denoiser sonrası uygulanan per-object
local/style gradientlerini gerçek gamma/eta ile ölçekleyip birleştir:
`d_C=(gamma*g_C,h, eta*g_C,s)`, `d_G` aynı biçimde. Kalibrasyon katsayısı
`alpha_M = median(||d_C||/||d_G||)`; clean veya etiket kullanılmaz. Her M için
aynı önceden seçilmiş küçük kalibrasyon nesnelerinde ölç, sonra dondur.
Bu, kalibrasyonda tipik normları eşler; her sonraki adımda exact eşitlik değildir.
Sıfır/çok küçük/finite olmayan normu sessizce büyütme; önceden tanımlı failure
veya inactive durumunu kaydet. Accuracy'den katsayı veya cutoff seçme.

PM checkpointlerinde `L_mix=.5*L_C+.5*alpha_M*L_G,M`; alpha detached/sabit.
Tek joint backward kullan. Diğer sampling adımlarında `L_C` katsayısı1 kalır.
Online exact norm eşleme ayrı VJP maliyeti doğurur; kullanıcı özellikle isterse
ayrı varyant olarak ölç. Eski fiziksel pilot rho.1 iken bu karışımın kalibre
göreli spektral normu1'dir; eski lambda802.5'i buraya kopyalama.

En az beş mantıksal kol:
1. Operasyonel tam-SCD+PM baseline.
2. Aynı live decoder yoluyla tam-SCD, spektral ağırlık0 (eşlenmiş ana baseline).
3. Aynı live decoder yoluyla yalnız PM adımlarında0.5*SCD, spektral ağırlık0.
4. M100 için0.5*SCD+0.5*alpha100*GSD.
5. M200 için0.5*SCD+0.5*alpha200*GSD.

Üçüncü kol, SCD azaltma etkisini GSD ekleme etkisinden ayırır; ikinci kol
observer değişiminin SCD ölçeğiyle karışmasını önler. Teknik aşamada
tam-SCD operational/live0 eşdeğerliğini ve aynı observer fresh-process
tekrarlarını ayrıca denetle. Norm eşitliği yön eşitliği değildir: dC/dG cosine,
karışım normu ve iptal oranını kalibrasyon/teknik aşamada kaydet. Bileşen
gradientlerini her ölçülen adımda tekrar hesaplamak ek VJP gerektirir; bu
probları deployment zamanlamasından ayır veya maliyetini açıkça say. Eşit norm için karışım normu
`||d_C||*sqrt((1+cos(theta))/2)` olur; zıt yönler birbirini söndürebilir.

## Sıralı uygulama ve kabul kapıları

A. Kod/CPU: `decoded_spectral.py`, `graph_spectral.py`, `tta.py`,
`scripts/diagnose_decoded_spectral.py` ve `shared_trajectory.py` yollarını izle.
Opt-in küçük değişiklik yap; varsayılanları değiştirme. Hard loss/gradient,
tekrarlı eigenspace, frame/shape, checkpoint seçimi, half-SCD'nin yalnız PM'de
etkinleşmesi ve sabit katsayıya yönelik anlamlı testler ekle. Kaynak snapshot'ı
ve dirty dosya hash'lerini dondur; yeni native/source sürümünü gizleme.

B. Colab: yalnız CLI; yeni VM'de `scripts/colab_environment_setup.ipynb` on kod
hücresi, cell1 ayrı restart/reconnect, sonra2–10 sırayla. Yalnız bilinen cell2
fastrlock hatası ertelenebilir. Native/CUDA/dataset/checkpoint hatasında dur.
Kendi monitor'ün tek cell sahibi olsun; Drive consent gerekiyorsa kullanıcı
tamamlar. Setup ve araştırma paketlerini ayır; tam notebook Run all çalıştırma.

C. Teknik: aynı cachedX/z0/h0/noise/RNG ile grad/FD/projection, model-state
korunumu ve fresh-process zero kontrolleri. Geçmiş sayısal kapıları gevşetme:
logitmax.05, per-object/checkpointRMS.01, firstdecodedRMS1e-4. Başarısız
tekrarlanabilirlikte accuracy sayımlarını betimsel raporla; küçük farkı guidance
etkisi diye sunma. Başka agent'ın native teşhis sonuçlarını uzlaştır.

D. Kalibrasyon ve pilot: önce somut IDs/seed/config/output ve bütün GPU fazı
süresini kaydet. Öneri: önceki kohortları dışlayan16 yeni geliştime nesnesi,
all15/s5/seed0/B2, beş kol; ufak etiketsiz kalibrasyonla toplam30–45min,
hard55min. Gerçek tahminle bütçeyi doğrula; tüm plan >60min olacaksa açık onay
al, işi kısa CLI çağrılarına bölerek bunu aşma. Aynı encoded/noise cache'lerini
ölçümden önce oluştur; bütün ölçülen kollar replay yapsın. M100/M200 arasında
aynı cache ve hazırlama RNG doğrulansın. Teknik başarısızlıkları koru.

E. Bağımsız analiz: PM ana readout, terminal ikincil. Her corruption ve all15
doğru/n/accuracy, operasyonel baseline ve half-SCD kontrolüne paired fark,
orijinal nesne-cluster bootstrapCI; bozulmaları bağımsız nesne sayma. Tek seed
pilotunu doğrulanmış genel iyileşme diye sunma. Fresh tüm pipeline maliyeti,
ortak hazırlık, kNN/eigh, loss, ek VJP, decoder/PM, terminal head, IO/workerwall
ve peak ölçüm kapsamı ayrı kaydedilsin. Forward reuse, backward maliyetini
ortadan kaldırmaz; yalnız U_M saklamak full eigh setup'ını hızlandırmaz.

## Arşiv ve teslim

Yeni result/Drive dizinleri, yedi standart dosya, prediction NPZ, tüm başlangıç
cache'leri, graph/rank, scalar/update norm/cosine, source/native byte payloadları,
başarılı/partial/failed ham ZIP'ler ve SHA doğrulanmış Drive/local kopyalar.
Eski kabul edilmiş koşuları yeni kanıt yerine tekrar çalıştırma.
Knowledge/findings/open_questions ve ilgili yöntem notlarını güncelle; [Code],
[Run], [Inference], [Open] ayrımını koru. Gerekli tamamlanan kod/compact kanıtı
temiz entegrasyon checkout'undan yalnız origin/thesis-integration'a normal push
ile yayımla; peer/user dosyalarını ve runtime'larını koru. Sonuç negatif/null
olabilir. GPU hız kazancı, accuracy, bit eşitliği ve gradyan yaklaşımını ayrı
sonuçlar olarak raporla. Runtime ancak kullanıcı kapatılmasını istediğinde kapanır.
