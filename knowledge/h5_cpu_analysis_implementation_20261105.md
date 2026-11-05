# H5 CPU analiz çekirdeği — dondurulmuş sözleşme

Tarih: 2026-11-05. Dal: `research/gsd-h5-reliability-20261105`.
Etiketler: **[Code] / [CPU] / [Open]**. Gerçek H5 araştırma verisi bu
implementasyon batch'inde üretilmedi/analiz edilmedi; aşağıdaki kurallar
oracle sonuçları görülmeden kaydedildi. [Ana plan](h5_reliability_study_20261105.md).

[Code] Giriş noktası `scripts/analyze_h5_reliability.py`:

```bash
python scripts/analyze_h5_reliability.py \
  --bundles REPEAT1.zip REPEAT2.zip REPEAT3.zip \
  --clean-subset oracle_clean.npz --clean-receipt oracle_clean_receipt.json \
  --output-dir FRESH_OUTPUT_DIRECTORY
```

[Code] ZIP'ler gerçekleşen run sırasıyla verilir; ilk repeat referans bazdır.
Mevcut accuracy analyzer `load_run` kontrol/weight0, model/state, kaynak/native,
backend, sayım ve prepared/cache/RNG eşitliğini doğrular. H5 flag'i,8 eski ID,
dört bozulma, bütün snapshot/basis/q/rank dosya/tensor hash'ları ayrıca denetlenir.
Scalar rank'ın `shape()` hash'i aynen korunur; `(1,)` biçimine yükseltilmez.
Her exactX için ilk repeat'in gerçek GPU eigensistemi üç repeat'e ortak
uygulanır. Diğer bazlar yalnız q/support/rank ve aynı X/Y tanıklarındaki
projector farkıyla kıyaslanır; eşleşmemiş coefficient karşılaştırması yapılmaz.

[Code] q yalnız normalize/clamp edilmiş Laplacian eigenvalue'larıdır;
H5 reliability ağırlığı değildir. Bant sınırları `[0,.25,.5,1,2,4,inf]`.
Numerik tekrar blokları, ilk eigenvalue'a sabitlenmiş
`effective_boundary_atol/mean_active_degree + eigenspace_rtol*|lambda_anchor|`
toleransıyla korunur; komşudan komşuya zincirlenmez. Blok alt eigenvalue'sunun
nominal bandına bütünüyle atanır. Gerçek sınırlar, mode indeksleri ve boş
bantlar yazılır. İzole vertex'ler projector dışında kalır; M100 uygulanmaz.

[Code] Online-uyumlu r/v yalnız mevcut Y, sabitX ve bir önceki gözlenen PM
checkpoint'ini okur. İlk checkpoint'te v ve s **UNAVAILABLE** olur.
`r=||P(Y-X)||²/(||PY||²+||PX||²+delta_r)`;
`v=||P(Y-Yprev)||²/(||PY||²+||PYprev||²+delta_v)`;
`s=r/(1+v)`. Burada `delta_r=1e-12*(1+||Y||²+||X||²)`;
delta_v aynı formda X yerine Yprev kullanır. Payda bu eşiği aşmıyorsa
feature unavailable'dır; keyfi ölçekle düzeltme yapılmaz.
Güvenilirlik skorları önceden sabit: frequency `-mean(lambda_band)`, residual
`-r`, combined `-s`. Fit, yeni skor taraması, applied q veya accuracy seçimi yoktur.

[Code] Bütün features.json satırları temiz veri değerleri okunmadan önce
yazılıp hash'lenir. Native feature variation tanısı sonradan hesaplanır ve
feature girdisi yapılmaz. Temiz NPZ/receipt mevcut kabul edilmiş oracle'ın
dataset ve subset pin'leriyle doğrulanır; yalnız CPU set-oracle'ıdır.

[Code] Oracle g=`2P(Y-X)/(3N)`, d=`-g/||g||`, epsilon=`.001||Y||`.
`||P(Y-X)||<=1e-6*(1+||Y||)` yönü unavailable yapar. Float32 GPU U değerleri
float64'te değerlendirilir; P=U_b U_bᵀ yeniden graph kurulmadan uygulanır.
Quadratic değer EᵀPE/(3N), exact finite-step değişimi
`-epsilon||g||+epsilon² dᵀPd/(3N)` ile denetlenir. Symmetry/joint permutation
kontrolleri vardır. Fark toleransı `1e-12*(1+|before|+|after|+|predicted|)`;
symmetry tanığı `1e-10*(1+||X||||Y||)` sınırını kullanır.

[Code] Sabit epsilon pozitif/zero quadratic değişim verirse **OVERSHOOT_UNAVAILABLE**
olarak, temiz mesafe hesaplanmadan kaydedilir. Epsilon küçültülmez, temiz
sonucun işaretine göre uygunluk seçilmez; bütün feature/satırlar tutulur.
Temiz geometri bozuk girdinin kaydedilmiş center/scale frame'inde
`((C-center)/scale)*3.3885`, ardından `(-z,y,x)` ile dönüştürülür; registration
ve classifier renormalization yoktur. İki yönlü squared mean-NN Chamfer,
eşit vertex sayısı/clean slot correspondence varsaymaz. Clean-CD değişimi
`1e-12*(1+baselineCD)` içinde kalırsa **UNRESOLVED_TIE** olur.

[Code] Frequency/residual/combined ana Spearman/AUROC karşılaştırması aynı
resolved-oracle ve üç sonlu skoru olan **ortak satır maskesini/sırasını** kullanır.
Bu nedenle ilk checkpoint üçlü karşılaştırmaya girmez. Ortak sıra hash'i ve
aynı n denetlenir. Feature başına tam coverage istatistikleri ayrıca
descriptive diye etiketlenir; farklı zaman coverage'ları ana kontrasta karışmaz.
Tanımsız korelasyon/tek-sınıf AUROC boş bırakılır; threshold/score seçilmez.

[Code] Beklenen2304 satır:3repeat×8ID×4corruption×4checkpoint×6band.
Boş/tiny/overshoot/tie durumları kaybolmaz. Çıktılar inputs.json, bands.json,
features.json, other_basis_comparisons.json, native_repeat_gates.json,
feature_variation.json, rows.jsonl ve summary.json'dur. Başarısızlıkta
failure.json ve yazılmış kısmi kanıt korunur; mevcut output dizini kullanılmaz.

[Code/Open] Üç repeat çiftinin mevcut logit/ilk-decode sayısal kapıları aynen
raporlanır. Known Background/native floor başarısızlığı giderilmiş sayılmaz.
Feature repeat-range ile adjacent temporal değişim ayrı tutulur; temporal
medyan0 ise oran tanımsız, native-dominance karşılaştırması yine görünürdür.
Sekiz original-object cluster vardır; repeat/checkpoint/band/bozulma satırları
bağımsız örnek değildir. Bu çekirdek cluster CI üretmez; ham satırları saklar.

[CPU] Future perturbation, whole-block rotation invariance, spectral-only
overshoot, brute NN, scalar hash, serialized capture/hash tampering, ortak
evaluation maskesi ve başarısız fresh-output kayıtları synthetic testlerle
kontrol edilir. Gerçek native H5 kabulü ve oracle sonucu henüz açık kalır.
