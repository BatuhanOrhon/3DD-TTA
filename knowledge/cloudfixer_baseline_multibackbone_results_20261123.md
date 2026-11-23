# CloudFixer XYZ N8: DGCNN, CurveNet ve PointNeXt tam test sonuçları

2026-11-23 — [Run/Validated] Tam 2.468 özgün nesne × 15 corruption, severity5, seed0. Her head için 37.020 corruption gözlemi; bu 37.020 ayrı özgün nesne değildir. 390 B96 blok, 135 accuracy ve 135 nesne-eşleşmeli karşılaştırma satırı bağımsız ham sayımdan doğrulandı.

## Sabit yöntem ve karşılaştırma

Resmî ModelNet40 CloudFixer XYZ diffusion prior kullanılır; LION kullanılmaz. Kendi yöntemimiz N8/K80/tau=.1/retained=.95 gradient-free proximal düzeltmedir. PM, düzeltme öncesi t20 ve t10 ile ek denoiser çağrısından gelen gerçek t0 softmax olasılıklarının eşit float64 ortalamasıdır. Toplam 9 prior çağrısı vardır. PM kapalı sütunu aynı N8 yörüngesinin t0 çıktısıdır; N32 değildir. Tek geometri üç donmuş head tarafından değerlendirilir; classifier ağırlıkları/adaptasyonu yoktur.

Orijinal CF, önceden tamamlanan plain CF/vote1 rotation/delta optimizasyonudur. Aynı veri, etiket/sınıf sırası, prior ve donmuş head-state kimlikleri ham kayıttan doğrulandı. CF B64/native sürekli RNG, yeni yöntem B96/ID-blok-adresli fork RNG kullanır. CF karşılaştırması nesne-ID/etiket eşleşmeli gözlenen pipeline kıyasıdır; ortak noise deneyi değildir. PM–terminal ise aynı hesaplanan geometri, noise ve terminal logitlerini paylaşır.

## All15 accuracy

| Source model | Source-only % | N8 terminal % | N8 PM % | Orijinal CF % | PM−terminal pp | PM−CF pp |
| --- | --- | --- | --- | --- | --- | --- |
| dgcnn | 58.835764 | 70.680713 | 71.139924 | 71.277688 | +0.459211 | -0.137763 |
| curvenet | 64.011345 | 73.841167 | 74.243652 | 75.931929 | +0.402485 | -1.688277 |
| pointnext | 65.837385 | 73.987034 | 74.278768 | 75.018909 | +0.291734 | -0.740140 |

Bütün hücrelerde 2.468 nesne bulunduğundan all15 macro ve micro aynıdır. Yüzdeler altı ondalığa yuvarlanır; aşağıdaki tamsayılar asıl sayımdır.

| Source model | Source doğru/37020 | Terminal doğru | PM doğru | CF doğru |
| --- | --- | --- | --- | --- |
| dgcnn | 21781 | 26166 | 26336 | 26387 |
| curvenet | 23697 | 27336 | 27485 | 28110 |
| pointnext | 24373 | 27390 | 27498 | 27772 |

## Eşleşmeli doğru/yanlış değişimleri

“Yalnız yöntemimiz doğru” ve “yalnız referans doğru” aynı corruption, özgün ID ve etiket üzerinde sayılır. Net = yalnız yöntemimiz − yalnız referans. PM–terminal satırında referans kendi t0 terminalimizdir.

| Model | Karşılaştırma | Yalnız yöntemimiz | Yalnız referans | Net doğru | Fark pp | Corruption kazan/eşit/kaybet |
| --- | --- | --- | --- | --- | --- | --- |
| dgcnn | proximal_n8_pm_vs_proximal_n8_terminal | 618 | 448 | 170 | +0.459211 | 12/1/2 |
| dgcnn | proximal_n8_pm_vs_cloudfixer_plain | 2073 | 2124 | -51 | -0.137763 | 10/0/5 |
| dgcnn | proximal_n8_terminal_vs_cloudfixer_plain | 2056 | 2277 | -221 | -0.596975 | 9/0/6 |
| curvenet | proximal_n8_pm_vs_proximal_n8_terminal | 523 | 374 | 149 | +0.402485 | 13/1/1 |
| curvenet | proximal_n8_pm_vs_cloudfixer_plain | 1525 | 2150 | -625 | -1.688277 | 5/0/10 |
| curvenet | proximal_n8_terminal_vs_cloudfixer_plain | 1518 | 2292 | -774 | -2.090762 | 5/0/10 |
| pointnext | proximal_n8_pm_vs_proximal_n8_terminal | 520 | 412 | 108 | +0.291734 | 12/0/3 |
| pointnext | proximal_n8_pm_vs_cloudfixer_plain | 1901 | 2175 | -274 | -0.740140 | 9/0/6 |
| pointnext | proximal_n8_terminal_vs_cloudfixer_plain | 1894 | 2276 | -382 | -1.031875 | 9/0/6 |

## Corruption bazında bütün sonuçlar

### dgcnn

| Corruption | Source % | Terminal % | PM % | CF % | PM−terminal pp | PM−CF pp |
| --- | --- | --- | --- | --- | --- | --- |
| uniform | 68.192869 | 88.006483 | 88.654781 | 88.168558 | +0.648298 | +0.486224 |
| gaussian | 75.000000 | 87.884927 | 89.059968 | 87.763371 | +1.175041 | +1.296596 |
| background | 13.614263 | 41.774716 | 43.192869 | 55.348460 | +1.418152 | -12.155592 |
| impulse | 55.024311 | 88.897893 | 88.897893 | 87.925446 | +0.000000 | +0.972447 |
| upsampling | 74.716370 | 89.100486 | 89.546191 | 88.452188 | +0.445705 | +1.094003 |
| distortion_rbf | 74.594814 | 80.956240 | 81.442464 | 82.009724 | +0.486224 | -0.567261 |
| distortion_rbf_inv | 75.202593 | 83.468395 | 83.792545 | 83.508914 | +0.324149 | +0.283630 |
| density | 51.620746 | 55.348460 | 56.239870 | 50.526742 | +0.891410 | +5.713128 |
| density_inc | 82.212318 | 85.291734 | 85.777958 | 83.995138 | +0.486224 | +1.782820 |
| shear | 76.782820 | 77.998379 | 77.674230 | 77.714749 | -0.324149 | -0.040519 |
| rotation | 57.252836 | 57.617504 | 58.144246 | 75.972447 | +0.526742 | -17.828201 |
| cutout | 57.617504 | 70.178282 | 70.461912 | 59.278768 | +0.283630 | +11.183144 |
| distortion | 76.458671 | 79.051864 | 79.011345 | 80.632091 | -0.040519 | -1.620746 |
| occlusion | 32.252836 | 39.303079 | 39.748784 | 36.264182 | +0.445705 | +3.484603 |
| lidar | 11.993517 | 35.332253 | 35.453809 | 31.604538 | +0.121556 | +3.849271 |

### curvenet

| Corruption | Source % | Terminal % | PM % | CF % | PM−terminal pp | PM−CF pp |
| --- | --- | --- | --- | --- | --- | --- |
| uniform | 67.301459 | 87.803890 | 88.209076 | 88.452188 | +0.405186 | -0.243112 |
| gaussian | 77.066451 | 88.249595 | 88.371151 | 88.492707 | +0.121556 | -0.121556 |
| background | 7.576985 | 29.700162 | 31.037277 | 49.918963 | +1.337115 | -18.881686 |
| impulse | 47.649919 | 88.330632 | 88.654781 | 88.209076 | +0.324149 | +0.445705 |
| upsampling | 70.097245 | 88.654781 | 88.897893 | 89.303079 | +0.243112 | -0.405186 |
| distortion_rbf | 78.606159 | 83.225284 | 83.752026 | 84.805511 | +0.526742 | -1.053485 |
| distortion_rbf_inv | 80.510535 | 84.846029 | 84.805511 | 86.466775 | -0.040519 | -1.661264 |
| density | 79.132901 | 77.269044 | 77.795786 | 74.027553 | +0.526742 | +3.768233 |
| density_inc | 88.168558 | 85.656402 | 85.656402 | 86.345219 | +0.000000 | -0.688817 |
| shear | 77.025932 | 80.307942 | 80.348460 | 81.401945 | +0.040519 | -1.053485 |
| rotation | 68.800648 | 70.583468 | 72.001621 | 83.306321 | +1.418152 | -11.304700 |
| cutout | 78.565640 | 81.807131 | 81.928687 | 79.821718 | +0.121556 | +2.106969 |
| distortion | 77.633712 | 81.442464 | 81.888169 | 83.022690 | +0.445705 | -1.134522 |
| occlusion | 35.615883 | 40.235008 | 40.478120 | 37.439222 | +0.243112 | +3.038898 |
| lidar | 26.418152 | 39.505673 | 39.829822 | 37.965964 | +0.324149 | +1.863857 |

### pointnext

| Corruption | Source % | Terminal % | PM % | CF % | PM−terminal pp | PM−CF pp |
| --- | --- | --- | --- | --- | --- | --- |
| uniform | 45.259319 | 88.168558 | 88.938412 | 87.965964 | +0.769854 | +0.972447 |
| gaussian | 54.943274 | 88.087520 | 89.303079 | 87.844408 | +1.215559 | +1.458671 |
| background | 85.251216 | 63.938412 | 65.032415 | 75.607780 | +1.094003 | -10.575365 |
| impulse | 72.366288 | 90.194489 | 90.316045 | 89.019449 | +0.121556 | +1.296596 |
| upsampling | 71.312804 | 89.384117 | 89.505673 | 88.290113 | +0.121556 | +1.215559 |
| distortion_rbf | 70.988655 | 79.497569 | 79.781199 | 79.983793 | +0.283630 | -0.202593 |
| distortion_rbf_inv | 74.351702 | 81.564019 | 81.645057 | 81.685575 | +0.081037 | -0.040519 |
| density | 85.615883 | 80.388979 | 80.794165 | 77.431118 | +0.405186 | +3.363047 |
| density_inc | 91.207455 | 88.209076 | 87.641815 | 85.494327 | -0.567261 | +2.147488 |
| shear | 78.808752 | 74.311183 | 74.230146 | 75.567261 | -0.081037 | -1.337115 |
| rotation | 39.059968 | 43.638574 | 44.205835 | 63.006483 | +0.567261 | -18.800648 |
| cutout | 85.737439 | 84.683955 | 84.967585 | 81.807131 | +0.283630 | +3.160454 |
| distortion | 73.946515 | 76.742301 | 76.904376 | 77.147488 | +0.162075 | -0.243112 |
| occlusion | 39.465154 | 41.045381 | 40.640194 | 38.573744 | -0.405186 | +2.066451 |
| lidar | 19.246353 | 39.951378 | 40.275527 | 35.858995 | +0.324149 | +4.416532 |

## Background ve Rotation açıkları

| Model | CF−PM toplam net doğru | Background+Rotation net açık | Diğer 13 net açık |
| --- | --- | --- | --- |
| dgcnn | 51 | 740 | -689 |
| curvenet | 625 | 745 | -120 |
| pointnext | 274 | 725 | -451 |

Pozitif açık CF üstünlüğü, negatif açık PM üstünlüğüdür. Bu ayrıştırma aritmetiktir: iki corruption üzerinde eşitlemenin kalan bütün sonuçları koruyacağına dair nedensel garanti vermez.

## Sınırlar ve kanıt

Tek seed ile gözlenen fark, PM’in her nesnede/corruption’da/seed’de veya her source modelde kesin üstünlüğünü kanıtlamaz. Geç PM penceresi önceki Point-MAE geliştirme analizinde keşifsel seçildi; bağımsız doğrulanmış optimum değildir. Bu tam testte pencere, tau, N veya yöntem test etiketlerine göre yeniden seçilmedi. Ayrı terminal-only uçtan uca gecikme ölçülmedi: saklanan trace bütün PM probe classifier çağrılarını içerir. Eski Point2Vec B1/seed2 ve Point-MAE 128 nesnelik geliştirme sonuçları bu üç head’in tam test ortalamasına karıştırılmaz.

- Çalıştırılan kaynak: `a979bcec11c24963ff31f0574d66decb85d680d3`.
- Kampanya: `e8df74eb3976517f74452c49363df953ac5d103e4ff0cc6fb265d327769b0427`.
- Bağımsız ham sayım: son arşivdeki `recount.json`.
- Son ham/source arşivi: `result/modelnet40_c/cloudfixer_baseline_heads/e8df74eb3976/final_20261123T112717Z.zip`, 406927870 B, SHA-256 `c392dce804c213a21e97769590e3a036ee80deefb9b0d6379689958530646265`.
- Drive yedekleme, gerçek worker çıkışı, ortam, kaynak/batch kararı, yayın ve owned-runtime kapatma ayrıntıları [operasyon kaydında](cloudfixer_baseline_multibackbone_20261123.md).

## Bulguların yorumu ve sonraki araştırma kararı

[Run] PM, bu sabit üç checkpoint/seed0 ölçümünde kendi gerçek N8 terminalinin all15 accuracy'sini üç modelde de artırdı. Bu, N8 PM ile N32 terminal arasındaki eski bütün-politika kıyasından farklı olarak **aynı N8 yörüngesinde PM readout etkisini** ölçer. PM'nin terminale göre net kazancı DGCNN'de 170, CurveNet'te 149 ve PointNeXt'te 108 corruption gözlemidir. Buna rağmen PM, terminalin doğru bildiği sırasıyla 448, 374 ve 412 gözlemi yanlış sınıfa çevirdi; yalnız PM'nin doğru bildiği sayılar 618, 523 ve 520'dir. Toplam kazanç, her nesnenin iyileştiği anlamına gelmez.

[Run] PM–terminal farkı her corruption'da pozitif değildir. DGCNN'de Shear −8 ve Distortion −1; CurveNet'te RBF inverse −1; PointNeXt'te Density increase −14, Shear −2 ve Occlusion −10 net doğru kaybı vardır. DGCNN Impulse ve CurveNet Density increase eşittir. Dolayısıyla “PM her corruption'da veya her timestep seçeneğinde daha iyidir” sonucu çıkarılamaz. Entropy, farklı PM ağırlıkları veya pencere bu testten yeniden seçilmedi.

[Run] Yeni CF-prior PM, orijinal CF'yi DGCNN'de 10/15, CurveNet'te 5/15 ve PointNeXt'te 9/15 corruption'da geçer. Ancak bütün15 toplamında CF hâlâ DGCNN'de 51, CurveNet'te 625 ve PointNeXt'te 274 fazla doğruya sahiptir. Özellikle DGCNN'de fark sadece 0.137763 pp olduğundan, eski LION kıyasındaki büyük CF üstünlüğünü bu yeni CF-denoiser baseline'a aynen taşımak doğru değildir. Bu kayıt istatistiksel eşdeğerlik veya üstünlük iddiası değildir; tek seed'deki gözlenen tamsayı farklarıdır.

[Run] **Background ve Rotation dışındaki 13 corruption'ın birleşiminde PM üç modelde de CF'den fazla doğru bilir:** DGCNN +689, CurveNet +120, PointNeXt +451. Background+Rotation ise sırasıyla 740, 745 ve 725 net CF avantajı üretir. Böylece bu iki corruption'ın kaybı, diğer 13'teki toplam kazancı aşar. Kullanıcının geometri düzeltmesini özellikle bu iki türde inceleme yönelimi yeni source modellerde de sayısal destek bulur.

[Inference; gerçekleşmiş deney değildir] Diğer 13 sonucu birebir koruyup yalnız Background ve Rotation'da CF'nin doğru sayısını yakalamak aşağıdaki all15 ortalamalarını verirdi. Bu sadece aritmetik bir hedef hesabıdır; uygulanabilir bir düzeltme veya gelecekteki iyileşme garantisi değildir. İki türe yapılacak bir müdahale başka türlerin sonuçlarını değiştirebilir.

| Model | Koşullu doğru sayısı | Koşullu all15 % | CF'ye koşullu fark pp |
| --- | --- | --- | --- |
| dgcnn | 27076/37020 | 73.138844 | +1.861156 |
| curvenet | 28230/37020 | 76.256078 | +0.324149 |
| pointnext | 28223/37020 | 76.237169 | +1.218260 |

[Run] PointNeXt Background source-only %85.251216 (2104 doğru), PM %65.032415 (1605 doğru), CF %75.607780 (1866 doğru). PM'nin terminale göre +27 net doğru kazancı vardır; buna rağmen source-only'ye göre 499 net doğru kaybolur. CF de source-only'den 238 net doğru geridedir. PointNeXt Density, Density increase, Shear ve Cutout'ta da source-only her iki düzeltmeden yüksektir. Bu bulgu, source-only kontrolünü korumanın ve düzeltmenin zararını ayrıca ölçmenin önemini gösterir; “her girdiyi mutlaka düzeltmek yararlıdır” varsayımını desteklemez.

[Code/Inference] PM aday tahminlerini birleştirir; örneklerin düzeltme geometrisini değiştirmez. Bu iki türdeki 10,6–18,9 yüzde puanlık CF açığını yalnız küçük PM–terminal readout kazancıyla kapatamadık. Orijinal CF'nin rotation/delta optimizasyonu ile bizim gözleme proximal bağlanmamızın pose/outlier etkileri olası mekanizmalardır; bu test bunları nedensel olarak ayırmaz. Sonraki mekanizma deneylerinde aynı donmuş prior/head/data ile pose, referans eşleşmesi ve background-anchor etkisi kontrollü olarak ayrılmalı; parametre/pencere seçimi ayrı geliştirme nesnelerinde yapılıp tam testten önce kilitlenmelidir. Buradaki tam test etiketleriyle yeniden tuning yapılmadı.

[Decision] Kayıtlı geliştirme referansı N8/K80/tau.1/retained.95 late equal PM[20,10,0] olarak korunur; bu görev aynı sabit politikanın üç yeni source modeldeki ölçümüdür. Aynı-yörünge terminal sütunu zorunlu karşılaştırma olarak tutulur. Sonuçlar CF'yi toplamda geçtiğimiz iddiasını desteklemez; sonraki geometri araştırmasının hem CF açığını hem source-only'ye verilen zararı izlemesi gerekir. Bu görev kapsamında ek geometri yöntemi veya yeni GPU deneyi başlatılmadı.

[Limit] Üç head aynı denoising geometrisini paylaşır; bunlar üç bağımsız geometri/seed tekrarı değildir. Ayrıca15 corruption aynı 2.468 özgün nesneyi kullanır. Sonuçlar aynı yerel donmuş source checkpoint'lerle matched kıyastır; makalenin classifier checkpoint'lerini birebir yeniden üretme iddiası değildir. Önceki CF A100 80 GB / B64, yeni yöntem A100 40 GB / B96 kullanmıştır; 75,0 dk ölçülen denetimli tam koşu süresinden doğrudan yöntem hız oranı çıkarılmadı. Terminal-only uçtan uca süre ayrıca ölçülmedi. Bağımsız seed/tekrar ve belirsizlik aralığı bu üç yeni head için henüz yoktur.

[Run/Artifact] Son406927870B raw/sourceZIP Drive→freshhost SHA/CRC/141safeüyekontrolünü geçti. Yürütücü0veall15kapsamı,code/knowledgebütün-branchreview0Critical/Important/Minor doğrulandı. Officialcloudflush/yayın/kapatma için operasyon kaydındaki sonraki timestamp izlenmelidir.
