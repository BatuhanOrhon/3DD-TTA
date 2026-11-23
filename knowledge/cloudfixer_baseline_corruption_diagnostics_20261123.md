# CF-XYZ N8: corruption bazında doğru sayıları ve eşleşmeli hata analizi

2026-11-23 — [Run/Derived] Önceden tamamlanan tam ModelNet40-C testinin mühürlü ham sayımından türetildi. Yeni GPU koşusu, model değişikliği veya test etiketleriyle tuning yapılmadı.

## Okuma ve kapsam

Her model/corruption hücresi **2.468 aynı özgün ID** içerir; severity 5, seed 0. Source-only, aynı N8 yörüngesinin gerçek ek-t0 terminali (T), eşit olasılık PM[20,10,0] ve orijinal plain CF/vote1 karşılaştırılır. PM N8/K80/tau.1/retained.95 ve 9 prior çağrısı kullanır. T, N32 değildir. Accuracy yüzdeleri ve yüzde puan farkları [ana sonuç kaydında](cloudfixer_baseline_multibackbone_results_20261123.md); burada yuvarlanmamış tamsayılar verilir.

İlk tablodaki net fark = soldaki yöntemin doğru sayısı − referansın doğru sayısı. İkinci tabloda `+` doğru, `−` yanlış tahmindir: örneğin **PM+/T−**, yalnız PM’nin doğru bildiği nesneleri sayar. İki yöntem de yanlışken farklı sınıflar seçmiş örnekler bu kolonlara girmez. PM+/T− − T+/PM− = PM−T net; diğer iki eşleşme de aynı şekilde okunur.

PM/T aynı geometri ve noise’u paylaşır. CF B64/native sürekli RNG, bizim baseline B96/ID-blok-fork RNG kullanır; CF eşleşmeleri aynı ID/etikette gözlenen sonuçlardır, ortak-noise deneyi değildir. Üç head aynı geometriyi paylaşır; tek seed veya üç model bağımsız denoising tekrarları olarak sunulmaz.

## uniform

| Model | Source doğru | T doğru | PM doğru | CF doğru | PM−T net | PM−CF net |
| --- | --- | --- | --- | --- | --- | --- |
| DGCNN | 1683 | 2172 | 2188 | 2176 | +16 | +12 |
| CurveNet | 1661 | 2167 | 2177 | 2183 | +10 | -6 |
| PointNeXt | 1117 | 2176 | 2195 | 2171 | +19 | +24 |

| Model | PM+/T− | T+/PM− | PM+/CF− | CF+/PM− | T+/CF− | CF+/T− |
| --- | --- | --- | --- | --- | --- | --- |
| DGCNN | 35 | 19 | 77 | 65 | 76 | 80 |
| CurveNet | 33 | 23 | 65 | 71 | 66 | 82 |
| PointNeXt | 34 | 15 | 93 | 69 | 85 | 80 |

[Run] PM, kendi terminalini üç modelde de yükseltir. CF karşısında DGCNN ve PointNeXt kazanırken CurveNet kaybeder; aynı geometri, bütün classifier’larda aynı sıralamayı üretmez.

## gaussian

| Model | Source doğru | T doğru | PM doğru | CF doğru | PM−T net | PM−CF net |
| --- | --- | --- | --- | --- | --- | --- |
| DGCNN | 1851 | 2169 | 2198 | 2166 | +29 | +32 |
| CurveNet | 1902 | 2178 | 2181 | 2184 | +3 | -3 |
| PointNeXt | 1356 | 2174 | 2204 | 2168 | +30 | +36 |

| Model | PM+/T− | T+/PM− | PM+/CF− | CF+/PM− | T+/CF− | CF+/T− |
| --- | --- | --- | --- | --- | --- | --- |
| DGCNN | 46 | 17 | 94 | 62 | 83 | 80 |
| CurveNet | 24 | 21 | 73 | 76 | 72 | 78 |
| PointNeXt | 44 | 14 | 108 | 72 | 90 | 84 |

[Run] PM–terminal farkı üç modelde de pozitiftir. DGCNN ve PointNeXt PM ile CF’yi geçer; CurveNet’in CF’ye küçük kaybı korunur. Bu satır Background sonucuna genellenmez.

## background

| Model | Source doğru | T doğru | PM doğru | CF doğru | PM−T net | PM−CF net |
| --- | --- | --- | --- | --- | --- | --- |
| DGCNN | 336 | 1031 | 1066 | 1366 | +35 | -300 |
| CurveNet | 187 | 733 | 766 | 1232 | +33 | -466 |
| PointNeXt | 2104 | 1578 | 1605 | 1866 | +27 | -261 |

| Model | PM+/T− | T+/PM− | PM+/CF− | CF+/PM− | T+/CF− | CF+/T− |
| --- | --- | --- | --- | --- | --- | --- |
| DGCNN | 78 | 43 | 142 | 442 | 143 | 478 |
| CurveNet | 53 | 20 | 66 | 532 | 63 | 562 |
| PointNeXt | 62 | 35 | 113 | 374 | 109 | 397 |

[Run] CF üç modelde de PM’den yüksektir; buna rağmen PM kendi terminalini üçünde de artırır. PointNeXt source-only kontrolü her iki düzeltmeden de daha iyidir. PM’nin terminale katkısı, source-only’ye göre zarar veya CF açığı ile aynı ölçüt değildir.

## impulse

| Model | Source doğru | T doğru | PM doğru | CF doğru | PM−T net | PM−CF net |
| --- | --- | --- | --- | --- | --- | --- |
| DGCNN | 1358 | 2194 | 2194 | 2170 | +0 | +24 |
| CurveNet | 1176 | 2180 | 2188 | 2177 | +8 | +11 |
| PointNeXt | 1786 | 2226 | 2229 | 2197 | +3 | +32 |

| Model | PM+/T− | T+/PM− | PM+/CF− | CF+/PM− | T+/CF− | CF+/T− |
| --- | --- | --- | --- | --- | --- | --- |
| DGCNN | 22 | 22 | 92 | 68 | 88 | 64 |
| CurveNet | 26 | 18 | 91 | 80 | 91 | 88 |
| PointNeXt | 24 | 21 | 93 | 61 | 86 | 57 |

[Run] PM, CF’yi üç modelde de geçer. DGCNN’de PM ve terminalin doğru sayıları eşittir; fakat PM yalnız 22 örneği doğru bilirken terminal de yalnız 22 başka örneği doğru bilir. Eşit accuracy, aynı nesnelerin doğru sınıflandırıldığı anlamına gelmez.

## upsampling

| Model | Source doğru | T doğru | PM doğru | CF doğru | PM−T net | PM−CF net |
| --- | --- | --- | --- | --- | --- | --- |
| DGCNN | 1844 | 2199 | 2210 | 2183 | +11 | +27 |
| CurveNet | 1730 | 2188 | 2194 | 2204 | +6 | -10 |
| PointNeXt | 1760 | 2206 | 2209 | 2179 | +3 | +30 |

| Model | PM+/T− | T+/PM− | PM+/CF− | CF+/PM− | T+/CF− | CF+/T− |
| --- | --- | --- | --- | --- | --- | --- |
| DGCNN | 26 | 15 | 79 | 52 | 79 | 63 |
| CurveNet | 29 | 23 | 55 | 65 | 53 | 69 |
| PointNeXt | 26 | 23 | 95 | 65 | 94 | 67 |

[Run] PM kendi terminalini üç modelde de artırır; CF karşısında DGCNN ve PointNeXt kazanır, CurveNet kaybeder. Bu aynı zamanda tam koşudaki en yüksek allocated GPU belleğinin görüldüğü corruption’dır; bu donanım gözlemi yöntemler arası hız üstünlüğü iddiası değildir.

## distortion_rbf

| Model | Source doğru | T doğru | PM doğru | CF doğru | PM−T net | PM−CF net |
| --- | --- | --- | --- | --- | --- | --- |
| DGCNN | 1841 | 1998 | 2010 | 2024 | +12 | -14 |
| CurveNet | 1940 | 2054 | 2067 | 2093 | +13 | -26 |
| PointNeXt | 1752 | 1962 | 1969 | 1974 | +7 | -5 |

| Model | PM+/T− | T+/PM− | PM+/CF− | CF+/PM− | T+/CF− | CF+/T− |
| --- | --- | --- | --- | --- | --- | --- |
| DGCNN | 43 | 31 | 115 | 129 | 114 | 140 |
| CurveNet | 38 | 25 | 95 | 121 | 97 | 136 |
| PointNeXt | 41 | 34 | 138 | 143 | 141 | 153 |

[Run] PM üç modelde de kendi terminalini iyileştirir; yine de CF üçünde de PM’den yüksektir. PM readout kazancı burada bütün düzeltme pipeline’ı arasındaki açığı kapatmaz.

## distortion_rbf_inv

| Model | Source doğru | T doğru | PM doğru | CF doğru | PM−T net | PM−CF net |
| --- | --- | --- | --- | --- | --- | --- |
| DGCNN | 1856 | 2060 | 2068 | 2061 | +8 | +7 |
| CurveNet | 1987 | 2094 | 2093 | 2134 | -1 | -41 |
| PointNeXt | 1835 | 2013 | 2015 | 2016 | +2 | -1 |

| Model | PM+/T− | T+/PM− | PM+/CF− | CF+/PM− | T+/CF− | CF+/T− |
| --- | --- | --- | --- | --- | --- | --- |
| DGCNN | 38 | 30 | 114 | 107 | 113 | 114 |
| CurveNet | 30 | 31 | 86 | 127 | 87 | 127 |
| PointNeXt | 27 | 25 | 120 | 121 | 122 | 125 |

[Run] CF karşısındaki PM kazancı yalnız DGCNN’dedir. CurveNet’te PM kendi terminalinden bir net doğru geriye düşer; PointNeXt’te PM–CF farkı yalnız bir doğru örnektir. Bu küçük tek-seed farkları evrensel bir sıralama olarak yorumlanmaz.

## density

| Model | Source doğru | T doğru | PM doğru | CF doğru | PM−T net | PM−CF net |
| --- | --- | --- | --- | --- | --- | --- |
| DGCNN | 1274 | 1366 | 1388 | 1247 | +22 | +141 |
| CurveNet | 1953 | 1907 | 1920 | 1827 | +13 | +93 |
| PointNeXt | 2113 | 1984 | 1994 | 1911 | +10 | +83 |

| Model | PM+/T− | T+/PM− | PM+/CF− | CF+/PM− | T+/CF− | CF+/T− |
| --- | --- | --- | --- | --- | --- | --- |
| DGCNN | 58 | 36 | 239 | 98 | 240 | 121 |
| CurveNet | 45 | 32 | 201 | 108 | 204 | 124 |
| PointNeXt | 49 | 39 | 203 | 120 | 205 | 132 |

[Run] PM, CF’yi üç modelde de geçer. Buna rağmen CurveNet ve PointNeXt’te source-only hem terminalden hem PM’den hem de CF’den yüksektir. DGCNN’de ise CF source-only’den düşükken bizim iki readout’umuz source-only’den yüksektir.

## density_inc

| Model | Source doğru | T doğru | PM doğru | CF doğru | PM−T net | PM−CF net |
| --- | --- | --- | --- | --- | --- | --- |
| DGCNN | 2029 | 2105 | 2117 | 2073 | +12 | +44 |
| CurveNet | 2176 | 2114 | 2114 | 2131 | +0 | -17 |
| PointNeXt | 2251 | 2177 | 2163 | 2110 | -14 | +53 |

| Model | PM+/T− | T+/PM− | PM+/CF− | CF+/PM− | T+/CF− | CF+/T− |
| --- | --- | --- | --- | --- | --- | --- |
| DGCNN | 33 | 21 | 129 | 85 | 125 | 93 |
| CurveNet | 33 | 33 | 88 | 105 | 85 | 102 |
| PointNeXt | 18 | 32 | 136 | 83 | 145 | 78 |

[Run] PM, CF karşısında DGCNN ve PointNeXt’te kazanır, CurveNet’te kaybeder. CurveNet PM ve terminal eşit doğru sayısına rağmen 33’er yalnız-doğru örneğe sahiptir. PointNeXt’te PM terminalden düşük olduğu hâlde CF’den yüksektir; CurveNet ve PointNeXt source-only kontrolleri de üç düzeltilmiş readout’tan yüksektir.

## shear

| Model | Source doğru | T doğru | PM doğru | CF doğru | PM−T net | PM−CF net |
| --- | --- | --- | --- | --- | --- | --- |
| DGCNN | 1895 | 1925 | 1917 | 1918 | -8 | -1 |
| CurveNet | 1901 | 1982 | 1983 | 2009 | +1 | -26 |
| PointNeXt | 1945 | 1834 | 1832 | 1865 | -2 | -33 |

| Model | PM+/T− | T+/PM− | PM+/CF− | CF+/PM− | T+/CF− | CF+/T− |
| --- | --- | --- | --- | --- | --- | --- |
| DGCNN | 32 | 40 | 141 | 142 | 146 | 139 |
| CurveNet | 27 | 26 | 111 | 137 | 117 | 144 |
| PointNeXt | 35 | 37 | 150 | 183 | 153 | 184 |

[Run] CF üç modelde de PM’den yüksektir. PM, DGCNN ve PointNeXt’te terminale zarar verir; CurveNet’te bir net doğru artırır. PointNeXt source-only kontrolü her iki yöntemin düzeltilmiş çıktılarından yüksektir.

## rotation

| Model | Source doğru | T doğru | PM doğru | CF doğru | PM−T net | PM−CF net |
| --- | --- | --- | --- | --- | --- | --- |
| DGCNN | 1413 | 1422 | 1435 | 1875 | +13 | -440 |
| CurveNet | 1698 | 1742 | 1777 | 2056 | +35 | -279 |
| PointNeXt | 964 | 1077 | 1091 | 1555 | +14 | -464 |

| Model | PM+/T− | T+/PM− | PM+/CF− | CF+/PM− | T+/CF− | CF+/T− |
| --- | --- | --- | --- | --- | --- | --- |
| DGCNN | 65 | 52 | 82 | 522 | 84 | 537 |
| CurveNet | 61 | 26 | 73 | 352 | 70 | 384 |
| PointNeXt | 48 | 34 | 67 | 531 | 74 | 552 |

[Run] CF üç modelde de açık biçimde daha fazla doğru bilir. PM, üç modelde de kendi terminalinden yüksektir; ancak bu readout farkı CF açığını kapatmaz. Bu gözlem pose mekanizmasını tek başına nedensel olarak teşhis etmez.

## cutout

| Model | Source doğru | T doğru | PM doğru | CF doğru | PM−T net | PM−CF net |
| --- | --- | --- | --- | --- | --- | --- |
| DGCNN | 1422 | 1732 | 1739 | 1463 | +7 | +276 |
| CurveNet | 1939 | 2019 | 2022 | 1970 | +3 | +52 |
| PointNeXt | 2116 | 2090 | 2097 | 2019 | +7 | +78 |

| Model | PM+/T− | T+/PM− | PM+/CF− | CF+/PM− | T+/CF− | CF+/T− |
| --- | --- | --- | --- | --- | --- | --- |
| DGCNN | 45 | 38 | 352 | 76 | 351 | 82 |
| CurveNet | 30 | 27 | 160 | 108 | 159 | 110 |
| PointNeXt | 35 | 28 | 173 | 95 | 170 | 99 |

[Run] PM, CF’yi üç modelde de geçer ve kendi terminalini de üçünde artırır. DGCNN’in en büyük pozitif PM–CF corruption farkı bu satırdadır. PointNeXt source-only yine PM ve CF’den yüksektir; göreli yöntem kazancı source-only’ye göre mutlak kazanç demek değildir.

## distortion

| Model | Source doğru | T doğru | PM doğru | CF doğru | PM−T net | PM−CF net |
| --- | --- | --- | --- | --- | --- | --- |
| DGCNN | 1887 | 1951 | 1950 | 1990 | -1 | -40 |
| CurveNet | 1916 | 2010 | 2021 | 2049 | +11 | -28 |
| PointNeXt | 1825 | 1894 | 1898 | 1904 | +4 | -6 |

| Model | PM+/T− | T+/PM− | PM+/CF− | CF+/PM− | T+/CF− | CF+/T− |
| --- | --- | --- | --- | --- | --- | --- |
| DGCNN | 39 | 40 | 110 | 150 | 112 | 151 |
| CurveNet | 38 | 27 | 104 | 132 | 103 | 142 |
| PointNeXt | 38 | 34 | 136 | 142 | 134 | 144 |

[Run] CF üç modelde de PM’den yüksektir. PM, CurveNet ve PointNeXt’te terminali artırırken DGCNN’de bir net doğru kaybettirir. Bu satır RBF ve RBF inverse ile karıştırılmadan ayrı tutulur.

## occlusion

| Model | Source doğru | T doğru | PM doğru | CF doğru | PM−T net | PM−CF net |
| --- | --- | --- | --- | --- | --- | --- |
| DGCNN | 796 | 970 | 981 | 895 | +11 | +86 |
| CurveNet | 879 | 993 | 999 | 924 | +6 | +75 |
| PointNeXt | 974 | 1013 | 1003 | 952 | -10 | +51 |

| Model | PM+/T− | T+/PM− | PM+/CF− | CF+/PM− | T+/CF− | CF+/T− |
| --- | --- | --- | --- | --- | --- | --- |
| DGCNN | 28 | 17 | 139 | 53 | 135 | 60 |
| CurveNet | 23 | 17 | 122 | 47 | 119 | 50 |
| PointNeXt | 14 | 24 | 116 | 65 | 125 | 64 |

[Run] PM, CF’yi üç modelde de geçer. Buna rağmen PointNeXt’te PM kendi terminalinden on net doğru düşüktür. CF’ye göre kazanç ile PM’nin aynı-yörünge terminaline katkısı bu örnekte ters yönlüdür.

## lidar

| Model | Source doğru | T doğru | PM doğru | CF doğru | PM−T net | PM−CF net |
| --- | --- | --- | --- | --- | --- | --- |
| DGCNN | 296 | 872 | 875 | 780 | +3 | +95 |
| CurveNet | 652 | 975 | 983 | 937 | +8 | +46 |
| PointNeXt | 475 | 986 | 994 | 885 | +8 | +109 |

| Model | PM+/T− | T+/PM− | PM+/CF− | CF+/PM− | T+/CF− | CF+/T− |
| --- | --- | --- | --- | --- | --- | --- |
| DGCNN | 30 | 27 | 168 | 73 | 167 | 75 |
| CurveNet | 33 | 25 | 135 | 89 | 132 | 94 |
| PointNeXt | 25 | 17 | 160 | 51 | 161 | 60 |

[Run] PM üç modelde de hem terminalden hem CF’den yüksektir. Bununla birlikte en yüksek PM accuracy bile %41’in altındadır; göreli kazanç, corruption’ın çözüldüğünü göstermez.

## Source-only kontrolünün koruduğu ek bilgi

[Run] Aşağıdaki hücrelerde en az bir düzeltme/readout source-only’den daha az doğru bilir. Bu, yalnız yöntemler arası kıyasın source-only’ye verilen zararı gizleyebileceğini gösterir. Negatif sayılar source-only’ye göre net doğru kaybıdır; her nesnede aynı zarar olduğunu göstermez.

| Corruption | Model | Source doğru | T−Source | PM−Source | CF−Source |
| --- | --- | --- | --- | --- | --- |
| background | PointNeXt | 2104 | -526 | -499 | -238 |
| density | DGCNN | 1274 | +92 | +114 | -27 |
| density | CurveNet | 1953 | -46 | -33 | -126 |
| density | PointNeXt | 2113 | -129 | -119 | -202 |
| density_inc | CurveNet | 2176 | -62 | -62 | -45 |
| density_inc | PointNeXt | 2251 | -74 | -88 | -141 |
| shear | PointNeXt | 1945 | -111 | -113 | -80 |
| cutout | PointNeXt | 2116 | -26 | -19 | -97 |
| occlusion | PointNeXt | 974 | +39 | +29 | -22 |

## Sonuçların sınırı ve araştırma kararı

[Run] PM üç modelde de all15 terminal toplamını yükseltir; tek tek corruption’larda ve nesnelerde kayıplar vardır. Background ve Rotation CF lehine; diğer 13 türün toplamı PM lehine kalır. Bu ayrıştırma [ana nottaki toplamlarla](cloudfixer_baseline_multibackbone_results_20261123.md#background-ve-rotation-açıkları) birlikte okunmalıdır.

[Inference/Open] Gelecek pose/background düzeltmesinin bu iki açığı kapatırken diğer türlerin kazanımlarını ve source-only kontrolünü koruması sınanmalıdır. Bu tablo mekanizmayı nedensel ayırmaz, test etiketleriyle parametre seçimi veya corruption kimliğine göre inference routing tanımlamaz. Yeni seçim ayrı geliştirme verisinde kilitlenmelidir. Bağımsız seed/CI ve standalone terminal-only süre burada ölçülmedi.

## Kanıt ve yeniden sayım

- Kaynak ZIP: `result/modelnet40_c/cloudfixer_baseline_heads/e8df74eb3976/final_20261123T112717Z.zip`,406927870B, SHA-256 `c392dce804c213a21e97769590e3a036ee80deefb9b0d6379689958530646265`.
- Arşiv üyesi: `recount.json`, SHA-256 `4abebb4d7d6b58c2d97f6fbf1d6d6f659e7b292deaa8bd1a36e2f2fa5f460fed`.
- Baseline kampanyası: `e8df74eb3976517f74452c49363df953ac5d103e4ff0cc6fb265d327769b0427`; çalıştırılan GPU kaynak commit’i `a979bcec11c24963ff31f0574d66decb85d680d3`.
- Kabul edilmiş CF kampanyası: `3596befa489bc8f05cd5f139a976c7a00b7429d390a19fec850595814cd3f224`.
- 135 accuracy kaydı, 135 eşleşme ve 9 all15 toplamı kapsam/sayı tutarlılığıyla kontrol edildi. Mevcut 45 accuracy satırının altışar yüzde/fark alanı mühürlü sayımla eşleştirildi.
- Yeni 45 doğru/net tablosu satırı ile 45 eşleşmeli tablo satırındaki 540 tamsayı ayrı okunup kaynakla karşılaştırıldı. Önceki arşivler değiştirilmedi; sayılar yeni bir deney sonucu değildir.
