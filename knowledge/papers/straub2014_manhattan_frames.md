# A Mixture of Manhattan Frames: Beyond the Manhattan World

[Paper] Julian Straub, Guy Rosman, Oren Freifeld, John J. Leonard ve John W. Fisher III. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014, s. 3770–3777. DOI 10.1109/CVPR.2014.488. [Yayın kaydı](https://doi.org/10.1109/CVPR.2014.488) · [Yazarın açık tam metni, sekiz sayfa](https://jstraub.github.io/download/straub2014mmf.pdf)

[Paper] Şekil 1 ve §3, insan yapımı bir sahnede yüzey normallerinin birim kürede birden fazla yerel koordinat takımının işaretli, birbirine dik eksenleri çevresinde kümelenmesini ele alır. Birden fazla Manhattan frame'i için olasılıksal bir karışım modeli kurarlar. Böylece aynı sahnede farklı yönlere bakan yapılar temsil edilebilir.

[Code/Inference] Tezde bu gözlemi, gözlenen nesnenin yüzeyleri zaten bir koordinat sistemini destekliyorsa gereksiz dönüşü önlemek amacıyla kullandık. Denoiser'ın t50 temiz tahmininden tek global PCA dönüşü çıkar. Gözlenen ham bulutta kimlik ve PCA dönüşünün eksen uyumları aynı normal yamalarında karşılaştırılır. 256 merkez, 16 komşu, düzlemsellik ağırlığı, iki ayrı grup, 0,015 doğrulama eşiği, SCD zincirine geri besleme ve üçlü PM bu projenin seçimleridir. Makalede bizim nokta bulutu sınıflandırıcımız, t50 kuralımız veya ModelNet40-C kazanımlarımız değerlendirilmez. [Yöntem ve pilot kanıtı](../cloudfixer_selective_rotation_20261123.md) · [A100 doğrulama protokolü](../cloudfixer_surface512_gpu_20261123.md)

[Artifact] Yazar kopyası 23 Kasım 2026'da indirildi. PDF SHA-256 d09779b957053e2e7cd3dbb78a6d7846ea3d9c981f38bc0ac89c8652e5ad921c. Çalışma ağacının knowledge/papers klasöründeki yerel dosya adı straub2014_manhattan_frames.pdf.
