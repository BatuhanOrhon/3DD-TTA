# ScanNet, ScanObjectNN-C ve diffusion eğitim alanları

22 Kasım 2026. [Paper/Code/Run/Inference/Open]
Kaynak worktree: `research/cloudfixer-transfer-20261114@d11b3483189f91197d24422cb1b72b883b5cd9c2`.
Resmî CloudFixer klonu `f043e24414fcf4165fdc892abff21215c9989141`;
inceleme sırasında temiz ve değiştirilmedi. Yeni inference/eğitim/GPU yok.

## 1. CloudFixer'ın ScanNet'i bizim ScanObjectNN-C ile aynı mı?

**[Paper/Code] Hayır, farklı benchmark/protokoller.** CloudFixer'ın
§5.1 ve Tablo5'teki ScanNet'i, **PointDA-10** içindeki gerçek-tarama alanıdır.
ModelNet/ShapeNet/ScanNet'in ortak **10 sınıfı** arasında doğal alan kayması
değerlendirilir. ScanNet alanı için `datasets/dataloader.py:PointDA10`
ve `load_scannet()` kullanılır; `PointDA_data/scannet/train_*.h5` / test
dosyaları ile çalışır. Bu deneylere ortak15 sentetik corruption uygulanmış
ScanObjectNN-C sonucu denmez.

[Paper] ScanObjectNN ayrı **15 sınıflı** nesne sınıflandırma benchmark'ıdır.
3DD-TTA'nın ScanObjectNN-C'si test nesnelerine15 corruption uygulanarak
oluşturulur. Bizim kabul edilmiş Scan paketi `main_split`,581 test nesnesi,
all15/s5 protokolüdür; eski geçersiz receipt kullanılmaz.

[Paper/Limit] İki veri kümesini tamamen ilişkisiz taramalar gibi anlatma:
ScanObjectNN, SceneNN ve ScanNet'ten türetilmiştir. Buna rağmen PointDA-10
ScanNet ve ScanObjectNN-C'nin nesne kimlikleri, sınıf seçimi, ayrımları ve
preprocessing'i aynı değildir. Veri kökeni paylaşımı nesne-ayrık olduklarını
da kanıtlamaz; olası scene/object overlap ayrıca audit gerektirir.

Kaynaklar:
[CloudFixer §5.1 ve Tablo5](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04603.pdf),
[Ek E.2, basılı s.3](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04603-supp.pdf),
[ScanObjectNN yazar sayfası](https://hkust-vgd.github.io/scanobjectnn/),
[SceneNN yazarlarının veri kökeni açıklaması](https://hkust-vgd.github.io/scenenn/),
[3DD-TTA §4.1](https://arxiv.org/html/2411.14495v1#S4.SS1).

## 2. CloudFixer, ScanNet testinde hangi diffusion modelini kullanıyor?

**[Paper] Denoiser, transfer yönünün kaynak alanında eğitiliyor.** Makale
§5.1'de her kaynak veri kümesi için ayrı diffusion eğitimi bildirir.
EkG, basılı s.5, ModelNet40, PointDA ModelNet, ShapeNet ve ScanNet'i ayrı
kaynaklar olarak sayar. Tablo5'te `M=ModelNet`, `S=ShapeNet`, `S*=ScanNet`.

| PointDA-10 yönü | Denoiser'ın eğitim alanı | İlgili yayımlanmış dosya adı |
|---|---|---|
| ModelNet → ScanNet | PointDA-10 ModelNet kaynağı | `diffusion_model_transformer_modelnet.npy` |
| ShapeNet → ScanNet | PointDA-10 ShapeNet kaynağı | `diffusion_model_transformer_shapenet.npy` |
| ScanNet → ModelNet | PointDA-10 ScanNet kaynağı | `diffusion_model_transformer_scannet.npy` |
| ScanNet → ShapeNet | PointDA-10 ScanNet kaynağı | `diffusion_model_transformer_scannet.npy` |

[Inference from Paper/Code] Dolayısıyla **ScanNet hedef test alanıyken**
ModelNet veya ShapeNet'te öğrenilmiş prior ile transfer yapılır. **ScanNet
kaynak alanken** ScanNet prior'ı kullanılır. Test veri kümesinin adı tek başına
denoiser'ın eğitim kümesini belirlemez. Yukarıdaki yön-dosya eşlemesi,
yayımlanmış kaynak-alan eğitim protokolü ve dosya adlarından çıkarılır;
her Tablo5 koşusunun özgün komut/log dosyaları yeniden elde edilmedi.

[Code] `scripts/train_dm.sh` ayrı `train_dm_modelnet40`,
`train_dm_modelnet`, `train_dm_shapenet`, `train_dm_scannet` fonksiyonlarını
tanımlar. PointDA ModelNet yolu `PointDA_data/modelnet`, 40-sınıflı ModelNet40
yolu `modelnet40_ply_hdf5_2048`'dir. Bu nedenle PointDA'daki ModelNet prior'ı
bizim A/B'de kullanılan `diffusion_model_transformer_modelnet40.npy` ile
özdeş sayılmaz. Ayrı dosyalar resmî Drive listesinde de vardır.

[Code] `scripts/hparam_tune_baselines.sh:hparam_tune_pointda()` classifier
ve diffusion yollarını `SOURCE_DOMAIN_LIST[idx]` üzerinden birlikte kurar.
Bu fonksiyon kaynak veri üzerinde tuning yapar; altı yönün final koşu
logu değildir. `adapt.py` verilen `--diffusion_dir`'i yükler; test alanına
bakarak otomatik olarak yeni bir diffusion checkpoint'ine geçmez.

Kod:
[eğitim reçeteleri](https://github.com/shimazing/CloudFixer/blob/f043e24414fcf4165fdc892abff21215c9989141/scripts/train_dm.sh),
[kaynakla eşlenen tuning yolları](https://github.com/shimazing/CloudFixer/blob/f043e24414fcf4165fdc892abff21215c9989141/scripts/hparam_tune_baselines.sh#L941),
[checkpoint yükleme](https://github.com/shimazing/CloudFixer/blob/f043e24414fcf4165fdc892abff21215c9989141/adapt.py#L414).

## 3. ScanObjectNN üzerinde eğitilmiş resmî CloudFixer denoiser'ı var mı?

**[Code/Artifact] İncelenen resmî yayın/kod/Drive envanterinde bulunmadı.**
22 Kasım'da README'nin herkese açık checkpoint klasörü yeniden GET ile
okundu: HTTP200,25 görünür checkpoint adı. Sekiz diffusion dosyası:
ModelNet40, PointDA ModelNet, ShapeNet, ScanNet, ShapeNetCore, Synthetic,
Kinect, Realsense. `scanobjectnn` adlı diffusion dosyası yok.
**ScanNet için yayımlanmış dosya var**:
`diffusion_model_transformer_scannet.npy`; bu ScanObjectNN eğitimi değildir.
[Tarihli envanter ve HTML hash'i](evidence/cloudfixer_scannet_lion_20261122/drive_inventory.json).

[Code] `train_dm.py:main()` diffusion eğitiminde ModelNet40, PointDA'nın
modelnet/shapenet/scannet alanları ve GraspNet synthetic/kinect/realsense
yollarını seçer; ScanObjectNN branch'i yoktur. Reponun başka yerlerinde
`ScanObjectNN` loader'ı ve Point-MAE ScanObjectNN YAML'leri bulunması,
bu veri kümesinde bir diffusion modeli eğitilip yayımlandığını kanıtlamaz.
`adapt.py`'nin ana veri seçimi de ModelNet40-C ve PointDA-10 ile sınırlıdır.

[Open/Limit] Bu, incelenen kamusal sürüm için negatif envanter bulgusudur;
yayımlanmamış bir modelin hiç var olmadığına dair evrensel iddia değildir.
Yeni ScanNet ağırlığı indirilmedi/deserileştirilmedi; dosyanın training logu,
epoch'u veya iç tensor kimliği doğrulanmadı. Klasörün dosya listesi doğrulandı.

## 4. Biz ScanObjectNN-C testinde hangi LION'u kullanıyoruz?

**[Paper/Code/Run] ShapeNet all55 üzerinde ön-eğitilmiş aynı LION prior'ını.**
ScanObjectNN'e özel eğitilmiş veya yeniden fine-tune edilmiş LION kullanılmıyor.
3DD-TTA §4 ve §4.1, ShapeNet ön-eğitimini ve ModelNet40/ScanObjectNN'e transferi
açıkça ayırır. Bizim kaynak ve kabul kayıtları bu model seçimini bağlar:

| Varlık | Kimlik / anlamı |
|---|---|
| LION checkpoint | `epoch_10999_iters_2100999.pt`,1508051672 bytes |
| Checkpoint SHA256 | `807f6732ad087a1ffdaeeaa456e32b4130c9a8a1708446cabc0059654a0a86c2` |
| LION config | `unconditional_all55_cfg.yml` |
| Config SHA256 | `9d4a826f35fe9b3053210105eb4857811c8bfd53cd8c6a998f8ea7953e975816` |
| Config eğitim alanı | `data/ShapeNetCore.v2.PC15k`, `dataset_type: shapenet15k`, `cates: all` |
| Scan Point-MAE classifier | `scanobject_jt.pth`,15 sınıflı farklı görev modeli |
| Scan Point-MAE SHA256 | `53f6c2e46d3c982c9c144a9c822acf5852ef626e2e3124ec0d165c64b9535dac` |

[Code] `scripts/eval_multibackbone_guidance.py` dataset'ten bağımsız tek
`LION_SHA256`/`LION_CONFIG_SHA256` sabiti kullanır. `load_guidance_runtime()`
bu kimlikleri doğrulayıp `lion.load_model(..., use_ema=False)` çağırır ve
VAE/prior parametrelerini dondurur. Aynı sabitler kabul edilmiş tam testin
bilimsel kaynağı `b651e259db9ec6341f39e606b01eef5254d90456` içinde de vardır.
Aktif root checkout'un veya eski notların varsayılanı üzerinden tahmin edilmedi.

[Run] 30 Ekim metadata binding'i, iki dataset'i ve tek LION dosyasını birlikte
kaydeder. 14 Kasım tam-veri kabulü, Scan581/all15/s5/seed0/B32 ile
ScanObjectNN'e özgü classifier dosyalarını ve frozen model kontrollerini
kaydeder. Bu inceleme bu kabul kayıtlarını/kaynaklarını okur;240 ham arşiv
ve1.5GB LION checkpoint'i yeniden indirip model inference'ı yapmaz.
[Kompakt kaynak/koşu kimlikleri](evidence/cloudfixer_scannet_lion_20261122/source_receipt.json).

## 5. Karşılaştırma açısından çıkarım

[Inference] LION'un ScanObjectNN-C başarısı, ShapeNet'te öğrenilmiş prior'ın
başka bir nesne veri kümesine transferidir. Classifier'ın ScanObjectNN'de
eğitilmiş olması denoiser'ın da orada eğitildiği anlamına gelmez.
CloudFixer'ın PointDA ScanNet doğrulukları, bizim15-sınıf ScanObjectNN-C
tablomuzla doğrudan karşılaştırılamaz. ScanNet prior'ını ScanObjectNN-C'ye
taşımak teknik olarak yeni bir transfer hipotezidir; source-matched prior
veya doğrulanmış performans kazancı diye adlandırılamaz. Ortak tarama kökeni
nedeniyle olası veri örtüşmesi ayrıca araştırılmalıdır.

[Decision/Open] Bu görev bir model/veri kökeni auditidir; yeni checkpoint
seçimi, pipeline değişikliği, fine-tuning, GPU fazı veya peer runtime işlemi
başlatmaz. ScanObjectNN'e eşlenmiş CloudFixer karşılaştırması gelecekte
istenirse dataset/class order/preprocessing/weights ve training-exposure
sözleşmesi ayrıca kurulmalıdır. Yeni kazanç veya genel mimari üstünlük yoktur.
