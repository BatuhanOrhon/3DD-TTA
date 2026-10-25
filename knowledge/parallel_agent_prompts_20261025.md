# Copyable prompts for two isolated Colab agents — 2026-10-25

[User report/Decision] These prompts define future assignments. No agent,
branch/worktree or runtime has been launched by preparing this file. Both
agents must read the original workspace's current instructions and the
[ownership contract](parallel_colab_agent_contract_20261025.md).

## Agent A — complete ModelNet trajectory readouts

```text
/Users/batuhan/Desktop/projects/3DD-TTA deposundaki AGENTS.md, RTK ve araştırma/
Colab skill’lerini oku; özellikle orijinal çalışma alanındaki güncel, henüz
commit edilmemiş operator skill ve monitor dosyalarını koru.

Önce knowledge/parallel_colab_agent_contract_20261025.md ve
knowledge/modelnet_multibackbone_trajectory_handoff_20261025.md dosyalarını oku.
Handoff’taki kabul edilmiş sonuç, protokol, checkpoint, yöntem/kod ve makale
referanslarını takip et.

gsd-multibackbone-study’nin commit edilmiş halinden gsd-modelnet-trajectory
dalını türet; /Users/batuhan/Desktop/projects/3DD-TTA-worktrees/modelnet-trajectory
worktree’sinde çalış. Mevcut kullanıcı değişikliklerini reset/overwrite etme.
Yalnızca tta-mn-trajectory-a100-<UTCdate>-r<N> adlı, endpoint sahipliğini
doğruladığın kendi Colab oturumunu ve tek monitorünü kullan. Diğer agent’ın
tta-guidance-components-* oturumlarına/branch’ine/Drive dosyalarına dokunma.

Görev: ModelNet40-C’de DGCNN, CurveNet ve PointNeXt için eksik GSD minimum
entropy ve eşit olasılık ortalaması sonuçlarını al. Severity5/seed0/all15,
2468 örnek/corruption; sabit birleşik GSD protokolünü koru. Her yöntem için
aynı koşunun terminalini raporla. Kaynaklar: scripts/eval_multibackbone_guidance.py,
shared_trajectory.py, tta_gsd_v2.py, scripts/analyze_scan_trajectory.py ve
scripts/analyze_legacy_trajectory.py. Scan-only/15-class sınırlarını strict
40-class ModelNet desteğiyle genişlet; Scan ve default terminal regresyonlarını
koru. Önce CPU kontrolleri, sonra GPU smoke ve tam koşu.

Source-only/original3DD, Scan ve Point-MAE tam koşularını tekrarlama; tuning,
guidance ablation veya logit ortalaması ekleme. Önceki ModelNet koşusu69.4dk:
bütün GPU fazını tahmin et ve >60dk için başlamadan tek kapsamlı onay al.
Colab CLI/on hücreli setup/gerçek snapshot-hash kurallarını izle. Kontrattaki
ayrı çıktı köklerini kullan; batch/corruption/model sonuçlarını Drive’a kaydet,
Drive/yerel SHA’yı doğrula. Knowledge güncelle; yalnızca kendi branch’ine
kabul edilen batch’leri push et. Diğer session’a hiçbir işlem yapma; kendi
runtime’ını da açık task-completion ve kapatma izni olmadan sonlandırma.
```

## Agent B — isolate guidance components and develop a supported candidate

```text
/Users/batuhan/Desktop/projects/3DD-TTA deposundaki AGENTS.md, RTK ve güncel
araştırma/Colab skill’lerini oku. knowledge/parallel_colab_agent_contract_20261025.md,
knowledge/guidance_component_study_handoff_20261025.md, knowledge/README.md ve
handoff’ın gösterdiği kabul edilmiş sonuç/makale/kod kayıtlarından başla.

gsd-multibackbone-study’nin commit edilmiş halinden gsd-guidance-components
dalını türet; /Users/batuhan/Desktop/projects/3DD-TTA-worktrees/guidance-components
worktree’sinde çalış. Kullanıcı dosyalarını ve peer worktree’yi değiştirme.
Yalnızca kendi tta-guidance-components-a100-<UTCdate>-r<N> Colab sessionunu,
doğrulanmış endpointini ve tek monitorünü kullan. tta-mn-trajectory-* dahil
diğer sessionlarda exec/setup/upload/stop veya monitor cleanup yapma.

Amaç: ModelNet40-C ve ScanObjectNN-C main_split’te, DGCNN/CurveNet/PointNeXt/
Point-MAE frozen classifier’larıyla guidance’ın accuracy/maliyet katkısını ayır.
Aynı GSD hostunda dört kolu karşılaştır: Chamfer+Spectral, Chamfer-only,
Spectral-only, ikisi kapalı DDIM. Her kolda aynı sabit clean checkpoint’ler ve
terminal üzerinden terminal/minimum entropy/eşit olasılık ortalamasını ölç.
Original3DD farklı bir hosttur; Chamfer-only ile özdeş sayma. Veri/checkpoint/
batch/horizon/scheduler/style/preprocessing ve mümkün olan input/noise draw’larını
eşleştir. Mevcut koşuları audit et; küçük sabit development subset’leriyle başla.

tta_gsd_v2.py, graph_spectral_v2.py, scripts/eval_multibackbone_guidance.py,
shared_trajectory.py ve mevcut analyzer/ablation araçlarını kullan. Eta local,
gamma style update; lambdaa/p Chamfer’da tutulan nokta oranıdır. Spectral ağırlık
ayrıdır. Sıfır spectral ağırlığı halen graph/eigh çalıştırır: gerçek hız için
inaktif hesapları atlamayı ayrıca doğrula, ortak backward maliyetini ölç.

Sabit ablation’dan sonra gerekçeli küçük değişiklikler ve sınırlı eta/gamma/p/
ağırlık denemeleri yapabilirsin. Önce aralık, trial sayısı, development/held-out
ID’ler ve compute bütçesini kaydet; karşılaştırıcılara eşit tuning fırsatı ver.
Final benchmark üzerinde en yüksek değeri seçip genelleme diye sunma. Balanced
dataset/backbone macro ortalaması, her hücre/corruption, süre/bellek ve olumsuz
sonuçları raporla. Spectral’ın kazanmasını varsayma; desteklenen çözümü seç.

Bütün önerilen GPU fazını tahmin et; >60dk ise başlamadan tek kapsamlı onay al.
Önce CPU ve GPU smoke, sonra bütçeli geliştirme ve dondurulmuş aday doğrulaması.
Kontrattaki ayrı Drive/yerel köklerini kullan; batch/corruption/model bazında
kaydet ve SHA doğrula. Kendi branch’inde knowledge/sonuçları ayrı commit’lerle
push et. Peer’in canlı runtime’ına erişerek kod veya sonuç alma; yalnızca kabul
edilmiş immutable bundle/commit’i inceleyerek yeniden kullan. Runtime kapatmak
için kullanıcıdan açık tamamlanma ve kapatma yetkisi gerekir.
```
