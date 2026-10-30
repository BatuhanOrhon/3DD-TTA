# Ortak tez dalı ve diğer agent'lar için entegrasyon promptu

Tarih: 2026-10-30. [User report/Decision]

`origin/thesis-integration`, kullanıcının `BatuhanOrhon/3DD-TTA` forkunda
**gerekli, tamamlanmış ve doğrulanmış** tez kodu ile knowledge kayıtlarını toplar.
Başlangıç tabanı `origin/gsd-multibackbone-study@f57b170` olarak seçilmiştir.
`main` ve `upstream` bu akışın yayın hedefi değildir. Agent'lar kendi worktree
ve dallarında çalışır; ortak dala bütün çalışma klasörünü taşımak zorunda değildir.

Kullanıcı, agent'ın gerekli gördüğü tamamlanmış kod/knowledge değişikliklerini
bu ortak dala entegre edip normal push yapmasına sürekli izin vermiştir. Her
seferinde yeniden merge/push onayı istenmez. Kodun testleri, import/config
bağımlılıkları ve bilimsel sonucun kompakt kanıtı da gerekli kapsama dahildir.
Başarısız/null sonuçlar silinmez. Geçici kurulum dosyaları, tekrarlı dump'lar,
büyük ham arşivler, cache ve ağırlıklar kendi arşivlerinde kalabilir; hangi
kapsamın neden dışarıda tutulduğu kaydedilir. Ayrıntılı kural:
[iterative thesis researcher skill](../.agents/skills/iterative-thesis-researcher/SKILL.md).

## Diğer agent'a verilecek prompt

```text
Bu projede tamamladığın çalışmanın yalnızca GEREKLİ GÖRDÜĞÜN kod ve knowledge
kayıtlarını, kullanıcının forkundaki origin/thesis-integration dalına entegre et
ve pushla. Bu işlem kullanıcı tarafından onaylandı; yeniden merge/push onayı isteme.

Önce kendi worktree/dal/HEAD/status'unu kaydet, origin'i fetch et ve güncel ortak
kuralı eski worktree'deki kopyaya güvenmeden oku:
  git show origin/thesis-integration:.agents/skills/iterative-thesis-researcher/SKILL.md
  git show origin/thesis-integration:knowledge/thesis_integration_workflow_20261030.md

Güncel ortak dalda zaten bulunan çalışmaları belirle. Yalnız gerekli, tamamlanmış
sonuçları seç: kodun bağımlılıkları/testleri/config'i ve araştırma notunun kompakt
kanıtı dahil olsun. Negatif/null/başarısız sonuçları da koru. Geçici dosyaları,
tekrarlı dump'ları, büyük ham ZIP/cache/weights dosyalarını sırf kendi dalında
bulunuyor diye taşıma. Hariç tuttukların kendi dalında/arşivinde korunsun.

Güncel origin/thesis-integration tabanlı temiz, izole bir entegrasyon dalında
çalış. Gerekirse seçici commit/cherry-pick hazırla; bütün dalı yalnız bütün
incoming diff gerekli ve doğrulanmışsa merge et. Başkasının knowledge/kodunu
eski dosya kopyasıyla ezme. Bilimsel anlamı ve tarihli geçmişi koruyarak
çatışmaları çöz; ilgili kontrolleri birleşmiş ağaçta çalıştır.

Son push öncesi remote'u tekrar kontrol et. Başka agent ilerlettiyse fetch edip
yeni değişiklikleri dahil et, etkilenen kontrolleri tekrarla ve normal push yap.
Force-push yapma. Hedef yalnız origin/thesis-integration; main/upstream'e yazma.

Kaynak dal/commit, dahil edilenler, hariç tutulanlar ve nedenleri, doğrulama
sonuçları ve remote entegrasyon SHA'sını raporla. Gerekli çalışma zaten
entegreyse bunu bildir; boş/tekrarlı merge üretme. Diğer worktree'leri, mevcut
kullanıcı değişikliklerini, agent dallarını ve runtime'ları koru. Bu görev yeni
GPU deneyi veya runtime kapatma yetkisi değildir.
```

## Temiz checkout bağımlılıkları

[Code/Run] İlk taban testi, Git'te eksik iki mevcut dosya nedeniyle4hata verdi.
`scanobjectnn_author_generator_107305f.py`, upstream107305f Git blob'undan
byte-identical olarak geri kondu (SHA3c18c0b3…cd8bb); severity1 profil runner'ı
ana çalışma alanındaki mevcut6899byte kopyadan değişmeden alındı
(SHA105d262c…67e59). Veri üretimi veya GPU profil çalışması başlatılmadı.
Bu iki bağımlılık eklendikten sonra temiz tabanın326CPU testi16.786s içinde geçti.
Agent B aktarımı ayrıca birleşmiş ağaçta doğrulanır.
