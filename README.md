# YAZLAB2 - Açıklanabilir Zaman Serisi Analizi

## 1. Proje Amacı
Bu projede zaman serisi verileri üzerinde **derin öğrenme tabanlı black-box modeller** ile **olasılıksal otomata tabanlı açıklanabilir modellerin** performansları karşılaştırılacaktır. Amaç, yalnızca yüksek doğruluk elde etmek değil, aynı zamanda model çıktılarının yorumlanabilirliğini incelemektir.

---

## 2. Proje Kapsamı
Proje kapsamında zaman serisi verileri üzerinde **anomali tespiti** yapılacaktır.  

Kullanılacak yöntemler:
- LSTM, GRU, 1D-CNN
- PAA, SAX, Sliding Window
- Levenshtein Distance
- Açıklanabilirlik analizi

---

## 3. Ekip
- Yasemin ATİŞ-231307023
- Şenay CENGİZ-231307027  

---

## 4. Veri Seti Analizi

Bu projede **WADI** ve **BATADAL** veri setleri kullanılmıştır.

---

### 4.1 BATADAL Veri Seti

**Dataset03 (Train):**
- Satır sayısı: 8761  
- Kolon sayısı: 45  
- Anomaly oranı: %0  

**Dataset04 (Test):**
- Satır sayısı: 4177  
- Kolon sayısı: 45  
- Anomaly oranı:
  - %94.75 normal  
  - %5.24 anomaly  

**Genel Özellikler:**
- Eksik veri bulunmamaktadır  
- Train veri tamamen normaldir  
- Test veri anomaly içerir  
- Veri seti düşük boyutlu ve düzenli yapıdadır  

---

### 4.2 WADI Veri Seti

**Train Verisi:**
- Satır sayısı: 784571  
- Kolon sayısı: 130  
- Label sütunu bulunmamaktadır  

**Attack Verisi:**
- Satır sayısı: 172803  
- Kolon sayısı: 131  
- Label sütunu:
  - `Attack LABLE (1:No Attack, -1:Attack)`

**Anomaly oranı:**
- %94.22 normal  
- %5.77 anomaly  

**Genel Özellikler:**
- Veri seti yüksek boyutlu ve büyük ölçeklidir  
- Bazı kolonlarda eksik veri bulunmaktadır  
- Anomaly yalnızca attack veri setinde yer almaktadır  

**Önemli Not:**
WADI attack veri setinde kolon isimleri düzgün okunmadığı için veri yükleme aşamasında ilk satır kolon isimleri olarak atanmış ve veri seti yeniden düzenlenmiştir.

---

### 4.3 Veri Setleri Karşılaştırması

| Özellik | BATADAL | WADI |
|--------|--------|------|
| Feature sayısı | 45 | 130 |
| Veri boyutu | Küçük | Büyük |
| Eksik veri | Yok | Var |
| Train anomaly | Yok | Yok |
| Test anomaly | Var | Attack datasetinde |
| Veri karmaşıklığı | Düşük | Yüksek |

---

### 4.4 Sayısal Karşılaştırma

- WADI veri seti, BATADAL veri setine göre yaklaşık **18 kat daha fazla veri içermektedir**.  
- Feature sayısı açısından WADI, BATADAL’a göre yaklaşık **3 kat daha fazladır** (130 vs 45).  
- Her iki veri setinde anomaly oranı benzer olup yaklaşık **%5 civarındadır**.  
- BATADAL veri setinde eksik veri bulunmazken, WADI veri setinde eksik değerler bulunmaktadır.  

---

### 4.5 Analitik Değerlendirme

- WADI veri setinin yüksek boyutlu ve büyük ölçekli olması, modelin daha fazla örüntü öğrenmesini sağlayabilir; ancak bu durum model karmaşıklığını artırarak eğitim süresini uzatabilir.  
- BATADAL veri seti daha küçük ve temiz olduğu için model geliştirme sürecinde hızlı denemeler yapılmasına olanak sağlar.  
- WADI veri setindeki yüksek feature sayısı, özellikle PCA gibi boyut indirgeme tekniklerinin kullanılmasını gerekli kılmaktadır.  
- Her iki veri setinde train verisinin tamamen normal olması, anomaly detection probleminin gerçekçi bir şekilde ele alındığını göstermektedir.  
- Test verilerinde yaklaşık %5 oranında anomaly bulunması, modellerin performansını değerlendirmek için dengeli bir senaryo sunmaktadır.  
- Veri setleri arasındaki boyut ve karmaşıklık farkı, modellerin genellenebilirlik performansını karşılaştırmak açısından önemli bir avantaj sağlamaktadır.
