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

---

## 5. Veri Ön İşleme (Preprocessing)

Zaman serisi verileri üzerinde modelleme yapılmadan önce veri ön işleme adımları uygulanmıştır.

### 5.1 Veri Bölme

Veri setleri aşağıdaki oranlarda sıralı şekilde bölünmüştür:

•⁠  ⁠Train: %60  
•⁠  ⁠Validation: %20  
•⁠  ⁠Test: %20  

Zaman serisi yapısını korumak amacıyla *shuffle işlemi uygulanmamıştır* ve veriler kronolojik sıraya göre bölünmüştür.

---

### 5.2 Veri Normalizasyonu

Bu aşamada veri setleri üzerinde **normalizasyon işlemi** uygulanmıştır. Amaç, farklı ölçeklerde bulunan özelliklerin aynı referans aralığına getirilerek modelin daha sağlıklı öğrenmesini sağlamaktır. Özellikle sensör verilerinin bulunduğu WADI ve BATADAL veri setlerinde, değişkenler arasında ciddi ölçek farkları bulunduğundan bu adım kritik öneme sahiptir.


### Kullanılan Yöntem

Normalizasyon işlemi için **Standard Scaler** yöntemi tercih edilmiştir. Bu yöntem ile her özellik için:

- Ortalama (mean) = 0
- Standart sapma (std) = 1

olacak şekilde dönüşüm yapılmaktadır.

### Veri Sızıntısını Önleme 

Bu projede hocanın özellikle vurguladığı en önemli kurallardan biri **veri sızıntısının (data leakage) engellenmesidir**. Bu nedenle normalizasyon işlemi aşağıdaki şekilde gerçekleştirilmiştir:

- Scaler **yalnızca train verisi üzerinde fit edilmiştir**
- Validation ve test verilerine **fit işlemi yapılmamış**, sadece aynı scaler ile **transform uygulanmıştır**

Bu yaklaşım sayesinde modelin test verisi hakkında önceden bilgi edinmesi engellenmiştir.

### Uygulama Akışı

Normalizasyon işlemi aşağıdaki sıraya göre gerçekleştirilmiştir:

1. Train veri seti alınır
2. Scaler bu veri üzerinde eğitilir (`fit`)
3. Train verisi dönüştürülür (`transform`)
4. Aynı scaler kullanılarak validation ve test verileri dönüştürülür


### Elde Edilen Çıktılar

- Normalize edilmiş **train**, **validation** ve **test** veri setleri oluşturulmuştur
- Kullanılan scaler modeli tekrar kullanılabilmesi için `.pkl` formatında saklanmıştır
- Tüm veri setleri aynı ölçeğe getirildiği için model eğitimi için uygun hale getirilmiştir

---

## 6. Boyut İndirgeme (PCA)

Normalizasyon işleminden sonra veri setleri üzerinde **boyut indirgeme (dimensionality reduction)** işlemi uygulanmıştır. Bu amaçla **Principal Component Analysis (PCA)** yöntemi kullanılmıştır.

### Amaç

Bu adımın temel amaçları:

- Çok boyutlu veriyi daha sade hale getirmek
- Gürültüyü (noise) azaltmak
- Hesaplama maliyetini düşürmek
- Zaman serisini tek boyutlu temsil ederek sonraki adımlar (SAX, state transition) için uygun hale getirmek

### Kullanılan Yöntem

PCA yöntemi kullanılarak veri **tek bileşene indirgenmiştir**:

- Sadece **birinci ana bileşen (Principal Component 1 - PC1)** kullanılmıştır
- Böylece her zaman adımı tek bir değer ile temsil edilmiştir

### Veri Sızıntısını Önleme 

Normalizasyonda olduğu gibi PCA uygulamasında da veri sızıntısını önlemek için şu kurala uyulmuştur:

- PCA modeli **sadece train verisi ile fit edilmiştir**
- Validation ve test verilerine **fit işlemi yapılmadan**, aynı PCA modeli ile sadece **transform uygulanmıştır**

Bu sayede modelin test verisinden bilgi öğrenmesi engellenmiştir.

### Uygulama Akışı

PCA işlemi aşağıdaki şekilde gerçekleştirilmiştir:

1. Normalize edilmiş train verisi alınır
2. PCA modeli bu veri ile eğitilir (`fit`)
3. Train verisi dönüştürülür (`transform`)
4. Aynı PCA modeli kullanılarak validation ve test verileri dönüştürülür


### Çıktı Özellikleri

- Veri seti çok boyutlu yapıdan **tek boyutlu yapıya indirgenmiştir**
- Her örnek için çıktı boyutu: (n_samples, 1)

- Tüm özellikler yerine artık sadece **en yüksek varyansı temsil eden tek bileşen (PC1)** kullanılmaktadır

### Proje Açısından Önemi

Bu adım, projenin ilerleyen aşamalarında uygulanacak olan:

- **Sliding Window**
- **PAA (Piecewise Aggregate Approximation)**
- **SAX (Symbolic Aggregate Approximation)**
- **Durum (state) ve geçiş (transition) analizi**

için temel veri temsilini oluşturmaktadır.

Tek boyutlu zaman serisi elde edilerek, verinin **yorumlanabilir (explainable)** hale getilmesi yönünde önemli bir adım atılmıştır.

---

## 6.1 PCA ve Normalizasyon Sonuçlarının Doğrulanması

Uygulanan normalizasyon ve PCA işlemlerinin doğruluğunu garanti altına almak amacıyla testler gerçekleştirilmiştir.

### Test Edilen Kriterler

Bu kapsamda aşağıdaki kontroller yapılmıştır:

- Normalizasyon sonrası verilerin aynı ölçeğe getirildiği doğrulanmıştır
- PCA sonrası veri boyutunun **tek bileşene indirildiği (n_components = 1)** kontrol edilmiştir
- PCA modelinin **sadece train verisi ile fit edildiği** doğrulanmıştır
- Validation ve test verilerinin yalnızca **transform işlemi ile dönüştürüldüğü** test edilmiştir

### Test Sonuçları

Yapılan testler sonucunda:

- PCA çıktısının her veri seti için `(n_samples, 1)` boyutunda olduğu gözlemlenmiştir
- Train, validation ve test veri setlerinde dönüşümün tutarlı olduğu doğrulanmıştır
- Veri sızıntısının oluşmadığı garanti altına alınmıştır

### Örnek Çıktı

Aşağıda PCA sonrası veri boyutuna ait örnek bir çıktı verilmiştir:

```
Train shape after PCA      : (5256, 1)
Validation shape after PCA : (1752, 1)
Test shape after PCA       : (1753, 1)
```

---


### Genel Değerlendirme

Uygulanan normalizasyon ve PCA işlemleri sonucunda veri, modelleme süreci için uygun hale getirilmiştir. Dönüşümlerin yalnızca train verisi üzerinden öğrenilmesi veri sızıntısını engelleyerek sonuçların güvenilirliğini artırmıştır.

PCA ile veri tek boyuta indirgenmiş ve en yüksek varyansı temsil eden bileşen korunmuştur. Bu sayede veri daha sade hale gelmiş ve sonraki adımlar (SAX, sliding window, state transition) için uygun bir yapı elde edilmiştir.


