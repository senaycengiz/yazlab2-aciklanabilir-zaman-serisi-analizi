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

Bu projede BATADAL Training Dataset 2 ve SKAB veri setleri kullanılmıştır.

---

## 4.1 BATADAL Veri Seti

Bu projede BATADAL veri seti içerisinden yalnızca hocanın proje isterlerinde belirtilen Training Dataset 2 kullanılmıştır.

Genel Bilgiler

* Satır sayısı: 4177
* Kolon sayısı: 45
* Model feature sayısı: 43
* Label sütunu: ATT_FLAG

Anomaly Dağılımı

* %94.75 normal
* %5.24 anomaly

Genel Özellikler

* Veri setinde eksik veri bulunmamaktadır.
* Veri seti düzenli ve düşük boyutlu yapıdadır.
* Anomaly bilgisi doğrudan ATT_FLAG sütununda tutulmaktadır.
* Sensör verileri farklı ölçeklerde olduğu için normalizasyon işlemi uygulanmıştır.
* PCA ile veri tek boyutlu zaman serisine indirgenmiştir.

---

## 4.2 SKAB Veri Seti

Bu projede SKAB veri seti içerisinden yalnızca:

* valve1
* valve2

klasörleri kullanılmıştır.

Bu klasörlerde yer alan tüm .csv dosyaları birleştirilerek tek bir veri seti oluşturulmuştur.

Genel Bilgiler

* Satır sayısı: 22474
* Kolon sayısı: 13
* Model feature sayısı: 8
* Label sütunu: anomaly
* Source file sayısı: 16

Kullanılan Kolonlar

SKAB veri setinde aşağıdaki temel sensör değişkenleri kullanılmıştır:

* Accelerometer1RMS
* Accelerometer2RMS
* Current
* Pressure
* Temperature
* Thermocouple
* Voltage
* Volume Flow RateRMS

Ek olarak veri takibi ve grup bazlı bölme işlemleri için:

* source_group
* source_file

kolonları eklenmiştir.

Anomaly Dağılımı

* %65.17 normal
* %34.82 anomaly

Genel Özellikler

* Veri setinde eksik veri bulunmamaktadır.
* Veri seti farklı sensörlerden oluşan çok değişkenli zaman serisi yapısındadır.
* Anomaly bilgisi anomaly sütununda tutulmaktadır.
* Veri seti birden fazla csv dosyasından oluştuğu için dosya bazlı veri bölme stratejisi uygulanmıştır.
* Aynı source file’ın hem train hem test verisinde bulunması engellenmiştir.

---

## 4.3 Veri Setleri Karşılaştırması

Özellik	BATADAL	SKAB
Feature sayısı	43	8
Veri boyutu	Küçük	Orta
Eksik veri	Yok	Yok
Label sütunu	ATT_FLAG	anomaly
Veri yapısı	Tek csv	Çoklu csv
Veri karmaşıklığı	Düşük	Orta
Bölme yöntemi	Sıralı split	GroupKFold

---

## 4.4 Sayısal Karşılaştırma

* SKAB veri seti, BATADAL veri setine göre daha fazla satır içermektedir.
* BATADAL veri seti daha fazla feature içermektedir.
* BATADAL veri setinde anomaly oranı yaklaşık %5 seviyesindedir.
* SKAB veri setinde anomaly oranı yaklaşık %35 seviyesindedir.
* Her iki veri setinde de eksik veri bulunmamaktadır.
* Veri setlerinin yapısal farklılıkları nedeniyle veri bölme stratejileri veri setine özel uygulanmıştır.

---

## 4.5 Analitik Değerlendirme

BATADAL veri seti daha küçük ve düzenli bir yapıya sahip olduğu için model geliştirme sürecinde hızlı deneyler yapılmasına olanak sağlamaktadır.

SKAB veri seti ise çoklu csv yapısı ve farklı sensör değişkenleri nedeniyle daha karmaşık bir yapı sunmaktadır. Bu nedenle veri bölme aşamasında klasik train/test ayrımı yerine GroupKFold yaklaşımı kullanılmıştır.

SKAB veri setinde aynı source file’ın hem eğitim hem test verisine düşmesi engellenerek veri sızıntısı (data leakage) riski azaltılmıştır.

Her iki veri setinde de sensör değişkenlerinin farklı ölçeklerde olması nedeniyle normalizasyon işlemi uygulanmıştır.

PCA ile veriler tek boyutlu zaman serisi temsiline indirgenmiş ve ilerleyen aşamalarda uygulanacak olan:

* Sliding Window
* PAA
* SAX
* State Transition
* Explainability

adımları için uygun veri yapısı elde edilmiştir.

---
## 5. Veri Ön İşleme (Preprocessing)

Zaman serisi verileri üzerinde modelleme yapılmadan önce veri ön işleme adımları uygulanmıştır.

---

## 5.1 Veri Bölme

BATADAL Veri Seti

BATADAL veri seti zaman sırası korunacak şekilde aşağıdaki oranlarda bölünmüştür:

* Train: %60
* Validation: %20
* Test: %20

Shuffle işlemi uygulanmamış ve veriler kronolojik sıraya göre ayrılmıştır.

BATADAL Split Sonuçları

* Train: 2506 satır
* Validation: 835 satır
* Test: 836 satır

---

SKAB Veri Seti

SKAB veri seti çoklu csv dosyalarından oluştuğu için klasik train/test bölmesi yerine GroupKFold yaklaşımı kullanılmıştır.

Bu yöntemde:

* source_file sütunu grup değişkeni olarak kullanılmıştır.
* Aynı csv dosyasının hem train hem test verisinde bulunması engellenmiştir.
* Veri sızıntısı oluşmaması hedeflenmiştir.

SKAB Fold Sonuçları

Fold 1

* Train: 17962 satır
* Test: 4512 satır

Fold 2

* Train: 17981 satır
* Test: 4493 satır

Fold 3

* Train: 17982 satır
* Test: 4492 satır

Fold 4

* Train: 18040 satır
* Test: 4434 satır

Fold 5

* Train: 17931 satır
* Test: 4543 satır

---

## 5.2 Veri Normalizasyonu

Bu aşamada veri setleri üzerinde normalizasyon işlemi uygulanmıştır. Amaç, farklı ölçeklerde bulunan özelliklerin aynı referans aralığına getirilerek modelin daha sağlıklı öğrenmesini sağlamaktır.

Özellikle sensör verilerinin bulunduğu BATADAL ve SKAB veri setlerinde değişkenler arasında ciddi ölçek farkları bulunduğundan bu adım kritik öneme sahiptir.

---

Kullanılan Yöntem

Normalizasyon işlemi için StandardScaler yöntemi tercih edilmiştir.

Bu yöntem ile her özellik için:

* Ortalama (mean) = 0
* Standart sapma (std) = 1

olacak şekilde dönüşüm yapılmaktadır.

---

Veri Sızıntısını Önleme

Bu projede veri sızıntısının (data leakage) önlenmesi temel kurallardan biridir.

Bu nedenle:

* Scaler yalnızca train verisi üzerinde fit edilmiştir.
* Validation ve test verilerine fit işlemi uygulanmamıştır.
* Aynı scaler kullanılarak yalnızca transform işlemi yapılmıştır.

Bu yaklaşım sayesinde modelin test verisinden önceden bilgi öğrenmesi engellenmiştir.

---

Uygulama Akışı

Normalizasyon işlemi aşağıdaki sıraya göre gerçekleştirilmiştir:

1. Train veri seti alınır
2. Scaler train verisi üzerinde fit edilir
3. Train verisi transform edilir
4. Aynı scaler validation ve test verilerine uygulanır

---

Elde Edilen Çıktılar

* Normalize edilmiş train/test/validation veri setleri oluşturulmuştur.
* Kullanılan scaler modelleri .pkl formatında kaydedilmiştir.
* Tüm veri setleri aynı ölçeğe getirilmiştir.
* Her fold için ayrı scaler modeli oluşturulmuştur.

---

## 6. Boyut İndirgeme (PCA)

Normalizasyon işleminden sonra veri setleri üzerinde boyut indirgeme işlemi uygulanmıştır.

Bu amaçla Principal Component Analysis (PCA) yöntemi kullanılmıştır.

---

Amaç

Bu adımın temel amaçları:

* Çok boyutlu veriyi sade hale getirmek
* Gürültüyü azaltmak
* Hesaplama maliyetini düşürmek
* Zaman serisini tek boyutlu temsil etmek
* SAX ve state transition adımları için uygun yapı oluşturmak

---

Kullanılan Yöntem

PCA yöntemi kullanılarak veri tek bileşene indirgenmiştir.

* Sadece birinci ana bileşen (PC1) kullanılmıştır.
* Her zaman adımı tek bir değer ile temsil edilmiştir.

---

Veri Sızıntısını Önleme

Normalizasyonda olduğu gibi PCA aşamasında da veri sızıntısını önlemek için:

* PCA modeli yalnızca train verisi ile fit edilmiştir.
* Validation ve test verilerine fit işlemi uygulanmamıştır.
* Aynı PCA modeli ile yalnızca transform işlemi yapılmıştır.

---

Uygulama Akışı

1. Normalize edilmiş train verisi alınır
2. PCA modeli train verisi ile fit edilir
3. Train verisi transform edilir
4. Aynı PCA modeli validation/test verilerine uygulanır

---

Çıktı Özellikleri

Veri setleri çok boyutlu yapıdan tek boyutlu yapıya indirgenmiştir.

Her örnek için temel çıktı:

* PC1

şeklinde elde edilmiştir.

Metadata kolonları ayrıca korunmuştur.

---

PCA Sonuçları

BATADAL

* Train: (2506, 3)
* Validation: (835, 3)
* Test: (836, 3)

Korunan kolonlar:

* PC1
* DATETIME
* ATT_FLAG

---

SKAB

Örnek fold çıktıları:

Fold 1

* Train: (17962, 6)
* Test: (4512, 6)

Fold 2

* Train: (17981, 6)
* Test: (4493, 6)

Fold 3

* Train: (17982, 6)
* Test: (4492, 6)

Fold 4

* Train: (18040, 6)
* Test: (4434, 6)

Fold 5

* Train: (17931, 6)
* Test: (4543, 6)

Korunan kolonlar:

* PC1
* datetime
* anomaly
* changepoint
* source_group
* source_file

---

## 6.1 PCA ve Normalizasyon Sonuçlarının Doğrulanması

Uygulanan normalizasyon ve PCA işlemlerinin doğruluğunu garanti altına almak amacıyla çeşitli kontroller gerçekleştirilmiştir.

---

Test Edilen Kriterler

Bu kapsamda aşağıdaki kontroller yapılmıştır:

* Normalizasyon sonrası verilerin aynı ölçeğe getirildiği doğrulanmıştır.
* PCA sonrası veri boyutunun tek bileşene indirildiği kontrol edilmiştir.
* PCA modelinin yalnızca train verisi ile fit edildiği doğrulanmıştır.
* Validation ve test verilerinin yalnızca transform işlemi ile dönüştürüldüğü test edilmiştir.
* GroupKFold yapısında aynı source file’ın train ve test verisinde tekrar etmediği kontrol edilmiştir.

---

Test Sonuçları

Yapılan kontroller sonucunda:

* Veri sızıntısının oluşmadığı doğrulanmıştır.
* PCA çıktılarının tutarlı olduğu gözlemlenmiştir.
* Fold yapısının doğru çalıştığı doğrulanmıştır.
* Train/test ayrımının source file bazlı gerçekleştirildiği doğrulanmıştır.

---

Genel Değerlendirme

Uygulanan preprocessing süreci sonucunda veri modelleme için uygun hale getirilmiştir.

Normalizasyon ve PCA işlemlerinin yalnızca train verisi üzerinden öğrenilmesi sayesinde veri sızıntısı engellenmiştir.

PCA ile veriler tek boyutlu zaman serisi temsiline dönüştürülmüş ve explainable state transition yapısı için gerekli temel veri yapısı oluşturulmuştur.


---

## 6.2 Açıklanabilirlik Tabanlı Sembolik Dönüşümler

Bu aşamada PCA sonrası elde edilen tek boyutlu zaman serisi verileri üzerinde açıklanabilir otomata yapısının oluşturulabilmesi amacıyla sırasıyla:

* PAA
* SAX
* Sliding Window
* State Transition

dönüşümleri uygulanmıştır.

Bu dönüşümlerin temel amacı sürekli zaman serisi verilerini sembolik ve yorumlanabilir yapılara dönüştürerek olasılıksal otomata modeli için uygun veri yapısı oluşturmaktır.

---

### 6.2.1 PAA (Piecewise Aggregate Approximation)

PCA sonrası elde edilen PC1 zaman serisi verileri üzerinde PAA dönüşümü uygulanmıştır.

Bu yöntemde zaman serisi belirli segmentlere ayrılmış ve her segmentin ortalama değeri alınarak veri boyutu azaltılmıştır.

Bu yaklaşım sayesinde:

* Gürültü azaltılmıştır.
* Veri daha sade hale getirilmiştir.
* SAX dönüşümü için uygun giriş yapısı oluşturulmuştur.

#### Uygulama Yapısı

Projede:

* PAA işlemi PCA sonrası elde edilen PC1 verisi üzerinde uygulanmıştır.
* Segment yapıları oluşturularak zaman serisi daha kısa ve temsil edilebilir hale getirilmiştir.
* Veri sızıntısını önlemek amacıyla dönüşüm sürecinde train/fold-train yapısı korunmuştur.
* SKAB veri setinde fold bazlı yapı, BATADAL veri setinde ise train/validation/test yapısı dikkate alınmıştır.

#### Kullanılan Parametreler

```python
PAA_SEGMENTS = 4
```

#### Örnek Dönüşüm

```text
Orijinal zaman serisi:
[1.2, 1.4, 1.5, 1.3, 0.9, 0.8]

PAA çıktısı:
[1.35, 0.85]
```

#### Oluşturulan Çıktılar

```text
data/processed/paa/
```

Bu klasör altında:

* SKAB fold bazlı PAA çıktıları
* BATADAL train/validation/test PAA çıktıları

saklanmıştır.

---

### 6.2.2 SAX (Symbolic Aggregate approXimation)

PAA dönüşümünden elde edilen sayısal segment değerleri SAX yöntemi ile sembolik yapıya dönüştürülmüştür.

Bu aşamada:

* Sayısal değerler belirli aralıklara bölünmüştür.
* Her aralık bir sembol ile temsil edilmiştir.
* Sürekli zaman serileri sembolik dizilere dönüştürülmüştür.

#### Kullanılan Parametreler

```python
ALPHABET_SIZE = 3
SAX_ALPHABET = ["a", "b", "c"]
```

Alphabet size parametresi merkezi config yapısına bağlanmıştır.

#### Veri Sızıntısını Önleme

SAX dönüşümünde:

* SAX sözlüğü yalnızca train/fold-train verisi kullanılarak oluşturulmuştur.
* Validation ve test verileri aynı train sözlüğü kullanılarak dönüştürülmüştür.
* Böylece test verisinden önceden bilgi öğrenilmesi engellenmiştir.

#### Örnek Dönüşüm

```text
PAA:
[1.35, 0.82, -0.15]

SAX:
[c, b, a]
```

#### Oluşturulan Çıktılar

```text
data/processed/sax/
```

Bu klasör altında:

* SAX sembolik veri çıktıları
* Fold bazlı SAX sözlükleri

saklanmıştır.

---

### 6.2.3 Sliding Window ve Pattern Üretimi

SAX dönüşümünden elde edilen sembolik veriler üzerinde sliding window yöntemi uygulanarak pattern listeleri oluşturulmuştur.

Bu yaklaşım sayesinde:

* Tek semboller yerine sembol dizileri oluşturulmuştur.
* Zaman serisi içerisindeki lokal davranış örüntüleri çıkarılmıştır.
* Automata state yapısı için anlamlı patternler elde edilmiştir.

#### Kullanılan Parametreler

```python
WINDOW_SIZE = 4
WINDOW_SIZE_OPTIONS = [3, 4, 5, 6]
```

Window size parametresi merkezi config yapısına bağlanmıştır.

#### Örnek Pattern Üretimi

```text
SAX dizisi:
[a, b, c, c, a]

Window size = 3

Üretilen patternler:
abc
bcc
cca
```

#### Pattern Analizi

Bu aşamada:

* Toplam pattern sayıları
* Benzersiz pattern sayıları
* Window size değişiminin pattern yoğunluğuna etkisi

analiz edilmiştir.

Window size değeri arttıkça üretilen toplam pattern sayısının azaldığı gözlemlenmiştir.

#### Oluşturulan Çıktılar

```text
data/processed/patterns/
results/pattern_count_analysis.csv
```

Bu çıktılar altında:

* Pattern listeleri
* Pattern yoğunluk analizleri
* Window size bazlı sonuçlar

saklanmıştır.

---

### 6.2.4 Automata State ve Transition Yapısı

Sliding window sonucunda elde edilen patternler automata state yapısına dönüştürülmüştür.

Bu aşamada:

* Her benzersiz pattern bir state olarak tanımlanmıştır.
* State yapıları `q0`, `q1`, `q2` formatında isimlendirilmiştir.
* Pattern sırasına göre state transition ilişkileri oluşturulmuştur.

#### State Oluşturma Mantığı

```text
Pattern:
abca

State:
q1
```

Her benzersiz pattern yalnızca bir state ile temsil edilmektedir.

#### Transition Yapısı

Pattern akışına göre state geçişleri oluşturulmuştur.

```text
abca → bcac → caca

Transition:
q1 → q5
q5 → q2
```

#### Transition Matrix

State transition geçiş sayıları hesaplanarak transition matrix yapısı oluşturulmuştur.

Bu yapıda:

* Satırlar başlangıç state’ini
* Sütunlar hedef state’i
* Hücre değerleri geçiş sayılarını

temsil etmektedir.

Bu yaklaşım sayesinde zaman serisinin davranış akışı yorumlanabilir hale getirilmiştir.

#### Oluşturulan Çıktılar

```text
data/processed/automata/
results/transition_matrices/
```

Bu klasörlerde:

* State listeleri
* Transition listeleri
* Transition matrix çıktıları

saklanmıştır.

---

### 6.2.5 Doğrulama ve Test Süreci

PAA, SAX, Sliding Window ve Automata dönüşümleri için birim testleri gerçekleştirilmiştir.

Bu testlerde:

* PAA segment yapıları
* SAX sembolik dönüşümleri
* Sliding window pattern üretimi
* Window size davranışı
* State oluşturma işlemleri
* Transition hesaplamaları
* Transition matrix doğruluğu

kontrol edilmiştir.

#### Kullanılan Test Yapısı

Projede pytest frameworkü kullanılmıştır.

Örnek test komutları:

```bash
python3 -m pytest tests
python3 -m pytest tests/test_sliding_window.py
python3 -m pytest tests/test_transition_analysis.py
```

---

### Genel Değerlendirme

Bu aşama sonucunda:

* Sürekli zaman serileri sembolik yapılara dönüştürülmüştür.
* Açıklanabilir automata state yapısı oluşturulmuştur.
* State transition ilişkileri çıkarılmıştır.
* Olasılıksal automata modelinin temel veri yapısı hazırlanmıştır.

Bu yapı ilerleyen aşamalarda:

* Unseen pattern kontrolü
* Transition probability hesaplamaları
* Explainability analizleri
* State davranış yorumlamaları

için kullanılacaktır.

---

# 7. Derin Öğrenme Model Eğitimleri

Bu aşamada PCA sonrası elde edilen tek boyutlu zaman serisi verileri üzerinde derin öğrenme tabanlı model eğitimleri gerçekleştirilmiştir.

Projede:

* LSTM
* GRU

modelleri kullanılmıştır.

Model eğitimleri PyTorch frameworkü kullanılarak gerçekleştirilmiştir.

---

## 7.1 Ortak Training Pipeline Yapısı

Derin öğrenme modellerinin eğitim süreçlerini standart hale getirmek amacıyla ortak bir training pipeline oluşturulmuştur.

Bu yapı sayesinde:

* Eğitim süreçleri merkezi hale getirilmiştir.
* Model bazlı tekrar eden kod yapısı azaltılmıştır.
* Validation kontrolü standartlaştırılmıştır.
* Early stopping mekanizması ortak yapı üzerinden yönetilmiştir.

---

### Kullanılan Eğitim Parametreleri

| Parametre | Değer |
|---|---|
| Epoch Sayısı | 50 |
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Early Stopping Patience | 5 |

---

### Kullanılan Yapılar

* BCEWithLogitsLoss
* Adam Optimizer
* Early Stopping
* Validation Loss Takibi
* PyTorch DataLoader

---

### Oluşturulan Dosyalar

```text
src/train.py
src/train_gru.py
src/train_lstm.py
src/data_loader_torch.py
```

---

## 7.2 GRU Model Eğitimi

Bu aşamada GRU modeli PCA sonrası elde edilen zaman serisi verileri üzerinde eğitilmiştir.

---

### Model Özellikleri

* PyTorch tabanlı GRU yapısı kullanılmıştır.
* Girdi formatı:

```text
(batch_size, sequence_length, input_size)
```

şeklinde düzenlenmiştir.

* PCA sonrası tek boyutlu veri yapısına uygun hale getirilmiştir.

---

### SKAB Eğitimi

SKAB veri seti üzerinde 5-fold yapısı kullanılmıştır.

Bu süreçte:

* Her fold için ayrı eğitim gerçekleştirilmiştir.
* Fold bazlı train/test yapısı korunmuştur.
* Fold sonuçları CSV formatında kaydedilmiştir.

---

### BATADAL Eğitimi

BATADAL veri setleri üzerinde zaman sıralı eğitim uygulanmıştır.

Bu süreçte:

* Shuffle işlemi uygulanmamıştır.
* Train-validation ayrımı korunmuştur.
* Dataset03 ve Dataset04 üzerinde ayrı eğitim gerçekleştirilmiştir.

---

### Kaydedilen Çıktılar

```text
results/gru/
models/gru/
```

Bu klasörlerde:

* Fold sonuçları
* Eğitim sonuçları
* Eğitilmiş model ağırlıkları

saklanmıştır.

---

## 7.3 LSTM Model Eğitimi

Bu aşamada LSTM modeli PCA sonrası oluşturulan zaman serisi verileri üzerinde eğitilmiştir.

---

### Model Özellikleri

* PyTorch tabanlı LSTM yapısı kullanılmıştır.
* Tek boyutlu zaman serisi yapısına uygun giriş formatı oluşturulmuştur.
* Validation loss takibi ile eğitim süreci kontrol edilmiştir.

---

### SKAB Eğitimi

SKAB veri seti üzerinde:

* 5-fold cross validation yapısı kullanılmıştır.
* Her fold için ayrı model eğitimi gerçekleştirilmiştir.
* Fold sonuçları CSV formatında kaydedilmiştir.

---

### BATADAL Eğitimi

BATADAL veri setleri üzerinde:

* Zaman sıralı eğitim uygulanmıştır.
* Validation veri yapısı korunmuştur.
* Dataset03 ve Dataset04 üzerinde ayrı eğitim gerçekleştirilmiştir.

---

### Kaydedilen Çıktılar

```text
results/lstm/
models/lstm/
```

Bu klasörlerde:

* Fold sonuçları
* Eğitim çıktıları
* Eğitilmiş model ağırlıkları

saklanmıştır.

---

## 7.4 Eğitim Süreci Değerlendirmesi

Model eğitim sürecinde validation loss değerleri takip edilmiştir.

Overfitting riskini azaltmak amacıyla early stopping mekanizması kullanılmıştır.

SKAB veri setinde fold bazlı eğitim uygulanarak modelin farklı veri grupları üzerindeki davranışı incelenmiştir.

BATADAL veri setinde ise zaman sırası korunarak gerçek zaman serisi senaryosuna uygun eğitim gerçekleştirilmiştir.

Model ağırlıkları `.pt` formatında kaydedilmiş ve tekrar kullanılabilir hale getirilmiştir.

--- 

## 7.5 1D-CNN Model Eğitimi

Bu aşamada 1D-CNN modeli PCA sonrası elde edilen tek boyutlu zaman serisi verileri üzerinde eğitilmiştir.

---

### Model Özellikleri

* PyTorch tabanlı 1D-CNN mimarisi kullanılmıştır.
* Conv1D katmanları ile zaman serisi örüntüleri öğrenilmiştir.
* PCA sonrası tek boyutlu veri yapısına uygun giriş formatı oluşturulmuştur.
* Adaptive pooling yapısı kullanılarak sabit boyutlu çıktı elde edilmiştir.

---

### SKAB Eğitimi

SKAB veri seti üzerinde:

* 5-fold GroupKFold yapısı kullanılmıştır.
* Her fold için ayrı CNN modeli eğitilmiştir.
* Fold bazlı eğitim sonuçları CSV formatında kaydedilmiştir.
* Eğitim süreçleri training loss ve validation loss değerleri üzerinden takip edilmiştir.

---

### BATADAL Eğitimi

BATADAL veri seti üzerinde:

* Zaman sıralı eğitim yaklaşımı uygulanmıştır.
* Validation veri yapısı korunmuştur.
* CNN modeli PCA sonrası oluşturulan tek boyutlu veri üzerinde eğitilmiştir.

---

### Eğitim Takibi ve Loglama

CNN eğitim sürecinde:

* Epoch bazlı training loss kayıtları tutulmuştur.
* Validation loss değerleri otomatik loglanmıştır.
* Eğitim geçmişleri CSV formatında kaydedilmiştir.
* Loss grafikleri otomatik olarak oluşturulmuştur.

---

### Oluşturulan Çıktılar

```text
models/cnn/
results/cnn/
logs/cnn/
results/plots/cnn/
```

Bu klasörlerde:

* Eğitilmiş model ağırlıkları
* Fold bazlı sonuçlar
* Eğitim geçmişi logları
* Loss analiz grafikleri

saklanmıştır.

---

### Eğitim Süreci Değerlendirmesi

CNN modeli özellikle SKAB veri setinde fold bazlı eğitim yapısı üzerinde test edilmiştir.

Validation loss değerleri takip edilerek early stopping mekanizması uygulanmıştır.

Eğitim süreçlerinin görselleştirilmesi sayesinde:

* Overfitting davranışı
* Fold bazlı performans değişimleri
* Eğitim kararlılığı

analiz edilmiştir.

---

# 8. Model Performans Analizi ve Görselleştirme

Bu aşamada eğitilen derin öğrenme modellerinin performanslarını değerlendirmek amacıyla çeşitli metrik hesaplama ve görselleştirme işlemleri gerçekleştirilmiştir.

Bu süreçte:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC Curve
* Precision-Recall Curve

analizleri uygulanmıştır.

---

## 8.1 Performans Metrikleri

Model performanslarını yalnızca doğruluk (accuracy) üzerinden değerlendirmek anomaly detection problemleri için yeterli değildir.

Bu nedenle aşağıdaki metrikler birlikte kullanılmıştır:

| Metrik | Açıklama |
|---|---|
| Accuracy | Toplam doğru tahmin oranı |
| Precision | Anomaly olarak tahmin edilen örneklerin doğruluk oranı |
| Recall | Gerçek anomalilerin yakalanma oranı |
| F1-Score | Precision ve recall değerlerinin dengeli ortalaması |

---

### Kullanılan Yapılar

Bu aşamada aşağıdaki sklearn metrik yapıları kullanılmıştır:

```python
accuracy_score
precision_score
recall_score
f1_score
```

---

### Oluşturulan Dosya

```text
src/evaluation/metrics.py
```

Bu modül içerisinde:

* Veri seti bazlı metrik hesaplama
* Fold bazlı değerlendirme
* Ortalama ve standart sapma hesaplama
* Zaman sıralı test çıktıları

işlemleri gerçekleştirilmiştir.

---

## 8.2 SKAB Fold Analizi

SKAB veri setinde GroupKFold yapısı kullanıldığı için her fold üzerinde ayrı performans değerlendirmesi yapılmıştır.

Bu süreçte:

* Her fold için accuracy hesaplanmıştır.
* Precision, recall ve F1-score değerleri çıkarılmıştır.
* Fold sonuçlarının ortalaması alınmıştır.
* Standart sapma hesaplanarak model kararlılığı incelenmiştir.

Bu yaklaşım sayesinde modelin farklı veri grupları üzerindeki davranışı daha güvenilir şekilde analiz edilmiştir.

---

## 8.3 BATADAL Zaman Sıralı Test Analizi

BATADAL veri setinde zaman sırası korunarak test değerlendirmesi yapılmıştır.

Bu süreçte:

* Shuffle işlemi uygulanmamıştır.
* Test sonuçları zaman sırasına göre kaydedilmiştir.
* Modelin zaman serisi üzerindeki anomaly detection davranışı analiz edilmiştir.

---

## 8.4 Confusion Matrix Analizi

Model tahminlerinin detaylı şekilde incelenebilmesi amacıyla confusion matrix görselleştirmeleri oluşturulmuştur.

Bu grafikler sayesinde:

* True Positive
* True Negative
* False Positive
* False Negative

değerleri analiz edilmiştir.

---

### Oluşturulan Dosya

```text
src/evaluation/visualization.py
```

Bu modül içerisinde:

* Confusion Matrix
* ROC Curve
* Precision-Recall Curve

grafikleri otomatik olarak oluşturulmaktadır.

---

## 8.5 ROC Curve Analizi

ROC eğrisi kullanılarak modellerin anomaly ve normal veri ayrım performansı incelenmiştir.

ROC eğrisi sayesinde:

* True Positive Rate
* False Positive Rate

arasındaki ilişki analiz edilmiştir.

Yüksek AUC değerine sahip modellerin daha başarılı ayrım yaptığı gözlemlenmiştir.

---

## 8.6 Precision-Recall Eğrisi

Anomaly detection problemlerinde veri dengesizliği önemli bir problem olduğu için precision-recall eğrileri ayrıca değerlendirilmiştir.

Bu analiz sayesinde:

* Precision-recall dengesi
* Düşük anomaly oranlarında model başarısı
* False positive davranışı

incelenmiştir.

---

## 8.7 Grafik Çıktıları

Üretilen görseller aşağıdaki klasörlerde saklanmıştır:

```text
results/figures/
results/plots/
```

Bu klasörlerde:

* Confusion matrix görselleri
* ROC curve grafikleri
* Precision-recall grafikleri
* Eğitim loss grafikleri

bulunmaktadır.

---

# 9. Model Karşılaştırma ve Analitik Değerlendirme

Bu aşamada:

* LSTM
* GRU
* 1D-CNN

modellerinin performansları karşılaştırılmıştır.

Karşılaştırmalar hem:

* SKAB
* BATADAL

veri setleri üzerinde gerçekleştirilmiştir.

---

## 9.1 Model Karşılaştırma Süreci

Bu aşamada modeller:

* Accuracy
* Precision
* Recall
* F1-score
* Validation loss

değerleri üzerinden analiz edilmiştir.

Karşılaştırma sonuçları tablo halinde kaydedilmiştir.

---

### Oluşturulan Dosya

```text
src/evaluation/model_comparison.py
```

Bu modül içerisinde:

* Model sonuçlarını birleştirme
* Veri seti bazlı karşılaştırma
* Fold sonuçlarını analiz etme
* Overfitting değerlendirmesi

işlemleri gerçekleştirilmiştir.

---

## 9.2 SKAB Veri Seti Değerlendirmesi

SKAB veri setinde:

* Fold bazlı performans değişimleri gözlemlenmiştir.
* Veri grupları arasındaki farklılıkların model sonuçlarını etkilediği görülmüştür.
* CNN modeli bazı foldlarda daha stabil sonuçlar üretmiştir.
* GRU ve LSTM modelleri uzun zaman bağımlılıklarını öğrenmede başarılı sonuçlar göstermiştir.

---

## 9.3 BATADAL Veri Seti Değerlendirmesi

BATADAL veri setinde:

* Veri yapısının daha düzenli olması nedeniyle modeller daha stabil sonuçlar üretmiştir.
* Anomaly oranının düşük olması precision ve recall dengesini önemli hale getirmiştir.
* Zaman sıralı yapı sayesinde gerçek anomaly detection senaryosuna yakın değerlendirme yapılmıştır.

---

## 9.4 Overfitting Analizi

Model eğitim süreçlerinde train ve validation sonuçları karşılaştırılmıştır.

Bu analiz sonucunda:

* Validation loss takibi yapılmıştır.
* Early stopping mekanizması uygulanmıştır.
* Bazı foldlarda overfitting eğilimleri gözlemlenmiştir.
* CNN modelinin bazı veri gruplarında daha kararlı öğrenme gerçekleştirdiği görülmüştür.

---

## 9.5 Genel Değerlendirme

Gerçekleştirilen analizler sonucunda:

* Derin öğrenme modellerinin anomaly detection performansları veri setine göre değişiklik göstermiştir.
* SKAB veri seti daha karmaşık yapısı nedeniyle modeller için daha zorlayıcı olmuştur.
* BATADAL veri setinde modeller daha stabil sonuçlar üretmiştir.
* Precision, recall ve F1-score metriklerinin anomaly detection problemlerinde accuracy değerinden daha kritik olduğu gözlemlenmiştir.
* Fold bazlı analizler model davranışlarını daha güvenilir şekilde incelemeye olanak sağlamıştır.

Bu aşama ile birlikte proje kapsamında derin öğrenme modellerinin performans analiz süreci tamamlanmıştır.
---

# 10. Olasılıksal Automata ve Transition Probability Analizi

Bu aşamada oluşturulan automata state yapıları kullanılarak durumlar arasındaki geçiş olasılıkları hesaplanmıştır.

Her state için geçiş olasılıkları frekans tabanlı yöntem kullanılarak elde edilmiştir.

Kullanılan formül:

```text
P(Si → Sj) =
Geçiş Sayısı / Toplam Çıkış Sayısı
```

Bu yöntem sayesinde her state'in hangi state'lere ne olasılıkla geçiş yaptığı belirlenmiştir.

---

## 10.1 Transition Probability Hesaplamaları

Transition count matrix yapıları kullanılarak probability matrixleri oluşturulmuştur.

Bu matrislerde:

* Satırlar başlangıç state'lerini
* Sütunlar hedef state'leri
* Hücre değerleri geçiş olasılıklarını

temsil etmektedir.

Üretilen bazı çıktılar:

| Veri Seti | Bölüm | State Sayısı | Matrix Boyutu |
|------------|------------|------------|------------|
| SKAB | Fold 1 Train | 34 | 34 × 34 |
| SKAB | Fold 4 Train | 34 | 34 × 34 |
| BATADAL Dataset04 | Train | 60 | 60 × 60 |

---

## 10.2 Smoothing Uygulaması

Eğitim verisinde gözlemlenmeyen state geçişlerinin sıfır olasılık üretmesi ilerleyen aşamalarda path probability hesaplamalarında problemlere neden olabilmektedir.

Bu nedenle Laplace Smoothing yöntemi uygulanmıştır.

Bu yaklaşım sayesinde:

* Zero probability problemi önlenmiştir.
* Görülmeyen geçişler için küçük olasılık değerleri üretilmiştir.
* Olasılık hesaplamalarının kararlılığı artırılmıştır.
* Unseen veri senaryoları için altyapı hazırlanmıştır.

---

## 10.3 Gerçek Çıktı Örnekleri

Transition probability matrisi içerisinden elde edilen bazı gerçek sonuçlar aşağıda verilmiştir:

```text
aaaa → aaaa : 0.8621
aaaa → aaab : 0.0678

aaab → aabb : 0.3098
aaab → aaba : 0.2394

abbb → bbbb : 0.4202
abbb → bbba : 0.1159
```

Bu sonuçlar bazı state geçişlerinin yüksek olasılıkla tekrarlandığını, bazı geçişlerin ise daha düşük frekansta gerçekleştiğini göstermektedir.

---

## 10.4 Test Sonuçları

Transition probability hesaplamalarının doğruluğu pytest kullanılarak doğrulanmıştır.

Çalıştırılan test:

```bash
pytest tests/test_transition_analysis.py
```

Sonuç:

```text
======================= 2 passed in 0.34s =======================
```

Bu sonuç transition count ve transition probability hesaplamalarının beklenen şekilde çalıştığını göstermektedir.

---

## 10.5 Oluşturulan Çıktılar

```text
results/transition_matrices/
results/transition_probabilities/
```

Bu klasörlerde:

* Transition count matrixleri
* Transition probability matrixleri
* Smoothing uygulanmış olasılık çıktıları

saklanmaktadır.

---

## 10.6 Genel Değerlendirme

Bu aşama sonucunda olasılıksal automata modelinin temel bileşenlerinden biri tamamlanmıştır.

State geçiş olasılıkları başarıyla hesaplanmış ve smoothing uygulanarak olasılık sistemi geliştirilmiştir.
