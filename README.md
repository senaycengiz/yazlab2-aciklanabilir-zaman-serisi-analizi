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

