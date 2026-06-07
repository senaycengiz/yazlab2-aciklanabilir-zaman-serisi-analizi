# YAZLAB2 - Açıklanabilir Zaman Serisi Analizi

---

## Proje Özeti

Bu projede zaman serileri üzerinde **anomali tespiti (Anomaly Detection)** gerçekleştirilmiştir.

Çalışmanın temel amacı, yüksek performans sağlayabilen ancak karar mekanizması doğrudan yorumlanamayan **Deep Learning tabanlı black-box modeller** ile karar sürecini açıklayabilen **Probabilistic Automata tabanlı açıklanabilir yaklaşımı** performans, dayanıklılık ve yorumlanabilirlik açısından karşılaştırmaktır.

Geliştirilen sistem yalnızca anomali tespiti yapmakla kalmamakta, aynı zamanda verdiği kararların nedenlerini state, pattern ve transition seviyesinde açıklayabilmektedir.

---

## Ekip

| Ad Soyad | Öğrenci No |
|-----------|-----------|
| Yasemin ATİŞ | 231307023 |
| Şenay CENGİZ | 231307027 |

---

# Proje Amacı

Bu çalışma kapsamında zaman serisi anomaly detection problemi üzerinde farklı modelleme yaklaşımları karşılaştırılmıştır.

Amaç yalnızca yüksek doğruluk elde etmek değil, aynı zamanda model kararlarının nasıl üretildiğini açıklayabilen yorumlanabilir bir yapı geliştirmektir.

Bu doğrultuda hem Deep Learning tabanlı yöntemler hem de açıklanabilir bir Probabilistic Automata yaklaşımı kullanılmıştır.

---

# Kullanılan Yaklaşımlar

## Deep Learning Modelleri

### LSTM

Long Short-Term Memory (LSTM), zaman serilerindeki uzun dönem bağımlılıkları öğrenebilen recurrent neural network yapısıdır. Geçmiş bilgileri hafızasında tutarak zamansal örüntüleri modelleyebilmektedir.

### GRU

Gated Recurrent Unit (GRU), LSTM'e benzer şekilde zamansal bağımlılıkları öğrenebilen ancak daha az parametre içerdiği için daha hızlı eğitilebilen bir recurrent neural network modelidir.

### 1D-CNN

1 Boyutlu Convolutional Neural Network (1D-CNN), zaman serileri üzerindeki yerel örüntüleri öğrenerek anomali tespiti gerçekleştiren convolutional tabanlı bir derin öğrenme modelidir.

---

## Açıklanabilir Yaklaşım

Probabilistic Automata modeli oluşturulurken aşağıdaki yöntemler kullanılmıştır:

- PCA (Principal Component Analysis)
- PAA (Piecewise Aggregate Approximation)
- SAX (Symbolic Aggregate approXimation)
- Sliding Window
- Probabilistic Automata
- Levenshtein Distance
- Explainability Analysis

Bu yapı sayesinde model yalnızca tahmin üretmekle kalmayıp, tahminin hangi pattern ve state geçişlerinden kaynaklandığını da açıklayabilmektedir.

---

# Kullanılan Veri Setleri

| Veri Seti | Açıklama |
|------------|------------|
| BATADAL Training Dataset 2 | Su dağıtım sistemi sensör verileri |
| SKAB | Endüstriyel sensör verileri |

---

# Veri Seti Özeti

| Özellik | BATADAL | SKAB |
|----------|----------:|----------:|
| Satır Sayısı | 4177 | 22474 |
| Kolon Sayısı | 45 | 13 |
| Feature Sayısı | 43 | 8 |
| Label | ATT_FLAG | anomaly |
| Anomaly Oranı | %5.24 | %34.82 |
| Veri Yapısı | Tek CSV | Çoklu CSV |
| Bölme Yöntemi | Zaman Sıralı Split | GroupKFold |

---

# Sistem Mimarisi

Proje modüler bir Python mimarisi üzerine geliştirilmiştir.

```text
src/
├── preprocessing_pipeline.py
├── prepare_skab_dataset.py
├── pca_transform.py
├── paa_transform.py
├── sax_transform.py
├── sliding_window.py
├── automata_builder.py
├── transition_analysis.py
├── automata_predict.py
├── unseen_mapper.py
├── train_lstm.py
├── train_gru.py
├── train_cnn.py
├── evaluate_normal_data.py
├── evaluate_noisy_deep_learning.py
├── evaluate_noisy_automata.py
├── evaluate_unseen_deep_learning.py
├── evaluate_unseen_automata.py
├── parameter_analysis.py
├── cross_dataset_analysis.py
├── cross_dataset_deep_learning.py
├── statistical_analysis.py
├── generate_explainability_outputs.py
└── generate_visualizations.py
```

Bu yapı sayesinde veri hazırlama, model eğitimi, deney çalıştırma, analiz ve görselleştirme süreçleri birbirinden bağımsız ve tekrar kullanılabilir şekilde tasarlanmıştır.

---

# Merkezi Konfigürasyon Yapısı

Projedeki tüm temel parametreler `config/config.py` dosyasında merkezi olarak tutulmaktadır. Bu yapı sayesinde deneylerin tekrarlanabilirliği sağlanmış ve tüm pipeline bileşenleri aynı parametreleri kullanmıştır.

Önemli konfigürasyon parametreleri aşağıda verilmiştir:

| Parametre | Değer |
|------------|------------:|
| RANDOM_SEED | 42 |
| WINDOW_SIZE | 4 |
| ALPHABET_SIZE | 3 |
| PAA_SEGMENTS | 4 |
| BATCH_SIZE | 32 |
| LEARNING_RATE | 0.001 |
| EPOCHS | 50 |
| EARLY_STOPPING_PATIENCE | 5 |
| UNSEEN_DISTANCE_THRESHOLD | 1 |

Parametre duyarlılık analizlerinde aşağıdaki seçenekler kullanılmıştır:

```python
WINDOW_SIZE_OPTIONS = [3, 4, 5, 6]
ALPHABET_SIZE_OPTIONS = [3, 4, 5, 6]
SEEDS = [42, 123, 2026, 7, 999]
```

Bu sayede parametre değişiklikleri tüm deney scriptleri tarafından ortak şekilde kullanılabilmiştir.

---

# Kurulum

Projeyi çalıştırmak için aşağıdaki adımlar uygulanmalıdır.

```bash
git clone https://github.com/senaycengiz/yazlab2-aciklanabilir-zaman-serisi-analizi.git

cd yazlab2-aciklanabilir-zaman-serisi-analizi

python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt
```

Veri setleri GitHub deposuna dahil edilmemiştir. SKAB ve BATADAL veri setlerinin ilgili klasörlere manuel olarak eklenmesi gerekmektedir.

---

# Kullanılan Teknolojiler

Proje geliştirme sürecinde aşağıdaki teknolojiler ve kütüphaneler kullanılmıştır.

| Teknoloji | Kullanım Amacı |
|------------|------------|
| Python | Temel geliştirme dili |
| Pandas | Veri işleme |
| NumPy | Sayısal hesaplamalar |
| Scikit-Learn | PCA, StandardScaler ve metrik hesaplamaları |
| PyTorch | LSTM, GRU ve 1D-CNN modelleri |
| Matplotlib | Grafik üretimi |
| Seaborn | Isı haritaları ve görselleştirme |
| NetworkX | Automata state diagram oluşturma |
| Levenshtein | Unseen pattern eşleme |
| Pytest | Birim testler |

---

# Veri Ön İşleme Süreci

Modelleme öncesinde hem Deep Learning hem de Probabilistic Automata tarafında veri ön işleme adımları uygulanmıştır.

## Ortak Ön İşleme Adımları

1. Veri setinin okunması
2. Feature ve label ayrımı
3. Train / Validation / Test bölünmesi
4. StandardScaler ile normalizasyon
5. Veri sızıntısını önlemek için yalnızca train üzerinde fit işlemi uygulanması

---

## Deep Learning Pipeline

```text
Raw Data
   │
   ▼
Train/Test Split
   │
   ▼
Normalization
   │
   ▼
Sequence Generation
   │
   ▼
LSTM / GRU / CNN
```

Deep Learning modelleri çok değişkenli sensör verileri üzerinde doğrudan çalışmaktadır.

---

## Probabilistic Automata Pipeline

```text
Raw Data
   │
   ▼
Normalization
   │
   ▼
PCA (PC1)
   │
   ▼
PAA
   │
   ▼
SAX
   │
   ▼
Sliding Window
   │
   ▼
Pattern Generation
   │
   ▼
State Generation
   │
   ▼
Transition Analysis
   │
   ▼
Prediction
   │
   ▼
Explainability
```

Automata modeli için PCA yalnızca eğitim verisi üzerinde öğrenilmiş ve tüm dönüşümler eğitimden elde edilen parametreler kullanılarak uygulanmıştır.

---

# Veri Bölme Stratejisi

Veri sızıntısını önlemek ve gerçek dünya senaryolarını daha doğru temsil etmek amacıyla veri setlerine özel bölme stratejileri uygulanmıştır.

---

## BATADAL

BATADAL veri setinde zaman serisi yapısının korunabilmesi için kronolojik bölme uygulanmıştır.

| Bölüm | Oran | Satır |
|---------|---------:|---------:|
| Train | %60 | 2506 |
| Validation | %20 | 835 |
| Test | %20 | 836 |

### Uygulanan Kurallar

- Shuffle uygulanmamıştır.
- Zaman sırası korunmuştur.
- Normalizasyon yalnızca train üzerinde fit edilmiştir.
- PCA yalnızca train üzerinde fit edilmiştir.
- Validation ve test verileri yalnızca transform edilmiştir.

---

## SKAB

SKAB veri setinde aynı dosyaya ait kayıtların hem eğitim hem test tarafına düşmesini önlemek amacıyla GroupKFold yaklaşımı uygulanmıştır.

Gruplama değişkeni:

```text
source_file
```

olarak belirlenmiştir.

| Fold | Train | Test |
|---------|---------:|---------:|
| Fold 1 | 17962 | 4512 |
| Fold 2 | 17981 | 4493 |
| Fold 3 | 17982 | 4492 |
| Fold 4 | 18040 | 4434 |
| Fold 5 | 17931 | 4543 |

### Uygulanan Kurallar

- Aynı CSV dosyası hem train hem test tarafında yer almamaktadır.
- Veri sızıntısı engellenmiştir.
- Her fold bağımsız değerlendirilmiştir.
- Sonuçlar fold ortalamaları şeklinde raporlanmıştır.

---

# Veri Sızıntısını Önleme Yaklaşımı

Proje boyunca aşağıdaki kurallar uygulanmıştır:

| İşlem | Uygulama |
|---------|---------|
| StandardScaler | Yalnızca train üzerinde fit edilmiştir |
| PCA | Yalnızca train üzerinde fit edilmiştir |
| Validation Dönüşümü | Train parametreleri ile transform |
| Test Dönüşümü | Train parametreleri ile transform |
| SKAB Split | source_file bazlı GroupKFold |
| BATADAL Split | Zaman sıralı ayrım |

Bu yaklaşım sayesinde test verisinin eğitim sürecine doğrudan veya dolaylı şekilde sızması engellenmiştir.

---
# Kullanılan Modeller

Bu proje kapsamında Deep Learning tabanlı black-box modeller ile açıklanabilir bir Probabilistic Automata yaklaşımı karşılaştırılmıştır.

---

### Kullanılan Eğitim Parametreleri

| Parametre | Değer |
|------------|------------:|
| Epoch | 50 |
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Early Stopping Patience | 5 |
| Optimizer | Adam |
| Loss Function | BCEWithLogitsLoss |
| Random Seed | 42 |

Deep Learning modelleri normalize edilmiş çok değişkenli sensör verileri üzerinde eğitilmiş ve Accuracy, Precision, Recall ve F1-score metrikleri ile değerlendirilmiştir.

---

## Açıklanabilir Model

| Model | Tür | Açıklama |
|---------|---------|---------|
| Probabilistic Automata | Explainable AI | State ve transition probability tabanlı açıklanabilir anomali tespit modeli |

Probabilistic Automata yaklaşımı, zaman serisini sembolik örüntülere dönüştürerek karar üretmektedir. Model yalnızca anomali tahmini yapmakla kalmamakta, aynı zamanda bu kararın hangi durum geçişleri ve hangi olasılıklar üzerinden oluştuğunu da açıklayabilmektedir.

---

# Açıklanabilir Automata Yaklaşımı

Bu çalışmada çok değişkenli sensör verileri önce normalize edilmiş, ardından PCA ile tek boyuta indirgenmiştir. İlk temel bileşen (PC1) kullanılarak zaman serisi sembolik forma dönüştürülmüş ve Probabilistic Automata modeli oluşturulmuştur.

### Kullanılan Temel Parametreler

| Parametre | Değer |
|------------|------------:|
| PCA Component | PC1 |
| PAA Segments | 4 |
| Window Size | 4 |
| Alphabet Size | 3 |
| Unseen Distance Threshold | 1 |

---

## Dönüşüm Zinciri

```text
Normalize Data
      │
      ▼
PCA (PC1)
      │
      ▼
PAA
      │
      ▼
SAX
      │
      ▼
Sliding Window
      │
      ▼
Pattern Generation
      │
      ▼
State Creation
      │
      ▼
Transition Matrix
      │
      ▼
Transition Probability
      │
      ▼
Prediction
      │
      ▼
Explainability
```

---

## Automata Model Akışı

```mermaid
flowchart TD

A[Normalized Features]
--> B[PCA PC1]

B --> C[PAA]
C --> D[SAX]
D --> E[Sliding Window]

E --> F[Pattern Generation]
F --> G[State Creation]

G --> H[Transition Matrix]
H --> I[Transition Probability]

I --> J[Laplace Smoothing]
J --> K[Prediction]

K --> L[Path Probability]
L --> M[Confidence Score]

M --> N[Explainability Output]
```

---

## Transition Probability Hesaplama

Bir durumdan başka bir duruma geçiş olasılığı aşağıdaki şekilde hesaplanmaktadır:

```text
P(Si → Sj)
=
Transition Count(Si → Sj)
/
Total Outgoing Transitions(Si)
```

Sıfır olasılık problemini azaltmak amacıyla Laplace Smoothing uygulanmıştır.

---

## Açıklanabilirlik Çıktıları

Automata modeli aşağıdaki bilgileri üretebilmektedir:

- State bilgisi
- Pattern bilgisi
- State geçişleri
- Transition probability değerleri
- Path probability
- Log path probability
- Average transition probability
- Confidence score
- Unseen pattern eşlemeleri
- Nihai karar (Normal / Anomaly)

Bu yapı sayesinde model yalnızca bir tahmin üretmekle kalmamakta, tahminin hangi geçişler ve hangi olasılıklar sonucunda oluştuğunu da açıklayabilmektedir.

---

# Unseen Pattern Yönetimi

Probabilistic Automata modeli, eğitim sırasında görülmeyen sembolik örüntüleri (unseen patterns) yönetebilmek için Levenshtein Distance tabanlı bir eşleme mekanizması kullanmaktadır.

Bu mekanizma sayesinde model, daha önce hiç karşılaşmadığı pattern'lar için de açıklanabilir ve deterministik kararlar üretebilmektedir.

---

## İşleyiş

1. Test sırasında gelen pattern eğitim sözlüğünde aranır.
2. Pattern sözlükte bulunuyorsa doğrudan ilgili state kullanılır.
3. Pattern sözlükte bulunmuyorsa **unseen** olarak işaretlenir.
4. Eğitim sözlüğündeki tüm pattern'lar ile Levenshtein Distance hesaplanır.
5. En küçük edit distance değerine sahip pattern bulunur.
6. Gelen pattern bu state'e eşlenir.
7. Transition analizi ve tahmin süreci eşlenen state üzerinden devam eder.
8. Açıklanabilirlik çıktısında eşleme bilgisi raporlanır.

---

## Örnek

```text
Train Pattern : abca

Incoming Pattern : abcb

Levenshtein Distance = 1

abcb → abca
```

Bu örnekte test sırasında gelen `abcb` pattern'i eğitim sırasında görülmemiştir. Sistem en yakın pattern olarak `abca` state'ini belirleyerek tahmin sürecine devam etmektedir.

---

# Explainability Çıktıları

Probabilistic Automata modeli yalnızca anomali kararı üretmekle kalmamakta, aynı zamanda kararın hangi state geçişleri ve hangi olasılıklar sonucunda oluştuğunu da raporlayabilmektedir.

Üretilen açıklama çıktılarında aşağıdaki bilgiler yer almaktadır:

- State bilgisi
- Pattern bilgisi
- Pattern durumu (seen / unseen)
- Eşlenen pattern bilgisi
- Levenshtein distance değeri
- Transition listesi
- Transition probability değerleri
- Path probability
- Log path probability
- Average transition probability
- Confidence score
- Nihai karar (Normal / Anomaly)

---

## Örnek Açıklama Çıktısı

```json
{
  "time_step": 1,
  "state": "abca",
  "pattern": "abcb",
  "status": "unseen",
  "mapped_to": "abca",
  "levenshtein_distance": 1,
  "transitions": [
    {
      "from": "abca",
      "to": "bcab",
      "probability": 0.2188
    }
  ],
  "path_probability": 0.2188,
  "confidence_score": 0.2188,
  "decision": "anomaly"
}
```

---

## Açıklanabilirlik Avantajları

Probabilistic Automata modeli aşağıdaki nedenlerle açıklanabilir bir yapı sunmaktadır:

- Karar süreci state bazında takip edilebilir.
- Hangi pattern'in anomaliye neden olduğu görülebilir.
- State geçişleri ve geçiş olasılıkları incelenebilir.
- Unseen pattern eşlemeleri açık şekilde raporlanabilir.
- Confidence score ile karar güveni yorumlanabilir.
- Transition probability yapısı görselleştirilebilir.

Bu özellikler sayesinde model yalnızca bir tahmin üretmekle kalmaz, aynı zamanda bu tahminin neden üretildiğini de açıklayabilir.

---

# Testler

Proje kapsamında veri ön işleme, PCA dönüşümü, PAA-SAX dönüşümleri, automata oluşturma, transition analizi, unseen pattern yönetimi, açıklanabilirlik modülü ve derin öğrenme bileşenleri için kapsamlı birim testler geliştirilmiştir.

Tüm testleri çalıştırmak için:

```bash
python -m pytest
```

Belirli testleri çalıştırmak için:

```bash
python -m pytest tests/test_automata_predict.py

python -m pytest tests/test_automata_prediction_accuracy.py

python -m pytest tests/test_explainability.py
```

---

## Test Sonuçları

Son doğrulama çalıştırmasında tüm testler başarıyla tamamlanmıştır.

| Ölçüt | Değer |
|--------|--------:|
| Toplam Test Sayısı | 47 |
| Başarılı Test Sayısı | 47 |
| Başarı Oranı | %100 |

<p align="center">
<img src="results/figures/tests/pytest_results.png.jpeg" width="900">
</p>

*Pytest sonuç ekranı (47/47 test başarılı).*

---

# Normal Veri Senaryosu Sonuçları

Bu deneyde modeller herhangi bir gürültü eklenmeden orijinal test verileri üzerinde değerlendirilmiştir.

---

## SKAB Sonuçları (5 Fold Ortalama)

| Model | Accuracy | Precision | Recall | F1-score |
|---------|---------:|---------:|---------:|---------:|
| LSTM | 0.670 | 0.564 | 0.063 | 0.099 |
| GRU | 0.673 | 0.574 | 0.087 | 0.135 |
| 1D-CNN | 0.675 | 0.596 | 0.104 | 0.160 |
| Automata | 0.600 | 0.276 | 0.104 | 0.137 |

---

## BATADAL Sonuçları

| Model | Accuracy | Precision | Recall | F1-score |
|---------|---------:|---------:|---------:|---------:|
| LSTM | 0.904 | 0.000 | 0.000 | 0.000 |
| GRU | 0.904 | 0.000 | 0.000 | 0.000 |
| 1D-CNN | 0.904 | 0.000 | 0.000 | 0.000 |
| Automata | 0.780 | 0.047 | 0.058 | 0.052 |

---

# Confusion Matrix

Aşağıdaki confusion matrix örneği Probabilistic Automata modelinin normal veri senaryosundaki sınıflandırma performansını göstermektedir.

<p align="center">
<img src="results/figures/confusion_matrix/confusion_matrix.png" width="700">
</p>

---

# Precision-Recall Curve

Precision ve Recall arasındaki ilişki aşağıdaki grafikte gösterilmektedir.

<p align="center">
<img src="results/figures/pr_curve/precision_recall_curve.png" width="700">
</p>

---

## Normal Veri Sonuçlarının Değerlendirilmesi

Elde edilen sonuçlar incelendiğinde Deep Learning modellerinin özellikle accuracy metriğinde Automata modelinden daha yüksek sonuçlar ürettiği görülmektedir.

SKAB veri setinde en yüksek F1-score değeri 1D-CNN modeli tarafından elde edilmiştir. CNN modeli zaman serilerindeki yerel örüntüleri öğrenmede başarılı sonuçlar vermiştir.

BATADAL veri setinde ise sınıf dengesizliği nedeniyle tüm Deep Learning modelleri yüksek accuracy üretmelerine rağmen anomali sınıfını yakalamakta zorlanmıştır. Bu durum precision, recall ve F1-score değerlerinin sıfıra yakın olmasına neden olmuştur.

Probabilistic Automata modeli Deep Learning modellerine kıyasla daha düşük performans metrikleri üretmesine rağmen karar mekanizmasını state geçişleri ve transition olasılıkları üzerinden açıklayabilmektedir.

Bu nedenle Automata yaklaşımının temel avantajı performanstan çok açıklanabilirlik ve karar izlenebilirliği olarak değerlendirilmiştir.

---

# Gürültü (Noise) Senaryosu

Modellerin veri bozulmalarına karşı dayanıklılığını ölçmek amacıyla test verilerine Gaussian Noise eklenmiştir.

Bu deneyde eğitim verileri değiştirilmemiş, yalnızca test verilerine belirli seviyede Gaussian Noise uygulanmıştır. Amaç modellerin daha gerçekçi ve gürültülü ortamlardaki davranışlarını incelemektir.

---

## Uygulanan Yaklaşım

- Eğitim verileri değiştirilmemiştir.
- Modeller yeniden eğitilmemiştir.
- Yalnızca test verilerine Gaussian Noise eklenmiştir.
- Deep Learning ve Automata modelleri aynı gürültülü veri üzerinde değerlendirilmiştir.
- Accuracy, Precision, Recall ve F1-score değişimleri incelenmiştir.

---

## Gürültü Senaryosu Akışı

```mermaid
flowchart TD

A[Orijinal Test Verisi]
--> B[Gaussian Noise]

B --> C[Gürültülü Test Verisi]

C --> D[LSTM]
C --> E[GRU]
C --> F[1D-CNN]
C --> G[Automata]

D --> H[Performans Analizi]
E --> H
F --> H
G --> H

H --> I[Karşılaştırma]
```

---

## SKAB Gürültü Sonuçları

| Model | Accuracy | Precision | Recall | F1-score |
|---------|---------:|---------:|---------:|---------:|
| LSTM | 0.655 | 0.441 | 0.110 | 0.179 |
| GRU | 0.655 | 0.412 | 0.099 | 0.160 |
| 1D-CNN | 0.403 | 0.355 | 0.519 | 0.422 |
| Automata | 0.598 | 0.275 | 0.102 | 0.135 |

---

## BATADAL Gürültü Sonuçları

| Model | Accuracy | Precision | Recall | F1-score |
|---------|---------:|---------:|---------:|---------:|
| LSTM | 0.000 | 0.000 | 0.000 | 0.000 |
| GRU | 0.000 | 0.000 | 0.000 | 0.000 |
| 1D-CNN | 0.000 | 0.000 | 0.000 | 0.000 |
| Automata | 0.600 | 0.000 | 0.000 | 0.000 |

---

## Gürültü Senaryosu Değerlendirmesi

- Gaussian Noise eklenmesi tüm modelleri etkilemiştir.
- SKAB veri setinde CNN modeli en yüksek Recall değerini üretmiştir.
- LSTM ve GRU modellerinde F1-score değerlerinde düşüş gözlemlenmiştir.
- Automata modeli sembolik dönüşüm yapısı sayesinde belirli seviyede dayanıklılık göstermiştir.
- BATADAL veri setinde Deep Learning modellerinin performansı ciddi şekilde düşmüştür.
- Gürültü arttıkça transition yapılarında bozulmalar meydana gelmiş ve confidence score değerleri azalmıştır.

---

# Unseen Veri Senaryosu

Bu deneyde eğitim sırasında görülmeyen yeni pattern'lar oluşturulmuştur.

Amaç, Automata modelinin daha önce karşılaşmadığı örüntüler karşısındaki davranışını incelemektir.

Modelin unseen pattern'ları yönetebilmesi için Levenshtein Distance tabanlı eşleme mekanizması kullanılmıştır.

---

## Unseen Senaryosu Akışı

```mermaid
flowchart TD

A[Test Pattern]
--> B{Pattern Seen?}

B -->|Yes| C[Direct State Mapping]

B -->|No| D[Levenshtein Distance]

D --> E[Nearest Pattern]

E --> F[State Mapping]

C --> G[Prediction]
F --> G

G --> H[Explainability Output]
```

---

## Unseen Veri Seti Özeti

| Veri Seti | Unseen Ratio | Açıklama |
|------------|------------:|------------|
| SKAB | 1.00 | Test patternlarının tamamı eğitimde görülmemiştir |
| BATADAL | 1.00 | Test patternlarının tamamı eğitimde görülmemiştir |

---

## Unseen Pattern Eşleme Sonuçları

| Veri Seti | Pattern Yönetimi | Eşleme Yöntemi | Karar Üretimi |
|------------|------------|------------|------------|
| SKAB | Başarılı | Levenshtein Distance | Devam etti |
| BATADAL | Başarılı | Levenshtein Distance | Devam etti |

---

## Açıklanabilirlik Çıktıları

| Veri Seti | Unseen Pattern Tespiti | Mapping Bilgisi | Transition Açıklaması | Confidence Score |
|------------|------------|------------|------------|------------|
| SKAB | Var | Var | Var | Var |
| BATADAL | Var | Var | Var | Var |

---

## Örnek Unseen Pattern Eşlemesi

| Gelen Pattern | En Yakın Eğitim Pattern'ı | Levenshtein Distance |
|--------------|--------------------------|----------------------:|
| abdc | abbc | 1 |
| cbad | cbcd | 1 |
| acbd | acad | 1 |

---

## Değerlendirme

- Eğitim sırasında görülmeyen pattern'lar başarıyla tespit edilmiştir.
- Levenshtein Distance kullanılarak en yakın eğitim pattern'ına eşleme yapılmıştır.
- State geçişleri korunmuştur.
- Karar süreci kesintiye uğramadan devam etmiştir.
- Mapping işlemleri explainability çıktılarında raporlanmıştır.
- Model tamamen yeni örüntüler için de açıklanabilir karar üretebilmiştir.

Bu sonuçlar, Probabilistic Automata yaklaşımının unseen pattern durumlarında deterministik ve açıklanabilir davranış sergileyebildiğini göstermektedir.

---

# Explainability Analizi

Probabilistic Automata modelinin en önemli avantajı yalnızca tahmin üretmesi değil, aynı zamanda karar sürecini açıklayabilmesidir.

Model;

- State bilgisi
- Pattern bilgisi
- Transition olasılıkları
- Path probability
- Confidence score
- Unseen pattern eşlemeleri

üreterek kararın nasıl oluştuğunu takip edilebilir hale getirmektedir.

---

## Automata State Diagram

Aşağıdaki grafik eğitim verisinden oluşturulan state yapısını ve state geçişlerini göstermektedir.

<p align="center">
<img src="results/figures/state_diagram/automata_state_diagram.png" width="700">
</p>

---

## Transition Probability Heatmap

Geçiş olasılıklarının yoğunluğu aşağıdaki heatmap üzerinde gösterilmektedir.

<p align="center">
<img src="results/figures/heatmap/transition_probability_heatmap.png" width="700">
</p>

---

## Örnek Açıklanabilirlik Çıktısı

```json
{
  "time_step": 25,
  "state": "abca",
  "pattern": "abca",
  "status": "seen",
  "path_probability": 0.00318,
  "confidence_score": 0.412,
  "decision": "normal"
}
```

Unseen pattern durumunda ise:

```json
{
  "pattern": "abda",
  "status": "unseen",
  "mapped_to": "abca",
  "levenshtein_distance": 1,
  "confidence_score": 0.287,
  "decision": "anomaly"
}
```

---

## Explainability Değerlendirmesi

Automata modeli:

- State bazlı çalışmaktadır.
- Geçiş olasılıklarını raporlayabilmektedir.
- Confidence score üretebilmektedir.
- Unseen pattern'ları açıklayabilmektedir.
- Levenshtein Distance ile pattern eşlemesi yapabilmektedir.
- Karar sürecini JSON çıktıları ile sunabilmektedir.

Bu özellikler deep learning modellerine kıyasla önemli yorumlanabilirlik avantajları sağlamaktadır.

---

# Parametre Duyarlılık Analizi

Automata modelinin davranışını incelemek amacıyla Window Size ve Alphabet Size parametreleri üzerinde duyarlılık analizi gerçekleştirilmiştir.

---

# Window Size Analizi

## SKAB Sonuçları

| Window Size | Accuracy | F1-score | State Count | Transition Density |
|------------:|----------:|----------:|------------:|-------------------:|
| 3 | 0.608 | 0.125 | 27.0 | 0.0977 |
| 4 | 0.600 | 0.137 | 71.2 | 0.0317 |
| 5 | 0.594 | 0.130 | Artış | Azalış |
| 6 | 0.588 | 0.129 | Artış | Azalış |

---

### SKAB

<p align="center">
<img src="results/figures/parameter_analysis/window_size_f1_sensitivity_SKAB.png" width="700">
</p>

---

### BATADAL

<p align="center">
<img src="results/figures/parameter_analysis/window_size_f1_sensitivity_BATADAL_dataset04.png" width="700">
</p>

---

## Değerlendirme

- Window Size arttıkça state sayısı artmıştır.
- Transition yoğunluğu azalmıştır.
- Model karmaşıklığı yükselmiştir.
- En dengeli sonuçlar Window Size = 4 için elde edilmiştir.

---

# Alphabet Size Analizi

## SKAB Sonuçları

| Alphabet Size | Accuracy | F1-score | State Count |
|--------------:|----------:|----------:|------------:|
| 3 | 0.600 | 0.137 | 71.2 |
| 4 | 0.615 | 0.148 | 130.6 |
| 5 | 0.613 | 0.170 | Artış |
| 6 | 0.600 | 0.157 | Artış |

---

### SKAB

<p align="center">
<img src="results/figures/parameter_analysis/alphabet_size_f1_sensitivity_SKAB.png" width="700">
</p>

---

### BATADAL

<p align="center">
<img src="results/figures/parameter_analysis/alphabet_size_f1_sensitivity_BATADAL_dataset04.png" width="700">
</p>

---

## Değerlendirme

- Alphabet Size arttıkça pattern çeşitliliği yükselmiştir.
- State sayısı artmıştır.
- Çok büyük alphabet değerlerinde performans düşüşü gözlemlenmiştir.
- En yüksek F1-score değerleri Alphabet Size = 5 civarında elde edilmiştir.

---

# Cross Dataset Analizi

Bu deneyde modellerin bir veri setinde eğitilip farklı veri setinde test edilmesiyle genellenebilirlikleri incelenmiştir.

---

## Probabilistic Automata Sonuçları

| Train Dataset | Test Dataset | Accuracy | Precision | Recall | F1-score |
|--------------|-------------|----------:|----------:|----------:|----------:|
| SKAB | BATADAL | 0.519 | 0.104 | 0.477 | 0.170 |
| BATADAL | SKAB | 0.590 | 0.362 | 0.170 | 0.222 |

---

## Deep Learning Cross Dataset Sonuçları

| Model | Train Dataset | Test Dataset | Accuracy | Precision | Recall | F1-score |
|---------|---------|---------|---------:|---------:|---------:|---------:|
| LSTM | SKAB | BATADAL | 0.751 | 0.150 | 0.333 | 0.204 |
| GRU | SKAB | BATADAL | 0.702 | 0.137 | 0.398 | 0.203 |
| CNN | SKAB | BATADAL | 0.737 | 0.140 | 0.335 | 0.195 |
| LSTM | BATADAL | SKAB | 0.652 | 0.000 | 0.000 | 0.000 |
| GRU | BATADAL | SKAB | 0.652 | 0.000 | 0.000 | 0.000 |
| CNN | BATADAL | SKAB | 0.652 | 0.000 | 0.000 | 0.000 |

---

## Cross Dataset Değerlendirmesi

- Veri setleri arasında doğrudan transfer başarısı sınırlı kalmıştır.
- SKAB üzerinde öğrenilen örüntüler BATADAL'a kısmen aktarılabilmiştir.
- BATADAL üzerinde öğrenilen modeller SKAB üzerinde başarılı sonuç verememiştir.
- Sensör yapılarındaki farklılıklar performansı önemli ölçüde etkilemiştir.
- Automata modeli de benzer şekilde veri setine bağımlı davranmıştır.

---

# İstatistiksel Analiz

Deneyler aşağıdaki random seed değerleri ile tekrarlanmıştır.

## Kullanılan Seed Değerleri

```text
42
123
2026
7
999
```

---

## Wilcoxon Signed-Rank Testi

Modeller arasındaki performans farklarını incelemek amacıyla Wilcoxon Signed-Rank testi uygulanmıştır.

| Karşılaştırma | Statistic | p-value | Yorum |
|-------------|----------:|---------:|---------|
| LSTM vs GRU | 1.0 | 0.125 | Anlamlı fark yok |
| LSTM vs CNN | 0.0 | 0.0625 | Sınıra yakın |
| GRU vs CNN | 1.0 | 0.125 | Anlamlı fark yok |

---

## Sonuç

- p-value değerleri 0.05'in üzerinde bulunmuştur.
- Modeller arasında istatistiksel olarak anlamlı fark gözlemlenmemiştir.
- Sonuçlar farklı random seed değerlerinde kararlı davranış göstermiştir.

---

# Genel Sonuç

Bu proje kapsamında:

- LSTM
- GRU
- 1D-CNN
- Probabilistic Automata

yaklaşımları zaman serisi anomali tespiti problemi üzerinde karşılaştırılmıştır.

Deep Learning modelleri özellikle SKAB veri setinde daha yüksek performans göstermiştir. CNN modeli SKAB üzerinde en yüksek F1-score değerini üretmiştir.

BATADAL veri setinde sınıf dengesizliği nedeniyle Accuracy değerleri yüksek olmasına rağmen Recall ve F1-score değerleri düşük kalmıştır.

Probabilistic Automata yaklaşımı ise:

- State yapıları
- Transition olasılıkları
- Confidence Score
- Path Probability
- Unseen Pattern Yönetimi
- Levenshtein Distance Eşlemesi
- Explainability Çıktıları

sayesinde karar sürecini açıklanabilir hale getirmiştir.

Sonuç olarak Deep Learning modelleri performans açısından avantaj sağlarken, Probabilistic Automata modeli yorumlanabilirlik açısından önemli avantajlar sunmuştur. Bu çalışma, zaman serisi anomali tespitinde performans ve açıklanabilirlik arasındaki dengeyi göstermektedir.
---
