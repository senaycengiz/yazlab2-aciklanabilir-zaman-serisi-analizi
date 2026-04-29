import pandas as pd

# Dosya yolları
train_path = "data/raw/BATADAL/BATADAL_dataset03.csv"
test_path = "data/raw/BATADAL/BATADAL_dataset04.csv"

# Verileri oku
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

# ===== TRAIN / DATASET03 =====
print("\n===== TRAIN (dataset03) =====")
print("Satır sayısı:", train_df.shape[0])
print("Kolon sayısı:", train_df.shape[1])

print("\nKolonlar:")
print(train_df.columns)

print("\nEksik veri:")
print(train_df.isnull().sum())

print("\nLabel/anomaly sütunu: ATT_FLAG")

print("\nAnomaly dağılımı:")
print(train_df["ATT_FLAG"].value_counts())

print("\nYüzdelik dağılım:")
print(train_df["ATT_FLAG"].value_counts(normalize=True) * 100)


# ===== TEST / DATASET04 =====
print("\n===== TEST (dataset04) =====")
print("Satır sayısı:", test_df.shape[0])
print("Kolon sayısı:", test_df.shape[1])

print("\nKolonlar:")
print(test_df.columns)

print("\nEksik veri:")
print(test_df.isnull().sum())

# Dataset04 içinde label kolonu var mı kontrol et
possible_label_columns = ["ATT_FLAG", "Label", "Attack", "anomaly", "class", "target"]

found_label = None

for col in possible_label_columns:
    if col in test_df.columns:
        found_label = col
        break

if found_label:
    print(f"\nLabel/anomaly sütunu: {found_label}")

    print("\nAnomaly dağılımı:")
    print(test_df[found_label].value_counts())

    print("\nYüzdelik dağılım:")
    print(test_df[found_label].value_counts(normalize=True) * 100)
else:
    print("\nDataset04 içinde ATT_FLAG veya benzeri bir label/anomaly sütunu bulunamadı.")
    print("Bu nedenle dataset04 için anomaly oranı doğrudan hesaplanamadı.")


# ===== KARŞILAŞTIRMA =====
print("\n===== DATASET03 - DATASET04 KARŞILAŞTIRMA =====")
print("Dataset03 satır sayısı:", train_df.shape[0])
print("Dataset03 kolon sayısı:", train_df.shape[1])
print("Dataset04 satır sayısı:", test_df.shape[0])
print("Dataset04 kolon sayısı:", test_df.shape[1])

print("\nFeature sayısı farkı:")
print(abs(train_df.shape[1] - test_df.shape[1]))