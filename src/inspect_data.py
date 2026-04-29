import pandas as pd

# BATADAL dosya yolunu kontrol et
batadal_path = "data/raw/BATADAL/BATADAL_dataset03.csv"

df = pd.read_csv(batadal_path)

print("\n===== BATADAL =====")
print("Satır sayısı:", df.shape[0])
print("Kolon sayısı:", df.shape[1])

print("\nKolonlar:")
print(df.columns)

print("\nEksik veri:")
print(df.isnull().sum())
