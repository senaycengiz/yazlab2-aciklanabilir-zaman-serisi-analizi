from data_loader import (
    load_csv,
    load_skab_dataset,
    print_basic_info,
    print_anomaly_info,
    find_label_column,
    get_feature_columns
)


batadal_path = "data/raw/BATADAL/BATADAL_dataset04.csv"


# BATADAL
batadal_df = load_csv(batadal_path)

print_basic_info(batadal_df, "BATADAL - Training Dataset 2")
print_anomaly_info(batadal_df, "BATADAL - Training Dataset 2")

batadal_label = find_label_column(batadal_df)
batadal_features = get_feature_columns(batadal_df)

print("\nBATADAL label sütunu:", batadal_label)
print("BATADAL model feature sayısı:", len(batadal_features))


# SKAB
skab_df = load_skab_dataset("data/raw/SKAB")

print_basic_info(skab_df, "SKAB - valve1 + valve2")
print_anomaly_info(skab_df, "SKAB - valve1 + valve2")

skab_label = find_label_column(skab_df)
skab_features = get_feature_columns(skab_df)

print("\nSKAB label sütunu:", skab_label)
print("SKAB model feature sayısı:", len(skab_features))
print("SKAB source_group dağılımı:")
print(skab_df["source_group"].value_counts())

print("\nSKAB source_file sayısı:", skab_df["source_file"].nunique())