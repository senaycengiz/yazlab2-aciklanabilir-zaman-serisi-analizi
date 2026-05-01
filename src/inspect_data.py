from data_loader import load_csv, load_wadi_attack, print_basic_info, print_anomaly_info

batadal_train_path = "data/raw/BATADAL/BATADAL_dataset03.csv"
batadal_test_path = "data/raw/BATADAL/BATADAL_dataset04.csv"

wadi_train_path = "data/raw/WADI/WADI_14days_new.csv"
wadi_attack_path = "data/raw/WADI/WADI_attackdataLABLE.csv"


# BATADAL
batadal_train = load_csv(batadal_train_path)
batadal_test = load_csv(batadal_test_path)

print_basic_info(batadal_train, "BATADAL TRAIN - dataset03")
print_anomaly_info(batadal_train, "BATADAL TRAIN - dataset03")

print_basic_info(batadal_test, "BATADAL TEST - dataset04")
print_anomaly_info(batadal_test, "BATADAL TEST - dataset04")


# WADI
wadi_train = load_csv(wadi_train_path)

wadi_attack = load_wadi_attack(wadi_attack_path)

print_basic_info(wadi_train, "WADI TRAIN")
print_anomaly_info(wadi_train, "WADI TRAIN")

print_basic_info(wadi_attack, "WADI ATTACK")
print_anomaly_info(wadi_attack, "WADI ATTACK")