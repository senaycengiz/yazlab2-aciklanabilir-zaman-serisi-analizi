# Genel proje konfigürasyonları

RANDOM_SEED = 42

# Veri bölme oranları
TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20
TEST_RATIO = 0.20

# Deney tekrarları için seed değerleri
SEEDS = [42, 123, 2026, 7, 999]

# Otomata modeli için varsayılan parametreler
WINDOW_SIZE = 4
ALPHABET_SIZE = 3

# Parametre analizi için değerler
WINDOW_SIZE_OPTIONS = [3, 4, 5, 6]
ALPHABET_SIZE_OPTIONS = [3, 4, 5, 6]

# Derin öğrenme eğitim parametreleri
EPOCHS = 50
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 5

# Klasör yolları
DATA_PATH = "data/"
SRC_PATH = "src/"
TESTS_PATH = "tests/"
RESULTS_PATH = "results/"
LOGS_PATH = "logs/"
