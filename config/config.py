# YAZLAB2 Proje 2 - Merkezi Konfigürasyon Dosyası

# Genel ayarlar
PROJECT_NAME = "Açıklanabilir Zaman Serisi Analizi"
RANDOM_SEED = 42
SEEDS = [42, 123, 2026, 7, 999]

# Veri bölme oranları
TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20
TEST_RATIO = 0.20

# Veri yolları
DATA_PATH = "data/"
RAW_DATA_PATH = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"
RESULTS_PATH = "results/"
LOGS_PATH = "logs/"
MODELS_PATH = "models/"

# Ön işleme parametreleri
NORMALIZATION_METHOD = "standard_scaler"
USE_PCA = True
PCA_COMPONENTS = 1

# Derin öğrenme eğitim parametreleri
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 5
VALIDATION_MONITOR = "val_loss"

# Kullanılabilecek derin öğrenme modelleri
DEEP_LEARNING_MODELS = ["LSTM", "GRU", "1D-CNN"]

# Otomata modeli varsayılan parametreleri
WINDOW_SIZE = 4
ALPHABET_SIZE = 3

# Parametre analizi değerleri
WINDOW_SIZE_OPTIONS = [3, 4, 5, 6]
ALPHABET_SIZE_OPTIONS = [3, 4, 5, 6]

# Deney senaryoları
EXPERIMENT_SCENARIOS = ["normal", "noise", "unseen"]

# Noise senaryosu parametreleri
NOISE_TYPE = "gaussian"
NOISE_MEAN = 0
NOISE_STD = 0.05

# SAX / PAA ayarları
PAA_SEGMENTS = 4
SAX_METHOD = "symbolic_aggregate_approximation"

# Unseen pattern yönetimi
UNSEEN_PATTERN_METHOD = "levenshtein"
DISTANCE_METRIC = "edit_distance"

# Değerlendirme metrikleri
METRICS = ["accuracy", "precision", "recall", "f1_score"]

# İstatistiksel analiz
USE_K_FOLD = True
K_FOLD_SPLITS = 5
STATISTICAL_TESTS = ["wilcoxon", "mcnemar"]

# Loglama
SAVE_EXPERIMENT_PARAMS = True
SAVE_METRICS = True
SAVE_RESULTS_AS = "csv"

# SAX parameters
SAX_ALPHABET = ["a", "b", "c"]

PAA_DATA_DIR = "data/processed/paa"
SAX_DATA_DIR = "data/processed/sax"

SAX_WORD_COLUMN = "sax_word"