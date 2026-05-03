import pandas as pd
from sklearn.preprocessing import StandardScaler


def test_scaler_is_fitted_only_on_train_data():
    train_data = pd.DataFrame({
        "sensor_1": [10, 20, 30, 40, 50],
        "sensor_2": [100, 200, 300, 400, 500],
    })

    scaler = StandardScaler()
    scaler.fit(train_data)

    assert scaler.mean_.tolist() == train_data.mean().tolist()


def test_validation_data_does_not_affect_scaler_mean():
    train_data = pd.DataFrame({
        "sensor_1": [10, 20, 30, 40, 50],
        "sensor_2": [100, 200, 300, 400, 500],
    })

    validation_data = pd.DataFrame({
        "sensor_1": [1000, 2000],
        "sensor_2": [10000, 20000],
    })

    scaler_train_only = StandardScaler()
    scaler_train_only.fit(train_data)

    scaler_with_leakage = StandardScaler()
    scaler_with_leakage.fit(pd.concat([train_data, validation_data]))

    assert scaler_train_only.mean_.tolist() != scaler_with_leakage.mean_.tolist()


def test_same_scaler_is_used_for_validation_and_test():
    train_data = pd.DataFrame({
        "sensor_1": [10, 20, 30, 40, 50],
        "sensor_2": [100, 200, 300, 400, 500],
    })

    validation_data = pd.DataFrame({
        "sensor_1": [15, 25],
        "sensor_2": [150, 250],
    })

    test_data = pd.DataFrame({
        "sensor_1": [35, 45],
        "sensor_2": [350, 450],
    })

    scaler = StandardScaler()

    normalized_train = scaler.fit_transform(train_data)
    normalized_validation = scaler.transform(validation_data)
    normalized_test = scaler.transform(test_data)

    assert normalized_train.shape == train_data.shape
    assert normalized_validation.shape == validation_data.shape
    assert normalized_test.shape == test_data.shape


def test_label_column_is_not_normalized():
    train_data = pd.DataFrame({
        "sensor_1": [10, 20, 30, 40, 50],
        "ATT_FLAG": [0, 0, 1, 0, 1],
    })

    feature_columns = ["sensor_1"]
    original_labels = train_data["ATT_FLAG"].copy()

    scaler = StandardScaler()
    train_data[feature_columns] = scaler.fit_transform(train_data[feature_columns])

    assert train_data["ATT_FLAG"].equals(original_labels)
