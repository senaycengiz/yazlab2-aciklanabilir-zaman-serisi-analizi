import numpy as np
from sklearn.decomposition import PCA


def test_pca_output_single_dimension():
    train_data = np.random.rand(100, 10)

    pca = PCA(n_components=1)
    transformed_data = pca.fit_transform(train_data)

    assert transformed_data.shape == (100, 1)


def test_pca_fit_only_train_transform_validation_and_test():
    train_data = np.random.rand(100, 10)
    validation_data = np.random.rand(30, 10)
    test_data = np.random.rand(40, 10)

    pca = PCA(n_components=1)
    pca.fit(train_data)

    validation_transformed = pca.transform(validation_data)
    test_transformed = pca.transform(test_data)

    assert validation_transformed.shape == (30, 1)
    assert test_transformed.shape == (40, 1)