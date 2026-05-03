import pandas as pd


def split_sequential(dataframe, train_ratio=0.60, val_ratio=0.20):
    total_size = len(dataframe)

    train_end = int(total_size * train_ratio)
    val_end = train_end + int(total_size * val_ratio)

    train_data = dataframe.iloc[:train_end]
    val_data = dataframe.iloc[train_end:val_end]
    test_data = dataframe.iloc[val_end:]

    return train_data, val_data, test_data


def test_split_ratios_are_correct():
    df = pd.DataFrame({"value": range(100)})

    train, val, test = split_sequential(df)

    assert len(train) == 60
    assert len(val) == 20
    assert len(test) == 20


def test_no_data_loss_after_split():
    df = pd.DataFrame({"value": range(100)})

    train, val, test = split_sequential(df)

    assert len(train) + len(val) + len(test) == len(df)


def test_sequential_order_is_preserved():
    df = pd.DataFrame({"value": range(100)})

    train, val, test = split_sequential(df)

    assert train["value"].tolist() == list(range(0, 60))
    assert val["value"].tolist() == list(range(60, 80))
    assert test["value"].tolist() == list(range(80, 100))


def test_shuffle_is_not_used():
    df = pd.DataFrame({"value": range(100)})

    train, val, test = split_sequential(df)

    combined = pd.concat([train, val, test])

    assert combined["value"].tolist() == df["value"].tolist()
