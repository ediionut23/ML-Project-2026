import pandas as pd
import numpy as np
from datetime import datetime

STANDALONE_SAUCES = [
    'Crazy Sauce', 'Cheddar Sauce', 'Extra Cheddar Sauce',
    'Garlic Sauce', 'Tomato Sauce', 'Blueberry Sauce',
    'Spicy Sauce', 'Pink Sauce'
]


def load_data(filepath):
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    return df


def extract_temporal_features(df):
    df = df.copy()
    df['data_bon'] = pd.to_datetime(df['data_bon'])
    df['hour'] = df['data_bon'].dt.hour
    df['day_of_week'] = df['data_bon'].dt.dayofweek + 1
    df['is_weekend'] = df['day_of_week'].isin([6, 7]).astype(int)
    df['date'] = df['data_bon'].dt.date
    return df


def create_receipt_features(df, exclude_products=None):
    if exclude_products is None:
        exclude_products = []

    df_features = df[~df['retail_product_name'].isin(exclude_products)]
    receipt_groups = df.groupby('id_bon')
    receipt_features = pd.DataFrame({
        'id_bon': df['id_bon'].unique()
    })

    cart_size = df_features.groupby('id_bon').size()
    receipt_features = receipt_features.merge(
        cart_size.rename('cart_size').reset_index(),
        on='id_bon',
        how='left'
    )
    receipt_features['cart_size'] = receipt_features['cart_size'].fillna(0)

    distinct_products = df_features.groupby('id_bon')['retail_product_name'].nunique()
    receipt_features = receipt_features.merge(
        distinct_products.rename('distinct_products').reset_index(),
        on='id_bon',
        how='left'
    )
    receipt_features['distinct_products'] = receipt_features['distinct_products'].fillna(0)

    total_value = df_features.groupby('id_bon')['SalePriceWithVAT'].sum()
    receipt_features = receipt_features.merge(
        total_value.rename('total_value').reset_index(),
        on='id_bon',
        how='left'
    )
    receipt_features['total_value'] = receipt_features['total_value'].fillna(0)

    temporal = df.groupby('id_bon').first()[['hour', 'day_of_week', 'is_weekend', 'date', 'data_bon']]
    receipt_features = receipt_features.merge(
        temporal.reset_index(),
        on='id_bon',
        how='left'
    )

    return receipt_features


def create_product_vector(df, receipt_ids, exclude_products=None):
    if exclude_products is None:
        exclude_products = []

    all_products = df[~df['retail_product_name'].isin(exclude_products)]['retail_product_name'].unique()

    pivot = df[~df['retail_product_name'].isin(exclude_products)].pivot_table(
        index='id_bon',
        columns='retail_product_name',
        aggfunc='size',
        fill_value=0
    )

    product_binary = (pivot > 0).astype(int)
    product_binary = product_binary.reindex(receipt_ids, fill_value=0)

    return product_binary


def create_product_count_vector(df, receipt_ids, exclude_products=None):
    if exclude_products is None:
        exclude_products = []

    pivot = df[~df['retail_product_name'].isin(exclude_products)].pivot_table(
        index='id_bon',
        columns='retail_product_name',
        aggfunc='size',
        fill_value=0
    )

    pivot = pivot.reindex(receipt_ids, fill_value=0)

    return pivot


def create_target_variable(df, target_product):
    receipts_with_product = df[df['retail_product_name'] == target_product]['id_bon'].unique()
    target = df.groupby('id_bon').first().index.to_series()
    target = target.isin(receipts_with_product).astype(int)
    return target


def filter_receipts_by_product(df, product_name):
    receipts_with_product = df[df['retail_product_name'] == product_name]['id_bon'].unique()
    return df[df['id_bon'].isin(receipts_with_product)]


def prepare_dataset_for_logistic_regression(df, target_sauce, filter_product=None):
    df = extract_temporal_features(df)

    if filter_product:
        df = filter_receipts_by_product(df, filter_product)

    receipt_ids = df['id_bon'].unique()

    receipt_features = create_receipt_features(df, exclude_products=[target_sauce])
    product_vectors = create_product_binary_vector(df, receipt_ids, exclude_products=[target_sauce])

    target = create_target_variable(df, target_sauce)
    target = target.reindex(receipt_ids)

    X = receipt_features.set_index('id_bon')
    X = X.join(product_vectors)

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]

    y = target.loc[X.index]

    return X, y, X.columns.tolist()


def create_product_binary_vector(df, receipt_ids, exclude_products=None):
    if exclude_products is None:
        exclude_products = []

    filtered_df = df[~df['retail_product_name'].isin(exclude_products)]

    if filtered_df.empty:
        return pd.DataFrame(index=receipt_ids)

    pivot = filtered_df.pivot_table(
        index='id_bon',
        columns='retail_product_name',
        aggfunc='size',
        fill_value=0
    )

    product_binary = (pivot > 0).astype(int)
    product_binary.columns = ['has_' + col for col in product_binary.columns]
    product_binary = product_binary.reindex(receipt_ids, fill_value=0)

    return product_binary


def train_test_split_by_receipt(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    receipt_ids = X.index.values
    np.random.shuffle(receipt_ids)

    n_test = int(len(receipt_ids) * test_size)
    test_ids = receipt_ids[:n_test]
    train_ids = receipt_ids[n_test:]

    X_train = X.loc[train_ids]
    X_test = X.loc[test_ids]
    y_train = y.loc[train_ids]
    y_test = y.loc[test_ids]

    return X_train, X_test, y_train, y_test


def train_test_split_temporal(X, y, df, split_date):
    df_temp = extract_temporal_features(df)
    receipt_dates = df_temp.groupby('id_bon')['date'].first()

    train_ids = receipt_dates[receipt_dates < split_date].index
    test_ids = receipt_dates[receipt_dates >= split_date].index

    X_train = X.loc[X.index.intersection(train_ids)]
    X_test = X.loc[X.index.intersection(test_ids)]
    y_train = y.loc[y.index.intersection(train_ids)]
    y_test = y.loc[y.index.intersection(test_ids)]

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    import os
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ap_dataset.csv')
    df = load_data(data_path)
    print(f"Loaded {len(df)} rows")

    df = extract_temporal_features(df)
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    df_crazy = filter_receipts_by_product(df, 'Crazy Schnitzel')
    print(f"Receipts with Crazy Schnitzel: {df_crazy['id_bon'].nunique()}")
