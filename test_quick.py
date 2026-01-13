import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import (
    load_data, extract_temporal_features, filter_receipts_by_product,
    create_receipt_features, create_product_binary_vector,
    train_test_split_by_receipt, STANDALONE_SAUCES
)
from logistic_regression import LogisticRegressionGD, LogisticRegressionNewton, standardize
from evaluation import accuracy_score, roc_auc_score, f1_score
import pandas as pd
import numpy as np

print("="*70)
print("QUICK TEST - Verifying code works")
print("="*70)

print("\n[1] Loading data...")
data_path = os.path.join(os.path.dirname(__file__), 'ap_dataset.csv')
df = load_data(data_path)
print(f"✓ Loaded {len(df)} rows, {df['id_bon'].nunique()} receipts")

df = extract_temporal_features(df)

print("\n[2] Filtering to Crazy Schnitzel receipts...")
df_crazy = filter_receipts_by_product(df, 'Crazy Schnitzel')
print(f"✓ Found {df_crazy['id_bon'].nunique()} receipts with Crazy Schnitzel")

print("\n[3] Preparing features...")
receipt_ids = df_crazy['id_bon'].unique()
receipt_features = create_receipt_features(df_crazy, exclude_products=['Crazy Sauce'])
product_vectors = create_product_binary_vector(df_crazy, receipt_ids,
                                                exclude_products=['Crazy Sauce'])

receipts_with_sauce = df_crazy[df_crazy['retail_product_name'] == 'Crazy Sauce']['id_bon'].unique()
target = pd.Series(index=receipt_ids, data=0)
target.loc[target.index.isin(receipts_with_sauce)] = 1

X = receipt_features.set_index('id_bon')
X = X.join(product_vectors)
X = X.drop(columns=['date', 'data_bon'], errors='ignore')
X = X.select_dtypes(include=[np.number])
y = target.loc[X.index]

print(f"✓ Feature matrix: {X.shape}")
print(f"✓ Positive class: {y.sum()} / {len(y)} ({100*y.mean():.1f}%)")

print("\n[4] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split_by_receipt(X, y, test_size=0.2, random_state=42)
print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")

X_train_scaled, X_test_scaled, mean, std = standardize(X_train.values, X_test.values)

print("\n[5] Testing Gradient Descent...")
model_gd = LogisticRegressionGD(learning_rate=0.1, max_iter=100, verbose=False)
model_gd.fit(X_train_scaled, y_train.values)
y_pred_gd = model_gd.predict(X_test_scaled)
y_proba_gd = model_gd.predict_proba(X_test_scaled)

acc_gd = accuracy_score(y_test.values, y_pred_gd)
auc_gd = roc_auc_score(y_test.values, y_proba_gd)
f1_gd = f1_score(y_test.values, y_pred_gd)

print(f"✓ GD - Accuracy: {acc_gd:.4f}, AUC: {auc_gd:.4f}, F1: {f1_gd:.4f}")

print("\n[6] Testing Newton's Method...")
model_newton = LogisticRegressionNewton(max_iter=20, verbose=False)
model_newton.fit(X_train_scaled, y_train.values)
y_pred_newton = model_newton.predict(X_test_scaled)
y_proba_newton = model_newton.predict_proba(X_test_scaled)

acc_newton = accuracy_score(y_test.values, y_pred_newton)
auc_newton = roc_auc_score(y_test.values, y_proba_newton)
f1_newton = f1_score(y_test.values, y_pred_newton)

print(f"✓ Newton - Accuracy: {acc_newton:.4f}, AUC: {auc_newton:.4f}, F1: {f1_newton:.4f}")

print("\n[7] Testing sauce list...")
print(f"✓ Standalone sauces ({len(STANDALONE_SAUCES)}):")
for sauce in STANDALONE_SAUCES:
    count = df[df['retail_product_name'] == sauce]['id_bon'].nunique()
    print(f"  - {sauce}: {count} receipts")

print("\n" + "="*70)
print("✓ ALL TESTS PASSED!")
print("="*70)
print("\nYou can now run the full experiments:")
print("  python3 src/task_crazy_sauce.py")
print("  python3 src/task_22_all_sauce.py")
print("  python3 run_all.py")
print("="*70 + "\n")
