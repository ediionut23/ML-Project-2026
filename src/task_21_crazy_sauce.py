import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from data_preprocessing import (
    load_data, extract_temporal_features, filter_receipts_by_product,
    create_receipt_features, create_product_binary_vector, create_target_variable,
    train_test_split_by_receipt, train_test_split_temporal, STANDALONE_SAUCES
)
from logistic_regression import (
    LogisticRegressionGD, LogisticRegressionNewton, LogisticRegressionMiniBatchGD,
    standardize
)
from evaluation import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, print_classification_report_with_auc,
    plot_confusion_matrix, plot_roc_curve, plot_loss_curve, plot_coefficient_importance
)

from sklearn.linear_model import LogisticRegression as SklearnLR

def prepare_crazy_sauce_dataset(df):

    df = extract_temporal_features(df)

    df_crazy = filter_receipts_by_product(df, 'Crazy Schnitzel')
    print(f"Receipts with Crazy Schnitzel: {df_crazy['id_bon'].nunique()}")

    receipt_ids = df_crazy['id_bon'].unique()

    receipt_features = create_receipt_features(df_crazy, exclude_products=['Crazy Sauce'])

    product_vectors = create_product_binary_vector(df_crazy, receipt_ids,
                                                    exclude_products=['Crazy Sauce'])

    receipts_with_crazy_sauce = df_crazy[df_crazy['retail_product_name'] == 'Crazy Sauce']['id_bon'].unique()
    target = pd.Series(index=receipt_ids, data=0)
    target.loc[target.index.isin(receipts_with_crazy_sauce)] = 1

    X = receipt_features.set_index('id_bon')
    X = X.join(product_vectors)

    drop_cols = ['date', 'data_bon']
    X = X.drop(columns=[c for c in drop_cols if c in X.columns])

    X = X.select_dtypes(include=[np.number])

    y = target.loc[X.index]

    return X, y, df_crazy

def run_experiment():

    print("="*70)
    print("Task 2.1: Logistic Regression - Crazy Sauce | Crazy Schnitzel")
    print("="*70)

    print("\n[1] Loading data...")
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ap_dataset.csv')
    df = load_data(data_path)
    print(f"Total rows: {len(df)}, Total receipts: {df['id_bon'].nunique()}")

    print("\n[2] Preparing dataset...")
    X, y, df_crazy = prepare_crazy_sauce_dataset(df)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Class distribution:")
    print(f"  - Class 0 (No Crazy Sauce): {sum(y == 0)} ({100*sum(y==0)/len(y):.1f}%)")
    print(f"  - Class 1 (Has Crazy Sauce): {sum(y == 1)} ({100*sum(y==1)/len(y):.1f}%)")

    feature_names = X.columns.tolist()
    print(f"\nFeatures ({len(feature_names)}):")
    for f in feature_names[:10]:
        print(f"  - {f}")
    if len(feature_names) > 10:
        print(f"  ... and {len(feature_names)-10} more")

    print("\n[3] Splitting data (80% train / 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split_by_receipt(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {len(X_train)} receipts")
    print(f"Test set: {len(X_test)} receipts")

    print("\n[4] Standardizing features...")
    X_train_scaled, X_test_scaled, mean, std = standardize(
        X_train.values, X_test.values
    )

    results = {}

    print("\n" + "="*70)
    print("[Model 1] Custom Logistic Regression - Gradient Descent")
    print("="*70)

    model_gd = LogisticRegressionGD(
        learning_rate=0.1,
        max_iter=2000,
        regularization=0.01,
        verbose=False
    )
    model_gd.fit(X_train_scaled, y_train.values)

    y_pred_gd = model_gd.predict(X_test_scaled)
    y_proba_gd = model_gd.predict_proba(X_test_scaled)

    results['Gradient Descent'] = print_classification_report_with_auc(
        y_test.values, y_pred_gd, y_proba_gd, "Custom LR (Gradient Descent)"
    )

    plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_loss_curve(model_gd.loss_history,
                    title='Training Loss - Gradient Descent',
                    save_path=os.path.join(plots_dir, 'task21_loss_gd.png'))

    print("\n" + "="*70)
    print("[Model 2] Custom Logistic Regression - Newton's Method")
    print("="*70)

    model_newton = LogisticRegressionNewton(
        max_iter=100,
        regularization=0.01,
        verbose=False
    )
    model_newton.fit(X_train_scaled, y_train.values)

    y_pred_newton = model_newton.predict(X_test_scaled)
    y_proba_newton = model_newton.predict_proba(X_test_scaled)

    results['Newton Method'] = print_classification_report_with_auc(
        y_test.values, y_pred_newton, y_proba_newton, "Custom LR (Newton's Method)"
    )

    plot_loss_curve(model_newton.loss_history,
                    title='Training Loss - Newton Method',
                    save_path=os.path.join(plots_dir, 'task21_loss_newton.png'))

    print("\n" + "="*70)
    print("[Model 3] Custom Logistic Regression - Mini-Batch Gradient Descent")
    print("="*70)

    model_mbgd = LogisticRegressionMiniBatchGD(
        learning_rate=0.1,
        max_iter=500,
        batch_size=32,
        regularization=0.01,
        verbose=False
    )
    model_mbgd.fit(X_train_scaled, y_train.values)

    y_pred_mbgd = model_mbgd.predict(X_test_scaled)
    y_proba_mbgd = model_mbgd.predict_proba(X_test_scaled)

    results['Mini-Batch GD'] = print_classification_report_with_auc(
        y_test.values, y_pred_mbgd, y_proba_mbgd, "Custom LR (Mini-Batch GD)"
    )

    print("\n" + "="*70)
    print("[Model 4] Sklearn Logistic Regression (Comparison)")
    print("="*70)

    sklearn_lr = SklearnLR(max_iter=2000, C=100)
    sklearn_lr.fit(X_train_scaled, y_train.values)

    y_pred_sklearn = sklearn_lr.predict(X_test_scaled)
    y_proba_sklearn = sklearn_lr.predict_proba(X_test_scaled)[:, 1]

    results['Sklearn LR'] = print_classification_report_with_auc(
        y_test.values, y_pred_sklearn, y_proba_sklearn, "Sklearn Logistic Regression"
    )

    print("\n" + "="*70)
    print("[Baseline] Majority Class Classifier")
    print("="*70)

    majority_class = int(y_train.mode()[0])
    y_pred_baseline = np.full(len(y_test), majority_class)
    y_proba_baseline = np.full(len(y_test), y_train.mean())

    results['Baseline (Majority)'] = print_classification_report_with_auc(
        y_test.values, y_pred_baseline, y_proba_baseline, "Majority Class Baseline"
    )

    print("\n[5] Generating visualizations...")

    plot_confusion_matrix(y_test.values, y_pred_newton,
                         title='Confusion Matrix - Newton Method',
                         save_path=os.path.join(plots_dir, 'task21_cm_newton.png'))

    plt.figure(figsize=(10, 8))

    for name, y_proba in [('Gradient Descent', y_proba_gd),
                          ('Newton Method', y_proba_newton),
                          ('Mini-Batch GD', y_proba_mbgd),
                          ('Sklearn LR', y_proba_sklearn)]:
        from evaluation import roc_curve
        fpr, tpr, _ = roc_curve(y_test.values, y_proba)
        auc = roc_auc_score(y_test.values, y_proba)
        plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={auc:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison - Task 2.1')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'task21_roc_comparison.png'), dpi=150)
    plt.close()

    plot_coefficient_importance(
        model_newton.weights,
        feature_names,
        title='Feature Importance (Newton Method Coefficients)',
        save_path=os.path.join(plots_dir, 'task21_feature_importance.png'),
        top_k=20
    )

    print("\n[6] Coefficient Interpretation...")
    print("\nTop 10 features that INCREASE probability of Crazy Sauce:")

    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': model_newton.weights
    })
    coef_df_sorted = coef_df.sort_values('coefficient', ascending=False)

    for _, row in coef_df_sorted.head(10).iterrows():
        print(f"  {row['feature']}: {row['coefficient']:.4f}")

    print("\nTop 10 features that DECREASE probability of Crazy Sauce:")
    for _, row in coef_df_sorted.tail(10).iterrows():
        print(f"  {row['feature']}: {row['coefficient']:.4f}")

    print("\n" + "="*70)
    print("SUMMARY RESULTS")
    print("="*70)

    summary_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [r['accuracy'] for r in results.values()],
        'Precision': [r['precision'] for r in results.values()],
        'Recall': [r['recall'] for r in results.values()],
        'F1': [r['f1'] for r in results.values()],
        'ROC-AUC': [r['roc_auc'] for r in results.values()]
    })

    print(summary_df.to_string(index=False))

    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    summary_df.to_csv(os.path.join(results_dir, 'task21_results.csv'), index=False)
    coef_df_sorted.to_csv(os.path.join(results_dir, 'task21_coefficients.csv'), index=False)

    print(f"\n[7] Results saved to {results_dir}")
    print(f"    Plots saved to {plots_dir}")

    return results, model_newton, feature_names

if __name__ == "__main__":
    results, best_model, feature_names = run_experiment()
