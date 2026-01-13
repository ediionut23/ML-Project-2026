import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from datetime import datetime

from data_preprocessing import (
    load_data, extract_temporal_features, filter_receipts_by_product,
    create_receipt_features, create_product_binary_vector, create_target_variable,
    train_test_split_by_receipt, STANDALONE_SAUCES
)
from logistic_regression import (
    LogisticRegressionGD, LogisticRegressionNewton, LogisticRegressionMiniBatchGD,
    standardize
)
from evaluation import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, print_classification_report_with_auc,
    plot_confusion_matrix, plot_roc_curve, hit_at_k, precision_at_k,
    recall_at_k, mean_reciprocal_rank, ndcg_at_k, evaluate_recommendations
)

from sklearn.linear_model import LogisticRegression as SklearnLR

class SauceRecommender:

    def __init__(self, sauces, model_type='newton', regularization=0.01):

        self.sauces = sauces
        self.model_type = model_type
        self.regularization = regularization
        self.models = {}
        self.scalers = {}
        self.feature_names = None
        self.training_stats = {}

    def _create_model(self):

        if self.model_type == 'gd':
            return LogisticRegressionGD(
                learning_rate=0.1,
                max_iter=2000,
                regularization=self.regularization,
                verbose=False
            )
        elif self.model_type == 'newton':
            return LogisticRegressionNewton(
                max_iter=100,
                regularization=self.regularization,
                verbose=False
            )
        elif self.model_type == 'minibatch':
            return LogisticRegressionMiniBatchGD(
                learning_rate=0.1,
                max_iter=500,
                batch_size=32,
                regularization=self.regularization,
                verbose=False
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def prepare_features(self, df, target_sauce):

        if 'hour' not in df.columns:
            df = extract_temporal_features(df)

        receipt_ids = df['id_bon'].unique()

        receipt_features = create_receipt_features(df, exclude_products=[target_sauce])

        product_vectors = create_product_binary_vector(
            df, receipt_ids,
            exclude_products=[target_sauce]
        )

        receipts_with_sauce = df[df['retail_product_name'] == target_sauce]['id_bon'].unique()
        target = pd.Series(index=receipt_ids, data=0)
        target.loc[target.index.isin(receipts_with_sauce)] = 1

        X = receipt_features.set_index('id_bon')
        X = X.join(product_vectors)

        drop_cols = ['date', 'data_bon']
        X = X.drop(columns=[c for c in drop_cols if c in X.columns])

        X = X.select_dtypes(include=[np.number])

        y = target.loc[X.index]

        return X, y

    def fit(self, df):

        print(f"\nTraining {len(self.sauces)} sauce models...")
        print("="*60)

        for i, sauce in enumerate(self.sauces):
            print(f"\n[{i+1}/{len(self.sauces)}] Training model for: {sauce}")

            X, y = self.prepare_features(df, sauce)

            if self.feature_names is None:
                self.feature_names = X.columns.tolist()

            n_positive = y.sum()
            n_total = len(y)
            print(f"    Samples: {n_total}, Positive: {n_positive} ({100*n_positive/n_total:.1f}%)")

            if n_positive < 10:
                print(f"    WARNING: Too few positive samples, skipping...")
                self.models[sauce] = None
                continue

            X_train, X_val, y_train, y_val = train_test_split_by_receipt(
                X, y, test_size=0.2, random_state=42
            )

            X_train_scaled, X_val_scaled, mean, std = standardize(
                X_train.values, X_val.values
            )
            self.scalers[sauce] = {'mean': mean, 'std': std, 'features': X.columns.tolist()}

            model = self._create_model()
            model.fit(X_train_scaled, y_train.values)

            y_pred = model.predict(X_val_scaled)
            y_proba = model.predict_proba(X_val_scaled)

            acc = accuracy_score(y_val.values, y_pred)
            auc = roc_auc_score(y_val.values, y_proba)
            f1 = f1_score(y_val.values, y_pred)

            print(f"    Validation - Accuracy: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")

            self.models[sauce] = model
            self.training_stats[sauce] = {
                'accuracy': acc,
                'auc': auc,
                'f1': f1,
                'n_samples': n_total,
                'n_positive': n_positive,
                'positive_rate': n_positive / n_total
            }

        return self

    def predict_sauce_probabilities(self, X_receipt, exclude_sauces=None):

        if exclude_sauces is None:
            exclude_sauces = set()
        else:
            exclude_sauces = set(exclude_sauces)

        probabilities = {}

        for sauce in self.sauces:
            if sauce in exclude_sauces:
                continue

            model = self.models.get(sauce)
            if model is None:
                probabilities[sauce] = 0.0
                continue

            scaler = self.scalers[sauce]
            features = scaler['features']

            X = np.zeros(len(features))
            for j, feat in enumerate(features):
                if feat in X_receipt.index:
                    X[j] = X_receipt[feat]

            X_scaled = (X - scaler['mean']) / scaler['std']
            X_scaled = X_scaled.reshape(1, -1)

            prob = model.predict_proba(X_scaled)[0]
            probabilities[sauce] = prob

        return probabilities

    def recommend(self, X_receipt, k=3, exclude_sauces=None):

        probs = self.predict_sauce_probabilities(X_receipt, exclude_sauces)

        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

        return sorted_probs[:k]

    def get_model_summary(self):

        summary = []
        for sauce in self.sauces:
            stats = self.training_stats.get(sauce, {})
            summary.append({
                'sauce': sauce,
                'trained': self.models.get(sauce) is not None,
                **stats
            })
        return pd.DataFrame(summary)

class PopularityBaseline:

    def __init__(self, sauces):

        self.sauces = sauces
        self.popularity = {}
        self.ranked_sauces = []

    def fit(self, df):

        receipt_counts = {}
        total_receipts = df['id_bon'].nunique()

        for sauce in self.sauces:
            receipts_with_sauce = df[df['retail_product_name'] == sauce]['id_bon'].nunique()
            receipt_counts[sauce] = receipts_with_sauce

        self.popularity = {s: c / total_receipts for s, c in receipt_counts.items()}

        self.ranked_sauces = sorted(self.popularity.keys(),
                                    key=lambda x: self.popularity[x],
                                    reverse=True)

        print("\nPopularity Baseline - Sauce Rankings:")
        for i, sauce in enumerate(self.ranked_sauces):
            print(f"  {i+1}. {sauce}: {100*self.popularity[sauce]:.2f}%")

        return self

    def recommend(self, k=3, exclude_sauces=None):

        if exclude_sauces is None:
            exclude_sauces = set()
        else:
            exclude_sauces = set(exclude_sauces)

        recommendations = []
        for sauce in self.ranked_sauces:
            if sauce not in exclude_sauces:
                recommendations.append((sauce, self.popularity[sauce]))
            if len(recommendations) >= k:
                break

        return recommendations

def evaluate_recommender(recommender, df_test, sauces, k_values=[1, 3, 5]):

    sauce_set = set(sauces)
    df_test = extract_temporal_features(df_test)

    receipt_sauces = df_test[df_test['retail_product_name'].isin(sauce_set)].groupby('id_bon')['retail_product_name'].apply(set)

    receipts_with_sauces = receipt_sauces[receipt_sauces.apply(len) > 0]

    print(f"\nEvaluating on {len(receipts_with_sauces)} receipts with sauces...")

    results = {k: {'hits': [], 'precisions': [], 'recalls': [], 'mrrs': [], 'ndcgs': []}
               for k in k_values}

    receipt_ids = receipts_with_sauces.index.tolist()
    receipt_features_df = create_receipt_features(df_test, exclude_products=sauces)
    product_vectors = create_product_binary_vector(df_test, df_test['id_bon'].unique(),
                                                    exclude_products=sauces)

    X_all = receipt_features_df.set_index('id_bon')
    X_all = X_all.join(product_vectors)
    drop_cols = ['date', 'data_bon']
    X_all = X_all.drop(columns=[c for c in drop_cols if c in X_all.columns])
    X_all = X_all.select_dtypes(include=[np.number])

    for receipt_id in receipt_ids:
        actual_sauces = receipts_with_sauces[receipt_id]

        if isinstance(recommender, SauceRecommender):
            if receipt_id not in X_all.index:
                continue
            X_receipt = X_all.loc[receipt_id]
            recs = recommender.recommend(X_receipt, k=max(k_values), exclude_sauces=None)
            recommended = [r[0] for r in recs]
        else:

            recs = recommender.recommend(k=max(k_values), exclude_sauces=None)
            recommended = [r[0] for r in recs]

        for k in k_values:
            top_k = recommended[:k]

            hit = 1 if len(set(top_k) & actual_sauces) > 0 else 0
            results[k]['hits'].append(hit)

            relevant_in_top_k = len(set(top_k) & actual_sauces)
            precision = relevant_in_top_k / k
            results[k]['precisions'].append(precision)

            recall = relevant_in_top_k / len(actual_sauces) if len(actual_sauces) > 0 else 0
            results[k]['recalls'].append(recall)

            mrr = 0
            for i, rec in enumerate(recommended):
                if rec in actual_sauces:
                    mrr = 1 / (i + 1)
                    break
            results[k]['mrrs'].append(mrr)

            dcg = 0
            for i, rec in enumerate(top_k):
                if rec in actual_sauces:
                    dcg += 1 / np.log2(i + 2)

            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(actual_sauces), k)))
            ndcg = dcg / idcg if idcg > 0 else 0
            results[k]['ndcgs'].append(ndcg)

    final_results = {}
    for k in k_values:
        final_results[k] = {
            'hit_rate': np.mean(results[k]['hits']),
            'precision': np.mean(results[k]['precisions']),
            'recall': np.mean(results[k]['recalls']),
            'mrr': np.mean(results[k]['mrrs']),
            'ndcg': np.mean(results[k]['ndcgs'])
        }

    return final_results

def plot_model_performance(training_stats, save_path=None):

    df = pd.DataFrame(training_stats).T
    df = df.dropna()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    sauces = df.index.tolist()
    aucs = df['auc'].values
    colors = plt.cm.viridis(np.linspace(0, 1, len(sauces)))
    bars = ax.barh(sauces, aucs, color=colors)
    ax.axvline(x=0.5, color='red', linestyle='--', label='Random (0.5)')
    ax.set_xlabel('ROC-AUC')
    ax.set_title('Model Performance by Sauce (ROC-AUC)')
    ax.legend()

    ax = axes[0, 1]
    pos_rates = df['positive_rate'].values * 100
    ax.barh(sauces, pos_rates, color=colors)
    ax.set_xlabel('Positive Rate (%)')
    ax.set_title('Sauce Prevalence in Dataset')

    ax = axes[1, 0]
    f1s = df['f1'].values
    ax.barh(sauces, f1s, color=colors)
    ax.set_xlabel('F1 Score')
    ax.set_title('Model F1 Scores by Sauce')

    ax = axes[1, 1]
    accs = df['accuracy'].values
    ax.barh(sauces, accs, color=colors)
    ax.axvline(x=1-df['positive_rate'].mean(), color='red', linestyle='--',
               label=f'Majority Baseline (~{100*(1-df["positive_rate"].mean()):.1f}%)')
    ax.set_xlabel('Accuracy')
    ax.set_title('Model Accuracy by Sauce')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_recommendation_comparison(model_results, baseline_results, k_values, save_path=None):

    metrics = ['hit_rate', 'precision', 'recall', 'mrr', 'ndcg']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]

        model_vals = [model_results[k][metric] for k in k_values]
        baseline_vals = [baseline_results[k][metric] for k in k_values]

        x = np.arange(len(k_values))
        width = 0.35

        bars1 = ax.bar(x - width/2, model_vals, width, label='LR Model', color='steelblue')
        bars2 = ax.bar(x + width/2, baseline_vals, width, label='Popularity Baseline', color='coral')

        ax.set_xlabel('K')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} @ K')
        ax.set_xticks(x)
        ax.set_xticklabels([f'K={k}' for k in k_values])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    axes[5].axis('off')

    plt.suptitle('Recommendation Performance: LR Model vs Popularity Baseline', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_coefficient_heatmap(recommender, top_n=15, save_path=None):

    coef_data = {}

    for sauce in recommender.sauces:
        model = recommender.models.get(sauce)
        if model is None:
            continue

        scaler = recommender.scalers[sauce]
        features = scaler['features']
        weights = model.weights

        coef_data[sauce] = dict(zip(features, weights))

    coef_df = pd.DataFrame(coef_data)

    max_abs_coef = coef_df.abs().max(axis=1)
    top_features = max_abs_coef.nlargest(top_n).index.tolist()

    coef_subset = coef_df.loc[top_features]

    plt.figure(figsize=(12, 10))
    sns.heatmap(coef_subset, cmap='RdBu_r', center=0, annot=True, fmt='.2f',
                linewidths=0.5, cbar_kws={'label': 'Coefficient'})
    plt.title('Top Feature Coefficients by Sauce Model')
    plt.xlabel('Sauce')
    plt.ylabel('Feature')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return coef_df

def run_experiment():

    print("="*70)
    print("Task 2.2: Logistic Regression for Each Sauce + Recommendations")
    print("="*70)

    print("\n[1] Loading data...")
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ap_dataset.csv')
    df = load_data(data_path)
    df = extract_temporal_features(df)
    print(f"Total rows: {len(df)}, Total receipts: {df['id_bon'].nunique()}")

    sauces = STANDALONE_SAUCES
    print(f"\nStandalone sauces ({len(sauces)}):")
    for s in sauces:
        count = df[df['retail_product_name'] == s]['id_bon'].nunique()
        print(f"  - {s}: {count} receipts ({100*count/df['id_bon'].nunique():.2f}%)")

    print("\n[2] Splitting data temporally...")
    dates = df['date'].unique()
    dates.sort()
    split_idx = int(len(dates) * 0.8)
    split_date = dates[split_idx]

    df_train = df[df['date'] < split_date].copy()
    df_test = df[df['date'] >= split_date].copy()

    print(f"Training period: {df_train['date'].min()} to {df_train['date'].max()}")
    print(f"Test period: {df_test['date'].min()} to {df_test['date'].max()}")
    print(f"Training receipts: {df_train['id_bon'].nunique()}")
    print(f"Test receipts: {df_test['id_bon'].nunique()}")

    print("\n[3] Training Multi-Sauce Recommender...")

    recommender = SauceRecommender(sauces, model_type='newton', regularization=0.01)
    recommender.fit(df_train)

    model_summary = recommender.get_model_summary()
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    print(model_summary.to_string(index=False))

    plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_model_performance(recommender.training_stats,
                          save_path=os.path.join(plots_dir, 'task22_model_performance.png'))

    print("\n[4] Training Popularity Baseline...")
    baseline = PopularityBaseline(sauces)
    baseline.fit(df_train)

    print("\n[5] Evaluating Recommendations...")

    k_values = [1, 3, 5]

    print("\n--- LR Model Evaluation ---")
    model_results = evaluate_recommender(recommender, df_test, sauces, k_values)

    print("\n--- Popularity Baseline Evaluation ---")
    baseline_results = evaluate_recommender(baseline, df_test, sauces, k_values)

    print("\n" + "="*70)
    print("RECOMMENDATION RESULTS COMPARISON")
    print("="*70)

    print("\n{:<15} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        "Model", "Hit@K", "Prec@K", "Recall@K", "MRR", "NDCG@K"))
    print("-"*70)

    for k in k_values:
        print(f"\nK = {k}:")
        print("  LR Model:     {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
            model_results[k]['hit_rate'],
            model_results[k]['precision'],
            model_results[k]['recall'],
            model_results[k]['mrr'],
            model_results[k]['ndcg']
        ))
        print("  Baseline:     {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
            baseline_results[k]['hit_rate'],
            baseline_results[k]['precision'],
            baseline_results[k]['recall'],
            baseline_results[k]['mrr'],
            baseline_results[k]['ndcg']
        ))

        hit_imp = (model_results[k]['hit_rate'] - baseline_results[k]['hit_rate']) / baseline_results[k]['hit_rate'] * 100 if baseline_results[k]['hit_rate'] > 0 else 0
        print(f"  Improvement:  Hit@{k}: {hit_imp:+.1f}%")

    plot_recommendation_comparison(model_results, baseline_results, k_values,
                                   save_path=os.path.join(plots_dir, 'task22_recommendation_comparison.png'))

    print("\n[6] Analyzing Model Coefficients...")

    coef_df = plot_coefficient_heatmap(recommender, top_n=15,
                                       save_path=os.path.join(plots_dir, 'task22_coefficient_heatmap.png'))

    print("\n" + "="*70)
    print("TOP PREDICTORS FOR EACH SAUCE")
    print("="*70)

    for sauce in sauces:
        model = recommender.models.get(sauce)
        if model is None:
            print(f"\n{sauce}: No model trained")
            continue

        scaler = recommender.scalers[sauce]
        features = scaler['features']
        weights = model.weights

        sorted_idx = np.argsort(np.abs(weights))[::-1]

        print(f"\n{sauce}:")
        print("  Positive predictors:")
        count = 0
        for idx in sorted_idx:
            if weights[idx] > 0 and count < 3:
                print(f"    + {features[idx]}: {weights[idx]:.4f}")
                count += 1

        print("  Negative predictors:")
        count = 0
        for idx in sorted_idx:
            if weights[idx] < 0 and count < 3:
                print(f"    - {features[idx]}: {weights[idx]:.4f}")
                count += 1

    print("\n[7] Example Recommendations...")

    test_receipt_ids = df_test['id_bon'].unique()[:5]

    receipt_features_df = create_receipt_features(df_test, exclude_products=sauces)
    product_vectors = create_product_binary_vector(df_test, df_test['id_bon'].unique(),
                                                    exclude_products=sauces)

    X_all = receipt_features_df.set_index('id_bon')
    X_all = X_all.join(product_vectors)
    drop_cols = ['date', 'data_bon']
    X_all = X_all.drop(columns=[c for c in drop_cols if c in X_all.columns])
    X_all = X_all.select_dtypes(include=[np.number])

    print("\n" + "="*70)
    print("EXAMPLE RECOMMENDATIONS")
    print("="*70)

    for receipt_id in test_receipt_ids:
        if receipt_id not in X_all.index:
            continue

        actual = df_test[(df_test['id_bon'] == receipt_id) &
                        (df_test['retail_product_name'].isin(sauces))]['retail_product_name'].tolist()

        other_products = df_test[(df_test['id_bon'] == receipt_id) &
                                (~df_test['retail_product_name'].isin(sauces))]['retail_product_name'].tolist()

        X_receipt = X_all.loc[receipt_id]
        recs = recommender.recommend(X_receipt, k=3)
        baseline_recs = baseline.recommend(k=3)

        print(f"\nReceipt {receipt_id}:")
        print(f"  Products: {', '.join(other_products[:5])}{'...' if len(other_products) > 5 else ''}")
        print(f"  Actual sauce(s): {', '.join(actual) if actual else 'None'}")
        print(f"  LR Recommendations: {', '.join([f'{r[0]} ({r[1]:.3f})' for r in recs])}")
        print(f"  Baseline Recs:      {', '.join([f'{r[0]} ({r[1]:.3f})' for r in baseline_recs])}")

        if actual:
            lr_hit = any(r[0] in actual for r in recs)
            bl_hit = any(r[0] in actual for r in baseline_recs)
            print(f"  Hit: LR={lr_hit}, Baseline={bl_hit}")

    print("\n[8] Saving results...")

    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    model_summary.to_csv(os.path.join(results_dir, 'task22_model_summary.csv'), index=False)

    rec_results = []
    for k in k_values:
        rec_results.append({
            'K': k,
            'Model': 'LR',
            **model_results[k]
        })
        rec_results.append({
            'K': k,
            'Model': 'Baseline',
            **baseline_results[k]
        })
    pd.DataFrame(rec_results).to_csv(os.path.join(results_dir, 'task22_recommendation_results.csv'), index=False)

    coef_df.to_csv(os.path.join(results_dir, 'task22_coefficients.csv'))

    print(f"\nResults saved to {results_dir}")
    print(f"Plots saved to {plots_dir}")

    return recommender, baseline, model_results, baseline_results

if __name__ == "__main__":
    recommender, baseline, model_results, baseline_results = run_experiment()
