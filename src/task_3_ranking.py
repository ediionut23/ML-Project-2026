"""
Task 3: Product Ranking for Upsell Recommendations
Score(p | cart) = P(p | cart) * price(p)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import (
    load_data, extract_temporal_features, 
    create_receipt_features, create_product_binary_vector,
    STANDALONE_SAUCES
)
from logistic_regression import LogisticRegressionNewton
from ranking_algorithms import (
    NaiveBayesClassifier, KNNClassifier, DecisionTreeID3, AdaBoostClassifier
)
from evaluation import roc_auc_score

CANDIDATE_CATEGORIES = {
    'sauces': STANDALONE_SAUCES,
    'drinks': ['Pepsi Cola 0.25L Doze', 'Mountain Dew 0.25L Doze', 'Aqua Carpatica Plata 0.5L',
               'Aqua Carpatica Minerala 0.5L', '7Up Lemon Lime 0.33L', 'Pepsi Cola Can 0.33L'],
    'sides': ['Baked potatoes', 'Crazy Fries with Cheddar Sauce', 'Crazy Fries with Parmesan']
}

class MultiProductRanker:
    def __init__(self, candidate_products, model_class, model_params=None):
        self.candidate_products = candidate_products
        self.model_class = model_class
        self.model_params = model_params or {}
        self.models = {}
        self.scalers = {}
        self.product_prices = {}
        self.feature_names = None
        
    def _prepare_features(self, df, target_product):
        receipt_ids = df['id_bon'].unique()
        receipt_features = create_receipt_features(df, exclude_products=[target_product])
        product_vectors = create_product_binary_vector(df, receipt_ids, exclude_products=[target_product])
        
        receipts_with_product = df[df['retail_product_name'] == target_product]['id_bon'].unique()
        target = pd.Series(index=receipt_ids, data=0)
        target.loc[target.index.isin(receipts_with_product)] = 1
        
        X = receipt_features.set_index('id_bon')
        X = X.join(product_vectors)
        drop_cols = ['date', 'data_bon']
        X = X.drop(columns=[c for c in drop_cols if c in X.columns])
        X = X.select_dtypes(include=[np.number])
        y = target.loc[X.index]
        return X, y
    
    def fit(self, df, verbose=True):
        df = extract_temporal_features(df)
        price_df = df.groupby('retail_product_name')['SalePriceWithVAT'].mean()
        self.product_prices = price_df.to_dict()
        
        if verbose:
            print(f"\nTraining {len(self.candidate_products)} product models...")
        
        for i, product in enumerate(self.candidate_products):
            if verbose:
                print(f"[{i+1}/{len(self.candidate_products)}] {product}...", end=" ")
            
            X, y = self._prepare_features(df, product)
            if self.feature_names is None:
                self.feature_names = X.columns.tolist()
            
            n_positive = y.sum()
            if n_positive < 5:
                if verbose:
                    print(f"SKIPPED")
                self.models[product] = None
                continue
            
            X_scaled = X.values
            if self.model_class in [LogisticRegressionNewton, KNNClassifier]:
                mean = np.mean(X_scaled, axis=0)
                std = np.std(X_scaled, axis=0)
                std[std == 0] = 1
                X_scaled = (X_scaled - mean) / std
                self.scalers[product] = {'mean': mean, 'std': std}
            
            model = self.model_class(**self.model_params)
            model.fit(X_scaled, y.values)
            self.models[product] = model
            
            if verbose:
                y_proba = model.predict_proba(X_scaled)
                auc = roc_auc_score(y.values, y_proba)
                print(f"AUC={auc:.4f}")
        return self
    
    def recommend(self, X_receipt, k=5, exclude_products=None, scoring='expected_revenue'):
        if exclude_products is None:
            exclude_products = set()
        else:
            exclude_products = set(exclude_products)
        
        rankings = []
        for product in self.candidate_products:
            if product in exclude_products:
                continue
            model = self.models.get(product)
            if model is None:
                continue
            
            X = np.zeros(len(self.feature_names))
            for j, feat in enumerate(self.feature_names):
                if feat in X_receipt.index:
                    X[j] = X_receipt[feat]
            
            if product in self.scalers:
                scaler = self.scalers[product]
                X = (X - scaler['mean']) / scaler['std']
            
            X = X.reshape(1, -1)
            prob = model.predict_proba(X)[0]
            price = self.product_prices.get(product, 0)
            
            if scoring == 'expected_revenue':
                score = prob * price
            elif scoring == 'probability':
                score = prob
            else:
                score = price
            
            rankings.append((product, score))
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings[:k]

class PopularityRanker:
    def __init__(self, candidate_products):
        self.candidate_products = candidate_products
        self.popularity = {}
        
    def fit(self, df, verbose=True):
        df = extract_temporal_features(df)
        total_receipts = df['id_bon'].nunique()
        for product in self.candidate_products:
            count = df[df['retail_product_name'] == product]['id_bon'].nunique()
            self.popularity[product] = count / total_receipts
        
        if verbose:
            print("\nPopularity Baseline:")
            for p, pop in sorted(self.popularity.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {p}: {pop*100:.2f}%")
        return self
    
    def recommend(self, X_receipt=None, k=5, exclude_products=None, scoring=None):
        if exclude_products is None:
            exclude_products = set()
        rankings = [(p, self.popularity.get(p, 0)) for p in self.candidate_products if p not in exclude_products]
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings[:k]

class RandomRanker:
    def __init__(self, candidate_products):
        self.candidate_products = candidate_products
    def fit(self, df, verbose=True):
        return self
    def recommend(self, X_receipt=None, k=5, exclude_products=None, scoring=None):
        available = [p for p in self.candidate_products if p not in (exclude_products or set())]
        np.random.shuffle(available)
        return [(p, 0.0) for p in available[:k]]

def evaluate_leave_one_out(ranker, df_test, candidate_products, k_values=[1,3,5,10], 
                           scoring='expected_revenue', verbose=True):
    df_test = extract_temporal_features(df_test)
    candidate_set = set(candidate_products)
    receipt_products = df_test.groupby('id_bon')['retail_product_name'].apply(set)
    valid_receipts = receipt_products[receipt_products.apply(lambda x: len(x & candidate_set) > 0)]
    
    if verbose:
        print(f"\nEvaluating on {len(valid_receipts)} receipts...")
    
    receipt_features_df = create_receipt_features(df_test, exclude_products=candidate_products)
    product_vectors = create_product_binary_vector(df_test, df_test['id_bon'].unique(), exclude_products=candidate_products)
    
    X_all = receipt_features_df.set_index('id_bon')
    X_all = X_all.join(product_vectors)
    X_all = X_all.drop(columns=[c for c in ['date', 'data_bon'] if c in X_all.columns])
    X_all = X_all.select_dtypes(include=[np.number])
    
    results = {k: {'hits': [], 'mrrs': [], 'ndcgs': []} for k in k_values}
    
    for receipt_id in valid_receipts.index:
        if receipt_id not in X_all.index:
            continue
        actual_products = valid_receipts[receipt_id] & candidate_set
        
        for removed_product in actual_products:
            X_receipt = X_all.loc[receipt_id].copy()
            try:
                recs = ranker.recommend(X_receipt, k=max(k_values), 
                                       exclude_products=actual_products - {removed_product},
                                       scoring=scoring)
                recommended = [r[0] for r in recs]
            except:
                continue
            
            for k in k_values:
                top_k = recommended[:k]
                hit = 1 if removed_product in top_k else 0
                results[k]['hits'].append(hit)
                
                mrr = 0
                for i, rec in enumerate(recommended):
                    if rec == removed_product:
                        mrr = 1 / (i + 1)
                        break
                results[k]['mrrs'].append(mrr)
                
                dcg = 0
                for i, rec in enumerate(top_k):
                    if rec == removed_product:
                        dcg = 1 / np.log2(i + 2)
                results[k]['ndcgs'].append(dcg)
    
    final_results = {}
    for k in k_values:
        if len(results[k]['hits']) > 0:
            final_results[k] = {
                'hit_rate': np.mean(results[k]['hits']),
                'precision': np.mean(results[k]['hits']) / k,
                'recall': np.mean(results[k]['hits']),
                'mrr': np.mean(results[k]['mrrs']),
                'ndcg': np.mean(results[k]['ndcgs']),
                'n_samples': len(results[k]['hits'])
            }
        else:
            final_results[k] = {'hit_rate': 0, 'precision': 0, 'recall': 0, 'mrr': 0, 'ndcg': 0, 'n_samples': 0}
    return final_results

def plot_algorithm_comparison(results_dict, metric='hit_rate', k_values=[1,3,5], title='Comparison', save_path=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(k_values))
    width = 0.8 / len(results_dict)
    colors = plt.cm.Set2(np.linspace(0, 1, len(results_dict)))
    
    for i, (algo_name, results) in enumerate(results_dict.items()):
        values = [results[k][metric] for k in k_values]
        offset = (i - len(results_dict)/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=algo_name, color=colors[i])
    
    ax.set_xlabel('K')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f'K={k}' for k in k_values])
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_metrics_heatmap(results_dict, k=5, save_path=None):
    metrics = ['hit_rate', 'mrr', 'ndcg']
    data = [[results[k].get(m, 0) for m in metrics] for results in results_dict.values()]
    df = pd.DataFrame(data, index=list(results_dict.keys()), columns=[m.upper() for m in metrics])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, fmt='.3f', cmap='YlGnBu', linewidths=0.5)
    plt.title(f'Performance Comparison (K={k})')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def run_comprehensive_ranking_experiment():
    print("=" * 80)
    print("TASK 3: COMPREHENSIVE PRODUCT RANKING FOR UPSELL")
    print("=" * 80)
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    plots_dir = os.path.join(base_dir, 'plots')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print("\n[1] Loading data...")
    data_path = os.path.join(base_dir, 'ap_dataset.csv')
    df = load_data(data_path)
    df = extract_temporal_features(df)
    print(f"    Total rows: {len(df)}, Receipts: {df['id_bon'].nunique()}")
    
    print("\n[2] Splitting data (80/20 temporal)...")
    dates = sorted(df['date'].unique())
    split_date = dates[int(len(dates) * 0.8)]
    df_train = df[df['date'] < split_date].copy()
    df_test = df[df['date'] >= split_date].copy()
    print(f"    Train: {df_train['id_bon'].nunique()} receipts, Test: {df_test['id_bon'].nunique()} receipts")
    
    all_candidates = CANDIDATE_CATEGORIES['sauces'] + CANDIDATE_CATEGORIES['drinks'] + CANDIDATE_CATEGORIES['sides']
    existing = set(df['retail_product_name'].unique())
    all_candidates = [p for p in all_candidates if p in existing]
    print(f"\n[3] Candidate products: {len(all_candidates)}")
    
    k_values = [1, 3, 5, 10]
    all_results = {}
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: ALGORITHM COMPARISON")
    print("=" * 80)
    
    # Naive Bayes
    print("\n--- Naive Bayes ---")
    nb = MultiProductRanker(all_candidates, NaiveBayesClassifier, {'alpha': 1.0})
    nb.fit(df_train)
    all_results['Naive Bayes'] = evaluate_leave_one_out(nb, df_test, all_candidates, k_values)
    
    # k-NN
    print("\n--- k-NN (k=5) ---")
    knn = MultiProductRanker(all_candidates, KNNClassifier, {'k': 5, 'weighted': True})
    knn.fit(df_train)
    all_results['k-NN'] = evaluate_leave_one_out(knn, df_test, all_candidates, k_values)
    
    # Decision Tree
    print("\n--- Decision Tree ---")
    dt = MultiProductRanker(all_candidates, DecisionTreeID3, {'max_depth': 8, 'min_samples_split': 10})
    dt.fit(df_train)
    all_results['Decision Tree'] = evaluate_leave_one_out(dt, df_test, all_candidates, k_values)
    
    # AdaBoost
    print("\n--- AdaBoost ---")
    ab = MultiProductRanker(all_candidates, AdaBoostClassifier, {'n_estimators': 50, 'learning_rate': 0.5})
    ab.fit(df_train)
    all_results['AdaBoost'] = evaluate_leave_one_out(ab, df_test, all_candidates, k_values)
    
    # Logistic Regression
    print("\n--- Logistic Regression ---")
    lr = MultiProductRanker(all_candidates, LogisticRegressionNewton, {'max_iter': 100, 'regularization': 0.01})
    lr.fit(df_train)
    all_results['Logistic Reg'] = evaluate_leave_one_out(lr, df_test, all_candidates, k_values)
    
    # Baselines
    print("\n--- Popularity Baseline ---")
    pop = PopularityRanker(all_candidates)
    pop.fit(df_train)
    all_results['Popularity'] = evaluate_leave_one_out(pop, df_test, all_candidates, k_values)
    
    print("\n--- Random Baseline ---")
    rand = RandomRanker(all_candidates)
    rand.fit(df_train)
    all_results['Random'] = evaluate_leave_one_out(rand, df_test, all_candidates, k_values)
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    for k in k_values:
        print(f"\n--- K = {k} ---")
        print(f"{'Algorithm':<18} {'Hit@K':<10} {'MRR':<10} {'NDCG':<10}")
        print("-" * 48)
        for algo, res in all_results.items():
            r = res[k]
            print(f"{algo:<18} {r['hit_rate']:<10.4f} {r['mrr']:<10.4f} {r['ndcg']:<10.4f}")
    
    # Plots
    print("\n[4] Generating plots...")
    plot_algorithm_comparison(all_results, metric='hit_rate', k_values=k_values, 
                             title='Hit Rate Comparison', save_path=os.path.join(plots_dir, 'task3_hit_rate.png'))
    plot_algorithm_comparison(all_results, metric='mrr', k_values=k_values,
                             title='MRR Comparison', save_path=os.path.join(plots_dir, 'task3_mrr.png'))
    plot_metrics_heatmap(all_results, k=5, save_path=os.path.join(plots_dir, 'task3_heatmap_k5.png'))
    
    # Hyperparameter analysis
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: HYPERPARAMETER ANALYSIS")
    print("=" * 80)
    
    print("\n--- k-NN: varying k ---")
    knn_results = {}
    for k_param in [1, 3, 5, 7, 10]:
        knn_test = MultiProductRanker(all_candidates, KNNClassifier, {'k': k_param, 'weighted': True})
        knn_test.fit(df_train, verbose=False)
        res = evaluate_leave_one_out(knn_test, df_test, all_candidates, [5], verbose=False)
        knn_results[k_param] = res[5]['hit_rate']
        print(f"    k={k_param}: Hit@5={res[5]['hit_rate']:.4f}")
    
    print("\n--- Decision Tree: varying depth ---")
    dt_results = {}
    for depth in [3, 5, 8, 10, 15]:
        dt_test = MultiProductRanker(all_candidates, DecisionTreeID3, {'max_depth': depth})
        dt_test.fit(df_train, verbose=False)
        res = evaluate_leave_one_out(dt_test, df_test, all_candidates, [5], verbose=False)
        dt_results[depth] = res[5]['hit_rate']
        print(f"    depth={depth}: Hit@5={res[5]['hit_rate']:.4f}")
    
    print("\n--- AdaBoost: varying n_estimators ---")
    ab_results = {}
    for n_est in [10, 30, 50, 75, 100]:
        ab_test = MultiProductRanker(all_candidates, AdaBoostClassifier, {'n_estimators': n_est})
        ab_test.fit(df_train, verbose=False)
        res = evaluate_leave_one_out(ab_test, df_test, all_candidates, [5], verbose=False)
        ab_results[n_est] = res[5]['hit_rate']
        print(f"    n_est={n_est}: Hit@5={res[5]['hit_rate']:.4f}")
    
    # Hyperparameter plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].plot(list(knn_results.keys()), list(knn_results.values()), 'bo-', lw=2)
    axes[0].set_xlabel('k'); axes[0].set_ylabel('Hit@5'); axes[0].set_title('k-NN')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(list(dt_results.keys()), list(dt_results.values()), 'go-', lw=2)
    axes[1].set_xlabel('depth'); axes[1].set_ylabel('Hit@5'); axes[1].set_title('Decision Tree')
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(list(ab_results.keys()), list(ab_results.values()), 'ro-', lw=2)
    axes[2].set_xlabel('n_estimators'); axes[2].set_ylabel('Hit@5'); axes[2].set_title('AdaBoost')
    axes[2].grid(True, alpha=0.3)
    plt.suptitle('Hyperparameter Sensitivity')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'task3_hyperparams.png'), dpi=150)
    plt.close()
    
    # Scoring comparison
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: SCORING METHOD COMPARISON")
    print("=" * 80)
    scoring_results = {}
    for scoring in ['expected_revenue', 'probability', 'price']:
        res = evaluate_leave_one_out(nb, df_test, all_candidates, [5], scoring=scoring, verbose=False)
        scoring_results[scoring] = res
        print(f"    {scoring}: Hit@5={res[5]['hit_rate']:.4f}")
    
    # Save results
    print("\n[5] Saving results...")
    rows = []
    for algo, res in all_results.items():
        for k in k_values:
            rows.append({'Algorithm': algo, 'K': k, **res[k]})
    pd.DataFrame(rows).to_csv(os.path.join(results_dir, 'task3_results.csv'), index=False)
    
    hyper_rows = []
    for k, v in knn_results.items():
        hyper_rows.append({'Algorithm': 'k-NN', 'Param': 'k', 'Value': k, 'Hit@5': v})
    for k, v in dt_results.items():
        hyper_rows.append({'Algorithm': 'DT', 'Param': 'depth', 'Value': k, 'Hit@5': v})
    for k, v in ab_results.items():
        hyper_rows.append({'Algorithm': 'AB', 'Param': 'n_est', 'Value': k, 'Hit@5': v})
    pd.DataFrame(hyper_rows).to_csv(os.path.join(results_dir, 'task3_hyperparams.csv'), index=False)
    
    print(f"\n    Results: {results_dir}")
    print(f"    Plots: {plots_dir}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    best_hit = max(all_results.keys(), key=lambda x: all_results[x][5]['hit_rate'])
    best_mrr = max(all_results.keys(), key=lambda x: all_results[x][5]['mrr'])
    print(f"  Best Hit@5: {best_hit} ({all_results[best_hit][5]['hit_rate']:.4f})")
    print(f"  Best MRR: {best_mrr} ({all_results[best_mrr][5]['mrr']:.4f})")
    
    return all_results, nb, pop

if __name__ == "__main__":
    all_results, best_ranker, baseline = run_comprehensive_ranking_experiment()
