import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def accuracy_score(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    return np.array([[TN, FP], [FN, TP]])

def precision_score(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))

    if TP + FP == 0:
        return 0.0
    return TP / (TP + FP)

def recall_score(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    if TP + FN == 0:
        return 0.0
    return TP / (TP + FN)

def f1_score(y_true, y_pred):

    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)

    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)

def specificity_score(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))

    if TN + FP == 0:
        return 0.0
    return TN / (TN + FP)

def roc_curve(y_true, y_scores, n_thresholds=100):

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    thresholds = np.linspace(0, 1, n_thresholds)
    fpr = []
    tpr = []

    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)

        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))

        tpr_val = TP / (TP + FN) if (TP + FN) > 0 else 0

        fpr_val = FP / (FP + TN) if (FP + TN) > 0 else 0

        tpr.append(tpr_val)
        fpr.append(fpr_val)

    return np.array(fpr), np.array(tpr), thresholds

def roc_auc_score(y_true, y_scores):

    fpr, tpr, _ = roc_curve(y_true, y_scores)

    sorted_indices = np.argsort(fpr)
    fpr_sorted = fpr[sorted_indices]
    tpr_sorted = tpr[sorted_indices]

    auc = np.trapz(tpr_sorted, fpr_sorted)

    return auc

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix', save_path=None):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_true, y_scores, title='ROC Curve', save_path=None):

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_loss_curve(loss_history, title='Training Loss', save_path=None):

    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_coefficient_importance(coefficients, feature_names, title='Feature Importance',
                                save_path=None, top_k=20):

    indices = np.argsort(np.abs(coefficients))[::-1][:top_k]

    plt.figure(figsize=(12, 8))
    colors = ['green' if c > 0 else 'red' for c in coefficients[indices]]
    plt.barh(range(len(indices)), coefficients[indices], color=colors)
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Coefficient Value')
    plt.title(title)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def print_classification_report(y_true, y_pred, model_name="Model"):

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    spec = specificity_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*50}")
    print(f"Classification Report: {model_name}")
    print(f"{'='*50}")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Precision:   {prec:.4f}")
    print(f"Recall:      {rec:.4f}")
    print(f"F1 Score:    {f1:.4f}")
    print(f"Specificity: {spec:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
    print(f"{'='*50}\n")

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'specificity': spec,
        'confusion_matrix': cm
    }

def print_classification_report_with_auc(y_true, y_pred, y_proba, model_name="Model"):

    report = print_classification_report(y_true, y_pred, model_name)
    auc = roc_auc_score(y_true, y_proba)
    print(f"ROC-AUC:     {auc:.4f}")
    report['roc_auc'] = auc
    return report

def hit_at_k(actual_items, recommended_items, k):

    top_k = set(recommended_items[:k])
    actual = set(actual_items) if not isinstance(actual_items, set) else actual_items
    return 1 if len(top_k & actual) > 0 else 0

def precision_at_k(actual_items, recommended_items, k):

    top_k = set(recommended_items[:k])
    actual = set(actual_items) if not isinstance(actual_items, set) else actual_items
    relevant = len(top_k & actual)
    return relevant / k if k > 0 else 0

def recall_at_k(actual_items, recommended_items, k):

    top_k = set(recommended_items[:k])
    actual = set(actual_items) if not isinstance(actual_items, set) else actual_items
    if len(actual) == 0:
        return 0
    relevant = len(top_k & actual)
    return relevant / len(actual)

def mean_reciprocal_rank(actual_items, recommended_items):

    actual = set(actual_items) if not isinstance(actual_items, set) else actual_items

    for i, item in enumerate(recommended_items):
        if item in actual:
            return 1 / (i + 1)
    return 0

def ndcg_at_k(actual_items, recommended_items, k):

    actual = set(actual_items) if not isinstance(actual_items, set) else actual_items

    dcg = 0
    for i, item in enumerate(recommended_items[:k]):
        if item in actual:
            dcg += 1 / np.log2(i + 2)

    n_relevant = min(len(actual), k)
    idcg = sum(1 / np.log2(i + 2) for i in range(n_relevant))

    if idcg == 0:
        return 0
    return dcg / idcg

def evaluate_recommendations(actual_items_list, recommendations_list, k_values=[1, 3, 5]):

    results = {}

    for k in k_values:
        hits = []
        precisions = []
        recalls = []
        mrrs = []
        ndcgs = []

        for actual, recs in zip(actual_items_list, recommendations_list):
            hits.append(hit_at_k(actual, recs, k))
            precisions.append(precision_at_k(actual, recs, k))
            recalls.append(recall_at_k(actual, recs, k))
            mrrs.append(mean_reciprocal_rank(actual, recs))
            ndcgs.append(ndcg_at_k(actual, recs, k))

        results[k] = {
            'hit_rate': np.mean(hits),
            'precision': np.mean(precisions),
            'recall': np.mean(recalls),
            'mrr': np.mean(mrrs),
            'ndcg': np.mean(ndcgs)
        }

    return results

def plot_recommendation_metrics(results, baseline_results=None, title='Recommendation Metrics',
                                save_path=None):

    k_values = list(results.keys())
    metrics = ['hit_rate', 'precision', 'recall', 'ndcg']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        model_values = [results[k][metric] for k in k_values]
        axes[i].plot(k_values, model_values, 'b-o', label='Model', linewidth=2, markersize=8)

        if baseline_results:
            baseline_values = [baseline_results[k][metric] for k in k_values]
            axes[i].plot(k_values, baseline_values, 'r--s', label='Baseline', linewidth=2, markersize=8)

        axes[i].set_xlabel('K')
        axes[i].set_ylabel(metric.replace('_', ' ').title())
        axes[i].set_title(f'{metric.replace("_", " ").title()} @ K')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xticks(k_values)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":

    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 0])
    y_proba = np.array([0.9, 0.2, 0.4, 0.8, 0.3, 0.7, 0.6, 0.1, 0.85, 0.25])

    print("Testing metrics:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1: {f1_score(y_true, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_true, y_proba):.4f}")
