import numpy as np
from collections import Counter, defaultdict

class NaiveBayesClassifier:

    def __init__(self, alpha=1.0):

        self.alpha = alpha
        self.class_priors = {}
        self.feature_probs = {}
        self.classes = None

    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y)

        self.classes = np.unique(y)
        n_samples, n_features = X.shape

        for c in self.classes:
            self.class_priors[c] = np.sum(y == c) / n_samples

        self.feature_probs = {}
        for c in self.classes:
            X_c = X[y == c]
            n_c = len(X_c)

            probs = (np.sum(X_c, axis=0) + self.alpha) / (n_c + 2 * self.alpha)
            self.feature_probs[c] = probs

        return self

    def predict_proba(self, X):

        X = np.array(X)
        n_samples = X.shape[0]

        log_probs = {}
        for c in self.classes:
            log_prior = np.log(self.class_priors[c])

            p = self.feature_probs[c]
            log_likelihood = np.sum(
                X * np.log(p + 1e-10) + (1 - X) * np.log(1 - p + 1e-10),
                axis=1
            )

            log_probs[c] = log_prior + log_likelihood

        log_prob_array = np.column_stack([log_probs[c] for c in self.classes])
        max_log_prob = np.max(log_prob_array, axis=1, keepdims=True)
        exp_probs = np.exp(log_prob_array - max_log_prob)
        probs = exp_probs / np.sum(exp_probs, axis=1, keepdims=True)

        class_1_idx = np.where(self.classes == 1)[0][0]
        return probs[:, class_1_idx]

    def predict(self, X, threshold=0.5):

        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

class KNNClassifier:

    def __init__(self, k=5, weighted=False):

        self.k = k
        self.weighted = weighted
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):

        self.X_train = np.array(X, dtype=np.float64)
        self.y_train = np.array(y)
        return self

    def _euclidean_distance(self, x1, x2):

        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _get_neighbors(self, x):

        distances = []
        for i, x_train in enumerate(self.X_train):
            dist = self._euclidean_distance(x, x_train)
            distances.append((dist, i, self.y_train[i]))

        distances.sort(key=lambda d: d[0])

        return distances[:self.k]

    def predict_proba(self, X):

        X = np.array(X, dtype=np.float64)
        probs = []

        for x in X:
            neighbors = self._get_neighbors(x)

            if self.weighted:

                weighted_votes = defaultdict(float)
                for dist, idx, label in neighbors:
                    weight = 1 / (dist + 1e-10)
                    weighted_votes[label] += weight

                total_weight = sum(weighted_votes.values())
                prob = weighted_votes.get(1, 0) / total_weight
            else:

                votes = [label for _, _, label in neighbors]
                prob = sum(votes) / len(votes)

            probs.append(prob)

        return np.array(probs)

    def predict(self, X, threshold=0.5):

        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

class DecisionTreeID3:

    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1):

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
        self.n_features = None

    def _entropy(self, y):

        if len(y) == 0:
            return 0

        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        return entropy

    def _information_gain(self, X_column, y, threshold):

        parent_entropy = self._entropy(y)

        left_mask = X_column <= threshold
        right_mask = ~left_mask

        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)
        n_total = len(y)

        if n_left == 0 or n_right == 0:
            return 0

        child_entropy = (
            (n_left / n_total) * self._entropy(y[left_mask]) +
            (n_right / n_total) * self._entropy(y[right_mask])
        )

        return parent_entropy - child_entropy

    def _find_best_split(self, X, y):

        best_gain = -1
        best_feature = None
        best_threshold = None

        n_features = X.shape[1]

        for feature_idx in range(n_features):
            column = X[:, feature_idx]
            thresholds = np.unique(column)

            for threshold in thresholds:
                gain = self._information_gain(column, y, threshold)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, depth=0):

        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            n_classes == 1):

            prob_1 = np.mean(y)
            return {'leaf': True, 'prob': prob_1, 'samples': n_samples}

        best_feature, best_threshold, best_gain = self._find_best_split(X, y)

        if best_gain <= 0:
            prob_1 = np.mean(y)
            return {'leaf': True, 'prob': prob_1, 'samples': n_samples}

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            prob_1 = np.mean(y)
            return {'leaf': True, 'prob': prob_1, 'samples': n_samples}

        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree,
            'gain': best_gain
        }

    def fit(self, X, y):

        X = np.array(X, dtype=np.float64)
        y = np.array(y)
        self.n_features = X.shape[1]
        self.tree = self._build_tree(X, y)
        return self

    def _predict_sample(self, x, node):

        if node['leaf']:
            return node['prob']

        if x[node['feature']] <= node['threshold']:
            return self._predict_sample(x, node['left'])
        else:
            return self._predict_sample(x, node['right'])

    def predict_proba(self, X):

        X = np.array(X, dtype=np.float64)
        return np.array([self._predict_sample(x, self.tree) for x in X])

    def predict(self, X, threshold=0.5):

        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

class AdaBoostClassifier:

    def __init__(self, n_estimators=50, learning_rate=1.0):

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.stumps = []
        self.alphas = []

    def _decision_stump_predict(self, X, feature, threshold, polarity):

        predictions = np.ones(len(X))
        if polarity == 1:
            predictions[X[:, feature] <= threshold] = -1
        else:
            predictions[X[:, feature] > threshold] = -1
        return predictions

    def _find_best_stump(self, X, y, sample_weights):

        n_samples, n_features = X.shape
        best_error = float('inf')
        best_stump = None

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                for polarity in [1, -1]:
                    predictions = self._decision_stump_predict(X, feature, threshold, polarity)

                    misclassified = predictions != y
                    error = np.sum(sample_weights * misclassified) / np.sum(sample_weights)

                    if error < best_error:
                        best_error = error
                        best_stump = {
                            'feature': feature,
                            'threshold': threshold,
                            'polarity': polarity
                        }

        return best_stump, best_error

    def fit(self, X, y):

        X = np.array(X, dtype=np.float64)
        y = np.array(y)

        y_transformed = np.where(y == 1, 1, -1)

        n_samples = len(y)
        sample_weights = np.ones(n_samples) / n_samples

        self.stumps = []
        self.alphas = []

        for t in range(self.n_estimators):

            stump, error = self._find_best_stump(X, y_transformed, sample_weights)

            error = np.clip(error, 1e-10, 1 - 1e-10)

            alpha = self.learning_rate * 0.5 * np.log((1 - error) / error)

            predictions = self._decision_stump_predict(
                X, stump['feature'], stump['threshold'], stump['polarity']
            )

            sample_weights *= np.exp(-alpha * y_transformed * predictions)
            sample_weights /= np.sum(sample_weights)

            self.stumps.append(stump)
            self.alphas.append(alpha)

        return self

    def predict_proba(self, X):

        X = np.array(X, dtype=np.float64)

        weighted_sum = np.zeros(len(X))

        for stump, alpha in zip(self.stumps, self.alphas):
            predictions = self._decision_stump_predict(
                X, stump['feature'], stump['threshold'], stump['polarity']
            )
            weighted_sum += alpha * predictions

        proba = 1 / (1 + np.exp(-weighted_sum))
        return proba

    def predict(self, X, threshold=0.5):

        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

class ProductRanker:

    def __init__(self, model, product_prices):

        self.model = model
        self.product_prices = product_prices

    def rank_products(self, X, candidate_products, feature_names):

        X = np.array(X).reshape(1, -1)

        rankings = []

        for product in candidate_products:

            X_with_product = X.copy()

            prob = self.model.predict_proba(X_with_product)[0]

            price = self.product_prices.get(product, 0)

            score = prob * price

            rankings.append({
                'product': product,
                'score': score,
                'probability': prob,
                'price': price
            })

        rankings.sort(key=lambda x: x['score'], reverse=True)

        return rankings

class PopularityBaseline:

    def __init__(self):
        self.popularity = {}
        self.ranked_products = []

    def fit(self, df, product_column='retail_product_name'):

        counts = df[product_column].value_counts()
        total = len(df)

        self.popularity = (counts / total).to_dict()
        self.ranked_products = list(counts.index)

        return self

    def recommend(self, n=5, exclude=None):

        if exclude is None:
            exclude = set()
        else:
            exclude = set(exclude)

        recommendations = []
        for product in self.ranked_products:
            if product not in exclude:
                recommendations.append(product)
            if len(recommendations) >= n:
                break

        return recommendations

    def get_probability(self, product):

        return self.popularity.get(product, 0)

if __name__ == "__main__":

    np.random.seed(42)

    n_samples = 500
    n_features = 10
    X = np.random.randint(0, 2, (n_samples, n_features))

    weights = np.random.randn(n_features)
    z = X @ weights
    y = (z > np.median(z)).astype(int)

    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print("Testing Naive Bayes:")
    nb = NaiveBayesClassifier()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    print(f"Accuracy: {np.mean(y_pred == y_test):.4f}")

    print("\nTesting k-NN:")
    knn = KNNClassifier(k=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(f"Accuracy: {np.mean(y_pred == y_test):.4f}")

    print("\nTesting Decision Tree (ID3):")
    dt = DecisionTreeID3(max_depth=5)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    print(f"Accuracy: {np.mean(y_pred == y_test):.4f}")

    print("\nTesting AdaBoost:")
    ab = AdaBoostClassifier(n_estimators=50)
    ab.fit(X_train, y_train)
    y_pred = ab.predict(X_test)
    print(f"Accuracy: {np.mean(y_pred == y_test):.4f}")
