import numpy as np

class LogisticRegressionGD:

    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-6,
                 regularization=0.0, verbose=False):

        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.regularization = regularization
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z):

        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _compute_loss(self, X, y):

        n = len(y)
        z = X @ self.weights + self.bias
        p = self._sigmoid(z)

        p = np.clip(p, 1e-15, 1 - 1e-15)

        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

        loss += (self.regularization / 2) * np.sum(self.weights ** 2)

        return loss

    def _compute_gradients(self, X, y):

        n = len(y)
        z = X @ self.weights + self.bias
        p = self._sigmoid(z)

        dw = (1 / n) * (X.T @ (p - y)) + self.regularization * self.weights

        db = np.mean(p - y)

        return dw, db

    def fit(self, X, y):

        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for i in range(self.max_iter):

            dw, db = self._compute_gradients(X, y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            loss = self._compute_loss(X, y)
            self.loss_history.append(loss)

            if i > 0 and abs(self.loss_history[-2] - self.loss_history[-1]) < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {i}")
                break

            if self.verbose and i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss:.6f}")

        return self

    def predict_proba(self, X):

        X = np.array(X, dtype=np.float64)
        z = X @ self.weights + self.bias
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):

        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def get_coefficients(self):

        return {
            'weights': self.weights.copy(),
            'bias': self.bias
        }

class LogisticRegressionNewton:

    def __init__(self, max_iter=100, tol=1e-6, regularization=0.0, verbose=False):

        self.max_iter = max_iter
        self.tol = tol
        self.regularization = regularization
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z):

        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _compute_loss(self, X, y):

        n = len(y)
        z = X @ self.weights + self.bias
        p = self._sigmoid(z)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        loss += (self.regularization / 2) * np.sum(self.weights ** 2)
        return loss

    def fit(self, X, y):

        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        n_samples, n_features = X.shape

        X_aug = np.column_stack([X, np.ones(n_samples)])

        theta = np.zeros(n_features + 1)
        self.loss_history = []

        for i in range(self.max_iter):

            z = X_aug @ theta
            p = self._sigmoid(z)

            gradient = (1 / n_samples) * (X_aug.T @ (p - y))

            gradient[:-1] += self.regularization * theta[:-1]

            S = p * (1 - p)
            H = (1 / n_samples) * (X_aug.T @ (S[:, np.newaxis] * X_aug))

            H[:-1, :-1] += self.regularization * np.eye(n_features)

            H += 1e-8 * np.eye(n_features + 1)

            try:
                delta = np.linalg.solve(H, gradient)
            except np.linalg.LinAlgError:

                delta = np.linalg.lstsq(H, gradient, rcond=None)[0]

            theta -= delta

            self.weights = theta[:-1]
            self.bias = theta[-1]

            loss = self._compute_loss(X, y)
            self.loss_history.append(loss)

            if np.linalg.norm(delta) < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {i}")
                break

            if self.verbose and i % 10 == 0:
                print(f"Iteration {i}, Loss: {loss:.6f}")

        return self

    def predict_proba(self, X):

        X = np.array(X, dtype=np.float64)
        z = X @ self.weights + self.bias
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):

        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def get_coefficients(self):

        return {
            'weights': self.weights.copy(),
            'bias': self.bias
        }

class LogisticRegressionMiniBatchGD:

    def __init__(self, learning_rate=0.01, max_iter=1000, batch_size=32,
                 tol=1e-6, regularization=0.0, verbose=False, random_state=42):

        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.tol = tol
        self.regularization = regularization
        self.verbose = verbose
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _compute_loss(self, X, y):
        n = len(y)
        z = X @ self.weights + self.bias
        p = self._sigmoid(z)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        loss += (self.regularization / 2) * np.sum(self.weights ** 2)
        return loss

    def fit(self, X, y):

        np.random.seed(self.random_state)
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        n_batches = max(1, n_samples // self.batch_size)

        for epoch in range(self.max_iter):

            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0

            for batch in range(n_batches):
                start_idx = batch * self.batch_size
                end_idx = min(start_idx + self.batch_size, n_samples)

                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                z = X_batch @ self.weights + self.bias
                p = self._sigmoid(z)

                n_batch = len(y_batch)
                dw = (1 / n_batch) * (X_batch.T @ (p - y_batch)) + self.regularization * self.weights
                db = np.mean(p - y_batch)

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            loss = self._compute_loss(X, y)
            self.loss_history.append(loss)

            if epoch > 0 and abs(self.loss_history[-2] - self.loss_history[-1]) < self.tol:
                if self.verbose:
                    print(f"Converged at epoch {epoch}")
                break

            if self.verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

        return self

    def predict_proba(self, X):
        X = np.array(X, dtype=np.float64)
        z = X @ self.weights + self.bias
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def get_coefficients(self):
        return {
            'weights': self.weights.copy(),
            'bias': self.bias
        }

def standardize(X_train, X_test=None):

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1

    X_train_scaled = (X_train - mean) / std

    if X_test is not None:
        X_test_scaled = (X_test - mean) / std
        return X_train_scaled, X_test_scaled, mean, std

    return X_train_scaled, mean, std

def normalize(X_train, X_test=None):

    min_val = np.min(X_train, axis=0)
    max_val = np.max(X_train, axis=0)
    range_val = max_val - min_val
    range_val[range_val == 0] = 1

    X_train_scaled = (X_train - min_val) / range_val

    if X_test is not None:
        X_test_scaled = (X_test - min_val) / range_val
        return X_test_scaled, min_val, range_val

    return X_train_scaled, min_val, range_val

if __name__ == "__main__":

    np.random.seed(42)

    n_samples = 1000
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    true_weights = np.array([1, -2, 0.5, 0, 1.5])
    z = X @ true_weights + 0.5
    y = (1 / (1 + np.exp(-z)) > 0.5).astype(int)

    print("Testing Gradient Descent:")
    model_gd = LogisticRegressionGD(learning_rate=0.1, max_iter=1000, verbose=True)
    model_gd.fit(X, y)
    y_pred = model_gd.predict(X)
    print(f"Accuracy: {np.mean(y_pred == y):.4f}")
    print(f"Learned weights: {model_gd.weights}")

    print("\nTesting Newton's Method:")
    model_newton = LogisticRegressionNewton(max_iter=100, verbose=True)
    model_newton.fit(X, y)
    y_pred = model_newton.predict(X)
    print(f"Accuracy: {np.mean(y_pred == y):.4f}")
    print(f"Learned weights: {model_newton.weights}")
