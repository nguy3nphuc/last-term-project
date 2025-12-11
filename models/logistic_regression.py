import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class CustomLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, eta=0.01, iterations=1000):
        self.eta = eta
        self.iterations = iterations
        self.w = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        num_samples, num_features = X.shape

        self.w = np.random.randn(num_features, 1)

        for _ in range(self.iterations):
            for i in range(num_samples):
                xi = X[i].reshape(-1, 1)
                yi = y[i]
                pi = self.sigmoid(np.dot(self.w.T, xi))
                self.w -= self.eta * (pi - yi) * xi

        return self

    def predict(self, X):
        X = np.array(X)
        preds = self.sigmoid(X @ self.w)
        return (preds >= 0.5).astype(int).flatten()
