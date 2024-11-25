import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# 獲取數據
glass_identification = fetch_ucirepo(id=42)
X = glass_identification.data.features.copy()
y = glass_identification.data.targets.copy()

bins = 10
# 特徵離散化
for col in X.columns:
    X[col] = pd.cut(X[col], bins, labels=False)

# 分割資料集
X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.2, random_state=42)

class NaiveBayesClassifier:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # 初始化條件概率和先驗概率
        self.class_counts = np.zeros(n_classes, dtype=np.float64)
        self.feature_counts = np.zeros((n_classes, n_features, bins), dtype=np.float64)  
        self.feature_totals = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.class_counts[idx] = X_c.shape[0]
            for i in range(n_features):
                self.feature_counts[idx, i, :] = np.bincount(X_c.iloc[:, i], minlength=10)
            self.feature_totals[idx] = X_c.shape[0]

        # 計算先驗概率和條件概率
        self.priors = self.class_counts / n_samples
        self.conditional_probs = (self.feature_counts + 1) / (self.feature_totals[:, None, None] + 10)  # Laplace平滑

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            class_conditional = np.sum(np.log(self.conditional_probs[idx, np.arange(len(x)), x]))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

# 建立並訓練模型
model = NaiveBayesClassifier()
model.fit(X_train, y_train)

# 在測試集上評估模型
y_pred = model.predict(X_test.values)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)

print("Test Accuracy:", accuracy)
print("Classification Report:\n", report)