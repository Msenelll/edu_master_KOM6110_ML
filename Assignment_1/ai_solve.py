# B
import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0

        for _ in range(self.epochs):
            for i in range(len(X)):
                prediction = self.predict(X[i])
                error = y[i] - prediction
                self.w += self.learning_rate * error * X[i]
                self.b += self.learning_rate * error

    def predict(self, X):
        return np.dot(X, self.w) + self.b

# Örnek veri
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([1, 0, 1])

# Model oluşturma ve eğitim
perceptron = Perceptron()
perceptron.fit(X, y)

# Tahmin yapma
predictions = perceptron.predict(X)

# Sonuçları yazdırma
print("Tahminler:", predictions)
print("Ağırlıklar:", perceptron.w)
print("Sabit terim:", perceptron.b)

# C

import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0

        for _ in range(self.epochs):
            for i in range(len(X)):
                prediction = self.predict(X[i])
                error = y[i] - prediction
                gradient = error * X[i]
                self.w -= self.learning_rate * gradient
                self.b -= self.learning_rate * error

    def predict(self, X):
        return np.dot(X, self.w) + self.b

# Örnek veri
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([1, 0, 1])

# Model oluşturma ve eğitim
perceptron = Perceptron()
perceptron.fit(X, y)

# Tahmin yapma
predictions = perceptron.predict(X)

# Sonuçları yazdırma
print("Tahminler:", predictions)

# C
import numpy as np

class DeltaRule:
    def __init__(self, learning_rate=0.01, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0

        for _ in range(self.epochs):
            for i in range(len(X)):
                prediction = self.predict(X[i])
                error = y[i] - prediction
                gradient = error * X[i]
                self.w += self.learning_rate * gradient
                self.b += self.learning_rate * error

    def predict(self, X):
        return np.dot(X, self.w) + self.b

# Örnek veri
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([1, 0, 1])

# Model oluşturma ve eğitim
delta_rule = DeltaRule()
delta_rule.fit(X, y)

# Tahmin yapma
predictions = delta_rule.predict(X)

# Sonuçları yazdırma
print("Tahminler:", predictions)

# D

import numpy as np

class DeltaRuleSGD:
    def __init__(self, learning_rate=0.01, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0

        for _ in range(self.epochs):
            for i in range(len(X)):
                # Rastgele bir veri noktası seç
                j = np.random.randint(0, len(X))

                prediction = self.predict(X[j])
                error = y[j] - prediction
                gradient = error * X[j]
                self.w -= self.learning_rate * gradient
                self.b -= self.learning_rate * error

    def predict(self, X):
        return np.dot(X, self.w) + self.b

# Örnek veri
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([1, 0, 1])

# Model oluşturma ve eğitim
delta_rule_sgd = DeltaRuleSGD()
delta_rule_sgd.fit(X, y)

# Tahmin yapma
predictions = delta_rule_sgd.predict(X)

# Sonuçları yazdırma
print("Tahminler:", predictions)
