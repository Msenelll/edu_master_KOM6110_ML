import numpy as np
import matplotlib.pyplot as plt
import os
import lib


class2_train_addr = os.path.join(os.path.dirname(__file__), 'assign1\class2_tr.txt')
class2_test_addr = os.path.join(os.path.dirname(__file__), 'assign1\class2_test.txt')

class3_train_addr = os.path.join(os.path.dirname(__file__), 'assign1\class3_tr.txt')
class3_test_addr = os.path.join(os.path.dirname(__file__), 'assign1\class3_test.txt')



# Read the data

cls2_test = lib.read_txt(class2_test_addr, plot=False)
cls2_train = lib.read_txt(class2_train_addr, plot=False)

cls3_test = lib.read_txt(class3_test_addr, plot=False)
cls3_train = lib.read_txt(class3_train_addr, plot=False)
    

# Ağırlık
w = [np.random.rand(), np.random.rand()]

# Eşik değeri
b = np.random.rand()

# Öğrenme oranı
eta = 0.1
epochs = 1000

X_tr = cls2_train[:, 0:2]
Y_tr = cls2_train[:, 2]

X_te = cls2_test[:, 0:2]
Y_te = cls2_test[:, 2]

y_pred = []

for epoch in range(epochs):
    for i in range(len(X_tr)):
        i = np.random.randint(0, len(X_tr))
        y_pred.append(lib.predict_sigmoid(X_tr[i], w, b))
        error = Y_tr[i] - y_pred[-1]

        for j in range(len(X_tr[0])):
            w[j] += eta * error * X_tr[i][j] * 2 / len(X_tr)
        b += eta * error * 2


# y_pred = 1 / (1 + np.exp(-(np.dot(X_te, w) + b)))

# # Doğruluk Hesaplama

# accuracy = np.mean(y_pred == Y_te)

# # Sonuçları Yazdırma
# print("Test Verisi Doğruluk:", accuracy)
# print("Tahminler:", y_pred)

lib.show_results(X_te, Y_te, w, b)