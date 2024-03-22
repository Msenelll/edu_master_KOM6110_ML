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

X_tr = cls2_train[:, 0:2]
Y_tr = cls2_train[:, 2]

X_te = cls2_test[:, 0:2]
Y_te = cls2_test[:, 2]

print(len(cls2_train[:, 2]))

for epoch in range(0,len(cls2_train)):
    # Tahmini hesaplayın
    # y = cls2_train[epoch, 2]
    # x = cls2_train[epoch, 0:2]
    # print(epoch)
    # print(x)

    # Rosenblatt’s Perceptron
    # y_pred = np.dot(X[epoch], w) + b
    y_pred = 1 if (X_tr[epoch,0]*w[0] + X_tr[epoch,1]*w[1] + b) > 0 else 0
    # # Hatayı hesaplayın
    error = Y_tr[epoch] - y_pred
    
    # # Ağırlıkları ve eşik değerini güncelleyin
    w[0] += eta * error * X_tr[epoch,0]
    w[1] += eta * error * X_tr[epoch,1]
    b += eta * error

# Tahmini çizmek için
# Verileri ve etiketleri tanımlayın

x_train = cls2_train[:, 0:2]
y_train = cls2_train[:, 2]

# Karar sınırını çizmek için
x_line = np.linspace(min(X_tr[:, 0]), max(X_tr[:, 1]), 100)


# y_line_0 = (-w[0] * x_line_0 - b) / w[0]
y_line = (-w[1] * x_line - w[0] * x_line + b)
plt.plot(x_line, y_line, color='red')

# # Verileri ve karar sınırını gösterin
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
plt.show()


# print(y_pred)
