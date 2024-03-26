import numpy as np
import lib
import os
import lib
import matplotlib.pyplot as plt

class2_train_addr = os.path.join(os.path.dirname(__file__), 'assign1\class2_tr.txt')
class2_test_addr = os.path.join(os.path.dirname(__file__), 'assign1\class2_test.txt')

class3_train_addr = os.path.join(os.path.dirname(__file__), 'assign1\class3_tr.txt')
class3_test_addr = os.path.join(os.path.dirname(__file__), 'assign1\class3_test.txt')



# Read the data
cls2_test = lib.read_txt(class2_test_addr, plot=False)
cls2_train = lib.read_txt(class2_train_addr, plot=False)

cls3_test = lib.read_txt(class3_test_addr, plot=False)
cls3_train = lib.read_txt(class3_train_addr, plot=False)

# Hyperparameters
ogrenme_orani = 0.1
epoch_sayisi = 1000

# Dataset parsing
x_train = cls3_train[:, 0:2]
y_train = cls3_train[:, 2:4]

# Gizli Katman Ağırlıkları ve Eğiklikler
w1 = np.random.rand(2, 3)
b1 = np.random.rand(3)
w2 = np.random.rand(3, 2)
b2 = np.random.rand(2)

# Gradient Descent
for epoch in range(epoch_sayisi):

    # Forward Propagation
    a1 = lib.sigmoid(np.dot(x_train, w1) + b1)
    a2 = lib.sigmoid(np.dot(a1, w2) + b2)

    # Error Calculation
    error = y_train - a2

    # Backpropagation
    d2 = error * (1 - a2) * a2
    dw2 = np.dot(a1.T, d2)
    db2 = np.sum(d2, axis=0)

    d1 = np.dot(d2, w2.T) * (1 - a1) * a1
    dw1 = np.dot(x_train.T, d1)
    db1 = np.sum(d1, axis=0)

    # Weight Update
    w1 -= ogrenme_orani * dw1
    b1 -= ogrenme_orani * db1
    w2 -= ogrenme_orani * dw2
    b2 -= ogrenme_orani * db2

predict = lib.sigmoid(np.dot(a1, w2[:,0]) + b2[0])

x_line = np.linspace(min(x_train[:, 0]), max(x_train[:, 1]), 100)

y_11 = ((-w2[0, 0] - w2[1, 0] - w2[2, 0]) * x_line - b2[0]) / w2[1, 0]
y_22 = ((-w2[0, 1] - w2[1, 1] - w2[2, 1]) * x_line - b2[1]) / w2[1, 1]

plt.plot(x_line, y_11, color='red')
plt.plot(x_line, y_22, color='red')

plt.scatter(x_train[:, 0], x_train[:, 1], c=cls3_train[:, 1:2])

plt.show()

print("Hata:", np.mean(np.square(error)))
print("Ağırlıklar:", w2)
print("Eğiklikler:", b2)


# sonucları grafikleme
# lib.plot_decision_boundary(cls3_train[:, 0:2], a2.T[0], w2[0], b2[0], label="1. sınıf")
# lib.plot_decision_boundary(cls3_train[:, 0:2], a2.T[0], w2[1], b2[0], label="2. sınıf")
# lib.plot_decision_boundary(cls3_train[:, 0:2], a2.T[0], w2[2], b2[0], label="3. sınıf")
# lib.plot_decision_boundary(cls3_train[:, 0:2], a2.T[1], w2[0], b2[1], label="4. sınıf")
# lib.plot_decision_boundary(cls3_train[:, 0:2], a2.T[1], w2[1], b2[1], label="5. sınıf")
# lib.plot_decision_boundary(cls3_train[:, 0:2], a2.T[1], w2[2], b2[1], label="6. sınıf")


# y_1 = (-w2[0, 0] * x_line - b2[0]) / w2[1, 1]
# y_2 = (-w2[1, 0] * x_line - b2[0]) / w2[1, 1]
# y_3 = (-w2[2, 0] * x_line - b2[0]) / w2[1, 1]
# y_4 = (-w2[0, 1] * x_line - b2[1]) / w2[1, 1]
# y_5 = (-w2[1, 1] * x_line - b2[1]) / w2[1, 1]
# y_6 = (-w2[2, 1] * x_line - b2[1]) / w2[1, 1]

# plt.plot(x_line, y_1, color='red')
# plt.plot(x_line, y_2, color='red')
# plt.plot(x_line, y_3, color='red')
# plt.plot(x_line, y_4, color='red')
# plt.plot(x_line, y_5, color='red')
# plt.plot(x_line, y_6, color='red')