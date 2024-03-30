from cv2 import mean
import numpy as np
import matplotlib.pyplot as plt
import os




def read_txt(addr, plot=False):
    filename = addr
    with open(filename, "r") as f:
        data = f.readlines()
    data_list = []
    for line in data:
        parts = line.split()
        if len(parts) == 0:
            continue
        data_list.append(np.float32(parts))
    data_array = np.array(data_list)
    if plot:
        plt.scatter(data_array[:, 0], data_array[:, 1], c=data_array[:, 2])
        plt.show()
    return data_array


def perceptron(X, Y, eta=0.1, epochs=1000, stochastic=False, w = [], b = np.random.rand()):
    
    if len(w) == 0:
        w = []
        for i in range(len(X[0])):
            w.append(np.random.rand())
    else:
        w = w

    for epoch in range(epochs):
        if stochastic:
            for i in range(len(X)):
                i = np.random.randint(0, len(X))
                y_pred = predict_threshold(X[i], w, b)
                error = Y[i] - y_pred
                for j in range(len(X[0])):
                    w[j] += eta * error * X[i][j]
                b += eta * error
        else:
            for i in range(len(X)):
                y_pred = predict_threshold(X[i], w, b)
                error = Y[i] - y_pred
                for j in range(len(X[0])):
                    w[j] += eta * error * X[i][j]
                b += eta * error
    print("Weights :" + str(w) + "\nBias :" + str(b))
    return w, b


def loss(y_true, y_pred):
  return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def sigmoid(z):
    return np.float64(1/(1+np.exp(-z)))

def predict_sigmoid(x, w, b):
    pred = 0
    for i in range(len(x)):
        pred += x[i] * w[i]
    pred += b
    return sigmoid(pred)

def predict_threshold(x, w, b):
    pred = 0
    for i in range(len(x)):
        pred += x[i] * w[i]
    pred += b
    return 1 if (pred) > 0 else 0

def decision_boundary(x, w, b):
    y_line = 0 
    x_line = np.linspace(min(x[:, 0]), max(x[:, 1]), 100)

    y_line = (-w[1] * x_line - w[0] * x_line - b) / w[1]

    return x_line, y_line 

def plot_decision_boundary(x, y, w, b, label="Label"): 
    if len(x[0]) == 2:
        x_line , y_line = decision_boundary(x, w, b) 
        plt.scatter(x[:, 0], x[:, 1], c=y)
        plt.plot(x_line, y_line, label=label)
    elif len(x[0]) == 3:
        plt.scatter(x[:, 0], x[:, 1], x[:,2], c=y)
    else:
        print("Error: x should have 2 or 3 columns")
    plt.legend()
    # plt.show()

def gradient_descent_sigmoid(X, Y, eta=0.1, epochs=1000, w = [], b = np.random.rand(),stochastic=False):
    y_pred = []
    if len(w) == 0:
        w = []
        for i in range(len(X[0])):
            w.append(np.random.rand())
    else:
        w = w
    if stochastic:
        for epoch in range(epochs):
            for i in range(len(X)):
                i = np.random.randint(0, len(X))
                y_pred.append(predict_sigmoid(X[i], w, b))
                error = Y[i] - y_pred[-1]

                for j in range(len(X[0])):
                    w[j] += eta * error * X[i][j] * 2 / len(X)
                b += eta * error * 2
    else:
        for epoch in range(epochs):
            for i in range(len(X)):
                y_pred.append(predict_sigmoid(X[i], w, b))
                error = Y[i] - y_pred[-1]

                for j in range(len(X[0])):
                    w[j] += eta * error * X[i][j] * 2 / len(X)
                b += eta * error * 2
    
    return w, b , y_pred

def show_results(X_te, Y_te, w, b, show_prediction = False):
    
    y_pred = 1 / (1 + np.exp(-(np.dot(X_te, w) + b)))
    error = 0
    for i in range(len(y_pred)):
        error += Y_te[i] - y_pred[i]

    print("Predictions: ", np.around(y_pred,3))
    print("Sum of dataset test error:", error)
    print("Weights: ", w)
    print("Bias: ", b)

    if show_prediction==True:
        print("Predictions: ", np.around(y_pred,3))
        # print("Real Values: ", Y_te)
    return y_pred


def gradient_with_two_labels(cls3_train):
    ogrenme_orani = 0.1
    epoch_sayisi = 100

    # Dataset parsing
    x_train = cls3_train[:, 0:2]
    y_train = cls3_train[:, 2:4]

    # Gizli Katman Ağırlıkları ve Eğiklikler
    w1 = np.random.rand(2, 3)
    b1 = np.random.rand(3)
    w2 = np.random.rand(3, 2)
    b2 = np.random.rand(2)

    error_arr = []
    # Gradient Descent
    for epoch in range(epoch_sayisi):

        # Forward Propagation
        a1 = sigmoid(np.dot(x_train, w1) + b1)
        a2 = sigmoid(np.dot(a1, w2) + b2)

        # Error Calculation
        error =  a2 - y_train

        error_arr.append(np.mean(np.square(error)))

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

    predict = sigmoid(np.dot(a1, w2[:,0]) + b2[0])

    x_line = np.linspace(min(x_train[:, 0]), max(x_train[:, 1]), 100)

    y_11 = ((-w2[0, 0] - w2[1, 0] - w2[2, 0]) * x_line - b2[0]) / w2[1, 1]
    y_22 = ((-w2[0, 1] - w2[1, 1] - w2[2, 1]) * x_line - b2[1]) / w2[1, 1]


    plt.plot(error_arr, color='blue')
    plt.show()
    plt.plot(x_line, y_11, color='red')
    plt.plot(x_line, y_22, color='red')

    plt.scatter(x_train[:, 0], x_train[:, 1], c=cls3_train[:, 1:2])



    print("Error:", np.mean(np.square(error)))
    print("Weights:", w2)
    print("Bias:", b2)
    plt.show()
    return w2, b2