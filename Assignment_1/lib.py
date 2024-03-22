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
                y_pred = predict_th(X[i], w, b)
                error = Y[i] - y_pred
                for j in range(len(X[0])):
                    w[j] += eta * error * X[i][j]
                b += eta * error
        else:
            for i in range(len(X)):
                y_pred = predict_th(X[i], w, b)
                error = Y[i] - y_pred
                for j in range(len(X[0])):
                    w[j] += eta * error * X[i][j]
                b += eta * error
    print("Weights :" + str(w) + "\nBias :" + str(b))
    return w, b


def loss(y_true, y_pred):
  return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def predict_th(x, w, b):
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




def sigmoid(z):
    return 1/(1+np.exp(-z))

def predict_sigmoid(x, w, b):
    pred = 0
    for i in range(len(x)):
        pred += x[i] * w[i]
    pred += b
    return sigmoid(pred)

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