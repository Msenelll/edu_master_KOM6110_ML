import lib
import os
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


if __name__ == "__main__":
    w_0,b_0 = lib.perceptron(cls2_train[:, 0:2], cls2_train[:, 2], eta=0.1, epochs=10, stochastic=False)
    lib.plot_decision_boundary(cls2_train[:, 0:2], cls2_train[:, 2], w_0, b_0, label="Perceptron_trainData")
    lib.plot_decision_boundary(cls2_test[:, 0:2], cls2_test[:, 2], w_0, b_0, label="Perceptron_testData")

    w_1,b_1 = lib.perceptron(cls2_train[:, 0:2], cls2_train[:, 2], eta=0.1, epochs=10, stochastic=True)
    lib.plot_decision_boundary(cls2_train[:, 0:2], cls2_train[:, 2], w_1, b_1, label="Perceptron Stochastic_trainData")
    lib.plot_decision_boundary(cls2_test[:, 0:2], cls2_test[:, 2], w_1, b_1, label="Perceptron Stochastic")

    w_2, b_2, _ = lib.gradient_descent_sigmoid(cls2_train[:, 0:2], cls2_train[:, 2],eta=0.1, epochs=10, stochastic=False)
    lib.plot_decision_boundary(cls2_train[:, 0:2], cls2_train[:, 2], w_2, b_2, label="Gradient Descent_trainData")
    lib.plot_decision_boundary(cls2_test[:, 0:2], cls2_test[:, 2], w_2, b_2, label="Gradient Descent")
    
    w_3, b_3, _ = lib.gradient_descent_sigmoid(cls2_train[:, 0:2], cls2_train[:, 2],eta=0.1, epochs=10, stochastic=True)
    lib.plot_decision_boundary(cls2_train[:, 0:2], cls2_train[:, 2], w_3, b_3, label="Gradient Descent Stochastic_trainData")
    lib.plot_decision_boundary(cls2_test[:, 0:2], cls2_test[:, 2], w_3, b_3, label="Gradient Descent Stochastic")
    
    plt.show()