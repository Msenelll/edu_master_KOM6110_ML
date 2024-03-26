import lib
import os
import matplotlib.pyplot as plt
import numpy as np


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

    while True:
        selection = int(input("1- 2 Feature, 1 Label\n2- 2 Feature 2 Label\n3- Exit\n"))
        print("Your selection is: ", selection)
        if selection == 1:

            X_tr = cls2_train[:, 0:2]
            Y_tr = cls2_train[:, 2]

            X_te = cls2_test[:, 0:2]
            Y_te = cls2_test[:, 2]

            print("Please select the algorithm you want to use")
            assign_selection = input("1- Perceptron\n2- Perceptron-Stochastic\n3- Delta\n4- Delta-Stochastic\n5- All\n0- Exit\n")

            if assign_selection == "1":
                w_0,b_0 = lib.perceptron(X_tr, Y_tr, eta=0.1, epochs=10, stochastic=False)
                lib.plot_decision_boundary(X_tr, Y_tr, w_0, b_0, label="Perceptron_trainData")
                lib.plot_decision_boundary(X_te, Y_te, w_0, b_0, label="Perceptron_testData")
                lib.show_results(X_te, Y_te, w_0, b_0,show_prediction=True)
                plt.show()
            elif assign_selection == "2":
                w_1,b_1 = lib.perceptron(X_tr, Y_tr, eta=0.1, epochs=10, stochastic=True)
                lib.plot_decision_boundary(X_tr, Y_tr, w_1, b_1, label="Perceptron Stochastic_trainData")
                lib.plot_decision_boundary(X_te, Y_te, w_1, b_1, label="Perceptron Stochastic")
                lib.show_results(X_te, Y_te, w_1, b_1,show_prediction=True)
                plt.show()
            elif assign_selection == "3":
                w_2, b_2, _ = lib.gradient_descent_sigmoid(X_tr, Y_tr,eta=0.1, epochs=10, stochastic=False)
                lib.plot_decision_boundary(X_tr, Y_tr, w_2, b_2, label="Gradient Descent_trainData")
                lib.plot_decision_boundary(X_te, Y_te, w_2, b_2, label="Gradient Descent")
                lib.show_results(X_te, Y_te, w_2, b_2,show_prediction=True)
                plt.show()
            elif assign_selection == "4":
                w_3, b_3, _ = lib.gradient_descent_sigmoid(X_tr, Y_tr,eta=0.1, epochs=10, stochastic=True)
                lib.plot_decision_boundary(X_tr, Y_tr, w_3, b_3, label="Gradient Descent Stochastic_trainData")
                lib.plot_decision_boundary(X_te, Y_te, w_3, b_3, label="Gradient Descent Stochastic")
                lib.show_results(X_te, Y_te, w_3, b_3,show_prediction=True)
                plt.show()
            elif assign_selection == "5":
                w_0,b_0 = lib.perceptron(X_tr, Y_tr, eta=0.1, epochs=10, stochastic=False)
                lib.plot_decision_boundary(X_tr, Y_tr, w_0, b_0, label="Perceptron_trainData")
                lib.plot_decision_boundary(X_te, Y_te, w_0, b_0, label="Perceptron_testData")
                print("Perceptron Results")
                lib.show_results(X_te, Y_te, w_0, b_0)
                w_1,b_1 = lib.perceptron(X_tr, Y_tr, eta=0.1, epochs=10, stochastic=True)
                lib.plot_decision_boundary(X_tr, Y_tr, w_1, b_1, label="Perceptron Stochastic_trainData")
                lib.plot_decision_boundary(X_te, Y_te, w_1, b_1, label="Perceptron Stochastic")
                print("Perceptron Stochastic Results")
                lib.show_results(X_te, Y_te, w_1, b_1)
                w_2, b_2, _ = lib.gradient_descent_sigmoid(X_tr, Y_tr,eta=0.1, epochs=10, stochastic=False)
                lib.plot_decision_boundary(X_tr, Y_tr, w_2, b_2, label="Gradient Descent_trainData")
                lib.plot_decision_boundary(X_te, Y_te, w_2, b_2, label="Gradient Descent")
                print("Gradient Descent Results")
                lib.show_results(X_te, Y_te, w_2, b_2)
                w_3, b_3, _ = lib.gradient_descent_sigmoid(X_tr, Y_tr,eta=0.1, epochs=10, stochastic=True)
                lib.plot_decision_boundary(X_tr, Y_tr, w_3, b_3, label="Gradient Descent Stochastic_trainData")
                lib.plot_decision_boundary(X_te, Y_te, w_3, b_3, label="Gradient Descent Stochastic")
                print("Gradient Descent Stochastic Results")
                lib.show_results(X_te, Y_te, w_3, b_3)
                plt.show()
            else:
                break
            
        elif selection == 2:
            X_tr = cls3_train[:, 0:2]
            Y_tr = cls3_train[:, 2]

            X_te = cls3_test[:, 0:2]
            Y_te = cls3_test[:, 2]

            w2, b2  = lib.gradient_with_two_labels(cls3_train)

        else:
            break