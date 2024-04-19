import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('./mnist_train.csv')

# print(data.head())

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

# print(m, n)

data_dev = data[0:100].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / .255

data_train = data[100:m].T
Y_train = data_train[0]
X_train = data[1:n]
X_train = X_train / .255

# print(Y_train)

def init_params():
    w1 = np.random.rand(10, 784) - 0.5 
    b1 = np.random.rand(10, 1) - 0.5
    w2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5

    return w1, b1, w2, b2

def ReLu(z):
    return np.maximum(0, z)

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def ReLu_prime(z):
    return z > 0

def forward_propagation(w1, b1, w2, b2, x, y):
    z1 = np.dot(w1, x) + b1
    a1 = ReLu(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2)

    return z1, a1, z2, a2

def backward_propagation(z1, a1, z2, a2, x, y, w2):
    one_hot_y = one_hot(y)
    dz2 = (a2 - one_hot_y)
    dw2 = 1 / m * a1.T.dot(dz2)
    db2 = 1 / m * np.sum(dz2)
    dz1 = w2.T.dot(dz2) * ReLu_prime(z1)
    dw1 = 1 / m * dz1.dot(x.T)
    db1 = 1 / m * np.sum(dz1)

    return dw1, db1, dw2, db2

def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1

    return w2, b2, w1, b1


def get_predictions(a):
    return np.argmax(a, 0)

def get_accuracy(predictions, y):
    print(predictions, y)
    return np.sum(predictions == y) / y.size

def gradient_descent(x, y, alpha, iterations):
    w1, b1, w2, b2 = init_params()
    for i in range(iterations):
        z1, a1, z2, a2 = forward_propagation(w1, b1, w2, b2, x, y)
        dw1, db1, dw2, db2 = backward_propagation(z1, a1, z2, a2, x, y, w2)
        w2, b2, w1, b1 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)

        if i % 10 == 0:
            predictions = get_predictions(a2)
            print(get_accuracy(predictions, y))
    return w1, b1, w2, b2

w1, b1, w2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)