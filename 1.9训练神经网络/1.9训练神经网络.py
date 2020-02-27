# _*_ coding: utf-8 _*_
import numpy as np


class ANN:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.parms = {}
        self.parms['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.parms['b1'] = np.zeros(hidden_size)
        self.parms['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.parms['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.parms['W1'], self.parms['W2']
        b1, b2 = self.parms['b1'], self.parms['b2']
        a1 = np.dot(x, W1) + b1
        z1 = self.sigmoid_function(a1)
        a2 = np.dot(z1, W2) + b2
        y = self.softmax_function(a2)
        return y

    def sigmoid_function(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax_function(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def cross_entropy_err(self,y_hat, y):
        delta = 1e-8
        return -np.sum(y * np.log(y_hat + delta))

    def loss(self, x, y):
        y_hat = self.predict(x)
        return self.cross_entropy_err(y_hat, y)

    def dfunc(self,f, x):
        h = 1e-4
        grad = np.zeros_like(x)
        it = np.nditer(x, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            tmp_val = x[idx]
            x[idx] = float(tmp_val) + h
            fxh1 = f(x)  # f(x+h)
            x[idx] = tmp_val - h
            fxh2 = f(x)  # f(x-h)
            grad[idx] = (fxh1 - fxh2) / (2 * h)
            x[idx] = tmp_val
            it.iternext()
        return grad

    def numerical_gradient(self, x, y):
        loss_W = lambda W: self.loss(x, y)
        grads = {}
        grads['W1'] = self.dfunc(loss_W, self.parms['W1'])
        grads['b1'] = self.dfunc(loss_W, self.parms['b1'])
        grads['W2'] = self.dfunc(loss_W, self.parms['W2'])
        grads['b2'] = self.dfunc(loss_W, self.parms['b2'])
        return grads


net = ANN(input_size=4, hidden_size=5, output_size=3)

print(net.parms['W1'].shape)
print(net.parms['b1'].shape)

print(net.parms['W2'].shape)
print(net.parms['b2'].shape)

import sklearn.datasets

iris = sklearn.datasets.load_iris()
x = iris.data
print(x)

y = np.zeros([len(iris.target), 3])
for idx, val in enumerate(iris.target):
    y[idx, val] = 1
print(y)

y_hat = net.predict(x)
print(y_hat)
print('----------------------------------------------------------')
epochs = 3000
lr = 0.01
train_loss = []

for i in range(epochs):
    grad = net.numerical_gradient(x,y)
    for key in ('W1','b1','W2','b2'):
        net.parms[key] = net.parms[key] - lr * grad[key]
    loss = net.loss(x,y)
    train_loss.append(loss)

print(train_loss)