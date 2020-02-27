# _*_ coding: utf-8 _*_
# 1---------threshold_function------------------------------
import numpy as np
import matplotlib.pyplot as plt


def threshold_function(x):
    y = x > 0
    return y.astype(int)


x = np.array([-1, 1, 2])
print(threshold_function(x))


# 2.--------sigmoid_function-------------------------------

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))


# x = np.array([-1, 1, 2])
# print(sigmoid_function(x))
x = np.arange(-10, 10, 0.1)
plt.plot(x, sigmoid_function(x), c='red')
plt.plot(x, np.zeros(len(x)) + 1, c='blue')
plt.show()


# 3.--------sigmoid_function-------------------------------

def tangent_function(x):
    return (1 - np.exp(-2 * x)) / (1 + np.exp(-2 * x))


x = np.array([-1, 1, 2])
print(tangent_function(x))
print(np.tanh(x))
# x = np.arange(-10, 10, 0.1)
# plt.plot(x, tangent_function(x), c='red')
# plt.plot(x, np.zeros(len(x)) - 1, c='blue')
# plt.plot(x, np.zeros(len(x)), c='blue')
# plt.plot(x, np.zeros(len(x)) + 1, c='blue')
# plt.plot(np.zeros(21),np.arange(-1,1.1,0.1), c='orange')
# plt.show()



# 4.--------relu_function-------------------------------

def relu_function(x):
    return np.maximum(0,x)

# x = np.array([-1, 1, 2])
# print(relu_function(x))
x = np.arange(-10, 10, 0.1)
plt.plot(x, relu_function(x), c='red')
plt.xticks()
plt.yticks()
plt.show()


# 5.--------单层神经网络向前传播过程--------------------------

import numpy as np
X = np.array([1,2])
W = np.array([[1,3,5],[2,4,6]])
b = np.array([7,8,9])
Y = np.add(np.dot(X,W),b)
print(Y)


# 6.--------两层神经网络向前传播过程--------------------------

import numpy as np
network ={}
network['w1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
network['b1'] = np.array([0.1,0.2,0.3])
network['w2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
network['b2'] = np.array([0.1,0.2])

x = np.array([1,0.5])
a = np.dot(x,network['w1'])+network['b1']
z = sigmoid_function(a)
y = np.dot(z,network['w2'])+network['b2']
print(y)

# 7.--------softmax_fucntion--------------------------------

def softmax_fucntion(x):
    return np.exp(x) / np.sum(np.exp(x))


print(softmax_fucntion(y))
print(np.sum(softmax_fucntion(y)))






