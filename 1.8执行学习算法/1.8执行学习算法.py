# _*_ coding: utf-8 _*_
# 1.----------均方误差------mse---------------------------
import numpy as np
import matplotlib.pyplot as plt


def mean_squard_err(y_hat, y):
    return 0.5 * np.sum((y_hat - y) ** 2)


y_hat = np.array([0, 0, 0, 0.2, 0.8])
y = np.array([0, 0, 0, 0, 1])

print(mean_squard_err(y_hat, y))


# 2.----------交叉熵------cross_entropy---------------------

def cross_entropy(y_hat, y):
    # 设置delta为一个小值
    delta = 1e-8
    return -np.sum(y * np.log(y_hat + delta))


print(cross_entropy(y_hat, y))


# 3.----------求微分---------------------------------------

def func(x):
    return x ** 2


def dfunc(f, x):
    h = 1e-4
    return (func(x + h) - f(x)) / (h)


print(dfunc(func, 3))


# 4.----------利用微分求出切线-----------------------------------

def tfunc(f, x, t):
    d = dfunc(f, x)
    y = f(x) - d * x
    return d * t + y


x = np.arange(-6, 6, 0.001)
y = func(x)
plt.plot(x, y)

x2 = np.arange(0, 5, 0.01)
y2 = tfunc(func, 3, x2)
plt.plot(x2, y2)
plt.show()


# 5.----------中间差分-----------------------------------

def dfunc(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


print(dfunc(func, 4))


# 6.----------计算微分通用公式--------------------------------
def tdfunc(f,x):
    h = 1e-4
    grad = np.zeros_like(x)
    it = np.nditer(x,flags = ['multi_index'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x[idx])
        x[idx] = float(tmp_val) - h
        fxh2 = f(x[idx])
        grad[idx] = (fxh2 - fxh1)/(2 * h)
        x[idx] = tmp_val
        it.iternext()
    return grad

x = np.arange(0,9,1).reshape((3,-1)).astype(np.float32)
print(tdfunc(func,x))


# 7.----------梯度下降公式------------------------------


def gradient_descent(func,init_x,lr = 0.3,epoch = 100):
    x = init_x
    res = [x]
    for i in range(epoch):
        grad = dfunc(func,x)
        x = x - grad * lr
        res.append(x)
    return np.array(res)

x = gradient_descent(func,-5,lr=0.9)
# print(x)

t = np.arange(-6,6,0.01)
plt.plot(t,func(t), c = 'b')
plt.plot(x,func(x), c = 'r')
plt.scatter(x,func(x), c = 'r')
plt.show()


# 8.----------计算神经网络梯度------------------------------

def softmax_fucntion(x):
    return np.exp(x) / np.sum(np.exp(x))

x = np.array([0.6,0.9])
np.random.seed(42)
weight = np.random.randn(2,3)
z = np.dot(x,weight)
y_hat = softmax_fucntion(z)

y = np.array([0,0,1])
print(cross_entropy(y_hat, y))

def predict(x):
    return np.dot(x,weight)

def loss(x,y):
    z = predict(x)
    y_hat = softmax_fucntion(z)
    loss = cross_entropy(y_hat,y)
    return loss

func = lambda w:loss(x,y)
print(tdfunc(func, weight))






