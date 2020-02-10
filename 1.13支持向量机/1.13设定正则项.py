# _*_ coding: utf-8 _*_
import sklearn.datasets
import sklearn.svm
import sklearn.linear_model
import numpy as np
import matplotlib.pyplot as plt

data = np.array(
    [[-1, 2, 0], [-2, 3, 0], [-2, 5, 0], [-3, -4, 0], [-0.1, 2, 0], [0.2, 1, 1], [0, 1, 1], [1, 2, 1], [1, 1, 1],
     [-0.4, 0.5, 1], [2, 5, 1]])

X = data[:, :2]
y = data[:, 2]

clf1 = sklearn.svm.SVC(kernel="linear",C=1.0)
clf1.fit(X, y)

clf2 = sklearn.svm.SVC(kernel="linear",C=10000000)
clf2.fit(X, y)


def plot_estimator(estimator, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.plot()
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.brg)
    plt.xlabel('Petal.Length')
    plt.ylabel('Petal.Width')
    plt.show()


plot_estimator(clf1, X, y)
plot_estimator(clf2, X, y)
