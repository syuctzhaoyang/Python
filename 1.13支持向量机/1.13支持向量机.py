# _*_ coding: utf-8 _*_
import sklearn.datasets
import sklearn.svm
import sklearn.linear_model
import numpy as np
import matplotlib.pyplot as plt

iris = sklearn.datasets.load_iris()

X = iris.data[1:100, [2, 3]]
y = iris.target[1:100]

clf1 = sklearn.svm.SVC(kernel="linear",C=1.0)
clf1.fit(X, y)

clf2 = sklearn.linear_model.LogisticRegression()
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
