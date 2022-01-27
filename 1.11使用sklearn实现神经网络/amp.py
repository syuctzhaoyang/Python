# _*_ coding: utf-8 _*_
import sklearn.datasets
import matplotlib.pyplot as plt
import sklearn.neural_network
import sklearn.preprocessing
import numpy as np

digits = sklearn.datasets.load_digits()

# fig = plt.figure(figsize=(8,8))
# fig.subplots_adjust(left=0,right = 1,bottom=0,top = 1,hspace=0.05,wspace=0.05)
# for i in range(36):
#     ax = fig.add_subplot(6,6,i+1,xticks = [],yticks = [])
#     ax.imshow(digits.images[i],cmap=plt.cm.binary,interpolation='nearest')
#     ax.text(0,7,str(digits.target[i]),color="red", fontsize = 20)
# plt.show()

scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(digits.data)               ##设置归一化模板
X_scaled = scaler.transform(digits.data)  ##应用归一化模板

mlp = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(30,30,30)
                                           ,activation='relu'
                                           ,max_iter=1000
                                           ,solver='sgd'
                                           ,learning_rate='constant'
                                           ,learning_rate_init=0.001)
mlp.fit(X_scaled,digits.target)

predicted = mlp.predict(X_scaled)
# fig = plt.figure(figsize=(8,8))
# fig.subplots_adjust(left=0,right = 1,bottom=0,top = 1,hspace=0.05,wspace=0.05)
# for i in range(36):
#     ax = fig.add_subplot(6,6,i+1,xticks = [],yticks = [])
#     ax.imshow(digits.images[i],cmap=plt.cm.binary,interpolation='nearest')
#     ax.text(0,7,str('{}-{}'.format(digits.target[i],predicted[i])),color="red", fontsize = 20)
# fig.show()

import sklearn.metrics

print(sklearn.metrics.accuracy_score(digits.target, predicted))

print(sklearn.metrics.confusion_matrix(digits.target, predicted))



