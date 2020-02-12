#_*_ coding: utf-8 _*_
import pandas as pd

df = pd.read_csv("customer_churn.csv", index_col=0, header=0)
df = df.iloc[:, 3:]
# print(df.columns)

# 数据清洗，将含有‘yes'或者'no'的数据转换为one_hot编码

cat_vars = ['international_plan', 'voice_mail_plan', 'churn']
for var in cat_vars:
    df[var] = df[var].map(lambda e: 1 if e == 'yes' else 0)
# 检查数据是否全部转化为数值型数据
print(df.info())

y = df.iloc[:, -1]
X = df.iloc[:, :-1]
# print(x.shape)
# print(y.shape)

# 区分训练与测试数据集

import sklearn.model_selection
#  random_state=123,相当于random类中的seed,设置可复现随机数据
(x_train, x_test, y_train, y_test) = sklearn.model_selection.train_test_split(X, y, test_size=0.33, random_state=123)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# 数据归一化
import sklearn.preprocessing
sc = sklearn.preprocessing.StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
#print(x_test)

import keras


def trainProcess(optimizer):
    model = keras.Sequential()
    model.add(keras.layers.Dense(units=8, kernel_initializer='uniform', activation='relu', input_shape=(16,)))
    model.add(keras.layers.Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    model.compile(
        # 因为此处输出为二元，0为留下，1为流失，所以使用binary_crossentropy作为代价函数
        loss='binary_crossentropy'
        , optimizer=optimizer
        , metrics=['accuracy']
    )
    history = model.fit(x_train, y_train
                        , batch_size=10
                        , epochs=100
                        , verbose=True
                        , validation_data=(x_test, y_test))

    return history
history1 = trainProcess('sgd')

history2 = trainProcess('RMSprop')
history3 = trainProcess('Adagrad')
history4 = trainProcess('Adadelta')
history5 = trainProcess('Adam')

import pickle

with open('history1.txt', 'wb') as file_pi:
    pickle.dump(history1.history, file_pi)
with open('history2.txt', 'wb') as file_pi:
    pickle.dump(history2.history, file_pi)
with open('history3.txt', 'wb') as file_pi:
    pickle.dump(history3.history, file_pi)
with open('history4.txt', 'wb') as file_pi:
    pickle.dump(history4.history, file_pi)
with open('history5.txt', 'wb') as file_pi:
    pickle.dump(history5.history, file_pi)

# with open('history1.txt','rb') as file_pi:
#     history1=pickle.load(file_pi)
# with open('history2.txt','rb') as file_pi:
#     history2=pickle.load(file_pi)
# with open('history3.txt','rb') as file_pi:
#     history3=pickle.load(file_pi)
# with open('history4.txt','rb') as file_pi:
#     history4=pickle.load(file_pi)
# with open('history5.txt','rb') as file_pi:
#     history5=pickle.load(file_pi)
# for key in history1.keys:
#     print(key)
# print(history2.keys())
#
# # for idx,value in enumerate(history1['accuracy']):
# #     print("{}---{}".format(idx,value))
# # print(enumerate(history1['accuracy']))
# import matplotlib.pyplot as plt
#
# plt.figure(num = 1,figsize=(8,6),dpi=80,facecolor='w',edgecolor='k')
# plt.plot(history1['val_loss'],label = 'SGD')
# plt.plot(history2['val_loss'],label = 'RMSprop')
# plt.plot(history3['val_loss'],label = 'Adagrad')
# plt.plot(history4['val_loss'],label = 'Adadelta')
# plt.plot(history5['val_loss'],label = 'Adam')
# plt.title('val_loss')
# plt.legend()
# plt.savefig('val_loss')
# plt.show()
#
# plt.figure(num = 2,figsize=(8,6),dpi=80,facecolor='w',edgecolor='k')
# plt.plot(history1['val_accuracy'],label = 'SGD')
# plt.plot(history2['val_accuracy'],label = 'RMSprop')
# plt.plot(history3['val_accuracy'],label = 'Adagrad')
# plt.plot(history4['val_accuracy'],label = 'Adadelta')
# plt.plot(history5['val_accuracy'],label = 'Adam')
# plt.title('val_accuracy')
# plt.legend()
# plt.savefig('val_accuracy')
# plt.show()
#
# plt.figure(num = 3,figsize=(8,6),dpi=80,facecolor='w',edgecolor='k')
# plt.plot(history1['loss'],label = 'SGD')
# plt.plot(history2['loss'],label = 'RMSprop')
# plt.plot(history3['loss'],label = 'Adagrad')
# plt.plot(history4['loss'],label = 'Adadelta')
# plt.plot(history5['loss'],label = 'Adam')
# plt.title('loss')
# plt.legend()
# plt.savefig('loss')
# plt.show()
#
# plt.figure(num = 4,figsize=(8,6),dpi=80,facecolor='w',edgecolor='k')
# plt.plot(history1['accuracy'],label = 'SGD')
# plt.plot(history2['accuracy'],label = 'RMSprop')
# plt.plot(history3['accuracy'],label = 'Adagrad')
# plt.plot(history4['accuracy'],label = 'Adadelta')
# plt.plot(history5['accuracy'],label = 'Adam')
# plt.title('accuracy')
# plt.legend()
# plt.savefig('accuracy')
# plt.show()


