# _*_ coding: utf-8 _*_
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

# model = keras.Sequential()
# model.add(keras.layers.Dense(units=8,kernel_initializer='uniform',activation='relu',input_shape=(16,)))
# model.add(keras.layers.Dense(units = 1,kernel_initializer='uniform',activation='sigmoid'))
#
# model.compile(
#     #因为此处输出为二元，0为留下，1为流失，所以使用binary_crossentropy作为代价函数
#     loss='binary_crossentropy'
#    ,optimizer='sgd'
#     ,metrics=['accuracy']
# )
#
# history = model.fit(x_train,y_train
#                     ,batch_size=10
#                     ,epochs=200
#                     ,verbose=True
#                     ,validation_data=(x_test,y_test))

# model.save('m2.h5')
model = keras.models.load_model('m2.h5')

y_pred = model.predict(x_test)
#大于0.5，客户会流失，小于0.5，客户会留下来
#flatten   2维数组拉平，变成1维的数组
#astype   使用0或1而不是用true或false 作为输出结果
y_pred = (y_pred>0.5).flatten().astype(int)

#预测测试数据的准确率
print(sum(y_pred == y_test)/len(y_test))
print('------------------------------')
print(sklearn.metrics.accuracy_score(y_test, y_pred))
print('------------------------------')
cm = sklearn.metrics.confusion_matrix(y_test,y_pred)
print(cm)

print('==============================')
print(sklearn.metrics.classification_report(y_test, y_pred))

