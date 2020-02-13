# _*_ coding: utf-8 _*_
import keras
import sklearn
import keras.wrappers.scikit_learn
import pandas as pd

df = pd.read_csv("customer_churn.csv", index_col=0, header=0)
df = df.iloc[:, 3:]

# 数据清洗，将含有‘yes'或者'no'的数据转换为one_hot编码

cat_vars = ['international_plan', 'voice_mail_plan', 'churn']
for var in cat_vars:
    df[var] = df[var].map(lambda e: 1 if e == 'yes' else 0)
# 检查数据是否全部转化为数值型数据
print(df.info())

y = df.iloc[:, -1]
X = df.iloc[:, :-1]

# 区分训练与测试数据集

import sklearn.model_selection

#  random_state=123,相当于random类中的seed,设置可复现随机数据
(x_train, x_test, y_train, y_test) = sklearn.model_selection.train_test_split(X, y, test_size=0.33, random_state=123)

# 数据归一化
import sklearn.preprocessing

sc = sklearn.preprocessing.StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


def trainProcess(optimizer):
    model = keras.Sequential()
    model.add(keras.layers.Dense(units=8, kernel_initializer='uniform', activation='relu', input_shape=(16,)))
    model.add(keras.layers.Dropout(rate=0.1))
    model.add(keras.layers.Dense(units=8, kernel_initializer='uniform', activation='relu'))
    model.add(keras.layers.Dropout(rate=0.1))
    model.add(keras.layers.Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    model.compile(
        # 因为此处输出为二元，0为留下，1为流失，所以使用binary_crossentropy作为代价函数
        loss='binary_crossentropy'
        , optimizer=optimizer
        , metrics=['accuracy']
    )
    return model


classifier = keras.wrappers.scikit_learn.KerasClassifier(
    build_fn=trainProcess
    , epochs=100
)
parameters = {'batch_size': [10, 15], 'optimizer': ['adam', 'rmsprop']}
grid_search = sklearn.model_selection.GridSearchCV(
    estimator=classifier
    , param_grid=parameters
    , scoring='accuracy'
    , cv=5
)

grid_search = grid_search.fit(x_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print(best_parameters)
print(best_accuracy)
