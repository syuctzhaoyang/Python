# _*_ coding: utf-8 _*_
import keras
# import matplotlib.pyplot as plt

n_hidden_1 = 256
n_hidden_2 = 256
n_input = 784
n_classes = 10
train_epochs = 15
batch_size = 100

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# plt.imshow(x_test[4])
# plt.show()
x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train, x_test = x_train / 255.0, x_test / 255.0

y_train = keras.utils.to_categorical(y_train,num_classes=n_classes)
y_test = keras.utils.to_categorical(y_test,num_classes=n_classes)

# print(y_test[4])
model = keras.Sequential()
model.add(keras.layers.Dense(n_hidden_1,activation='relu',input_shape=(n_input,)))
model.add(keras.layers.Dense(n_hidden_2,activation='relu'))
model.add(keras.layers.Dense(n_classes,activation='softmax'))

model.compile(
    loss='categorical_crossentropy'
   ,optimizer='sgd'
    ,metrics=['accuracy']
)

history = model.fit(x_train,y_train
                    ,batch_size=batch_size
                    ,epochs=train_epochs
                    ,verbose=True
                    ,validation_data=(x_test,y_test))









