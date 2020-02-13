# _*_ coding: utf-8 _*_
import keras

# import matplotlib.pyplot as plt

n_hidden_1 = 256
n_hidden_2 = 256
n_input = 784
n_classes = 10
train_epochs = 5
batch_size = 100

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# plt.imshow(x_test[4])
# plt.show()
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train, x_test = x_train / 255.0, x_test / 255.0

y_train = keras.utils.to_categorical(y_train, num_classes=n_classes)
y_test = keras.utils.to_categorical(y_test, num_classes=n_classes)

# print(y_test[4])
model = keras.Sequential()
model.add(keras.layers.Conv2D(
    filters=16
    , kernel_size=(3, 3)
    , padding='same'
    , input_shape=(28, 28, 1)
    , activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(keras.layers.core.Flatten())
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(
    loss='categorical_crossentropy'
    , optimizer='adam'
    , metrics=['accuracy']
)

history = model.fit(x_train, y_train
                    , batch_size=batch_size
                    , epochs=train_epochs
                    , verbose=True
                    , validation_data=(x_test, y_test))

# import pickle

# with open('history.txt', 'wb') as file_pi:
#     pickle.dump(history.history, file_pi)
# with open('history.txt', 'rb') as file_pi:
#     history = pickle.load(file_pi)

import matplotlib.pyplot as plt

plt.figure(num = 1,figsize=(8,6),dpi=80,facecolor='w',edgecolor='k')
plt.plot(history.history['val_accuracy'],label = 'SGD')
plt.title('val_loss')
plt.legend()
plt.savefig('val_loss')
plt.show()
