'''
输入：单项
输出： 独热二维值
功能：二分类预测
'''

import tensorflow as tf
import numpy as np

def train_data():
    x = [[0.2], [0.4], [0.7], [1.2], [1.4], [1.8], [1.9], [2], [0.11], [0.16], [0.5]]
    y = [[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [1, 0]]
    return (x, y)

def test_data():
    x = [[0.3], [0.6], [0.8], [1.3], [1.5]]
    y = [[1, 0], [1, 0], [1, 0], [0, 1], [0, 1]]
    return (x, y)


def weight_variable(shape,num):
    w = tf.get_variable('weight_%d' % num, shape=shape, initializer=tf.random_normal_initializer(stddev=1, dtype=tf.float32))
    return w

def bias_variable(shape,num):
    b = tf.get_variable('bias_%d' % num, shape=shape, initializer=tf.random_normal_initializer(stddev=1, dtype=tf.float32))
    return b


x = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='x_input')
y_ = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='y_input')

w1 = weight_variable([1, 3],1)
b1 = bias_variable([3],1)
w2 = weight_variable([3, 2],2)
b2 = bias_variable([2],2)

layer1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1)
y = tf.matmul(layer1, w2) + b2

loss = tf.nn.l2_loss(y - y_)

train_op = tf.train.GradientDescentOptimizer(
    learning_rate=0.01).minimize(loss)

correct_predition = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predition, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x_train = train_data()[0]
    y_train = train_data()[1]

    for i in range(1):
        sess.run(train_op,feed_dict={x:x_train,y_:y_train})
    print(sess.run(accuracy,feed_dict = {x:test_data()[0],y_:test_data()[1]}))