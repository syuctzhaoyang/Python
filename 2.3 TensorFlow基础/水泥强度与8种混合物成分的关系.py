# _*_ coding: utf-8 _*_
'''
数据集concrete.csv共有1030条数据，9个字段，包含混凝土抗压强度的数据，字段及说明如下：
        字段	说明
        cement	使用的水泥
        slag	矿渣
        ash	灰
        water	水
        superplastic	超塑化剂
        coarseagg	粗集料
        fineagg	细集料
        age	老化时间
        strength	混凝土强度
请按要求完成下列操作：
利用TensorFlow建立全连接神经网络模型，探究strength与8种混合物成分的关系
迭代轮数设为1000，损失函数设为 l2l2 损失，优化方法设为Adam
将数据集分为训练集和测试集，测试集的比例为0.15
训练模型后，请在测试集上做出预测，将预测结果保存在ndarray对象y_pred中，并输出测试集的 l2l2 损失，保存在变量testing_loss中
正误判定变量：testing_loss
'''

# 导入需要的包
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd

# 读取数据集
concrete = pd.read_csv("concrete.csv")

# 进行数据的归一化
concrete = concrete.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
X = concrete.drop('strength', axis=1).values
y = concrete['strength'].values

# 将数据集分为训练集和测试集，测试集的比例为0.15
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=10)

# 设置学习率
learning_rate = 0.000001
# 设置训练轮数
training_epochs = 1000
## 每50轮展示当前模型的损失
display_step = 50

# 定义输入输出的张量占位符
x_ = tf.placeholder(shape=[None, 8], dtype=tf.float32, name="X_train")
y_ = tf.placeholder(shape=[None], dtype=tf.float32, name="y_train")

# 定义权重和偏置
with tf.variable_scope("DNN", reuse=tf.AUTO_REUSE):
    # 设置模型的权重和偏置
    W1 = tf.get_variable(initializer=tf.truncated_normal(shape=[8, 16], stddev=0.1), name="weight1")  # 生成权重
    b1 = tf.get_variable(initializer=tf.truncated_normal(shape=[16], stddev=0.1), name="bias1")  # 生成偏置
    W2 = tf.get_variable(initializer=tf.truncated_normal(shape=[16, 1], stddev=0.1), name="weight2")  # 生成权重
    b2 = tf.get_variable(initializer=tf.truncated_normal(shape=[1], stddev=0.1), name="bias2")  # 生成偏置

    layer1 = tf.nn.relu(tf.matmul(x_, W1) + b1)
    y = tf.matmul(layer1, W2) + b2

# 定义损失，使用的是 l2 损失
loss = tf.nn.l2_loss(y - y_);

# 定义训练的优化方法，优化方法使用Adam，优化目标为最小化损失函数
with tf.variable_scope("Adam", reuse=tf.AUTO_REUSE):
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# 建立会话运行程序
with tf.Session() as sess:
    # 初始化变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # 拟合训练数据
    for epoch in range(training_epochs):
        # 带入数据进行训练模型
        _, cost = sess.run([train_op, loss], feed_dict={x_: X_train, y_: y_train})

        # 展示训练的日志，输出损失
        if (epoch + 1) % display_step == 0:
            print("Epoch:", '%4d' % (epoch + 1), "loss=", "{:.3f}".format(cost ** 0.5))

    print("Optimization Finished!")

    # 输出测试集预测结果
    y_pred = sess.run(y, feed_dict={x_: X_test, y_: y_test})

    # 计算最终损失函数
    testing_loss = sess.run(loss, feed_dict={x_: X_train, y_: y_train})
    print("Testing loss = %.3f" % testing_loss ** 0.5)