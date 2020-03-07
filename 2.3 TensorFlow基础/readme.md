# Tensor形态

## Constant(常数)

## Variable(变量)

可以训练的变量，比如模型的权重(weights)或偏置(bias)

## Placeholder(站位符)

Tensorflow 中是先建好图在决定资料的输入与输出，Placeholder在还没有数据的时候先占个位置，以得到之后传递进来的输入值



## Tensorflow 2.0 语法中不在有Session、 placeholder等，可以视Tensorflow语法与numpy一致就可以

--------------------------------------------------------------------------------
## 6-1卷积神经网络应用于MNIST数据集分类.py

  - 利用tensorflow 1.15 编写的卷积神经网络
  
     
         结构：卷积->relu激活->最大池化->卷积->relu激活->最大池化->全连接层->relu激活->dropout->全连接层->softmax激活化迭代
  
                |                                                                                      |
           
                ------------------<----------利用adam方式优化----<--计算交叉熵---------<------------------


## 7-2递归神经网络RNN.py

  - 利用tensorflow 1.15 编写的递归神经网络RNN完成MNIST数据集的分类

## 8-1训练和保存RNN模型.py

  - saver = tf.train.Saver()    对象实例化一个保存模型的对象
  - saver.save(sess,'net/my_net.ckpt') 调用save方法保存训练好的模型数据
  
## 8-2调用训练好的RNN模型.py

  - saver = tf.train.Saver()    对象实例化一个保存模型的对象
  - saver.restore(sess,'net/my_net.ckpt') 从文件中反向序列化到sess对象中
  - acc = sess.run(accuacy, feed_dict={x: mnist.test.images, y: mnist.test.labels}) 调用模型预测测试集结果
   
## 牛津大学图片数据集   http://www.robots.ox.ac.uk/~vgg/data/
