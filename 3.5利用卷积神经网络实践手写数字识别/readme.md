## 利用卷积神经网络实践手写数字识别
- 把x_train = x_train.reshape(60000, 28, 28, 1)和x_test = x_test.reshape(10000, 28, 28, 1)reshape成2维图像数据。
- 利用keras.layers.Conv2D做2维卷积
- filter = 16,意为使用16个不同的卷积核对原始图片进行卷积，输出也是16个通道数据。从而获取不同角度的图像特征。
- kernel_size=(3, 3)，卷积核为3*3
- 采用最大池化，池化核为2*2。keras.layers.MaxPool2D(pool_size=(2, 2)。
