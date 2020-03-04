# 代价函数

  常使用的代价函数有均方误差(mean-square-error,简称mse)、二次代价函数(quadratic cost)和交叉熵代价函数(cross-entropy），交叉熵代价函数中又可细化2-class和n-class等。
  
## 均方误差

  mse = ((y实际值 - y预测值）** 2）/n
  
   tf.losses.mean_squared_error(y - prediction)
   
   tf.reduce_mean(tf.square(y - prediction))
   
## 二次代价函数

  C = sum(||y - a||**2)/(2 * n)

  C表示代价， y表示样本，a表示输出值，n表示样本的个数
  
  * 个人理解：均方误差与二次代价函数本质上是一致的，二次代价函数在求导时可以消去分母上的2，使导数是多项式形式并且最高项系数为1，有利于下一步激活函数的计算
  
  tf.reduce_mean(tf.square(y - prediction))
 
## 交叉熵代价函数

  均方误差和二次代价函数的权值和偏置值的调整与激活函数的导数成正比，交叉熵代价函数的权值和偏置值的调整与激活函数的导数是无关的。如果输出神经元的激活函数是线性的(relu)，二次代价函数是一种合适的选择。如果输出神经元的激活函数是S型函数(sigmoid或tanh)，比较适用交叉熵代价函数。
  
## 对数似然代价函数(log-likehood cost)

  对数似然代价函数常用来作为softmax回归的代价函数。深度学习中更普遍的做法是将softmax作为最后一层，此时常用的代价函数是对数似然代价函数。
  
  对数似然代价函数与softmax的组合和交叉熵与sigmoid函数组合非常相似。对数似然代价函数在二分类时可以简化为交叉熵代价函数的形式。
  
  在Tensorflow中用：
  
  * tf.nn.sigmoid_cross_entropy_with_logits()来表示和sigmoid搭配使用的交叉熵。
  
  * tf.nn.softmax_cross_entropy_with_logits()来表示和softmax搭配使用的交叉熵。
  
  
