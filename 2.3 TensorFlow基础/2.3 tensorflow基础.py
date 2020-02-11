# _*_ coding: utf-8 _*_
import tensorflow as tf

a = tf.constant([[1,2],[3,4]])
b = tf.constant([[3,4],[2,1]])

print(tf.matmul(a,b))
z = tf.zeros_like(a)
print(tf.eye(2, num_columns=3))
print(z)
