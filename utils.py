import tensorflow as tf
import numpy as np 


def conv2d(input,ksize_h,ksize_w,channel_output,stride,name,with_bias=True,padding="SAME",mean=0.0,stddev=0.01,reuse=False):
    with tf.variable_scope(name,reuse=reuse) as scope:
        filter = tf.get_variable("filter",shape=[ksize_h,ksize_w,input.get_shape()[-1],channel_output],
                                dtype=tf.float32,initializer=tf.random_normal_initializer(mean,stddev,seed=520))
        conv = tf.nn.conv2d(input,filter,[1,stride,stride,1],padding=padding)

        if with_bias == True:
            bias = tf.get_variable("bias",shape=[channel_output],dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv,bias)
        
        return conv

