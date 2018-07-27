'''
Author Akiyama
First edit date: 2018/07/10
Note that different network for different task. Alexnet are useless for mnist and cifar dataset.

logs:
    2018/07/10: the result is not good , whoes accuracy is 64% for the best
    2018/07/12: change the inference according to the tensorflow cifar cnn and add weight decay
    2018/07/13: add learning rate decay for this model

'''
import tensorflow as tf 
import numpy as np 
import Load_data as data
import matplotlib.pyplot as plt


# model class
class AlexNet():
    def __init__(self,batch_size,drop_rate,bn,norm,weight_decay):
        self.batch_size = batch_size
        self.drop_rate = drop_rate
        self.bn = bn
        self.norm=norm
        self.weight_decay = weight_decay

    def conv(self,inputs,output_channel,ksize,strides,name,regularizer=None,training=True):
        if self.bn==False:
            use_bias = True
        else:
            use_bias = False

        output = tf.layers.conv2d(inputs,output_channel,[ksize,ksize],(strides,strides),
                                padding="SAME",use_bias=use_bias,
                                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                bias_initializer=tf.constant_initializer(0.0),
                                kernel_regularizer=regularizer,name=name)
        if self.bn == True:
            output = tf.layers.batch_normalization(output,training=training,name="bn_"+name)
        output = tf.nn.relu(output)
        return output
    
    def dense(self,inputs,units,name,regularizer=None,training=True,active=True):
        if self.bn:
            use_bias=False
        else:
            use_bias=True

        output = tf.layers.dense(inputs,units,use_bias=use_bias,
                                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                bias_initializer=tf.constant_initializer(0.0),
                                kernel_regularizer=regularizer,name=name)
        if self.bn == True:
            output = tf.layers.batch_normalization(output,training=training,name="bn_"+name)
        if active ==True:
            output = tf.nn.relu(output)
        return output

    def ALEX(self,inputs,class_num,training=False,reuse=False):
        with tf.variable_scope("alex",reuse=reuse):
            if self.norm==True:
                regularizer = tf.contrib.layers.l2_regularizer(self.weight_decay)
            else:
                regularizer = None
            
            conv1 = self.conv(inputs,96,11,4,"conv1",regularizer,training)
            pooling1 = tf.layers.max_pooling2d(conv1,(3,3),(2,2),padding="SAME",name="maxpooling1")

            conv2 = self.conv(pooling1,256,5,1,"conv2",regularizer,training)
            pooling2 = tf.layers.max_pooling2d(conv2,(3,3),(2,2),padding="SAME",name="maxpooling2")

            conv3 = self.conv(pooling2,384,3,1,"conv3",regularizer,training)

            conv4 = self.conv(conv3,384,3,1,"conv4",regularizer,training)

            conv5 = self.conv(conv4,256,3,1,"conv5",regularizer,training)
            pooling5 = tf.layers.max_pooling2d(conv5,(3,3),(2,2),padding="SAME",name="maxpooling5")

            reshape = tf.reshape(pooling5,[self.batch_size,-1])
            fc6 = self.dense(reshape,4096,"fc6",regularizer,training)
            fc6_drop = tf.layers.dropout(fc6,rate=self.drop_rate,training=training,name="dropout1")
            fc7 = self.dense(fc6_drop,4096,"fc7",regularizer,training)
            fc7_drop = tf.layers.dropout(fc7,rate=self.drop_rate,training=training,name="dropout7")
            fc8 = self.dense(fc7_drop,class_num,"output",regularizer,training,active=False)
           
            return fc8 


