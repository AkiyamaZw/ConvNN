import tensorflow as tf 
import numpy as np 

class AlexNet():
    def __init__(self,classes_num,learning_rate):
        self.classes_num = classes_num
        self.learning_rate = learning_rate
    
    def inference(inputs,drop_rate=0.5,training=False):
        with tf.variable_scope("alex"):
            # conv1
            conv1 = tf.layers.conv2d(inputs,96,[11,11],stride=(4,4),
                                    padding="SAME",activation=tf.nn.relu,
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                    name="conv1")
            conv1_pooling = tf.layers.max_pooling2d(conv1,pool_size=(3,3),strides=(2,2),padding="SAME")
            conv1_lrn = tf.nn.local_response_normalization(conv1_pooling)
            # LRN

            # conv2
            conv2 = tf.layers.conv2d(conv1_lrn,256,[5,5],strides=(1,1),
                                    padding="SAME",activation=tf.nn.relu,
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                    name="conv2")
            conv2_pooling = tf.layers.max_pooling2d(conv2,pooling_size=(3,3),strides=(2,2),padding="SAME")
            conv2_lrn = tf.nn.local_response_normalization(conv2_pooling)
            # LRN

            # conv3
            conv3 = tf.layers.conv2d(conv2_lrn,384,[3,3],strides=(1,1),
                                    padding="SAME",activation=tf.nn.relu,
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                    name="conv3")
            
            # conv4
            conv4 = tf.layers.conv2d(conv3,384,[3,3],stride=(1,1),
                                    padding="SAME",activation=tf.nn.relu,
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                    name="conv4")
            
            # conv5
            conv5 = tf.layers.conv2d(conv4,256,[3,3],stride=(1,1),
                                    padding="SAME",activation=tf.nn.relu,
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                    name="conv5")
            
            conv5_pooling = tf.layers.max_pooling2d(conv5,pooling_size=(3,3),strides=(2,2),padding="SAME")

            # dense6
            shape = conv5_pooling.get_shape()
            dense_dim = [shape[0],shape[1]*shape[2]*shape[3]]
            conv5_reshape = tf.reshape(conv5_pooling,shape=dense_dim)
            dense6 = tf.layers.dense(conv5_reshape,units=4096,activation=tf.nn.relu,
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.02),name="dense6")
            
            dense6_drop = tf.layers.dropout(dense6,rate=drop_rate,training=training,name="dropout6")
            
            # dense7
            dense7 = tf.layers.dense(dense6_drop,units=4096,activation=tf.nn.relu,
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.02),name="dense7")

            dense7_drop = tf.layers.dropout(dense7,rate=drop_rate,training=training,name="dropout7")

            # dense8 ouput
            dense8 = tf.layers.dense(dense7_drop,units=self.classes_num,kernel_initializer=tf.random_normal_initializer(stddev=0.02),name="dense7")

            return dense8
    
    def build_loss(logits,label):
        global_step = tf.train.get_or_create_global_step()
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label,logits=logits,name="loss")
        tf.summary.scalar("loss",loss)

        train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss,global_step=global_step)
        return train_op

    def train():
        pass



