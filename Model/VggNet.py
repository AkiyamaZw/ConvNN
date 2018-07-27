'''
Author: AkiyamaZw
First edit data:2018-07-23
Logs:
2018-07-23: create this python file and build base inference function;It seems vgg model need dropout layer and finetune.
'''

import tensorflow as tf 
import Load_data


class VGGNet():
    def __init__(self,batch_size,drop_rate,bn,norm,weight_decay):
        self.batch_size = batch_size
        self.drop_rate=drop_rate
        self.bn = bn
        self.norm = norm 
        self.weight_decay = weight_decay

    def conv(self,inputs,filterNum,name,filterSize=3,regularizer=None,training=True):
        if self.bn:
            use_bias=False
        else:
            use_bias=True

        output = tf.layers.conv2d(inputs,filterNum,(filterSize,filterSize),
                                strides=(1,1),padding="SAME",use_bias=use_bias,
                                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                bias_initializer=tf.constant_initializer(0.0),
                                kernel_regularizer=regularizer,name=name)
        if self.bn == True:
            output = tf.layers.batch_normalization(output,training=training,name="bn_"+name)
        
        output = tf.nn.relu(output)
        return output

    def dense(self,inputs,units,name,regularizer=None,bn=False,training=True,active=True):
        if self.bn:
            use_bias=False
        else:
            use_bias=True

        #use_bias = True
        output = tf.layers.dense(inputs,units,use_bias=use_bias,
                                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                bias_initializer=tf.constant_initializer(0.0),
                                kernel_regularizer=regularizer,name=name)
        if self.bn == True:
            output = tf.layers.batch_normalization(output,training=training,name="bn_"+name)
        if active ==True:
            output = tf.nn.relu(output)
        return output

    def VGG19(self,inputs,class_num,training=False,reuse=False):
        with tf.variable_scope("vgg",reuse=reuse):
            maxPoolingWindow = [2,2]
            maxPoolingStride = (2,2)
            if self.norm!=False:
                regularizer = tf.contrib.layers.l2_regularizer(self.weight_decay)
            else:
                regularizer = None
  
            conv1 = self.conv(inputs,64,"conv1",3,regularizer,training)       
            conv2 = self.conv(conv1,64,"conv2",3,regularizer,training)
            maxPooling1 = tf.layers.max_pooling2d(conv2,maxPoolingWindow,strides=maxPoolingStride,padding="SAME",name="maxpooling1")
            
            conv3 = self.conv(maxPooling1,128,"conv3",3,regularizer,training)
            conv4 = self.conv(conv3,128,"conv4",3,regularizer,training)
            maxPooling2 = tf.layers.max_pooling2d(conv4,maxPoolingWindow,strides=maxPoolingStride,padding="SAME",name="maxpooling2")

            conv5 = self.conv(maxPooling2,256,"conv5",3,regularizer,training)
            conv6 = self.conv(conv5,256,"conv6",3,regularizer,training)
            conv7 = self.conv(conv6,256,"conv7",3,regularizer,training)
            conv8 = self.conv(conv7,256,"conv8",3,regularizer,training)
            maxPooling3 = tf.layers.max_pooling2d(conv8,maxPoolingWindow,strides=maxPoolingStride,padding="SAME",name="maxpooling3")

            conv9 = self.conv(maxPooling3,512,"conv9",3,regularizer,training)
            conv10 = self.conv(conv9,512,"conv10",3,regularizer,training)
            conv11 = self.conv(conv10,512,"conv11",3,regularizer,training)
            conv12 = self.conv(conv11,512,"conv12",3,regularizer,training)
            maxPooling4 = tf.layers.max_pooling2d(conv12,maxPoolingWindow,strides=maxPoolingStride,padding="SAME",name="maxpooling4")

            conv13 = self.conv(maxPooling4,512,"conv13",3,regularizer,training)
            conv14 = self.conv(conv13,512,"conv14",3,regularizer,training)
            conv15 = self.conv(conv14,512,"conv15",3,regularizer,training)
            conv16 = self.conv(conv15,512,"conv16",3,regularizer,training)
            maxPooling5 = tf.layers.max_pooling2d(conv16,maxPoolingWindow,strides=maxPoolingStride,padding="SAME",name="maxpooling5")

            reshape = tf.reshape(maxPooling5,[self.batch_size,-1])
            fc17 = self.dense(reshape,4096,"fc17",regularizer,training)
            fc17_drop = tf.layers.dropout(fc17, rate=self.drop_rate, training=training, name="dropout1")
            fc18 = self.dense(fc17_drop,4096,"fc18",regularizer,training)
            fc18_drop = tf.layers.dropout(fc18, rate=self.drop_rate, training=training, name="dropout4")
            fc19 = self.dense(fc18_drop,class_num,"output",regularizer,training,active=False)
        return fc19

    def VGG16_D(self,inputs,class_num,training=False,reuse=False):
        with tf.variable_scope("vgg",reuse=reuse):
            maxPoolingWindow = [2,2]
            maxPoolingStride = (2,2)
            if self.norm!=False:
                regularizer = tf.contrib.layers.l2_regularizer(self.weight_decay)
            else:
                regularizer = None
  
            conv1 = self.conv(inputs,64,"conv1",3,regularizer,training)       
            conv2 = self.conv(conv1,64,"conv2",3,regularizer,training)
            maxPooling1 = tf.layers.max_pooling2d(conv2,maxPoolingWindow,strides=maxPoolingStride,padding="SAME",name="maxpooling1")
            
            conv3 = self.conv(maxPooling1,128,"conv3",3,regularizer,training)
            conv4 = self.conv(conv3,128,"conv4",3,regularizer,training)
            maxPooling2 = tf.layers.max_pooling2d(conv4,maxPoolingWindow,strides=maxPoolingStride,padding="SAME",name="maxpooling2")

            conv5 = self.conv(maxPooling2,256,"conv5",3,regularizer,training)
            conv6 = self.conv(conv5,256,"conv6",3,regularizer,training)
            conv7 = self.conv(conv6,256,"conv7",3,regularizer,training)
            maxPooling3 = tf.layers.max_pooling2d(conv7,maxPoolingWindow,strides=maxPoolingStride,padding="SAME",name="maxpooling3")

            conv8 = self.conv(maxPooling3,512,"conv8",3,regularizer,training)
            conv9 = self.conv(conv8,512,"conv9",3,regularizer,training)
            conv10 = self.conv(conv9,512,"conv10",3,regularizer,training)
            maxPooling4 = tf.layers.max_pooling2d(conv10,maxPoolingWindow,strides=maxPoolingStride,padding="SAME",name="maxpooling4")

            conv11 = self.conv(maxPooling4,512,"conv11",3,regularizer,training)
            conv12 = self.conv(conv11,512,"conv12",3,regularizer,training)
            conv13 = self.conv(conv12,512,"conv13",3,regularizer,training)
            maxPooling5 = tf.layers.max_pooling2d(conv13,maxPoolingWindow,strides=maxPoolingStride,padding="SAME",name="maxpooling5")

            reshape = tf.reshape(maxPooling5,[self.batch_size,-1])
            fc14 = self.dense(reshape,4096,"fc14",regularizer,training)
            fc14_drop = tf.layers.dropout(fc14, rate=self.drop_rate, training=training, name="dropout1")
            fc15 = self.dense(fc14_drop,4096,"fc15",regularizer,training)
            fc15_drop = tf.layers.dropout(fc15, rate=self.drop_rate, training=training, name="dropout2")
            fc16 = self.dense(fc15_drop,class_num,"output",regularizer,training,active=False)
        return fc16

    def VGG16_C(self,inputs,class_num,drop_rate,training=False,reuse=False):
        with tf.variable_scope("vgg",reuse=reuse):
            maxPoolingWindow = [2,2]
            maxPoolingStride = (2,2)
            if self.norm!=False:
                regularizer = tf.contrib.layers.l2_regularizer(self.weight_decay)
            else:
                regularizer = None

            conv1 = self.conv(inputs,64,"conv1",3,regularizer,training)       
            conv2 = self.conv(conv1,64,"conv2",3,regularizer,training)
            maxPooling1 = tf.layers.max_pooling2d(conv2,maxPoolingWindow,strides=maxPoolingStride,padding="SAME",name="maxpooling1")
            
            conv3 = self.conv(maxPooling1,128,"conv3",3,regularizer,training)
            conv4 = self.conv(conv3,128,"conv4",3,regularizer,training)
            maxPooling2 = tf.layers.max_pooling2d(conv4,maxPoolingWindow,strides=maxPoolingStride,padding="SAME",name="maxpooling2")

            conv5 = self.conv(maxPooling2,256,"conv5",3,regularizer,training)
            conv6 = self.conv(conv5,256,"conv6",3,regularizer,training)
            conv7 = self.conv(conv6,256,"conv7",1,regularizer,training)
            maxPooling3 = tf.layers.max_pooling2d(conv7,maxPoolingWindow,strides=maxPoolingStride,padding="SAME",name="maxpooling3")

            conv8 = self.conv(maxPooling3,512,"conv8",3,regularizer,training)
            conv9 = self.conv(conv8,512,"conv9",3,regularizer,training)
            conv10 = self.conv(conv9,512,"conv10",1,regularizer,training)
            maxPooling4 = tf.layers.max_pooling2d(conv10,maxPoolingWindow,strides=maxPoolingStride,padding="SAME",name="maxpooling4")

            conv11 = self.conv(maxPooling4,512,"conv11",3,regularizer,training)
            conv12 = self.conv(conv11,512,"conv12",3,regularizer,training)
            conv13 = self.conv(conv12,512,"conv13",1,regularizer,training)
            maxPooling5 = tf.layers.max_pooling2d(conv13,maxPoolingWindow,strides=maxPoolingStride,padding="SAME",name="maxpooling5")

            reshape = tf.reshape(maxPooling5,[self.batch_size,-1])
            fc14 = self.dense(reshape,4096,"fc14",regularizer,training)
            fc14_drop = tf.layers.dropout(fc14, rate=self.drop_rate, training=training, name="dropout1")
            fc15 = self.dense(fc14_drop,4096,"fc15",regularizer,training)
            fc15_drop = tf.layers.dropout(fc15, rate=self.drop_rate, training=training, name="dropout2")
            fc16 = self.dense(fc15_drop,class_num,"output",regularizer,training,active=False)
        return fc16





        



            
    