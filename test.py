import tensorflow as tf 
import numpy as np
import utils 
import data
import cv2

tf.app.flags.DEFINE_string("model_name","cnn","the name")
tf.app.flags.DEFINE_integer("layers",12,"layers")

FLAGS = tf.app.flags.FLAGS


def inference_short(self, inputs, classes_num, drop_rate=0.0, is_training=False, reuse=False):
    with tf.variable_scope("alex_short", reuse=reuse):
        regularizer = tf.contrib.layers.l2_regularizer(FLAGS.weight_decay)
        # ++++++++++++++++
        # conv_2
        conv_2 = tf.layers.conv2d(inputs, 128, [5, 5], strides=(1, 1),
                                  padding="SAME", activation=tf.nn.relu, use_bias=True,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                  bias_initializer=tf.constant_initializer(0.0),
                                  kernel_regularizer=regularizer, name="conv_2")
        # pooling_2
        conv_2_pooling = tf.layers.max_pooling2d(conv_2, pool_size=(3, 3),
                                                 strides=(2, 2), padding="SAME", name="pooling_2")
        # conv_1
        conv_1 = tf.layers.conv2d(conv_2_pooling, 128, [5, 5], strides=(1, 1),
                                  padding="SAME", activation=tf.nn.relu, use_bias=True,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                  bias_initializer=tf.constant_initializer(0.0),
                                  kernel_regularizer=regularizer, name="conv_1")
        # pooling_1
        conv_1_pooling = tf.layers.max_pooling2d(conv_1, pool_size=(3, 3),
                                                 strides=(2, 2), padding="SAME", name="pooling_1")
        # conv0
        conv0 = tf.layers.conv2d(conv_1_pooling, 128, [5, 5], strides=(1, 1),
                                 padding="SAME", activation=tf.nn.relu, use_bias=True,
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                 bias_initializer=tf.constant_initializer(0.0),
                                 kernel_regularizer=regularizer, name="conv0")
        # pooling0
        conv0_pooling = tf.layers.max_pooling2d(conv0, pool_size=(3, 3),
                                                strides=(2, 2), padding="SAME", name="pooling0")
        # +++++++++++
        # conv1
        conv1 = tf.layers.conv2d(conv0_pooling, 256, [5, 5], strides=(1, 1),
                                 padding="SAME", activation=tf.nn.relu, use_bias=True,
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                 bias_initializer=tf.constant_initializer(0.0),
                                 kernel_regularizer=regularizer, name="conv1")
        # pooling1
        conv1_pooling = tf.layers.max_pooling2d(conv1, pool_size=(3, 3),
                                                strides=(2, 2), padding="SAME", name="pooling1")

        # norm1
        norm1 = tf.nn.lrn(conv1_pooling, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm1")

        # conv2
        conv2 = tf.layers.conv2d(norm1, 256, [5, 5], strides=(1, 1),
                                 padding="SAME", activation=tf.nn.relu, use_bias=True,
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                 bias_initializer=tf.constant_initializer(0.0),
                                 kernel_regularizer=regularizer, name="conv2")

        # norm2
        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm2")

        # pooling2
        conv2_pooling = tf.layers.max_pooling2d(norm2, pool_size=(3, 3),
                                                strides=(2, 2), padding="SAME", name="pooling2")

        # local3
        reshape = tf.reshape(conv2_pooling, [FLAGS.batch_size, -1])
        dense3 = tf.layers.dense(reshape, units=512, activation=tf.nn.relu, use_bias=True,
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                 bias_initializer=tf.constant_initializer(0.0),
                                 kernel_regularizer=regularizer, name="dense3")
        dense3_drop = tf.layers.dropout(dense3, rate=drop_rate, training=is_training, name="dropout3")

        # local4
        dense4 = tf.layers.dense(dense3_drop, units=256, activation=tf.nn.relu, use_bias=True,
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                 bias_initializer=tf.constant_initializer(0.0),
                                 kernel_regularizer=regularizer, name="dense4")

        dense4_drop = tf.layers.dropout(dense4, rate=drop_rate, training=is_training, name="dropout4")

        # local 5
        dense5 = tf.layers.dense(dense4_drop, units=classes_num, use_bias=True,
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                                 bias_initializer=tf.constant_initializer(0.0),
                                 kernel_regularizer=regularizer, name="dense5")
        return dense5
def testsharingvariable():
    with tf.variable_scope("scp0"):
        w = tf.get_variable('w0',initializer=1.0)
    with tf.variable_scope("scp0",reuse=True):
        w_re = tf.get_variable('w0',initializer=2.0)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        w_,w_re_ = sess.run([w,w_re])
    print(w.name,w_re.name)
    print(w_,w_re_)

def testconv():
    inputs = tf.Variable(np.ones([1,3,3,1]),dtype=tf.float32)
    
    print(inputs)
    api_output = tf.layers.conv2d(inputs,4,[3,3],strides=(2,2),padding="SAME",activation=None,kernel_initializer=tf.random_normal_initializer(0,0.01,seed=520))
    output = utils.conv2d(inputs,3,3,4,2,"conv")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        api_,output_ = sess.run([api_output,output])
        print(api_)
        print(output_)


def in_top_k():
    tf.enable_eager_execution()
    logits = np.array([[0.1,0.2,0.3,0.4],[0.4,0.2,0.2,0.2]])
    label = np.array([3,0])
    top_k_op = tf.nn.in_top_k(logits,label,1)

    print(tf.reduce_sum(tf.cast(top_k_op,tf.int32)))

    right_cout = tf.reduce_sum(tf.cast(tf.equal(label,tf.argmax(logits,axis=1)),tf.int32))
    print(right_cout)

def lr():
    
    global_step = tf.train.get_or_create_global_step()
    variable = tf.Variable([10.0],dtype=tf.float32)
    loss = variable * variable
    learning_rate = 0.1
    lr = tf.train.exponential_decay(learning_rate,global_step,decay_steps=100,decay_rate=0.1,staircase=True)
    train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss,global_step=global_step)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        _,l,lr_,gs = sess.run([train_op,loss,lr,global_step])
        if i %1 == 0:
            print("epoch %d,loss %.4f,gs %d, lr %.8f"%(i,l,gs,lr_))
        
def datas():
    cifar = data.Data("cifar10")
    batch_size=5678
    with tf.Session() as sess:
        print(cifar.num_examples_per_epoch_for_train//batch_size)
        print(cifar.num_examples_per_epoch_for_eval // batch_size)

        image,label, train_data_op,val_data_op = cifar.get_train_val_data(batch_size)
        for epoch in range(10):
            sess.run(train_data_op)
            for iters in range(cifar.num_examples_per_epoch_for_train//batch_size):
                i,l = sess.run([image,label])
                print("train:epoch %d, iter %d, image_shape %s"%(epoch,iters,np.shape(i)))
                # img = i[0,:,:,:]
                # img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                # img = cv2.resize(img,(256,256))
                # cv2.imshow("image",img)
                # cv2.waitKey(0)

                #print(l)
            
            sess.run(val_data_op)
            for iters in range(cifar.num_examples_per_epoch_for_eval // batch_size):
                i,l = sess.run([image,label])
                print("Val:epoch %d, iter %d, image_shape %s"%(epoch,iters,np.shape(i)))
                img = i[0,:,:,:]
                img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                img = cv2.resize(img,(256,256))
                cv2.imshow("image",img)
                cv2.waitKey(0)
                #print(l)
def main(_):
    # print(FLAGS.model_name)
    # print(FLAGS.layers)
    # #testsharingvariable()
    # testconv()
    in_top_k()
    #datas()
if __name__ == "__main__":
    tf.app.run()
