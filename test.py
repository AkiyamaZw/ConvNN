import tensorflow as tf 
import numpy as np
import utils 

tf.app.flags.DEFINE_string("model_name","cnn","the name")
tf.app.flags.DEFINE_integer("layers",12,"layers")

FLAGS = tf.app.flags.FLAGS
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

def main(_):
    print(FLAGS.model_name)
    print(FLAGS.layers)
    #testsharingvariable()
    testconv()
if __name__ == "__main__":
    tf.app.run()
