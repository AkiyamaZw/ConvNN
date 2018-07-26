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
# hyperparameter
tf.flags.DEFINE_string("dataset","cifar10","dataset name")
tf.flags.DEFINE_integer("batch_size",128,"batch_size")
tf.flags.DEFINE_integer("epoch_num",35,"epoch num")
tf.flags.DEFINE_float("learning_rate",0.01,"init_learning_rate")
tf.flags.DEFINE_float("learning_decay_rate",0.1,"init_learning_rate")
tf.flags.DEFINE_float("learning_decay_step",1000,"the weight_decay step")
tf.flags.DEFINE_string("log_dir","./log/AlexNet/","dir of checkpoint stored")
tf.flags.DEFINE_string("checkpoint_path","./log/AlexNet/model.ckpt","define the model name")
tf.flags.DEFINE_float("weight_decay",0.0005,"the weight_decay rate")
tf.flags.DEFINE_float("drop_rate",0.5,"the rate of dropout ps. drop_rate=1-rate")
FLAGS = tf.app.flags.FLAGS

# model class
class AlexNet():
    def __init__(self):
        # self.classes_num = classes_num
        # self.learning_rate = learning_rate
        pass
    def inference_short(self,inputs,classes_num,drop_rate=0.0,is_training=False,reuse=False):
        with tf.variable_scope("alex_short",reuse=reuse):
            regularizer = tf.contrib.layers.l2_regularizer(FLAGS.weight_decay)

            # conv1
            conv1 = tf.layers.conv2d(inputs,256,[5,5],strides=(1,1),
                                    padding="SAME",activation=tf.nn.relu,use_bias=True,
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                    bias_initializer=tf.constant_initializer(0.0),
                                    kernel_regularizer=regularizer,name="conv1")
            # pooling1
            conv1_pooling = tf.layers.max_pooling2d(conv1,pool_size=(3,3),
                                                    strides=(2,2),padding="SAME",name="pooling1")

            # norm1 
            norm1 = tf.nn.lrn(conv1_pooling,4,bias=1.0,alpha=0.001/9.0,beta=0.75,name="norm1")
            
            # conv2
            conv2 = tf.layers.conv2d(norm1,256,[5,5],strides=(1,1),
                                    padding="SAME",activation=tf.nn.relu,use_bias=True,
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                    bias_initializer=tf.constant_initializer(0.0),
                                    kernel_regularizer=regularizer,name="conv2")

            # norm2
            norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm2")

            # pooling2
            conv2_pooling = tf.layers.max_pooling2d(norm2,pool_size=(3,3),
                                                    strides=(2,2),padding="SAME",name="pooling2")
            


            # local3
            reshape = tf.reshape(conv2_pooling,[FLAGS.batch_size,-1])
            dense3 = tf.layers.dense(reshape,units=512,activation=tf.nn.relu,use_bias=True,
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                    bias_initializer=tf.constant_initializer(0.0),
                                    kernel_regularizer=regularizer,name="dense3")
            dense3_drop = tf.layers.dropout(dense3,rate=drop_rate,training=is_training,name="dropout3")

            # local4
            dense4 = tf.layers.dense(dense3_drop,units=256,activation=tf.nn.relu,use_bias=True,
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                    bias_initializer= tf.constant_initializer(0.0),
                                    kernel_regularizer=regularizer,name="dense4")

            dense4_drop = tf.layers.dropout(dense4,rate=drop_rate,training=is_training,name="dropout4")

            # local 5
            dense5 = tf.layers.dense(dense4_drop,units=classes_num,use_bias=True,
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                                    bias_initializer=tf.constant_initializer(0.0),
                                    kernel_regularizer=regularizer,name="dense5")
            return dense5

    def inference(self,inputs,classes_num,drop_rate=0.0,is_training=False,reuse=False):
        with tf.variable_scope("alex",reuse=reuse):
            regularizer = tf.contrib.layers.l2_regularizer(FLAGS.weight_decay)
            # conv1
            conv1 = tf.layers.conv2d(inputs, 256, [5,5], strides=(1, 1),
                                      padding="SAME", use_bias=True,
                                      kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                      bias_initializer=tf.constant_initializer(0.0),
                                      kernel_regularizer=regularizer, name="conv1")
            bn1 =tf.layers.batch_normalization(conv1,training=is_training,name="bn1")
            conv1 = tf.nn.relu(bn1)
            #conv1 = tf.nn.relu(conv1)

            # pooling1
            conv1_pooling = tf.layers.max_pooling2d(conv1, pool_size=(3, 3),
                                                     strides=(2, 2), padding="SAME", name="pooling1")


            # conv2
            conv2 = tf.layers.conv2d(conv1_pooling, 256, [3, 3], strides=(1, 1),
                                      padding="SAME",use_bias=True,
                                      kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                      bias_initializer=tf.constant_initializer(0.0),
                                      kernel_regularizer=regularizer, name="conv2")
            bn2 = tf.layers.batch_normalization(conv2, training=is_training, name="bn2")
            conv2 = tf.nn.relu(bn2)
            #conv2 = tf.nn.relu(conv2)
            # pooling2
            conv2_pooling = tf.layers.max_pooling2d(conv2, pool_size=(3, 3),
                                                     strides=(2, 2), padding="SAME", name="pooling2")

            # conv3
            conv3 = tf.layers.conv2d(conv2_pooling, 384, [3, 3], strides=(1, 1),
                                     padding="SAME", use_bias=True,
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                     bias_initializer=tf.constant_initializer(0.0),
                                     kernel_regularizer=regularizer, name="conv3")
            bn3 = tf.layers.batch_normalization(conv3, training=is_training, name="bn3")
            conv3 = tf.nn.relu(bn3)
            #conv3 = tf.nn.relu(conv3)
            # pooling3
            conv3_pooling = tf.layers.max_pooling2d(conv3, pool_size=(3, 3),
                                                    strides=(2, 2), padding="SAME", name="pooling3")

            # conv4
            conv4 = tf.layers.conv2d(conv3_pooling, 384, [3, 3], strides=(1, 1),
                                     padding="SAME", use_bias=True,
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                     bias_initializer=tf.constant_initializer(0.0),
                                     kernel_regularizer=regularizer, name="conv4")

            bn4 = tf.layers.batch_normalization(conv4, training=is_training, name="bn4")
            conv4 = tf.nn.relu(bn4)
            #conv4 = tf.nn.relu(conv4)
            # pooling4
            conv4_pooling = tf.layers.max_pooling2d(conv4, pool_size=(3, 3),
                                                    strides=(2, 2), padding="SAME", name="pooling4")

            # norm1
            #norm1 = tf.nn.lrn(conv1_pooling, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm1")

            # conv5
            conv5 = tf.layers.conv2d(conv4_pooling, 256, [3, 3], strides=(1, 1),
                                     padding="SAME", use_bias=True,
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                     bias_initializer=tf.constant_initializer(0.0),
                                     kernel_regularizer=regularizer, name="conv5")
            bn5 = tf.layers.batch_normalization(conv5, training=is_training, name="bn5")
            conv5 = tf.nn.relu(bn5)
            #conv5 = tf.nn.relu(conv5)
            # norm2
            #norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm2")

            # pooling5
            conv5_pooling = tf.layers.max_pooling2d(conv5, pool_size=(3, 3),
                                                    strides=(2, 2), padding="SAME", name="pooling5")

            # local6
            reshape = tf.reshape(conv5_pooling, [FLAGS.batch_size, -1])
            dense6 = tf.layers.dense(reshape, units=4096, activation=tf.nn.leaky_relu, use_bias=True,
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                     bias_initializer=tf.constant_initializer(0.0),
                                     kernel_regularizer=regularizer, name="dense6")
            dense6_drop = tf.layers.dropout(dense6, rate=drop_rate, training=is_training, name="dropout6")

            # local7
            dense7 = tf.layers.dense(dense6_drop, units=4096, activation=tf.nn.leaky_relu, use_bias=True,
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                     bias_initializer=tf.constant_initializer(0.0),
                                     kernel_regularizer=regularizer, name="dense7")

            dense7_drop = tf.layers.dropout(dense7, rate=drop_rate, training=is_training, name="dropout7")

            # local 8
            dense8 = tf.layers.dense(dense7_drop, units=classes_num, use_bias=True,
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                                     bias_initializer=tf.constant_initializer(0.0),
                                     kernel_regularizer=regularizer, name="dense8")

            if is_training == True:
                print("conv_pool1 shape:", conv1.get_shape().as_list())
                print("conv_pool2 shape:", conv2.get_shape().as_list())
                print("conv_pool3 shape:", conv3.get_shape().as_list())
                print("conv_pool4 shape:", conv4.get_shape().as_list())
                print("conv_pool5 shape:", conv5.get_shape().as_list())
                print("dense1 shape:", dense6.get_shape().as_list())
                print("dense2 shape:", dense7.get_shape().as_list())
                print("dense3 shape:", dense8.get_shape().as_list())
            return dense8


    def learning_rate_control(self,lr,global_step):
        lr = tf.train.exponential_decay(lr,global_step,decay_steps=FLAGS.learning_decay_step,decay_rate=FLAGS.learning_decay_rate,staircase=True)
        return lr

    def eval(self,img,label,classes_num):
        logits = self.inference(img,classes_num,reuse=True)
        num = tf.reduce_sum(tf.cast(tf.nn.in_top_k(logits, label, 1), tf.int32))
        return num

    def show_all_variable(self):
        variables_op = [v for v in tf.trainable_variables()]
        print("=====variable in this model======")
        for idx,v in enumerate(variables_op):
            print("var{:3}:name {},shape{:15}".format(idx,v.name,str(v.get_shape())))

    def build_loss(self,logits,learning_rate,label,global_step):
        with tf.name_scope("loss"):
            # get weight decay
            reg_term = tf.losses.get_regularization_loss()
            # define loss
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label,logits=logits,name="loss"))
            loss += reg_term
            tf.summary.scalar("loss",loss)
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        #optimizer = tf.train.AdamOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            grad = optimizer.compute_gradients(loss)
            train_op = optimizer.apply_gradients(grad, global_step=global_step)

        grad_conv0 = grad[0][0]
        print(grad[0][1].name)


        tf.summary.histogram("gradient_conv0",grad_conv0)
        #train_op = optimizer.minimize(loss,global_step=global_step)
        return train_op,loss

    def train(self):
        # define global_step
        global_step = tf.train.get_or_create_global_step()
        

        # define learning_rate
        learning_rate = self.learning_rate_control(FLAGS.learning_rate,global_step)
        #learning_rate = FLAGS.learning_rate
        tf.summary.scalar("learning_rate",learning_rate)

        # prepare data
        # if need train data, sess run train_data_op,
        # if need validation data, sess run val_data_op
        datas = data.Data()
        image,label,train_data_op,val_data_op = datas.form_dataset(FLAGS.dataset,FLAGS.batch_size)

        # inference
        logits = self.inference(image,datas.class_num,FLAGS.drop_rate,is_training=True)
        train_op,loss = self.build_loss(logits,learning_rate,label,global_step)
        
        # test this batch
        #pro = tf.nn.softmax(logits)
        #right_count = tf.reduce_sum(tf.cast(tf.equal(label,tf.cast(tf.argmax(pro,axis=1),dtype=tf.int32)),tf.int32))
        right_count = tf.reduce_sum(tf.cast(tf.nn.in_top_k(logits,label,1),tf.int32))
        accu_this_batch = right_count / FLAGS.batch_size

        # eval
        num = self.eval(image,label,datas.class_num)


        # debug
        self.show_all_variable()
        #raise ValueError("debug stop")

        # prepare ops
        num_batches_per_epoch_train = datas.num_examples_per_epoch_for_train // FLAGS.batch_size
        num_batches_per_epoch_eval = datas.num_examples_per_epoch_for_eval // FLAGS.batch_size

        saver = tf.train.Saver()
        merge_all = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))
        init = tf.global_variables_initializer()

        # display acc
        plt.ion()
        acc_list=[]
        fig = plt.figure("acc")
        axe = fig.add_subplot(111)
        accuracy = 0
        with tf.Session() as sess:
            sess.run(init)
            sum_writer = tf.summary.FileWriter(FLAGS.log_dir,graph=sess.graph)
            gstep=0
            maxacc = 0
            for i in range(FLAGS.epoch_num):
                sess.run(train_data_op)
                # show acc
                acc_list.append(accuracy)
                axe.cla()
                axe.set_title("accuracy every epoch")
                axe.set_xlabel("epoch")
                axe.set_ylabel("acc")
                axe.set_xlim(-0.02,len(acc_list)+0.5)
                axe.set_ylim(0, 1.05)
                axe.scatter(range(len(acc_list)), acc_list, c="black")
                axe.plot(range(len(acc_list)), acc_list, c="red")
                axe.text(0.1, 0.1, "best acc: %.4f" % maxacc)



                plt.pause(0.1)
                # show acc end

                for j in range(num_batches_per_epoch_train):
                    # train
                    gstep = sess.run(global_step)
                    if gstep % 100 == 0:
                        _,loss_, acc, merge_ = sess.run([train_op,loss, accu_this_batch, merge_all])
                        print("global_step %d, epoch %d, batch %d, loss %.4f, train_acc %.4f"%(gstep,i,j,loss_,acc))
                        sum_writer.add_summary(merge_, global_step=gstep)
                    else:
                        _, = sess.run([train_op])

                # do eval when a epoch end
                print("start to validation...")
                sess.run(val_data_op)
                right_count = 0
                count = 0
                for k in range(num_batches_per_epoch_eval):
                    num_ = sess.run([num])
                    right_count += num_[0]
                    count += FLAGS.batch_size

                accuracy = right_count / (num_batches_per_epoch_eval * FLAGS.batch_size)
                if accuracy > maxacc:
                    maxacc = accuracy
                print("test_accuracy:%.4f"%(accuracy))


                
                # save the model when a epoch end
                saver.save(sess,FLAGS.checkpoint_path,global_step=gstep)
        
        # end plot acc
        plt.ioff()
        plt.show()


        
if __name__ == "__main__":
    net = AlexNet()
    net.train()


