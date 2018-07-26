import tensorflow as tf 
import Load_data as data
import utils
import VggNet
import matplotlib.pyplot as plt
import os

class vgg_hparam19():
    def __init__(self):
        # this works which got 89.45% accuracy on validate set of cifar10
        self.dataset = "cifar10"
        self.batch_size = 128
        self.epoch_num = 100
        self.drop_rate = 0.5
        self.learning_rate=0.001
        self.learning_decay_rate=0.1
        self.learning_decay_step = 1000
        self.log_dir = r"./log/VGGNet/"
        self.checkpoint_path = r"./log/VGGNet/VGG.ckpt"
        self.weight_decay = 0.0005
        self.norm = False
        self.bn = True
        self.optimizer = "adam"
        self.inference = "vgg19"

class vgg_hparam16D():
    def __init__(self):
        # this works which got 90.38% accuracy on validate set of cifar10
        self.dataset = "cifar10"
        self.batch_size = 128
        self.epoch_num = 100
        self.drop_rate = 0.5
        self.learning_rate=0.001
        self.learning_decay_rate=0.1
        self.learning_decay_step = 1000
        self.log_dir = r"./log/VGGNet16D/"
        self.checkpoint_path = r"./log/VGGNet16D/VGG.ckpt"
        self.weight_decay = 0.0005
        self.norm = False
        self.bn = True
        self.optimizer = "adam"
        self.inference = "vgg16d"

class vgg_hparam16C():
    def __init__(self):
        # this works which got 90.34% accuracy on validate set of cifar10
        self.dataset = "cifar10"
        self.batch_size = 128
        self.epoch_num = 100
        self.drop_rate = 0.5
        self.learning_rate=0.001
        self.learning_decay_rate=0.1
        self.learning_decay_step = 1000
        self.log_dir = r"./log/VGGNet16C/"
        self.checkpoint_path = r"./log/VGGNet16C/VGG.ckpt"
        self.weight_decay = 0.0005
        self.norm = False
        self.bn = True
        self.optimizer = "adam"
        self.inference = "vgg16c"

def VGG_train():

    param = vgg_hparam16C()
    # define global_step
    global_step = tf.train.get_or_create_global_step()

    # define learning_rate
    if param.optimizer =="adam":
        learning_rate = param.learning_rate
    else:
        learning_rate = utils.learning_rate_control(param.learning_rate,global_step,
                                                    param.learning_decay_step,
                                                    param.learning_decay_rate,
                                                    staircase=False)

    tf.summary.scalar("learning_rate",learning_rate)

    # prepare data
    # if need train data, sess run train_data_op,
    # if need validation data, sess run val_data_op
    datas = data.Data()
    image,label,train_data_op,val_data_op = datas.form_dataset(param.dataset,param.batch_size)

    # build model
    vgg = VggNet.VGGNet(param.batch_size,param.weight_decay)
    if param.inference =="vgg19":
        inference=vgg.VGG19
    elif param.inference== "vgg16d":
        inference = vgg.VGG16_D
    elif param.inference == "vgg16c":
        inference = vgg.VGG16_C
    else:
        raise ValueError("don't support this network architechture!")

    logits = inference(image,datas.class_num,param.drop_rate,training=True,norm=param.norm,bn=param.bn)
    train_op,loss = vgg.build_loss(logits,param.optimizer,learning_rate,label,global_step,param.norm,param.bn)

    # test this batch
    right_count = tf.reduce_sum(tf.cast(tf.nn.in_top_k(logits,label,1),tf.int32))
    acc_this_batch = right_count / param.batch_size

    # eval
    logits_val = inference(image, datas.class_num,param.drop_rate,reuse=True,bn=param.bn,norm=param.bn)
    num = tf.reduce_sum(tf.cast(tf.nn.in_top_k(logits_val, label, 1), tf.int32))


    # show all variable
    utils.show_all_variable()

    # prepare ops
    num_batches_per_epoch_train = datas.num_examples_per_epoch_for_train // param.batch_size
    num_batches_per_epoch_eval = datas.num_examples_per_epoch_for_eval // param.batch_size
    
    saver = tf.train.Saver()
    merge_all = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))
    init = tf.global_variables_initializer()

    # clean summuary file
    if tf.gfile.Exists(param.log_dir):
        tf.gfile.DeleteRecursively(param.log_dir)

    # display acc
    plt.ion()
    acc_list=[]
    fig = plt.figure("acc")
    axe = fig.add_subplot(111)
    accuracy = 0

    with tf.Session() as sess:
        sess.run(init)
        sum_writer = tf.summary.FileWriter(param.log_dir,graph=sess.graph)
        gstep=0
        maxacc = 0
        for i in range(param.epoch_num):
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
                    _,loss_, acc, merge_ = sess.run([train_op,loss, acc_this_batch, merge_all])
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
                count += param.batch_size

            accuracy = right_count / (num_batches_per_epoch_eval * param.batch_size)
            if accuracy > maxacc:
                maxacc = accuracy
            print("test_accuracy:%.4f"%(accuracy))


                
            # save the model when a epoch end
            #saver.save(sess,param.checkpoint_path,global_step=gstep)
        
    # end plot acc
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    VGG_train()




