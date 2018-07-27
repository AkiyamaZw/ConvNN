import tensorflow as tf
import ops
import matplotlib.pyplot as plt
import Load_data as data
# plot acc images
def plot_acc_helper(axe,acc_list,maxacc):
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

def train(param,inference):

    # define global_step
    global_step = tf.train.get_or_create_global_step()

    # define learning_rate
    learning_rate = ops.learning_rate_control(param,global_step)
    tf.summary.scalar("learning_rate",learning_rate)

    # prepare data
    # if need train data, sess run train_data_op,
    # if need validation data, sess run val_data_op
    datas = data.Data()
    image,label,train_data_op,val_data_op = datas.form_dataset(param.dataset,param.batch_size)


    logits = inference(image,datas.class_num,training=True)
    train_op,loss = ops.build_loss(logits,param.optimizer,learning_rate,label,global_step,param.norm,param.bn)

    # test this batch
    acc_this_batch = ops.acc_this_batch(logits,label,param.batch_size)

    # eval_batch
    num = ops.eval_batch(inference,image,label,datas.class_num)

    # show all variable
    ops.show_all_variable()

    # prepare ops
    num_batches_per_epoch_train = datas.num_examples_per_epoch_for_train // param.batch_size
    num_batches_per_epoch_eval = datas.num_examples_per_epoch_for_eval // param.batch_size
    
    saver = tf.train.Saver()
    merge_all = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))
    init = tf.global_variables_initializer()

    #gstep=0
    maxacc=0
    
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

        for i in range(param.epoch_num):
            # show acc
            acc_list.append(accuracy)
            plot_acc_helper(axe,acc_list,maxacc)
            # show acc end

            # training epoch
            sess.run(train_data_op)
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
            for k in range(num_batches_per_epoch_eval):
                num_ = sess.run([num])
                right_count += num_[0]

            accuracy = right_count / (num_batches_per_epoch_eval * param.batch_size)
            acc_sum = tf.summary.scalar("accuracy", tf.constant(accuracy,dtype=tf.float32))
            acc_sum_ = sess.run(acc_sum)
            sum_writer.add_summary(acc_sum_,global_step=i)

            if accuracy > maxacc:
                maxacc = accuracy
            print("test_accuracy:%.4f"%(accuracy))
  
            # save the model when a epoch end
            #saver.save(sess,param.checkpoint_path,global_step=gstep)
        
    # end plot acc
    plt.ioff()
    plt.show()