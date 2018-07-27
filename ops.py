# sharing functions about nerual network
import tensorflow as tf 


# learning decay
def learning_rate_decay(lr,global_step,decay_steps,decay_rate,staircase):
    lr = tf.train.exponential_decay(lr,global_step,
                                    decay_steps=decay_steps,
                                    decay_rate=decay_rate,
                                    staircase=staircase)
    return lr

# learning decay for different optimizer
def learning_rate_control(param,global_step):
    optimizer = param.optimizer
    if optimizer == "adam" or optimizer == "adagrad" or optimizer=="rmsprop":
        learning_rate = param.learning_rate
    else:
        learning_rate = learning_rate_decay(param.learning_rate,
                                              global_step,
                                              param.learning_decay_step,
                                              param.learning_decay_rate,
                                              param.staircase )
    return learning_rate

# show all variable, used to debug
def show_all_variable():
    variables_op = [v for v in tf.trainable_variables()]
    print("=====variable in this model======")
    for idx,v in enumerate(variables_op):
        print("var{:3}:name {},shape{:15}".format(idx,v.name,str(v.get_shape())))


# accuracy in this batch
def acc_this_batch(logits,label,batch_size):
    right_count = tf.reduce_sum(tf.cast(tf.nn.in_top_k(logits,label,1),tf.int32))
    acc_this_batch = right_count / batch_size
    return acc_this_batch

# eval batch image and reture the num of right prediction.
def eval_batch(inference,image,label,class_num):
    logits_val = inference(image, class_num,reuse=True)
    num = tf.reduce_sum(tf.cast(tf.nn.in_top_k(logits_val, label, 1), tf.int32))
    return num 


# build loss
def build_loss(logits,optimizer,learning_rate,label,global_step,norm=True,bn=False):
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label,logits=logits,name="loss"))
        if norm==True:
            # get weight decay
            reg_term = tf.losses.get_regularization_loss()
            # define loss
            loss += reg_term
        tf.summary.scalar("loss",loss)
    if optimizer == "adam":
        optimizer = tf.train.AdamOptimizer(learning_rate)
    elif optimizer == "gradient":
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif optimizer == "Momentum":
        optimizer = tf.train.MomentumOptimizer(learning_rate,0.9)
    else:
        raise ValueError("don't define this optimizer")

    if bn==True:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            grad = optimizer.compute_gradients(loss)
            train_op = optimizer.apply_gradients(grad, global_step=global_step)
    else:
        grad = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grad, global_step=global_step)

    grad_conv0 = grad[0][0]
    #print(grad[0][1].name)
    tf.summary.histogram("gradient_conv0",grad_conv0)
    #train_op = optimizer.minimize(loss,global_step=global_step)
    return train_op,loss
