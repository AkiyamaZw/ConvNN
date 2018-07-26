import tensorflow as tf
import numpy as np 
import cv2



class Data():
    def __init__(self):
        pass

    def raw_data(self,data_name):
        if data_name == "cifar10":
            self.num_examples_per_epoch_for_train= 50000
            self.num_examples_per_epoch_for_eval = 10000
            self.image_size = [32,32,3]
            self.class_num = 10

            (x_train,y_train),(x_test,y_test) = tf.keras.datasets.cifar10.load_data()
            y_train = np.reshape(y_train,(y_train.shape[0]))
            y_test = np.reshape(y_test,(y_test.shape[0]))

        elif data_name == "cifar100":
            raise ValueError("I will implement this function!")
            self.num_examples_per_epoch_for_train= None
            self.num_examples_per_epoch_for_eval = None
            self.image_size = None
            self.class_num = 100

            (x_train,y_train),(x_test,y_test) = tf.keras.datasets.cifar100.load_data()
        elif data_name == "mnist":
            self.num_examples_per_epoch_for_train= 60000
            self.num_examples_per_epoch_for_eval = 10000
            self.image_size = [28,28,1]
            self.class_num = 10

            (x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
            x_train = np.reshape(x_train,(60000,28,28,1))
            x_test = np.asarray(x_test)
            x_test = np.reshape(x_test,(10000,28,28,1))
        
        else:
            raise ValueError("not support this ")

        print("======================================<%s>==========================================="%data_name)
        print("train data image shape %s, type %s, dtype %s"%(x_train.shape,type(x_train),x_train.dtype))
        print("train data label shape %s, type %s, dtype %s:"%(y_train.shape,type(y_train),y_train.dtype))
        print("validation data image shape %s, type %s, dtype %s"%(x_test.shape,type(x_test),x_test.dtype))
        print("validation data image shape %s, type %s, dtype %s"%(y_test.shape,type(y_test),y_test.dtype))
        print("======================================================================================")
        return x_train,y_train,x_test,y_test

    def form_dataset(self, dataset_name,batch_size):
        # get raw data(numpy type)
        x_train,y_train,x_test,y_test = self.raw_data(dataset_name)

        # create dataset
        train_dataset = tf.data.Dataset.from_tensor_slices({"image":x_train,"label":y_train})
        validation_dataset = tf.data.Dataset.from_tensor_slices({"image":x_test,"label":y_test})
        
        #dataset infomation#
        #self._debug_dataset_ditial(train_dataset)
        #self._debug_dataset_ditial(validation_dataset)
        
        # map,batch,shuffle dataset
        train_dataset = self.map_batch_shuffle_dataset(train_dataset,batch_size,True)
        validation_dataset = self.map_batch_shuffle_dataset(validation_dataset,batch_size,False)

        # create iterator
        iterator = tf.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)
        batch_image,batch_label = iterator.get_next()

        train_init_op = iterator.make_initializer(train_dataset)
        validation_init_op = iterator.make_initializer(validation_dataset)

        return batch_image,batch_label,train_init_op,validation_init_op

    
    def map_batch_shuffle_dataset(self,dataset,batch_size,is_training,buffer_size=10000):
        dataset = dataset.map(lambda record:self.process_data(record,is_training))
        if is_training == True:
            dataset = dataset.shuffle(buffer_size)
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        return dataset

        #return image,label,train_op_init,validation_op_init
    def process_data(self,record,is_training):

        image = record["image"]
        label = record["label"]
        #print(label.shape)
        #raise ValueError("stop")
        # change label and image data dtype
        label = tf.cast(label,tf.int32)
        image = tf.cast(image,tf.float32)
       
        # processes image
        image = self.preprocess_image(image,is_training)

        return image,label

    def preprocess_image(self,image,is_training):
        if is_training:
            # Resize the image to add four extra pixels on each side
            image = tf.image.resize_image_with_crop_or_pad(image,self.image_size[0]+8,self.image_size[1]+8)

            # Randomly crop a section of the image
            image = tf.random_crop(image,self.image_size)
            
            # Randomly flip the image horizontally
            image = tf.image.random_flip_left_right(image)

        # Subtract off the mean and divide by the variance of the pixels
        image = tf.image.per_image_standardization(image)
        return image 

    def _debug_dataset_ditial(self,dataset):
        print("dataset.output_type:",dataset.output_types)
        print("dataset.output_shapes:",dataset.output_shapes)
        print("dataset.output_classes:",dataset.output_classes)

if __name__ == "__main__":
    cifar10 = Data()
    batch_size= 5120
    image,label,train_init_op,validation_init_op = cifar10.form_dataset("cifar10",batch_size)
    with tf.Session() as sess:
        train_batch_num = cifar10.num_examples_per_epoch_for_train // batch_size
        validation_batch_num = cifar10.num_examples_per_epoch_for_eval // batch_size
        print(train_batch_num,validation_batch_num)
        for epoch in range(10):
            sess.run(train_init_op)
            for iters in range(train_batch_num):
                i,l = sess.run([image,label])
                # print("train:epoch %d, iter %d, image_shape %s"%(epoch,iters,np.shape(i)))
                # img = i[0,:,:,:].astype(np.uint8)
                # img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                # img = cv2.resize(img,(256,256))
                # cv2.imshow("image",img)
                # cv2.waitKey(0)
                #
                # print(l)
            
            sess.run(validation_init_op)
            for iters in range(validation_batch_num):
                i,l = sess.run([image,label])
                print("Val:epoch %d, iter %d, image_shape %s"%(epoch,iters,np.shape(i)))
                img = i[0,:,:,:].astype(np.uint8)
                img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                img = cv2.resize(img,(256,256))
                cv2.imshow("image",img)
                cv2.waitKey(0)
                print(l)

