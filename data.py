import tensorflow as tf 
import numpy as np 
import os 
import PIL.Image as Image 
import urllib
import pickle 
import sys 
import tarfile
import PIL.Image as Image

import cv2

data_dir = "./data"
CIFAR10_DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

class Data():
    def __init__(self,data_name):
        if data_name == "cifar10":
            # get cifar10 data
            self.classes_num = 10
            self.data_dir = os.path.join(data_dir,"cifar10")
            self.num_examples_per_epoch_for_train = 50000
            self.num_examples_per_epoch_for_eval = 10000
            self.image_size = [32,32,3]
            # download cifar10 data if need
            self.maybe_download_and_extract()
            pass
        elif data_name == 'cifar100':
            # get cifar100 data
            self.classes_num = 100
            self.data_dir = os.path.join(data_dir,"cifar100")
            pass
        elif data_name == "mnist":
            # get mnist data
            self.classes_num = 10
            self.data_dir = os.path.join(data_dir,"mnist10")
            pass
        else:
            raise ValueError("don't support this dataset:%s"%data_name)
# =============================cifar10 dataset========================================================================
    # download and extract cifar10 data
    def maybe_download_and_extract(self):
        '''will download cifar10 data and extract it'''
        dest_directory = self.data_dir
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)
        filename = CIFAR10_DATA_URL.split('/')[-1]
        filepath = os.path.join(dest_directory,filename)
        if not os.path.exists(filepath):
            def _progress(count,block_size,total_size):
                sys.stdout.write('\r>> Download %s %.1f%%' % (filename,float(count*block_size)/float(total_size)*100.0))
                sys.stdout.flush()
            filepath,_ = urllib.request.urlretrieve(CIFAR10_DATA_URL,filepath,_progress)
            print()
            statinfo = os.stat(filepath)
            print("Successfully download",filename,statinfo.st_size,"bytes.")
        if os.path.exists(filepath):
            tarfile.open(filepath,'r:gz').extractall(dest_directory)
    
    # get path of cifar10 train or evalidation dataset 
    def get_filenames(self,is_training):
        data_dir = os.path.join(self.data_dir,"cifar-10-batches-bin")
        assert os.path.exists(data_dir),('Run cifar10_download_and_extract function first to download and extract the data')

        if is_training:
            filename_list = [os.path.join(data_dir,'data_batch_%d.bin'%i) for i in range(1,6)]
        else:
            filename_list = [os.path.join(data_dir,"test_batch.bin")]

        return filename_list

    # get cifar10 trainning data
    def read_cifar10_data_train(self,batch_size,record_bytes):
        # get bin data dir
        cifar10_list_train = self.get_filenames(is_training=True)
        dataset = tf.data.FixedLengthRecordDataset(cifar10_list_train,record_bytes=record_bytes)
        dataset = dataset.map(lambda value:self.process_cifar10_data(value,is_training=True))
        dataset = dataset.shuffle(buffer_size=5000)
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        return dataset
    
    def read_cifar10_data_val(self,batch_size,record_bytes):
        cifar10_list_val = self.get_filenames(is_training=False)
        dataset = tf.data.FixedLengthRecordDataset(cifar10_list_val,record_bytes=record_bytes)
        dataset = dataset.map(lambda value:self.process_cifar10_data(value,is_training=False))
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        return dataset

    def get_train_val_data(self,batch_size):
        # get record_bytes 
        image_bytes = self.image_size[0]*self.image_size[1]*self.image_size[2]
        label_bytes = 1 # cifar100 2
        record_bytes = image_bytes + label_bytes

        train_dataset = self.read_cifar10_data_train(batch_size,record_bytes)
        val_dataset = self.read_cifar10_data_val(batch_size,record_bytes)

        iterator = tf.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)
        image,label = iterator.get_next()
        
        train_init_op = iterator.make_initializer(train_dataset)
        val_init_op = iterator.make_initializer(val_dataset)

        return image,label,train_init_op,val_init_op

    
    def process_cifar10_data(self,records,is_training):

        # Convert bytes to a vector of uint8 that is record_bytes long.
        record = tf.decode_raw(records,tf.uint8)
    
        # The first byte represents the label, which we convert from uint8 to int32
        label = tf.cast(record[0],tf.int32)

        # The remaining bytes after the label represent the image, which we reshape 
        # from [depth * height * width] to [depth,height,width]
        depth_major = tf.reshape(record[1:],[self.image_size[2],self.image_size[0],self.image_size[1]])

        # convert from [depth, height, width] to [height,width,depth], and cast as tf.float32
        image = tf.cast(tf.transpose(depth_major,[1,2,0]),tf.float32)

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
#===============================================cifar10 dataset=============================================
if __name__ == "__main__":
    data = Data("cifar10")
    tf.enable_eager_execution()
    image,label = data.read_cifar10_data(batch_size=16,is_training=True)
    print(image[1])
    image = np.array(image).astype(np.uint8)
    # for i in range(16):
    #     img = image[i]
    #     img = Image.fromarray(np.uint8(img))
    #     img.show("%d"%i)

    for i in range(16):
        img = image[i,:,:,:]
        print(img)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        #img = cv2.resize(img,(256,256))
        cv2.imshow("image",img)
        cv2.waitKey(0)



