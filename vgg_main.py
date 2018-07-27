import tensorflow as tf 
import Load_data as data
import utils
import Model.VggNet as VggNet
import matplotlib.pyplot as plt
import ops

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
        self.staircase = False

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
        self.staircase = False

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
        self.staircase = False


def get_inference(param,vgg):
    if param.inference =="vgg19":
        inference=vgg.VGG19
    elif param.inference== "vgg16d":
        inference = vgg.VGG16_D
    elif param.inference == "vgg16c":
        inference = vgg.VGG16_C
    else:
        raise ValueError("don't support this network architechture!")
    return inference 

def VGG_train(Param):
       param = Param()
       vgg = VggNet.VGGNet(param.batch_size,param.drop_rate,param.bn,param.norm,param.weight_decay)
       inference = get_inference(param.inference,vgg)
       utils.train(param,inference)


if __name__ == "__main__":
    VGG_train(vgg_hparam16D)




