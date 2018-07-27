import utils
import Model.AlexNet as AlexNet


class alex_hparam():
    def __init__(self):
        self.dataset = "cifar10"
        self.batch_size = 128
        self.epoch_num = 100
        self.drop_rate = 0.5
        self.learning_rate=0.001
        self.learning_decay_rate=0.1
        self.learning_decay_step = 1000
        self.log_dir = r"./log/AlexNet/"
        self.checkpoint_path = r"./log/AlexNet/alex.ckpt"
        self.weight_decay = 0.0005
        self.norm = False
        self.bn = True
        self.optimizer = "adam"
        self.inference = "vgg19"
        self.staircase = False


def get_inference(param,alex):
    return alex.ALEX
    
def alex_train(Param):
    param = Param()
    alex = AlexNet.AlexNet(param.batch_size,param.drop_rate,param.bn,param.norm,param.weight_decay)
    inference = get_inference(param,alex)
    utils.train(param,inference)

if __name__ == "__main__":
    alex_train(alex_hparam)
