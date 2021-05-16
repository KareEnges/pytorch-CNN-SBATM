from os import close
import torch
from torch._C import device
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter



classes = ('NO', 'YES')




# 定义一个类，用来建立一个简单神经网络，属于固定套路
# 可以看这个博客：https://blog.csdn.net/monk1992/article/details/89947267
class Net(nn.Module):                                                               # 继承 torch 的 nn.Module（所有神经网络模块的基类）
                                                                                    # 官网API说明：https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module
    def __init__(self):                                                             
        super().__init__()                                                 # 继承 nn.Module 的内置方法，
        
        self.conv1 = nn.Conv2d(3, 6, 5)                                             # 卷积层第一层，这里是二维卷积，即卷积在二维平面上移动，图像是二维的嘛
                                                                                    # in_channels（输入图像中的通道数，高度）, out_channels（卷积产生的通道数）, kernel_size（卷积内核的大小，这里是5*5像素）
                                                                                    # 官网API说明：https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
                                                                                    # 莫烦教程https://www.pytorchtutorial.com/4-1-cnn/ ，视频里面用硬币讲得很好

        self.pool = nn.MaxPool2d(2, 2)                                              # 池化层，对应的是二维卷积的池化层，在我看来就是对卷积进行筛选
                                                                                    # kernel_size（窗口大小，也就是用的大小2*2像素特征去筛选），stride（步长，默认值是kernel_size）
                                                                                    # 官网API说明：https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html

        self.conv2 = nn.Conv2d(6, 16, 5)                                            # 卷积层第二层，这里是二维卷积
                                                                                    # in_channels（第一层的输出）, out_channels（卷积产生的通道数）, kernel_size（卷积内核的大小，这里是5*5像素）

        # 线性变换，对传入数据应用线性变换，用于设置网络中的全连接层
        # 在基本的CNN网络中，全连接层的作用是将经过多个卷积层和池化层的图像特征图中的特征进行整合，获取图像特征具有的高层含义，之后用于图像分类
        # in_features（输入的二维张量的大小或者上层神经元个数），out_features（本层神经元个数）
        # 官网API说明：https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.fc0 = nn.Linear(1296, 400)  
        self.fc1 = nn.Linear(400, 120)                                              # 可以通过print(x.size())查看张量大小，这里为什么输入是400，因为经过第二次卷积后，图像5*5，16通道，5*5*16=400
        self.fc2 = nn.Linear(120, 84)                                               # 输入是上一次的输出，输出是下一次的输入
        self.fc3 = nn.Linear(84, 2)                                                 # 输入是上一次的输出，输出......这里为什么是2
                                                                                    # 这边为什么是三层......

    def forward(self, x):                                                           # 前向函数，怎么调用的可以看 https://www.cnblogs.com/llfctt/p/10967651.html , 相当于是 __call__ 方法调用

        # 这里的self.pool(F.relu(tenser))相当于也是一个__call__方法，具体可以看各自源码
        # 官方API说明：https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU
        # 好像是继承过去的一个方法：https://pytorch.org/docs/stable/nn.functional.html
        # 莫烦教程，激励函数：https://www.pytorchtutorial.com/2-3-activation/
        #(32-5+0)/1+1 = 28 28/2=14
        x = self.pool(F.relu(self.conv1(x)))                                        # 首先对输入层做一次卷积运算，再进行激励（非线性映射），再池化
        # 4*6*14*14
        # (14-5)/1 + 1 = 10 10/2=5
        x = self.pool(F.relu(self.conv2(x)))                                        # 然后对第一次操作（卷积、激励、池化）的结果做第二次卷积，同样卷积、激励、池化                                                                 
        # 4*16*5*5                                                                      
        #print(x.size())
        #exit()
        x = x.view(-1, 1296)                                                         # 将 torch 变量转为一个400列的张量
        # 4*400                                                                        # 官网API说明：https://pytorch.org/docs/stable/tensor_view.html
                                                                                    # 这个博客讲得很好：https://nickhuang1996.blog.csdn.net/article/details/86569501
        #0,1
        #1,0
        #0.1 0.9

        #如果输出是一个维度
        # 通过一个sigmoid函数，输出大小范围在0-1之间，以0.5为界限，》0.5是一类，小于0.5是一类
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x






# 首先定义一个函数，用来获取测试用的图像, 这里是批数据训练
# 具体注释看 train.py 里面
def loadtestdata():
    path = r"C:/Users/Landian04/Desktop/pytorch/项目/奥特曼识别/test"
    testset = torchvision.datasets.ImageFolder( path,
                                                transform=transforms.Compose([
                                                    transforms.Resize((50,50)),              # 将图片缩放到指定大小（h,w）
                                                    transforms.ToTensor()])                     # 将PIL图像或 numpy.ndarray 转换为 tenser 张量
                                               )
    testloader = torch.utils.data.DataLoader(   testset,                                        # torch TensorDataset format，即前面创建的 Dataset
                                                batch_size=25,                                  # 指一次拿多少数据出来，这里每步都导出了25个数据进行学习
                                                shuffle=True,                                   # 要不要打乱数据 (打乱比较好)
                                                num_workers=2)                                  # 多线程来读数据
    return testloader



# 取出训练的整个神经网络的模型结构以及参数，net.pkl 是在 train.py 里面保存的
# 具体注释看 train.py 里面
def reload_net():
    net = Net()
    net = net.cuda()
    checkpoint = torch.load('net_params.pkl')       
    net.load_state_dict(checkpoint)                                       
    return net



# 用来显示图片
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()





def test():
    testloader = loadtestdata()                                                                 # 取出测试数据
    net = reload_net()                                                                          # 取出神经网络参数
    dataiter = iter(testloader)                                                                 # iter返回了基于可迭代对象dataloader生成的迭代器dataloaderiter 而enumerate是在dataloader里直接加了i的索引
    images, labels = dataiter.next()                                                            # 迭代器通过next()来遍历元素
    
    print('GroundTruth: ', " ".join('%-10s' % classes[labels[j]] for j in range(25)))            # 打印25个图的实际结果
    outputs = net(Variable(images).cuda())                                                      # 对目标图像进行测试
    #print(outputs)
    _, predicted = torch.max(outputs.data, 1)                                                   # torch.max返回输入张量中所有元素的最大值
    #print(torch.argmax(outputs,dim = 1))
    #print(predicted)
                                                                                                # 官网API说明：https://pytorch.org/docs/stable/generated/torch.max.html
                                                                                                # 可以看这个博客：https://blog.csdn.net/weixin_48249563/article/details/111387501

    print('  Predicted: ', " ".join('%-10s' % classes[predicted[j]] for j in range(25)))            # 打印25个图的预测结果

    imshow(torchvision.utils.make_grid(images, nrow=5))                                         # 显示测试的图像
                                                                                                # make_grid的作用是将若干幅图像拼成一幅图像，在需要展示一批数据时有用，nrow代表网格中每行显示的图像数


if __name__ == '__main__':
    print(torch.cuda.is_available())
    test()


