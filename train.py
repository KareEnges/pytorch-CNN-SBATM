from os import close
import torch
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



# 首先定义一个函数，用来获取训练用的图像, 这里是批数据训练
# 莫烦教程https://www.pytorchtutorial.com/3-5-data-loader/，视频里面的硬币说有意思
def loadtraindata():
    path = r"C:/Users/Landian04/Desktop/pytorch/项目/奥特曼识别/train"

    # torchvision.datasets.ImageFolder：创建一个数据加载器(dataloader)，就是将东西转换成 torch 能识别的 Dataset
    # API说明https://pytorch.org/vision/stable/datasets.html#imagefolder
    trainset = torchvision.datasets.ImageFolder(path,                               # 图像目录路径
                                                transform=transforms.Compose([      # transforms.Compose：接收PIL图像并返回转换版本的函数/转换（将几个变换组合在一起），参数是一个transforms对象
                                                                                    # API说明：https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.Compose
                                                    transforms.Resize((100, 100)),  # 将图片缩放到指定大小（h,w）
                                                    transforms.CenterCrop(32),      # 剪裁
                                                    transforms.ToTensor()])         # 将PIL图像或 numpy.ndarray 转换为 tenser 张量
                                                )


    # torch.utils.data.DataLoader：加载数据, DataLoader 是 torch 给你用来包装你的数据的工具, 他们帮你有效地迭代数据
    # API说明https://pytorch.org/docs/stable/data.html
    trainloader = torch.utils.data.DataLoader(  trainset,                           # torch TensorDataset format，即前面创建的 Dataset
                                                batch_size=4,                       # 指一次拿多少数据出来，这里每步都导出了4个数据进行学习
                                                shuffle=True,                       # 要不要打乱数据 (打乱比较好)
                                                num_workers=2                       # 多线程来读数据
                                            )
    return trainloader





# 定义一个类，用来建立一个简单神经网络，属于固定套路
# 可以看这个博客：https://blog.csdn.net/monk1992/article/details/89947267
class Net(nn.Module):                                                               # 继承 torch 的 nn.Module（所有神经网络模块的基类）
                                                                                    # 官网API说明：https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module
    def __init__(self):                                                             
        super(Net, self).__init__()                                                 # 继承 nn.Module 的 __init__ 方法，
        
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
        self.fc1 = nn.Linear(400, 120)                                              # 可以通过print(x.size())查看张量大小，这里为什么输入是400，因为经过第二次卷积后，图像5*5，16通道，5*5*16=400
        self.fc2 = nn.Linear(120, 84)                                               # 输入是上一次的输出，输出是下一次的输入
        self.fc3 = nn.Linear(84, 2)                                                 # 输入是上一次的输出，输出......这里为什么是2
                                                                                    # 这边为什么是三层......

    def forward(self, x):                                                           # 前向函数，怎么调用的可以看 https://www.cnblogs.com/llfctt/p/10967651.html , 相当于是 __call__ 方法调用

        # 这里的self.pool(F.relu(tenser))相当于也是一个__call__方法，具体可以看各自源码
        # 官方API说明：https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU
        # 好像是继承过去的一个方法：https://pytorch.org/docs/stable/nn.functional.html
        # 莫烦教程，激励函数：https://www.pytorchtutorial.com/2-3-activation/
        x = self.pool(F.relu(self.conv1(x)))                                        # 首先对输入层做一次卷积运算，再进行激励（非线性映射），再池化
        x = self.pool(F.relu(self.conv2(x)))                                        # 然后对第一次操作（卷积、激励、池化）的结果做第二次卷积，同样卷积、激励、池化
                                                                                    
                                                                                    
        x = x.view(-1, 400)                                                         # 将 torch 变量转为一个400列的张量
                                                                                    # 官网API说明：https://pytorch.org/docs/stable/tensor_view.html
                                                                                    # 这个博客讲得很好：https://nickhuang1996.blog.csdn.net/article/details/86569501


        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



classes = ('不是奥特曼', '奥特曼')


def loadtestdata():
    path = r"./test"
    testset = torchvision.datasets.ImageFolder(path,
                                               transform=transforms.Compose([
                                                   transforms.Resize((100, 100)),
                                                   transforms.ToTensor()])
                                               )
    testloader = torch.utils.data.DataLoader(testset, batch_size=25,
                                             shuffle=True, num_workers=2)
    return testloader



# 开始训练的函数
def trainandsave():  
    writer = SummaryWriter('runs/aoteman_1')                                        # 实例化summery，将条目直接写入日志目录中的事件文件，以供TensorBoard使用
                                                                                    # 默认情况下，Writer将输出到./runs/目录
                                                                                    # 官网API说明：https://pytorch.org/docs/stable/tensorboard.html

    trainloader = loadtraindata()                                                   # 获取训练用图像
    net = Net()                                                                     # 实例化神经网络对象
    net.cuda()# Moves all model parameters and buffers to the GPU.

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)                 # 实例化神经网络优化器，主要是为了优化我们的神经网络，使他在我们的训练过程中快起来，节省社交网络训练的时间
                                                                                    # 莫烦教程：https://www.pytorchtutorial.com/3-6-optimizer/
                                                                                    # 官网API说明：https://pytorch.org/docs/stable/optim.html
                                                                                    
    criterion = nn.CrossEntropyLoss()                                               # 实例化损失函数
                                                                                    # 官网API说明：https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss

    for epoch in range(1000):                                                         # 此循环即为训练过程，这里是训练1000步
        running_loss = 0.0                                                          # 每次训练损失初值置为0
        for i, data in enumerate(trainloader, 0):                                   # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中
                                                                                    # 看这个博客：https://blog.csdn.net/qq_29893385/article/details/84640581

            #print('第%d张图片' % (i))#小测试，主要看看多少张图片，看样子应该是每张图片都拿出来训练1000次

            inputs, labels = data                                                   # 在 Torch 中的 Variable 就是一个存放会变化的值的地理位置，里面的值会不停的变化。Variable是可更改的，而Tensor是不可更改的。
            inputs, labels = Variable(inputs), Variable(labels)                     # 莫烦教程：https://www.pytorchtutorial.com/2-2-variable/
            inputs = inputs.cuda() # Tensor on GPU
            labels = labels.cuda() # Tensor on GPU


            img_grid = torchvision.utils.make_grid(inputs)                          # make_grid的作用是将若干幅图像拼成一幅图像，在需要展示一批数据时有用
                                                                                    # 官网API说明：https://pytorch.org/vision/stable/utils.html
            writer.add_image('aoteman', img_grid)                                   # 将图像数据添加到summary，以供TensorBoard使用
            

            optimizer.zero_grad()                                                   # 将所有优化的张量的梯度设置为零
            outputs = net(inputs)                                                   # 相当于调用Net()的forward方法
            loss = criterion(outputs, labels)                                       # 损失函数，具体看源码去，它主要刻画的是实际输出（概率）与期望输出（概率）的距离，也就是交叉熵的值越小，两个概率分布就越接近
            loss.backward()                                                         # 反向传播，计算当前梯度，backward()在这里https://pytorch.org/docs/stable/autograd.html，
                                                                                    # 应该是调用损失函数时返回了些什么才会使loss有这个方法
            optimizer.step()                                                        # 进行单次优化
            running_loss += loss.item()                                             # 计算这次训练loss的平均值，前面有清零操作，item()在这里https://pytorch.org/docs/stable/tensors.html?highlight=item#torch.Tensor.item
                                                                                    # 八成也应该是调用损失函数时返回了些什么才会使loss有这个方法

            if i % 40 == 39:                                                        # 打印出loss值
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 40))
                running_loss = 0.0

    print('Finished Training')                                                      
    torch.save(net, 'net.pkl')                                                      # 保存整个神经网络的模型结构以及参数
    torch.save(net.state_dict(), 'net_params.pkl')                                  # 只保存模型参数
                                                                                    # 可以看这个博客：https://blog.csdn.net/caiweibin2246/article/details/107559524
                                                                                    # 莫烦教程：https://www.pytorchtutorial.com/3-4-save-and-restore-model/
                                                                                    # 官网API说明：https://pytorch.org/docs/stable/generated/torch.save.html



    


if __name__ == '__main__':
    print(torch.cuda.is_available())
    trainandsave()
