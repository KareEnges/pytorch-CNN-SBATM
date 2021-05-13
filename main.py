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



# 首先获取训练用的图像, 这里是批数据训练
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





# 建立一个简单神经网络，属于固定套路
# 可以看莫烦教程https://www.pytorchtutorial.com/3-1-regression/
class Net(nn.Module):                                                               # 继承 torch 的 nn.Module（所有神经网络模块的基类）
                                                                                    # 官网API说明：https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module
    def __init__(self):                                                             
        super(Net, self).__init__()                                                 # 继承 nn.Module 的 __init__ 方法，
        
        self.conv1 = nn.Conv2d(3, 6, 5)                                             # 卷积层第一层，这里是二维卷积
                                                                                    # in_channels（输入图像中的通道数，高度）, out_channels（卷积产生的通道数）, kernel_size（卷积内核的大小，这里是5*5像素）
                                                                                    # 官网API说明：https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
                                                                                    # 莫烦教程https://www.pytorchtutorial.com/4-1-cnn/ ，视频里面用硬币讲得很好

        self.pool = nn.MaxPool2d(2, 2)                                              # 池化层，对应的是二维卷积的池化层，在我看来就是筛选
                                                                                    # kernel_size（窗口大小，也就是用的大小2*2像素特征去筛选），stride（步长，默认值是kernel_size）
                                                                                    # 官网API说明：https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html

        self.conv2 = nn.Conv2d(6, 16, 5)                                            # 卷积层第二层，这里是二维卷积
                                                                                    # in_channels（第一层的输出）, out_channels（卷积产生的通道数）, kernel_size（卷积内核的大小，这里是5*5像素）

        self.fc1 = nn.Linear(400, 120)                                              # 全连接层
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):                                                           # 前向函数，怎么调用的可以看 https://www.cnblogs.com/llfctt/p/10967651.html , 相当于是 __call__ 方法调用
        x = self.pool(F.relu(self.conv1(x)))                                        # 
        x = self.pool(F.relu(self.conv2(x)))                                        # 
        x = x.view(-1, 400)                                                         # 

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 陈俊宇 - 0 黄志恒 - 1 林杨宗 - 2 丘海仁 - 3 詹艺锦 - 4
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


def trainandsave():  # 训练
    writer = SummaryWriter('runs/aoteman_1')
    trainloader = loadtraindata()
    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1000):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)

            img_grid = torchvision.utils.make_grid(inputs)
            writer.add_image('aoteman', img_grid)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 40 == 39:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 40))
                running_loss = 0.0

    print('Finished Training')
    torch.save(net, 'net.pkl')
    torch.save(net.state_dict(), 'net_params.pkl')


def reload_net():
    trainednet = torch.load('net.pkl')
    return trainednet


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def test():
    testloader = loadtestdata()
    net = reload_net()
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images, nrow=5))
    print('GroundTruth: '
          , " ".join('%5s' % classes[labels[j]] for j in range(25)))  # 打印前25个测试图
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)

    print('Predicted: ', " ".join('%5s' % classes[predicted[j]] for j in range(25)))


def test_one(img_path):
    model = reload_net()
    transform_valid = transforms.Compose([
        transforms.Resize((32, 32), interpolation=2),
        transforms.ToTensor()
    ]
    )
    img = Image.open(img_path)
    img_ = transform_valid(img).unsqueeze(0)
    outputs = model(img_)
    _, indices = torch.max(outputs, 1)
    result = classes[indices]
    print('predicted:', result)
    print(outputs)


def main():
    trainandsave()
    # test()
    # test_one(r'./test1.jpg')


if __name__ == '__main__':
    main()
