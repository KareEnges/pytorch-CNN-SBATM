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


def loadtraindata():
    path = r"./train"
    trainset = torchvision.datasets.ImageFolder(path,
                                                transform=transforms.Compose([
                                                    transforms.Resize((100, 100)),  # 将图片缩放到指定大小（h,w）

                                                    transforms.CenterCrop(32),
                                                    transforms.ToTensor()])
                                                )

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
    return trainloader


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # 卷积层
        self.pool = nn.MaxPool2d(2, 2)  # 池化层
        self.conv2 = nn.Conv2d(6, 16, 5)  # 卷积层
        self.fc1 = nn.Linear(400, 120)  # 全连接层
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 400)

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
