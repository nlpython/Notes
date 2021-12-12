# 									PyTorch



## 1. random

### 1.1 rand

```python
# 返回[0, 1]之间的随机数
torch.rand(4, 3)
tensor([[3.5660e-01, 2.4175e-01, 3.3922e-01],
        [7.6469e-01, 3.1886e-01, 2.6293e-01],
        [4.3449e-01, 1.9652e-04, 9.9456e-01],
        [3.5111e-01, 4.3965e-01, 5.5827e-01]])
```

### 1.2 randn

```python
# 返回一组满足N(0, 1)正态分布的随机数
torch.randn(4, 3)
tensor([[ 0.1169, -0.6394,  0.8261],
        [-1.9318, -1.1511, -0.2720],
        [ 0.6991, -1.0631,  1.4761],
        [ 1.0109, -0.9506,  0.6074]])
```

### 1.3 randint

```python
# 返回[low, high)之间的随机整数
torch.randint(1, 3, [4, 3])
tensor([[1, 2, 2],
        [1, 2, 2],
        [1, 2, 1],
        [1, 2, 2]])
```



## 2. math

### 2.1 add

````python
torch.add(x, y)
tensor([[2., 2.],
        [2., 2.],
        [2., 2.]])

# 创建一个空的result
result = torch.empty(3, 2)
torch.add(x, y, out=result)	# 将结果保存在result中

# y -= x
y.add_(x) 
````

### 2.2 other

同add操作



## 3. shape

```python
x = torch.randn(4, 4)

# -1表示自动匹配
x.view(-1, 2)
tensor([[-0.2235,  0.7340],
        [ 0.0827, -0.1139],
        [-0.1384,  0.9476],
        [ 0.4965, -0.0498],
        [ 0.1948,  0.1302],
        [ 0.6812, -0.7222],
        [-1.6224, -1.5482],
        [-1.0436,  0.3090]])

x.view(16)
tensor([-0.2235,  0.7340,  0.0827, -0.1139, -0.1384,  0.9476,  0.4965, -0.0498,
         0.1948,  0.1302,  0.6812, -0.7222, -1.6224, -1.5482, -1.0436,  0.3090])

x.view(1, 16)
tensor([[-0.2235,  0.7340,  0.0827, -0.1139, -0.1384,  0.9476,  0.4965, -0.0498,
          0.1948,  0.1302,  0.6812, -0.7222, -1.6224, -1.5482, -1.0436,  0.3090]])
```



## 4. data

### 4.1 item()

```python
x = torch.randn(1)
# 只有x中只有一个值的时候才能调用此函数
x.item()
-0.9429922699928284
```

### 4.2 numpy()

```python
x = torch.ones(3, 2)
# 将tensor转换成numpy类型
x.numpy()
array([[1., 1.],
       [1., 1.],
       [1., 1.]], dtype=float32)
```

**tensor和numpy相同, 存储空间共享.**

### 4.3 from_numpy

```python
# numpy => tensor
torch.from_numpy(x)
tensor([1., 1., 1., 1., 1.], dtype=torch.float64)
```

### 4.4 Cuda

```python
# 如果服务器上已经安装了GPU和CUDA
if torch.cuda.is_available():
	# 定义一个对象, 这里指定CUDA, 即使用GPU
    device = torch.device('cuda')
    # 直接在GPU上创建一个Tensor
    y = torch.randn(x, device=device)
    # 将在CPU上的x张量移动到GPU
    x = x.to(device)
    # x和y都在GPU上, 才能支持加法运算
    z = x + y
    # 也可以将z移动到CPU上, 并同时指定tensor的数据类型
    z.to('cpu', torch.double)
```



## 5. autograd

在整个pytorch框架中, 所有的神经网络本质都是一个autograd package(自动求导工具包).

### 5.1 Tensor

关于torch.Tensor:

- torch.Tensor是整个package中的核心类, 如果属性.requires_grad设置为True, 它将追踪在这个类上定义的所有操作. 当代码要进行反向传播的时候, 直接调用.backward()就可以自动计算所有的梯度,在整个Tensor上的所有梯度将被累加进属性.grad中.
- 如果想终止一个Tensor在计算图中追溯回溯, 只需要执行.detach()就可以将该Tensor从计算图中撤下.
- 除了.detach(), 如果想终止对计算图的回溯, 也就是不再进行方向传播求导数的过程, 也可以采用代码块的方式with torch.no_grad(): 这种方式非常适用于对模型进行预测的时候, 因为预测阶段不再需要对梯度进行计算.

```python
torch.randn(3, 3, requires_grad=True)
tensor([[ 0.3937, -1.0660,  1.4021],
        [ 0.1661, -0.4292,  1.0090],
        [ 0.7087,  0.1737,  0.4018]], requires_grad=True)
x.requires_grad = True
x.requires_grad_(False)
```



### 5.2 Function

关于torch.Function:

- Function类是和Tensor类同等重要的一个核心类, 它和Tensor共同创建了一个完整类, 每一个Tensor拥有一个.grad_fn属性, 代表引用了那个具体的Function创建了该Tensor.
- 如果某个张量Tensor是用户自定义(即定义的, 而不是由某个计算衍生出的), 则其对应的grad_fn is None.



## 6. network

### 6.1 torch.nn

关于torch.nn:

- 使用PyTorch来构建神经网络, 主要的工具都在torch.nn中.
- nn依赖于autograd来定义模型, 并对其自动求导.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义一个简单的网络类
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 定义第一层卷积神经网络, 输入通道为1, 输出通道为6, filter.size = [3, 3]
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # 定义三个全连接层
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 在[2, 2]的池化窗口下执行最大池化操作
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        # 计算size, 除了第0个维度上的batch_size
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

if __name__ == '__main__':
    net = Net()
    print(net)
    params = list(net.parameters())
    # print(len(params))
    # [nSamples, nChannels, Height, Width]
    input = torch.randn(3, 1, 32, 32)
    out = net(input)
    tensor([[ 0.0512, -0.0136,  0.1191, -0.0747,  0.0925, -0.0610, -0.0331,  0.0334,
         -0.0111, -0.0627],
        [ 0.0396, -0.0320,  0.1175, -0.0860,  0.1061, -0.0593, -0.0343,  0.0255,
         -0.0295, -0.0715],
        [ 0.0168, -0.0235,  0.1095, -0.0762,  0.0905, -0.0754, -0.0225,  0.0301,
         -0.0121, -0.0577]], grad_fn=<AddmmBackward0>)

```

### 6.2 nn.MSELoss

均方损失(Mean Squre)

```python
output = net(input)
target = torch.randn(30)

# 改变target的形状为二维张量, 为了和output匹配
target = target.view(3, -1)
criterion = nn.MSELoss()

loss = criterion(output, target)
loss
tensor(0.5107, grad_fn=<MseLossBackward0>)
```

### 6.3 nn.CrossEntropyLoss

```python
# 定义交叉熵损失
criterion = nn.CrossEntropyLoss()
loss = criterion(outputs, labels)
loss.backward()

# 分两步
output = F.log_softmax(x, dim=1)
loss = F.null_loss(output, target)
```



### 6.4 zero_grad

```python
# 梯度清0, 否则梯度会在不同批次间的数据间累加
net.zero_grad()
# 进行一次反向传播
loss.backward()
```

### 6.5 SGD

更新网络参数

- 更新参数最简单的算法就是SGD(随机梯度下降)
- 具体的算法公式表达为weights = weights - learning_rate * gradint

```python
# SGD传统python代码
learning_rate = 0.01
for f in net.parameters():
	f.data.sub_(f.grad.data * learning_rate)
```

利用torch

```python
# 首先导入优化器的包, optim中含有若干优化算法, 比如SGD,Adam
import torch.optim as optim

# 通过optim创建优化器对象
optimizer = optim.SGD(net.parameters(), lr=0.01)
# 梯度清零
optimizer.zero_grad()

output = net(input)
loss = criterion(output, target)

# 对损失执行反向传播操作
loss.backward()
# 参数的更新通过一次标准代码来执行
optimizer.step()
```

### 6.6 nn.LSTM

LSTM和GRU都由torch.nn提供

**torch.nn.LSTM(inputs_size, hidden_size, num_layers, batch_first, dropout, bidirectional)**

- inputs_size: 输入数据的形状, 即embedding_dim.
- hidden_size: 隐藏层的数量, 即每层有多少个LSTM单元.
- num_layers: 即LSTM的层数.
- batch_first: 默认值为False, 输入的数据需要[seq_len, batch, featrue], 如果为True, 则为[batch, seq_len, feature].
- dropout: dropout的比例, 默认值为0. 可让部分参数随机失活, 能够提高训练速度, 防止过拟合.
- bidirectional: 是否使用双向LSTM, 默认是False.

实例化后LSTM对象之后, **不仅需要传入数据, 还需要前一次的h_0(前一次隐藏状态)和c_0(前一次memory).**(默认初始化为0)

即: LSTM(input, (h_0, c_0))

LSTM的默认输出为: output, (h_n, c_n)

- output: (seq_len, batch, num_directions * hidden_size)
- h_n: (num_layers * num_directions, batch, hidden_size)
- c_n: (num_layers * num_directions, batch, hidden_size)

```python
"""
lstm使用实例
"""
import torch
import torch.nn as nn

# 超参数
batch_size = 100
seq_len = 20
vocab_size = 100
embedding_dim = 30

hidden_size = 18
num_layer = 1

input = torch.randint(low=0, high=100, size=[batch_size, seq_len])

# 数据经过embedding处理
embeddiing = nn.Embedding(vocab_size, embedding_dim)
# => [batch_size, seq_len, embedding_dim]
input_embeded = embeddiing(input)

# 将input_embeded数据传入lstm
lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layer, batch_first=True)
output, (h_n, c_n) = lstm(input_embeded)

print(output)
print(output.size())    # [100, 20, 18]
print(h_n.size())       # [1, 100, 18]
print(c_n.size())       # [1, 100, 18]

# 获取最后一次时间步上输出
print(output[:, -1, :].size())	# [100, 18]
# 获取最后一次时间步上的hidden_state
print(h_n[-1:, :, :].size())	# [1, 100, 18]
```

### 6.7 nn.Sequential

nn.Sequential是一个有序的容器, 其中传入的是构造类(各种用来处理的input的类), 最终input会被Sequential中的构造器一次执行.

例如:

```python
layers = nn.Sequential(
    nn.Linear(input_dim, hidden_size_1),
    nn.ReLU(True),      # inplace = False, 是否对输出就地修改, 默认为False
    nn.Linear(hidden_size_1, hidden_size2),
    nn.ReLU(True),
    nn.Linear(hidden_size_2, output_dim)
)
```

### 6.8 nn.BatchNorm1d

batch normalization翻译成中文就是批规范化, 即在每个batch训练的过程中, 对参数进行归一化处理, 从而达到加快训练速度的效果.

以sigmoid激活函数为例, 它在反向传播的过程中, 在值0, 1的时候, 梯度接近于0, 导致参数被更新的幅度很小, 训练速度很慢. 但是如果进行归一化之后, 就会尽可能地把数据拉到[0, 1]地范围, 从而让参数更新地幅度变大, 提高训练速度.

**batchNorm一般会放到激活函数之后, 即对输出进行激活处理之后再进入batchNorm.**

```python
layers = nn.Sequential(
    nn.Linear(input_dim, hidden_size_1),
    nn.ReLU(True),      # inplace = False, 是否对输出就地修改, 默认为False
    nn.BatchNorm1d(hidden_size_1),
    
    nn.Linear(hidden_size_1, hidden_size2),
    nn.ReLU(True),
	nn.BatchNorm1d(hidden_size_2),
    
    nn.Linear(hidden_size_2, output_dim)
)
```





## 7. Function

### 7.1 unsqueeze

```python
x = torch.randn(3, 5, 5)
x.unsqueeze(0)
tensor([[[[ 1.1409,  0.6631,  0.1097, -0.1283, -0.9606],
          [ 0.4644, -0.5534, -0.5051, -1.0581, -0.6318],
          [ 1.0638, -0.0821, -0.0307, -0.3892, -0.2530],
          [-1.7476,  0.4410,  0.2487, -0.5119,  1.7660],
          [ 1.3575, -0.1486,  1.5306, -2.0209,  0.1642]],
         [[-0.4325,  1.3418, -0.7083, -1.0927,  0.4047],
          [-1.6599, -0.1249, -0.1248, -0.8173,  0.7993],
          [ 0.0891, -0.7809,  0.5308, -0.1077,  0.3994],
          [-0.5067, -1.1756,  1.3608, -0.2987,  0.8566],
          [ 0.2299,  2.0760, -0.2104, -0.8309, -0.7438]],
         [[-0.3864, -0.6580, -0.5890, -1.4323, -1.0220],
          [ 0.1295, -0.5992,  0.5012, -0.0481, -0.4859],
          [ 0.3186, -0.8763,  1.8741,  0.8163,  0.2274],
          [-0.1309, -0.0512, -0.4789, -0.7026,  1.4465],
          [ 1.6590,  1.0816, -0.7832, -0.3551, -1.5969]]]])
```

### 7.2 max

```python
# 第一个参数返回axis上的最大值, 第二个返回最大值的位置索引
outputs = torch.randn(4, 10)
_, predicted = torch.max(outputs, axis=1)

predicted
tensor([8, 7, 4, 3])
_
tensor([2.3335, 2.1936, 1.4814, 1.2780])
```

### 7.3 save

```python
# 保存模型
torch.save(network.state_dict(), './model/lstm.pkl')
torch.save(optimizer.state_dict(), './model/optim.pkl')
```

### 7.4 load_state_dict

```python
# 加载模型
network.load_state_dict(torch.load('./model/lstm.pkl'))
network = network.to(lib.device)
```

### 7.5 cat

```python
x = torch.ones(2, 3)
y = torch.ones(1, 3)

torch.cat([x, y], dim=0)
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]])
```

### 7.6 squeeze

````python
# 去除第0个维度
x = torch.tensor([[1, 2]])
x.squeeze(0)
tensor([1, 2])
````

### 7.7 unsqueeze

```python
# 在最后一个维度上增加一个维度
x.unsqueeze(-1)
tensor([[[1],
         [2]]])
# 在第0个维度上增加一个维度
x.unsqueeze(0)
tensor([[[1, 2]]])
```

### 7.8 bmm

```python
# [n, a, b] @ [n, b, c] = [n, a, c]
X = torch.ones((2, 1, 4))
Y = torch.ones((2, 4, 6))

torch.bmm(X, Y).shape
torch.Size([2, 1, 6])
```

### 7.9 repeat_interleave

```python
x = torch.tensor([1, 2, 3])

torch.repeat_interleave(x, 2)
Out[4]: tensor([1, 1, 2, 2, 3, 3])
```

### 7.10 transpose

```python
# 只能交换两个维度
x = torch.randn(3, 2, 4)
x.shape
Out[6]: torch.Size([3, 2, 4])
x.transpose(1, 2).shape
Out[7]: torch.Size([3, 4, 2])
```

### 7.11 permute

```python
# 交换任意维度 
x = torch.randn(3, 2, 4)
x.shape
Out[6]: torch.Size([3, 2, 4])
x.permute(1, 0, 2).shape
Out[8]: torch.Size([2, 3, 4])
```





## 8. Library

### 8.1 jieba

#### 8.1.1 cut

```python
import jieba
text = "小明去买了一个苹果"
seq_list = jieba.cut(text)
# cut返回一个类对象
jieba.lcut(text)
# lcht返回一个列表
['小明', '去', '买', '了', '一个', '苹果']
result = ' '.join(seq_list)
小明 去 买 了 一个 苹果
Library
```

#### 8.1.2 pesg

```python
# 词性标注
import jieba.posseg as pseg
pseg.lcut('我爱北京天安门')
[pair('我', 'r'), pair('爱', 'v'), pair('北京', 'ns'), pair('天安门', 'ns')]
```

### 8.2 pickle

模型的加载和保存.

```python
import pickle
# 保存模型, ws是一个对象
pickle.dump(ws, open('./ws.pkl', 'wb'))

# 加载模型
ws = pickle.load(path='./ws.pkl', 'rb')
```











## 9.Models

### 9.1 CIFAR10

#### 9.1.1 dataset

CIFAR10, 每张图片的尺寸是3 * 32 * 32, 一共有十种不同的分类['airplane', 'bird', 'cat', 'dog', ...]

#### 9.1.2 Setup

1) 使用torchvision下载CIFAR10数据集

```python
import torch
import torchvision
import torchvision.transforms as transforms
```

- 下载数据集并对图片进行调整, 因为torchvision数据集的输出是PILImage格式, 数据域在[0, 1], 我们将其转换为标准格式数据域[-1, 1].

```python
transforms = transforms.Compose(
	[transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2) # num_workers表示多线程

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

- 展示图片

```python
# 展示图片
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(trainloader)
images, labels = dataiter.next()

# 展示图片
imshow(torchvision.utils.make_grid(images))
```

![image-20211119131340740](C:\Users\86185\AppData\Roaming\Typora\typora-user-images\image-20211119131340740.png)



- 定义神经网络

```python
# 定义一个简单的网络类
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 定义第一层卷积神经网络, 输入通道为3, 输出通道为6, filter.size = [3, 3]
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # 定义三个全连接层
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        # 定义池化窗口
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # 在[2, 2]的池化窗口下执行最大池化操作
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @staticmethod
    def num_flat_features(x):
        # 计算size, 除了第0个维度上的batch_size
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
```

- 定义损失函数

```python
import torch.optim as optim
# 定义交叉熵损失
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
```

- 反向传播+梯度更新

```python
for epoch in range(2):
    
    running_loss = 0
    for i, data in enumerate(trainloader, 0):
        # data中包含了输入图像张量inputs, 标签张量labels
        inputs, labels = data
        
        # 首先将优化器梯度归零
        optimizer.zero_grad()
        
        # 输入图像张量进网络, 得到输出张量
        outputs = net(inputs)
        
        # 利用网络的输出outputs和标签labels计算损失值
        loss = criterion(outputs, labels)
        
        # 反向传播+梯度更新
        loss.backward()
        optimizer.step()
        
        # 打印轮次和损失值
        running_loss += loss.item()
        if (i + 1) % 500 == 0:
            print(f'epoch: {epoch + 1} | step: {i + 1} | loss: {running_loss / 500}')
            running_loss = 0
            
print('Finished Training.')
```

- 调用GPU

```python
device = torch.device('cuda:0')
net = Net().to(device)
# inputs, label也要改变, 否则无法运行
inputs, labels = data[0].to(device), data[1].to(device)
```

- 加载模型

```python
# 首先实例化模型的类对象
net = Net()
# 加载训练阶段保存好的模型的状态字典
net.load_state_dict(torch.load(PATH))

# 利用模型对图片进行预测
outputs = net(image)

# 共有10个类别, 采用模型计算出的概率最大作为预测的类别
_, predictd = torch.max(outputs, 1)

# 打印标签结果
print('Predictd: ', ' '.join('%5s', % classes[predicted[j]] for j in range(4)))
```

- 接下来查看在全部测试集上的表现

```python
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, axis=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Accuracy: {correct / total}')
```

### 9.2 IMDB

#### 9.2.1 数据集处理:

```python
from torch.utils.data import DataLoader, Dataset
import os
import re
from wordSequence import Word2Sequence
# import pickle
from lib import ws


def tokenlize(content):
    filters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>', '\?',
               '@'
        , '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”', '“', "'"]
    text = re.sub('<.*?>', " ", content)
    content = re.sub('|'.join(filters), ' ', content)
    tokens = [i.strip().lower() for i in content.split()]
    return tokens

class ImdbDataset(Dataset):
    def __init__(self, train=True):
        self.train_data_path = '../data/aclImdb/train'
        self.test_data_path = '../data/aclImdb/test'
        data_path = self.train_data_path if train else self.test_data_path

        # 把所有的文件名放入列表
        temp_data_path = [os.path.join(data_path, 'pos'), os.path.join(data_path, 'neg')]
        self.total_file_path = []    # 所有评论文件的路径
        for path in temp_data_path:
            file_name_list = os.listdir(path)
            file_path_list = [os.path.join(path, i) for i in file_name_list if i.endswith('.txt')]
            self.total_file_path.extend(file_path_list)

    def __getitem__(self, idx):
        file_path = self.total_file_path[idx]
        # 获取label
        label_str = file_path.split('\\')[-2]
        label = 0 if label_str == 'neg' else 1
        # 获取内容
        tokens = tokenlize(open(file_path, encoding='utf-8').read())
        return tokens, label

    def __len__(self):
        return len(self.total_file_path)

def collate_fn(batch):
    """
    :param batch: 一个getitem的结果 [[token, label], [token, label]..]
    :return:
    """
    import torch
    ret, label = zip(*batch)
    # print('ret', ret)
    content = [ws.transform(i, max_len=20) for i in ret]
    # print(content)
    content = torch.LongTensor(content)
    label = torch.LongTensor(label)
    return content, label


def get_data_loader(is_train=True):
    imdb_dataset = ImdbDataset(is_train)
    data_loader = DataLoader(imdb_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn, drop_last=True)
    return data_loader

if __name__ == '__main__':
    import os
    from dataset import tokenlize
    from dataset import ImdbDataset
    import pickle
    from tqdm import tqdm

    # 保存ws
    ws = Word2Sequence()
    path = '../data/aclImdb/train'
    temp_data_path = [os.path.join(path, 'pos'), os.path.join(path, 'neg')]
    for data_path in temp_data_path:
        file_paths = [os.path.join(data_path, file_name) for file_name in os.listdir(data_path) if file_name.endswith('.txt')]
        for file_path in tqdm(file_paths):
            sentence = tokenlize(open(file_path, encoding='utf-8').read())
            ws.fit(sentence)
            # print(ws.dict)

    ws.build_vocab(min=10)

    pickle.dump(ws, open('./ws.pkl', 'wb'))
```

#### 9.2.2 文本序列化

文本 -> 数字 -> 向量

这里我们考虑把文本中的每个词语和其对应的数字, 使用字典保存, 同时实现方法把句子通过字典映射为包含数字的列表.

实现文本序列化之前, 应考虑以下几点:

- 如何使用字典把词语和数字对应
- 不同的词语出现的次数不同, 是否需要对低频或高频词过滤, 词典的总数是否需要限制
- 得到词典后, 如何把句子转化为数字序列, 如何把数字序列转化为句子
- 不同句子长度不同, 每个batch的句子如何构造成相同的长度(可以对短句进行填充, 长句进行切割)
- 对于新出现的词语在词典中没有怎么办(可用特殊字符代替)

```python
class Word2Sequence:
    # 特殊字符
    UNK_TAG = 'UNK'
    PAD_TAG = 'PAD'
    UNK = 0
    PAD = 1

    def __init__(self):
        self.dict = {
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD
        }
        self.inverse_dict = {
            self.UNK: self.UNK_TAG,
            self.PAD: self.PAD_TAG
        }
        self.count = {}

    def fit(self, sentence):
        """
        :param sentence: [word1, word2, word3,..]
        :return:
        """
        for word in sentence:
            if word not in self.count.keys():
                self.count[word] = 1
                self.dict[word] = len(self.dict)
                self.inverse_dict[len(self.inverse_dict)] = word
            else:
                self.count[word] += 1


    def build_vocab(self, min=5, max=None):
        """
        生成词典
        :param min: 最少出现次数
        :param max: 最大出现次数
        :param max_features: 一共保留多少词语
        :return:
        """
        # 删除count中小于min的词
        self.count = {word: value for word, value in self.count.items() if value > min}

        max_features = len(self.count)

        idx = 2
        for word in self.count.keys():
            self.dict[word] = idx
            idx += 1
        # 得到一个反转字典
        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def transform(self, sentence, max_len=30):
        # idx = 0
        # for key, value in self.dict.items():
        #     print(key, value)
        #     idx += 1
        #     if idx == 100:
        #         break
        # exit()
        id_list = []
        for word in sentence:
            id_list.append(self.dict.get(word, self.UNK))

        # 对句子进行填充或裁剪
        if len(id_list) > max_len:
            id_list = id_list[:max_len]
        else:
            id_list.extend([self.PAD for _ in range(max_len - len(id_list))])
        return id_list

    def inverse_transform(self, indices):
        word_list = []
        for idx in indices:
            word_list.append(self.inverse_dict.get(idx, self.UNK_TAG))
        return word_list

    def __len__(self):
        return len(self.dict)

if __name__ == '__main__':
    pass
```

#### 9.2.3 model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from dataset import get_data_loader
from wordSequence import Word2Sequence
import pickle
from lib import ws


max_len = 20
# print(ws.dict)
# print(len(ws))

class IMDB(nn.Module):

    def __init__(self):
        super(IMDB, self).__init__()
        self.embedding = nn.Embedding(len(ws), 100)
        self.fc1 = nn.Linear(max_len * 100, 2)

    def forward(self, input):
        """
        :param input: [batch_size, max_len]
        :return:
        """
        # [batch_size, max_len] => [batch_size, max_len, 100]
        x = self.embedding(input)
        # print('x:', x.size())
        x = x.view(128, -1)
        # print(x.size(
        out = self.fc1(x)

        return out


network = IMDB()
optimizer = optim.Adam(network.parameters(), lr=0.001)
def train(epochs):
    for epoch in range(epochs):
        for idx, (input, target) in enumerate(get_data_loader(True)):
            # 梯度归零
            optimizer.zero_grad()
            output = network(input)

            criteria = nn.CrossEntropyLoss()
            loss = criteria(output, target)
            loss.backward()
            optimizer.step()

            if idx % 10 == 0:
                print(f'epoch: {epoch+1} | step: {idx+1} | loss: {loss.item()}')

def test():
    test_loss = 0
    correct = 0
    mode = False
    network.eval()
    test_dataloader = get_data_loader(mode)
    with torch.no_grad():
        for target, input, input_lenght in test_dataloader:
            output = network(input)
            test_loss += F.nll_loss(output, target, reduction='sum')
            pred = torch.max(output, dim=-1)[-1]
            correct = pred.eq(target.data).sum()
        test_loss = test_loss / len(test_dataloader.datset)
        print(f'loss: {test_loss}, acc: {correct / len(test_dataloader.dataset)}')

if __name__ == '__main__':
    # print(network)
    train(2)
    test()
```

#### 9.2.4 eval

```python
def test():
    test_loss = 0
    correct = 0
    mode = False
    network.eval()
    test_dataloader = get_data_loader(mode)
    with torch.no_grad():
        for target, input, input_lenght in test_dataloader:
            output = network(input)
            test_loss += F.nll_loss(output, target, reduction='sum')
            pred = torch.max(output, dim=-1)[-1]
            correct = pred.eq(target.data).sum()
        test_loss = test_loss / len(test_dataloader.datset)
        print(f'loss: {test_loss}, acc: {correct / len(test_dataloader.dataset)}')
```

### 9.3 lstmImdb

为了达到更好的效果, 我们加入lstm层, 并对模型做如下修改:

- max_len = 200
- 构建dataset的过程, 把数据转化成2分类, pos = 1, neg = 0
- 实例化lstm时, dropout=0.5, 在model.eval()的过程中, dropout自动为0.

#### 9.3.1 model

```python
# 超参数
max_len = 200
hidden_size = 128
num_layers = 2
bidriectional = True
dropout = 0.5

class LSTMIMDB(nn.Module):

    def __init__(self):
        super(LSTMIMDB, self).__init__()
        self.embedding = nn.Embedding(len(ws), 100)
        # 加入lstm层
        self.lstm = nn.LSTM(input_size=100, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=bidriectional, dropout=dropout)

        self.fc1 = nn.Linear(hidden_size * 2, 2)

    def forward(self, input):
        """
        :param input: [batch_size, max_len]
        :return:
        """
        # [batch_size, max_len] => [batch_size, max_len, 100]
        x = self.embedding(input)
        # x: [batch_size, max_len, 2 * hidden_size], h_n: [2 * 2, batch_size, hidden_size]
        x, (h_n, c_n) = self.lstm(x)
        # 获取两个方向最后一次的output, 进行concat
        output_f = h_n[-2, :, :]    # 正向最后一次输出
        output_b = h_n[-1, :, :]    # 反向最后一次输出
        output = torch.cat([output_f, output_b], dim=-1)    # [batch_size, hidden_size * 2]

        # [batch_size, 2]
        out = self.fc1(output)
```

#### 9.3.2 train

```python
def train(epochs):
    for epoch in range(epochs):

        correct, total = 0, 0
        for idx, (input, target) in enumerate(get_data_loader(True)):
            # 调用cuda
            input = input.to(lib.device)
            target = target.to(lib.device)

            # 梯度归零
            optimizer.zero_grad()
            output = network(input)

            criteria = nn.CrossEntropyLoss()
            loss = criteria(output, target)
            loss.backward()
            optimizer.step()

            pred = torch.max(output, dim=-1)[-1]
            correct += (pred == target).sum().item()
            total += output.size(0)

            if idx % 10 == 0:
                print(f'epoch: {epoch+1} / {epochs} | step: {idx+1} | loss: {loss.item()} | accuracy: {correct / total}')
                correct, total = 0, 0
```

#### 9.3.3 eval

```python
def test():
    test_loss = 0
    correct = 0
    total = 0
    mode = False
    network.eval()
    test_dataloader = get_data_loader(mode)
    with torch.no_grad():
        for input, target in tqdm(test_dataloader, 'Testing'):
            input = input.to(lib.device)
            target = target.to(lib.device)
            output = network(input)

            criteria = nn.CrossEntropyLoss()
            test_loss += criteria(output, target)
            pred = torch.max(output, dim=-1)[-1]
            correct += (pred == target).sum().item()
            total += output.size(0)

        print('On test set:', f'loss: {test_loss / total}, accuracy: {correct / total}')
```

### 9.4 聊天机器人

#### 9.4.1 综述

**分类:** 

1. QA BOT (问答机器人): 回答问题
     	 1) 代表: 智能客服
          	 2) 比如: 提问和回答
2.  TASK BOT (任务机器人): 完成任务
   1. 代表: siri
   2. 比如: 设置明天早上8点的闹钟
3. CHAT BOT (聊天机器人): 通用, 开放聊天
   1. 代表: 微软小冰

**QA实现:** 

 1. 信息检索, 搜索 (简单, 效果一般, 对数据问答对的要求高)
    	关键词: TF-IDF, SVM, 朴素贝叶斯, RNN, CNN

 2. 知识图谱

    ​    在图形数据库中存储知识和知识间的关系, 把问答转换为查询语句, 能够实现推理

**TASK实现:**

 	1. 语音转文字
 	2. 意图识别, 领域识别, 文本分类
 	3. 槽位填充
 	4. 回话管理, 会话策略
 	5. 自然语言生成
 	6. 文本转语音

**CHAT实现**:

	1. 信息检索
	2. seq2seq的变种



阿里小蜜-电商智能助理:  https://juejin.cn/post/6844903504918609934



### 9.5 Seq2Seq

- Dataset

```python
"""
准备数据集
"""
import torch
import numpy as np
import lib
from torch.utils.data import DataLoader, Dataset


class NumDataset(Dataset):

    def __init__(self):
        super(NumDataset, self).__init__()
        # 使用numpy随机创建一个数
        self.data = np.random.randint(1, 1e8, size=[100000])

    def __getitem__(self, idx):
        input = list(str(self.data[idx]))    # ['1', '8', '0', '6', '2', '9', '7']
        label = input + ['0']
        input_length = len(input)
        label_length = len(label)
        return input, label, input_length, label_length

    def __len__(self):
        return len(self.data)

def collate_fn(batch):
    # batch: [(input, label, input_length, label_length), (input, label, input_length, label_length)...]

    batch = sorted(batch, key=lambda x: x[3], reverse=True)   # 根据
    content, label, content_len, label_len = zip(*batch)
    # ret: ([input, input,,,], [label, label,,,])
    content = torch.LongTensor([lib.ns.transform(i, max_len=lib.max_len) for i in content])
    label = torch.LongTensor([lib.ns.transform(i, max_len=lib.max_len+1) for i in label])
    content_len = torch.LongTensor(content_len)
    label_len = torch.LongTensor(label_len)
    return content, label, content_len, label_len


def get_data_loader(is_train=True):
    num_dataset = NumDataset()
    return DataLoader(num_dataset, batch_size=lib.batch_size, shuffle=is_train, drop_last=True, collate_fn=collate_fn)

if __name__ == '__main__':
    # num_dataset = NumDataset()
    for input, label, len1, len2 in get_data_loader():
        print(input)
        print(label)
        print(len1)
        break
```

- Num_sequence

```python

class Num_sequnece:
    PAD_TAG = "PAD"
    PAD = 10
    UNK_TAG = "UNK"
    UNK = 11
    SOS_TAG = 'SOS'
    SOS = 12
    EOS_TAG = 'EOS'
    EOS = 13

    def __init__(self):
        self.dict = {str(i): i for i in range(10)}
        self.dict[self.PAD_TAG] = self.PAD
        self.dict[self.UNK_TAG] = self.UNK
        self.dict[self.SOS_TAG] = self.SOS
        self.dict[self.EOS_TAG] = self.EOS

        self.inverse_dict = {value: key for key, value in self.dict.items()}

    def transform(self, sentence, max_len=9, add_eos=False):
        if len(sentence) > max_len:
            sentence = sentence[: max_len]

        sentence_len = len(sentence)

        if add_eos:
            sentence = sentence + [self.EOS]
        if sentence_len < max_len:
            sentence = sentence + [self.PAD for _ in range(max_len - sentence_len)]

        return [self.dict.get(i, self.UNK) for i in sentence]

    def inverse_transform(self, indices):
        return [self.inverse_dict.get(i, self.UNK_TAG) for i in indices]


if __name__ == '__main__':
    ns = Num_sequnece()
    print(ns.dict)
    print(ns.inverse_dict)

```

#### 9.5.1 Encoder

```python
"""
编码器
"""
import torch.nn as nn
import lib
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from dataset import get_data_loader

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(lib.ns), embedding_dim=lib.embedding_dim,
                                      padding_idx=lib.ns.PAD)   # 传入后PAD相应的权重将不参与更新
        self.gru = nn.GRU(input_size=lib.embedding_dim, num_layers=lib.num_layers, hidden_size=lib.hidden_size,
                          batch_first=True, dropout=lib.dropout)



    def forward(self, input, input_length):
        """
        :param input: [batch_size, max_len]
        :return:
        """
        embeded = self.embedding(input)
        # 打包, 加入gru计算
        embeded = pack_padded_sequence(embeded, input_length, batch_first=True)
        out, hidden = self.gru(embeded)
        out, out_length = pad_packed_sequence(out, batch_first=True, padding_value=lib.ns.PAD)

        # out: [batch_size, seq_len, hidden_size]
        # hidden: [1 * 2, batch_size, hidden_size]
        return out, hidden, out_length
```

#### 9.5.2 Decoder

**如何计算损失:**

1. 每次输出是一个分类问题, 选择概率最大的词进行输出
2. 真实值的形状为[batch_size, max_len], 从而我们知道输出的结果需要是[batch_size, max_len, vocab_size]
3. 即预测值的最后一个维度进行计算log_softmax, 然后和真实值进行相乘, 从而得到损失

**如何将编码结果[1, batch_size, hidden_size]进行操作, 得到预测值. 解码器也是一个RNN, 即也可以使用LSTM的结构, 所以在解码器中:**

1. 通过循环, 每次计算一个time step的内容
2. 编码器的结果作为初始的隐藏状态, 定义一个[batch_size, 1]的全为sos的数据作为最开始的输入, 告诉解码器该工作了
3. 通过解码器预测一个输出[batch_size, hidden_size] (会进行形状的调整为[batch_size, vocab_size]), 把这个输出作为输入再使用解码器进行解码
4. 上述是一个循环, 循环max_len次
5. 把所有的输出concat, 得到[batch_size, max_len, vocab_size]

**在RNN的训练过程中, 使用前一个预测的结果作为下一个step的输入, 可能会导致一步错, 步步错, 那么该如何提升?**

1. 可以考虑在预训练的过程中, 将真实值作为下一步的输入
2. 使用真实值的过程中, 仍然使用预测值, 两种输入随机使用
3. 上述机制称做Teacher forcing

```python
"""
解码器
"""
import torch
import torch.nn as nn
import lib
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from dataset import get_data_loader
import torch.nn.functional as F

class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(lib.ns), embedding_dim=lib.embedding_dim,
                                      padding_idx=lib.ns.PAD)
        self.gru = nn.GRU(input_size=lib.embedding_dim, hidden_size=lib.hidden_size, batch_first=True,
                          num_layers=lib.num_layers)
        self.fc = nn.Linear(lib.hidden_size, len(lib.ns))


    def forward(self, target, encoder_hidden):
        # 1.获取encoder的输出, 作为隐藏状态
        decoder_hidden = encoder_hidden
        # 2.准备decoder第一个时间步上的输入, [batch_size, 1] SOS作为输入
        decoder_input = torch.LongTensor(torch.ones([target.size(0), 1], dtype=torch.int64) * lib.ns.SOS).to(lib.device)
        # 3.在第一个时间步上进行计算, 得到第一个时间步的输出, hidden_state

        # 预测结果  [batch_size, max_len+2, vocab_size]
        decoder_outputs = torch.zeros([lib.batch_size, lib.max_len + 2, len(lib.ns)]).to(lib.device)
        for t in range(lib.max_len + 2):
            # decoder_output_t: [batch_size, vocab_size]
            decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs[:, t :] = decoder_output_t.unsqueeze(1)

            value, index = torch.topk(decoder_output_t, 1)
            decoder_input = index    #[batch_size, 1]

        return decoder_outputs, decoder_hidden


    def forward_step(self, decoder_input, decoder_hidden):
        """
        计算每个时间步上的输出
        :param decoder_input: [batch_size, 1]
        :param decoder_hidden: [1, batch_size, hidden_size]
        :return:
        """
        decoder_input_embedding = self.embedding(decoder_input) # [batch_size, 1, embedding_size]

        # out: [batch_size, 1, hidden_size]
        # decoder_hidden: [1, batch_size, hidden_size]
        out, decoder_hidden = self.gru(decoder_input_embedding)

        out = out.squeeze(1) # [batch_size, hidden_size]
        output = F.log_softmax(self.fc(out), dim=-1)
        # output: [batch_size, vocab_size]
        return output, decoder_hidden

if __name__ == '__main__':
    from encoder import Encoder
    decoder = Decoder()
    encoder = Encoder()
    for input, target, input_size, target_size in get_data_loader():
        out, encoder_hidden, _ = encoder(input, input_size)
        decoder(target, encoder_hidden)

```

#### 9.5.3 Seq2Seq

```python
"""
Seq2Seq
"""
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
import lib
from torch.optim import Adam
from dataset import get_data_loader
import torch.nn.functional as F

class Seq2Seq(nn.Module):

    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, input, target, input_length, target_length):
        encoder_outputs, encoder_hidden = self.encoder(input, input_length)
        decoder_outputs, decoder_hidden = self.decoder(target, encoder_hidden)
        return decoder_outputs, decoder_hidden

net = Seq2Seq().to(lib.device)
optimizer = Adam(net.parameters(), lr=lib.learning_rate)
data_loader = get_data_loader(True)

def train(epochs):

    for epoch in range(epochs):
        for idx, (input, target, input_size, target_size) in enumerate(data_loader):
            # 调用cuda
            input, target, input_size, target_size = input.to(lib.device), target.to(lib.device), \
                                                     input_size.to(lib.device), target_size.to(lib.device)

            optimizer.zero_grad()
            decoder_outputs, _ = net(input, target, input_size, target_size)
            decoder_outputs = decoder_outputs.view(decoder_outputs.size(0)*decoder_outputs.size(1), -1) # [batch_size*seq_len, -1]
            target = target.view(-1) # [batch_size*seq_len]
            loss = F.nll_loss(decoder_outputs, target)
            loss.backward()
            optimizer.step()

            if idx % 20 == 0:
                print(f'epoch: {epoch+1} / {epochs} | step: {idx} | loss: {loss.item()}')


if __name__ == '__main__':
    train(lib.epochs)
```







## 10. class

### 10.1 Dataset

在torch中提供了数据集的基类torch.utils.data.Dataset, 继承这个基类, 我们能够非常快速的实现对数据的加载.

```python
import torch
from torch.utils.data import Dataset

data_path = './data/SMSSpamCollection'

# 完成数据集类
class MyDataSet(Dataset):

    def __init__(self):
        self.lines = open(data_path, encoding='utf-8').readlines()

    def __getitem__(self, index):
        # 获取索引对应位置的一条数据
        return self.lines[index]

    def __len__(self):
        # 返回数据的总数量
        return len(self.lines)

if __name__ == '__main__':
    dataset = MyDataSet()
    print(len(dataset))
    print(dataset[67])
5574
spam	Urgent UR awarded a complimentary trip to EuroDisinc Trav, Aco&Entry41 Or £1000. To claim txt DIS to 87121 
	
```

### 10.2 DataLoader

和Dataset配合使用

```python
# 自定义collate_fn函数
def collate_fn(batch):
    # batch: [(input, label), (input, label),,]
    ret = zip(*batch)
    # ret: ([input, input,,,], [label, label,,,])
    return ret

def get_data_loader(is_train=True):
    num_dataset = NumDataset()
    return DataLoader(num_dataset, batch_size=lib.batch_size, shuffle=is_train, drop_last=True, collate_fn=collate_fn)

for input, label in get_data_loader():
        print(input)
        print(label)
        break
```

### 10.3 torchvision/text

pytorch自带数据集:

1. torchvision: 图像   torchvision.datasets
2. torchtext: 文本

```python
import torchvision.transforms as transforms
transforms = transforms.Compose(
	[transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
```

### 10.4  TensorDataset

```python
from torch.utils.data import TensorDataset

a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = torch.tensor([44, 55, 66, 44, 55, 66, 44, 55, 66, 44, 55, 66])

# TensorDataset对tensor进行打包
train_ids = TensorDataset(a, b) 
for x_train, y_label in train_ids:
    print(x_train, y_label)

# dataloader进行数据封装
print('=' * 80)
train_loader = DataLoader(dataset=train_ids, batch_size=4, shuffle=True)
for i, data in enumerate(train_loader, 1):  
# 注意enumerate返回值有两个,一个是序号，一个是数据（包含训练数据和标签）
    x_data, label = data
    print(' batch:{0} x_data:{1}  label: {2}'.format(i, x_data, label))
```







