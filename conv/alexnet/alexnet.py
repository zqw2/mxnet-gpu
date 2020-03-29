import d2lzh as d2l
from mxnet import gluon, init, nd
from mxnet.gluon import data as gdata, nn
import mxnet as mx
import os
import sys
net = nn.Sequential()
# 使⽤较⼤的11 x 11窗⼝来捕获物体。同时使⽤步幅4来较⼤幅度减⼩输出⾼和宽。这⾥使⽤的输出通
# 道数⽐LeNet中的也要⼤很多
net.add(nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
    nn.MaxPool2D(pool_size=3, strides=2),
    # 减⼩卷积窗⼝，使⽤填充为2来使得输⼊与输出的⾼和宽⼀致，且增⼤输出通道数
    nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'),
    nn.MaxPool2D(pool_size=3, strides=2),
    # 连续3个卷积层，且使⽤更⼩的卷积窗⼝。除了最后的卷积层外，进⼀步增⼤了输出通道数。
    # 前两个卷积层后不使⽤池化层来减⼩输⼊的⾼和宽
    nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
    nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
    nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'),
    nn.MaxPool2D(pool_size=3, strides=2),
    # 这⾥全连接层的输出个数⽐LeNet中的⼤数倍。使⽤丢弃层来缓解过拟合
    nn.Dense(4096, activation="relu"), nn.Dropout(0.5),
    nn.Dense(4096, activation="relu"), nn.Dropout(0.5),
    # 输出层。由于这⾥使⽤Fashion-MNIST，所以⽤类别数为10，⽽⾮论⽂中的1000
    nn.Dense(10))

# X = nd.random.uniform(shape=(1, 1, 224, 224))
# net.initialize()
# for layer in net:
#     X = layer(X)
#     print(layer.name, 'output shape:\t', X.shape)

##############################################################################
# 加载数据
# 读取数据的时候我们额外做了⼀步将图像⾼和
# 宽扩⼤到AlexNet使⽤的图像⾼和宽224。这个可以通过Resize实例来实现。也就是说，我们
# 在ToTensor实例前使⽤Resize实例，然后使⽤Compose实例来将这两个变换串联以⽅便调⽤
##############################################################################
def load_data_fashion_mnist(batch_size, resize=None, root=os.path.join(
'~', '.mxnet', 'datasets', 'fashion-mnist')):
    root = os.path.expanduser(root) # 展开⽤⼾路径'~'
    transformer = []
    if resize:
        transformer += [gdata.vision.transforms.Resize(resize)]
    transformer += [gdata.vision.transforms.ToTensor()]
    transformer = gdata.vision.transforms.Compose(transformer)
    mnist_train = gdata.vision.FashionMNIST(root=root, train=True)
    mnist_test = gdata.vision.FashionMNIST(root=root, train=False)
    num_workers = 0 if sys.platform.startswith('win32') else 4
    train_iter = gdata.DataLoader(
    mnist_train.transform_first(transformer), batch_size, shuffle=True,num_workers=num_workers)
    test_iter = gdata.DataLoader(
    mnist_test.transform_first(transformer), batch_size, shuffle=False,num_workers=num_workers)
    return train_iter, test_iter

batch_size = 128
# 如出现“ out of memory”的报错信息，可减⼩batch_size或resize
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)

##############################################################################
# 开始训练
##############################################################################
lr, num_epochs, ctx = 0.01, 5, mx.cpu()
# d2l.try_gpu()
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)
