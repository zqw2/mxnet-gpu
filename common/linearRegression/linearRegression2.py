from matplotlib import pyplot as plt
from mxnet import autograd, nd
import random
from d2lzh import *

from mxnet.gluon import data as gdata
from mxnet.gluon import nn
from mxnet import init
from mxnet.gluon import loss as gloss
from mxnet import gluon

###################
# 得到初始数据
###################
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)

def draw():
    set_figsize()
    plt.scatter(features[:, 1].asnumpy(), labels.asnumpy(), 1)
    plt.show()




#######################
# 利用gluon api训练数据
#######################
def train():
    ###################
    # 定义几个超参
    ###################
    batch_size = 10
    num_epochs = 3
    lr = 0.03

    # 将训练数据的特征和标签组合
    dataset = gdata.ArrayDataset(features, labels)
    data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)

    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=0.01))
    loss = gloss.L2Loss() # 平⽅损失⼜称L2范数损失

    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})

    for epoch in range(1, num_epochs + 1):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        l = loss(net(features), labels)
        print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))

    print("true_w=%s, true_b=%s" % (true_w, true_b))
    dense = net[0]
    print("after train: w=%s, b=%s" % (dense.weight.data(), dense.bias.data()))


#######################
# 不使用gluon api训练数据
#######################
def train_old():
    ###################
    # 定义几个超参
    ###################
    batch_size = 10
    lr = 0.03
    num_epochs = 3
    net = linreg
    loss = squared_loss

    ###################
    # 开始迭代
    ###################
    w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
    b = nd.zeros(shape=(1,))

    w.attach_grad()
    b.attach_grad()

    for epoch in range(num_epochs):  # 训练模型⼀共需要num_epochs个迭代周期
        # 在每⼀个迭代周期中，会使⽤训练数据集中所有样本⼀次（假设样本数能够被批量⼤⼩整除）。 X
        # 和y分别是⼩批量样本的特征和标签
        for X, y in data_iter(batch_size, features, labels):
            with autograd.record():
                l = loss(net(X, w, b), y)  # l是有关⼩批量X和y的损失
            l.backward()  # ⼩批量的损失对模型参数求梯度
            sgd([w, b], lr, batch_size)  # 使⽤⼩批量随机梯度下降迭代模型参数
        train_l = loss(net(features, w, b), labels)
        print('epoch %d, loss %f' % (epoch + 1, train_l.mean().asnumpy()))

    print("true_w=%s, true_b=%s" % (true_w, true_b))
    print("after train: w=%s, b=%s" % (w.asnumpy(),b.asnumpy()))

if __name__ == "__main__":
    # 图形化展示生成的数据(仅一个维度)
    # draw()

    print("===========METHOD1=================")
    #调用gluon api进行训练
    train()

    print("===========METHOD2=================")
    #普通线性回归进行训练，自己构建
    train_old()


