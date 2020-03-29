import d2lzh as d2l
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn

########################
#  准备几个超参
########################
batch_size = 256
num_epochs, lr = 5, 0.5
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

########################
#  设置网络形式
########################
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'),
        nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
########################
#  设置loss函数形式
########################
loss = gloss.SoftmaxCrossEntropyLoss()

########################
#  开始训练
########################

d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, trainer)