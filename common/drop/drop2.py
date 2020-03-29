import d2lzh as d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn

drop_prob1, drop_prob2 = 0.2, 0.5

######################
#  设置网络
######################
net = nn.Sequential()
net.add(nn.Dense(256, activation="relu"),
nn.Dropout(drop_prob1), # 在第⼀个全连接层后添加丢弃层
nn.Dense(256, activation="relu"),
nn.Dropout(drop_prob2), # 在第⼆个全连接层后添加丢弃层
nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))

######################
#  开始运行
######################
num_epochs, lr, batch_size = 5, 0.5, 256
loss = gloss.SoftmaxCrossEntropyLoss()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, trainer)