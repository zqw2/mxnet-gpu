import d2lzh as d2l
from mxnet import gluon, init, autograd
from mxnet.gluon import loss as gloss, nn


########################
#  准备几个超参
########################
num_epochs = 10
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


########################
#  设置网络形式
########################
net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))

########################
#  设置loss函数形式
########################
loss = gloss.SoftmaxCrossEntropyLoss()

########################
#  设置优化算法
########################
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

########################
#  设置预测准确率计算方法
########################
def accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        y = y.astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y).sum().asscalar()
        n += y.size
    return acc_sum / n
########################
#  开始训练
########################
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, trainer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
        if trainer is None:
            d2l.sgd(params, lr, batch_size)
        else:
            trainer.step(batch_size)
        y = y.astype('float32')
        train_l_sum += l.asscalar()
        train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
        n += y.size

        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))



train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, trainer)

########################
#  训练完成后，图形化展示结果
########################
for X, y in test_iter:
    break

true_labels = d2l.get_fashion_mnist_labels(y.asnumpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
d2l.show_fashion_mnist(X[0:9], titles[0:9])