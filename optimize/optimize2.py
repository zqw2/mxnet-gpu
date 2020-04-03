import d2lzh as d2l
from mxnet import nd

eta = 0.4

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
d2l.plt.show()

eta = 0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
d2l.plt.show()


########################
# 动量法
########################
def momentum_2d(x1, x2, v1, v2):
    v1 = gamma * v1 + eta * 0.2 * x1
    v2 = gamma * v2 + eta * 4 * x2
    return x1 - v1, x2 - v2, v1, v2
eta, gamma = 0.4, 0.5
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
d2l.plt.show()

eta = 0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
d2l.plt.show()

# vt γvt−1 + ηtgt