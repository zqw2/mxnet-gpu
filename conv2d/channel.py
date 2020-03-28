import d2lzh as d2l
from mxnet import nd

##############################
# 含通道的输入与含通道的卷积核运算
##############################
def corr2d_multi_in(X, K):
    # ⾸先沿着X和K的第0维（通道维）遍历。然后使⽤*将结果列表变成add_n函数的位置参数
    # （ positional argument）来进⾏相加
    return nd.add_n(*[d2l.corr2d(x, k) for x, k in zip(X, K)])

X = nd.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
K = nd.array([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])
print(corr2d_multi_in(X, K))

################################################
# 含通道的输入与含通道的卷积核运算，半且是多输出版本的
################################################
def corr2d_multi_in_out(X, K):
    # 对K的第0维遍历，每次同输⼊X做互相关计算。所有结果使⽤stack函数合并在⼀起
    return nd.stack(*[corr2d_multi_in(X, k) for k in K])

K = nd.stack(K, K + 1, K + 2)
print(corr2d_multi_in_out(X, K))