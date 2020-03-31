import d2lzh as d2l
from mxnet import nd
from mxnet.gluon import rnn
(corpus_indices, char_to_idx, idx_to_char,
vocab_size) = d2l.load_data_jay_lyrics()

num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
ctx = d2l.try_gpu()
def get_params():
    def _one(shape):
        return nd.random.normal(scale=0.01, shape=shape, ctx=ctx)
    def _three():
        return (_one((num_inputs, num_hiddens)),
    _one((num_hiddens, num_hiddens)),
    nd.zeros(num_hiddens, ctx=ctx))
    W_xi, W_hi, b_i = _three() # 输⼊⻔参数
    W_xf, W_hf, b_f = _three() # 遗忘⻔参数
    W_xo, W_ho, b_o = _three() # 输出⻔参数
    W_xc, W_hc, b_c = _three() # 候选记忆细胞参数
    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = nd.zeros(num_outputs, ctx=ctx)
    # 附上梯度
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
    b_c, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params

def init_lstm_state(batch_size, num_hiddens, ctx):
    return (nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx),
        nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx))

def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
    W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = nd.sigmoid(nd.dot(X, W_xi) + nd.dot(H, W_hi) + b_i)
        F = nd.sigmoid(nd.dot(X, W_xf) + nd.dot(H, W_hf) + b_f)
        O = nd.sigmoid(nd.dot(X, W_xo) + nd.dot(H, W_ho) + b_o)
        C_tilda = nd.tanh(nd.dot(X, W_xc) + nd.dot(H, W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * C.tanh()
        Y = nd.dot(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H, C)

num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']

# result = d2l.train_and_predict_rnn(lstm, get_params, init_lstm_state, num_hiddens,
#     vocab_size, ctx, corpus_indices, idx_to_char,
#     char_to_idx, False, num_epochs, num_steps, lr,
#     clipping_theta, batch_size, pred_period, pred_len,
#     prefixes)
# print(result)


######################
# 利用简捷实现
######################
lstm_layer = rnn.LSTM(num_hiddens)
model = d2l.RNNModel(lstm_layer, vocab_size)
d2l.train_and_predict_rnn_gluon(model, num_hiddens, vocab_size, ctx,
    corpus_indices, idx_to_char, char_to_idx,
    num_epochs, num_steps, lr, clipping_theta,
    batch_size, pred_period, pred_len, prefixes)