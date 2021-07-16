from model import get_model
import json
import numpy as np
from bert4keras.snippets import sequence_padding
import keras
import sys
from bert4keras.tokenizers import Tokenizer
def write_excel(path_target,data,sheetname='Sheet1'):
    import xlwt
    # 创建一个workbook 设置编码
    workbook = xlwt.Workbook(encoding='utf-8')
    # 创建一个worksheet
    worksheet = workbook.add_sheet(sheetname)
    # 写入excel
    # 参数对应 行, 列, 值
    rows,cols = len(data),len(data[0])
    for i in range(rows):
        for j in range(cols):
            #worksheet.write(i, j, label=str(data[i][j]))
            worksheet.write(i, j, label=data[i][j])
    # 保存
    workbook.save(path_target)
bert_model = 'chinese_simbert_L-4_H-312_A-12'
path_model = '/search/odin/guobk/data/my_rdrop_bert4/model_001.h5'
path_docs = "/search/odin/guobk/data/bert_semantic/finetuneData_new_test/Docs.json"
path_queries = "/search/odin/guobk/data/Tab3_test/Q-20210629.json"
path_target = "/search/odin/guobk/data/Tab3_test/Q-20210629-rdrop.json"
dict_path = '/search/odin/guobk/data/model/{}/vocab.txt'.format(bert_model)
tag = 'rec_rdrop'
maxQ = 100
path_model,bert_model,tag,path_docs,path_queries,maxQ,path_target = sys.argv[1:]
model = get_model(bert_model)
model.load_weights(path_model)
tokenizer = Tokenizer(dict_path, do_lower_case=True)
with open(path_docs,'r') as f:
    D = json.load(f)
with open(path_queries,'r') as f:
    Q = json.load(f)
def emb(encoder,Sents, batch_size = 128):
    V = []
    X, S = [], []
    for t in Sents:
        x, s = tokenizer.encode(t)
        X.append(x)
        S.append(s)
    X = sequence_padding(X)
    S = sequence_padding(S)
    Z = encoder.predict([X, S],verbose=True)
    Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
    return Z
maxQ = int(maxQ)
Queries = Q[:maxQ]
maxRec = 10
SentsQ = [d['input'] for d in Queries]
SentsD = [d['content'] for d in D]
V_q = emb(model,SentsQ)
V_d = emb(model,SentsD)
s = V_q.dot(np.transpose(V_d))
idx = np.argsort(-s,axis=-1)
for j in range(len(Queries)):
    score = [s[j][ii] for ii in idx[j][:maxRec]]
    contents = [SentsD[ii] for ii in idx[j][:maxRec]]
    Queries[j][tag] = [contents[k]+'\t%0.4f'%score[k] for k in range(len(score))]
with open(path_target,'w') as f:
    json.dump(Queries,f,ensure_ascii=False,indent=4)

def test0():
    import json
    with open('/search/odin/guobk/data/Tab3_test/Q-20210629-tmp.json','r') as f:
        D = json.load(f)
    R = [['index','query','model-base (simcse-bert-12layer)','score-base','accuracy-base','model-simbert (4layer)','score-simbert','accuracy-simbert']]
    k0 = 'rec_bert_cls_base'
    k1 = 'rec_simbert_l4'
    ii = 0
    maxRec = 10
    for d in D:
        r0 = d[k0][:maxRec]
        r1 = d[k1][:maxRec]
        r0 = r0 + ['']*(maxRec-len(r0))
        r1 = r1 + ['']*(maxRec-len(r1))
        r = [[ii,d['input']]+r0[0].split('\t') + [''] + r1[0].split('\t') + ['']]
        for i in range(1,maxRec):
            if r0[i]!='':
                r00 = r0[i].split('\t') + ['']
            else:
                r00 = ['', '', '']
            if r1[i]!='':
                r11 = r1[i].split('\t') + ['']
            else:
                r11 = ['','','']
            r.append(['','']+r00+r11)
        ii += 1
        R.extend(r)
    #####################################
    for i in range(len(R)):
        if R[i][2] and R[i][2][0]=='*':
            R[i][4] = 1
            R[i][2] = R[i][2][1:]
        if R[i][5] and R[i][5][0]=='*':
            R[i][7] = 1
            R[i][5] = R[i][5][1:]
    #####################################
    write_excel('/search/odin/guobk/data/Tab3_test/Q-20210629-tmp.xls',R)





###################################
import tensorflow as tf
from bert4keras.backend import keras, search_layer, K
from keras.losses import kullback_leibler_divergence as kld
tf.reset_default_graph()
bert_model = 'chinese_simbert_L-4_H-312_A-12'
path_model = '/search/odin/guobk/data/my_rdrop_bert4/model_006.h5'
model = get_model(bert_model)
model.load_weights(path_model)
dict_path = '/search/odin/guobk/data/model/{}/vocab.txt'.format(bert_model)
tokenizer = Tokenizer(dict_path, do_lower_case=True)

batch_size = 32
y_pred = model.output
idxs = K.arange(0, K.shape(y_pred)[0])
idxs_1 = idxs[None, :]
idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
labels = K.equal(idxs_1, idxs_2)
y_true = K.cast(labels, K.floatx())
y_pred1 = K.l2_normalize(y_pred, axis=1)  # 句向量归一化
similarities = K.dot(y_pred1, K.transpose(y_pred1))  # 相似度矩阵
similarities = similarities - K.eye(K.shape(y_pred1)[0]) * 1e12  # 排除对角线
similarities = similarities * 30  # scale
loss1 = K.mean(K.categorical_crossentropy(
    y_true, similarities, from_logits=True
))
# K-L loss
# loss2 = kld(y_pred1[::2], y_pred1[1::2]) + kld(y_pred1[1::2], y_pred1[::2])
# loss2_0 = kld(y_pred1[::4], y_pred1[2::4]) + kld(y_pred1[2::4], y_pred1[::4])
# loss2_1 = kld(y_pred1[1::4], y_pred1[3::4]) + kld(y_pred1[3::4], y_pred1[1::4])
# loss2 = loss2_0 + loss2_1
# loss = loss1 + K.mean(loss2) / 4 * alpha
# MSE
# loss2_0 = K.mean(K.square(y_pred1[::4]-y_pred1[2::4]))
# loss2_1 = K.mean(K.square(y_pred1[1::4]-y_pred1[3::4]))
# loss2 = loss2_0 + loss2_1
# loss = loss1/2 + loss2*0.3
# 对比loss
y1 = y_pred1[:batch_size] # 第1次model predict后的emb
y2 = y_pred1[batch_size:] # 第1次model predict后的emb
# 计算第1次emb的相似性矩阵
similarities1 = K.dot(y1, K.transpose(y1))  # 相似度矩阵
similarities1 = similarities1 - K.eye(K.shape(y1)[0]) * 1e12  # 排除对角线
similarities1 = similarities1 * 30  # scale
p_similarities1 = K.softmax(similarities1)
# 计算第2次emb的相似性矩阵
similarities2 = K.dot(y2, K.transpose(y2))  # 相似度矩阵
similarities2 = similarities2 - K.eye(K.shape(y2)[0]) * 1e12  # 排除对角线
similarities2 = similarities2 * 30  # scale
p_similarities2 = K.softmax(similarities2)
loss2_1 = K.mean(K.categorical_crossentropy(
    p_similarities1, similarities2, from_logits=True
))
loss2_2 = K.mean(K.categorical_crossentropy(
    p_similarities2, similarities1, from_logits=True
))
loss = loss1 + (loss2_1+loss2_2)/4*alpha
input_ids, input_segment = model.input

valid_generator = data_generator(valid_data, batch_size)
iter = valid_generator.forfit()
x,y = next(iter)

sess = tf.keras.backend.get_session()
#init_op = tf.global_variables_initializer()
#sess.run(init_op)
# model.load_weights('/search/odin/guobk/data/my_rdrop_bert4/model_006.h5')
feed_dict = {input_ids:x[0],input_segment:x[1]}
similarities_, loss1_,loss2_1_,loss2_2_,loss_,y_,y1_,p_similarities1_,p_similarities2_ = sess.run([similarities,loss1,loss2_1,loss2_2,loss,y_pred,y_pred1,p_similarities1,p_similarities2],feed_dict=feed_dict)

import numpy as np
import scipy.stats
KL = scipy.stats.entropy(y1_[::4], y1_[2::4]) 


#############################################
path_model = '/search/odin/guobk/data/my_rdrop_bert4-test/model_001.h5'
import tensorflow as tf
input_ids, input_segment = model.input

y_pred = model.output

idxs = K.arange(0, K.shape(y_pred)[0])
idxs_1 = idxs[None, :]
idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
labels = K.equal(idxs_1, idxs_2)
y_true = K.cast(labels, K.floatx())
y_pred1 = K.l2_normalize(y_pred, axis=1)  # 句向量归一化
similarities = K.dot(y_pred1, K.transpose(y_pred1))  # 相似度矩阵
similarities = similarities - K.eye(K.shape(y_pred1)[0]) * 1e12  # 排除对角线
similarities = similarities * 30  # scale
loss1 = K.mean(K.categorical_crossentropy(
    y_true, similarities, from_logits=True
))
y1 = y_pred1[:batch_size] # 第1次model predict后的emb
y2 = y_pred1[batch_size:] # 第1次model predict后的emb
# 计算第1次emb的相似性矩阵
similarities1 = K.dot(y1, K.transpose(y1))  # 相似度矩阵
similarities1 = similarities1 - K.eye(K.shape(y1)[0]) * 1e12  # 排除对角线
similarities1 = similarities1 * 3  # scale
p_similarities1 = K.softmax(similarities1)
# 计算第2次emb的相似性矩阵
similarities2 = K.dot(y2, K.transpose(y2))  # 相似度矩阵
similarities2 = similarities2 - K.eye(K.shape(y2)[0]) * 1e12  # 排除对角线
similarities2 = similarities2 * 3  # scale
p_similarities2 = K.softmax(similarities2)
loss2_1 = K.mean(K.categorical_crossentropy(
    p_similarities1, similarities2, from_logits=True
))
loss2_2 = K.mean(K.categorical_crossentropy(
    p_similarities2, similarities1, from_logits=True
))
loss2 = loss2_1 + loss2_2
loss = loss1 + loss2/4*alpha


sess = tf.keras.backend.get_session()

y_, loss1_, loss2_, loss_ = sess.run([y_pred,loss1,loss2,loss],feed_dict=feed_dict)

y_pred1_, y_true_, similarities_, similarities1_, similarities2_ = sess.run([y_pred1,y_true,similarities,similarities1,similarities2],feed_dict=feed_dict)





train_generator = data_generator(train_data[:(batch_size*2)], batch_size)



model.load_weights(path_model)

input_ids, input_segment = model.input
y_pred = model.output

iter = train_generator.forfit()
x,y = next(iter)

feed_dict = {input_ids:x[0],input_segment:x[1]}



idxs = K.arange(0, K.shape(y_pred)[0])
idxs_1 = idxs[None, :]
idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
labels = K.equal(idxs_1, idxs_2)
y_true = K.cast(labels, K.floatx())
y_pred1 = K.l2_normalize(y_pred, axis=1)  # 句向量归一化
similarities = K.dot(y_pred1, K.transpose(y_pred1))  # 相似度矩阵
similarities = similarities - K.eye(K.shape(y_pred1)[0]) * 1e12  # 排除对角线
similarities = similarities * 30  # scale
loss1 = K.mean(K.categorical_crossentropy(
    y_true, similarities, from_logits=True
))

sess = tf.keras.backend.get_session()

y_, loss1_, y_pred1_,y_true_,similarities_ = sess.run([y_pred,loss1,y_pred,y_true,similarities],feed_dict=feed_dict)


def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum
    return softmax
