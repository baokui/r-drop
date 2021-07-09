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
path_model = '/search/odin/guobk/data/my_rdrop_bert4/model_006.h5'
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
loss1 = K.categorical_crossentropy(
    y_true, similarities, from_logits=True
)
# K-L loss
# loss2 = kld(y_pred1[::2], y_pred1[1::2]) + kld(y_pred1[1::2], y_pred1[::2])
loss2_0 = kld(y_pred1[::4], y_pred1[2::4]) + kld(y_pred1[2::4], y_pred1[::4])
loss2_1 = kld(y_pred1[1::4], y_pred1[3::4]) + kld(y_pred1[3::4], y_pred1[1::4])
loss2 = loss2_0 + loss2_1
loss = loss1 + K.mean(loss2) / 4 * alpha
input_ids, input_segment = model.input

iter = train_generator.forfit()
x,y = next(iter)

sess = tf.keras.backend.get_session()
#init_op = tf.global_variables_initializer()
#sess.run(init_op)
# model.load_weights('/search/odin/guobk/data/my_rdrop_bert4/model_006.h5')
feed_dict = {input_ids:x[0],input_segment:x[1]}
similarities_, loss1_,loss2_0_,loss2_1_,loss2_,loss_,y_,y1_ = sess.run([similarities,loss1,loss2_0,loss2_1,loss2,loss,y_pred,y_pred1],feed_dict=feed_dict)

import numpy as np
import scipy.stats
KL = scipy.stats.entropy(y1_[::4], y1_[2::4]) 