#! -*- coding:utf-8 -*-
# 通过R-Drop增强模型的泛化性能
# 数据集：TNEWS 短文本分类 (https://github.com/CLUEbenchmark/CLUE)
# 博客：https://kexue.fm/archives/8496

import json
import numpy as np
from bert4keras.backend import keras, search_layer, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.layers import Lambda, Dense
from keras.losses import kullback_leibler_divergence as kld
from tqdm import tqdm
from modules import truncate
import os
maxlen = 128
batch_size = 32
dim = 312
alpha = 4

# BERT base
config_path = '/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/bert_config.json'
checkpoint_path = '/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/bert_model.ckpt'
dict_path = '/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/vocab.txt'

path_train = '/search/odin/guobk/data/simcse/20210621/train_d_drop.txt'
path_dev = '/search/odin/guobk/data/simcse/20210621/dev_d_drop.txt'
path_model = '/search/odin/guobk/data/my_rdrop_bert4'

def load_data(filename):
    D = []
    with open(filename) as f:
        for i, l in enumerate(f):
            l = json.loads(l)
            #text, syns = l['text'], l['synonyms']
            D.append(l)
    return D


# 加载数据集
train_data = load_data(path_train)
valid_data = load_data(path_dev)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

class data_generator(DataGenerator):
    """数据生成器
    """
    def __init__(self, *args, **kwargs):
        super(data_generator, self).__init__(*args, **kwargs)
        self.some_samples = []
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, d in self.sample(random):
            text, synonyms = d['text'], d['synonyms']
            synonyms = [text] + synonyms
            np.random.shuffle(synonyms)
            text, synonym = synonyms[:2]
            text, synonym = truncate(text,maxlen), truncate(synonym,maxlen)
            self.some_samples.append(text)
            if len(self.some_samples) > 1000:
                self.some_samples.pop(0)
            for ii in range(2):
                token_ids, segment_ids = tokenizer.encode(
                    text, synonym, maxlen=maxlen * 2
                )
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append(0)
                token_ids, segment_ids = tokenizer.encode(
                    synonym, text, maxlen=maxlen * 2
                )
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append(0)
            if len(batch_token_ids) == self.batch_size*2 or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []
# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)

# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    dropout_rate=0.3,
    return_keras_model=False,
)

output = Lambda(lambda x: x[:, 0])(bert.model.output)
# output = Dense(
#     units=dim,
#     activation='tanh',
#     kernel_initializer=bert.initializer
# )(output)
output = keras.layers.Dense(dim,activation='tanh')(output)
model = keras.models.Model(bert.model.input, output)
model.summary()

def simcse_loss(y_true, y_pred):
    """用于SimCSE训练的loss
    """
    # 构造标签
    idxs = K.arange(0, K.shape(y_pred)[0]/2)
    idxs_1 = idxs[None, :]
    idxs_2 = idxs[:, None]
    y_true = K.equal(idxs_1, idxs_2)
    y_true = K.cast(y_true, K.floatx())
    # 计算相似度
    outputA = Lambda(lambda x: x[:,:dim])(y_pred)
    outputB = Lambda(lambda x: x[:,dim:])(y_pred)
    outputA = outputA[::2] #取偶数行，即取A句的featureA
    outputB = outputB[1::2] #取奇数行，即取B句的featureB
    outputA = K.l2_normalize(outputA, axis=1)
    outputB = K.l2_normalize(outputB, axis=1)
    similarities = K.dot(outputA, K.transpose(outputB))
    similarities = similarities * 2
    loss = K.categorical_crossentropy(y_true, similarities, from_logits=True)
    return K.mean(loss)
def crossentropy_with_rdrop(y_true, y_pred):
    """配合R-Drop的交叉熵损失
    https://spaces.ac.cn/archives/8496
    """
    # 相似性loss
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
    y0_1 = y_pred[0::4] # 第1次encoding对query的emb
    y1_1 = y_pred[1::4] # 第1次encoding对doc的emb
    y0_2 = y_pred[2::4] # 第2次encoding对query的emb
    y1_2 = y_pred[3::4] # 第2次encoding对doc的emb
    # 计算第1次emb的相似性矩阵
    similarities1 = K.dot(y0_1, K.transpose(y1_1))  # 相似度矩阵
    similarities1 = similarities1 - K.eye(K.shape(y0_1)[0]) * 1e12  # 排除对角线
    similarities1 = similarities1 * 30  # scale
    # 计算第2次emb的相似性矩阵
    similarities2 = K.dot(y0_2, K.transpose(y1_2))  # 相似度矩阵
    similarities2 = similarities2 - K.eye(K.shape(y0_2)[0]) * 1e12  # 排除对角线
    similarities2 = similarities2 * 30  # scale
    loss2_1 = K.mean(K.categorical_crossentropy(
        similarities1, similarities2, from_logits=True
    ))
    loss2_2 = K.mean(K.categorical_crossentropy(
        similarities2, similarities1, from_logits=True
    ))
    loss = loss1 + (loss2_1+loss2_2)/4*alpha
    return loss

loss = crossentropy_with_rdrop(None, model.output)
model.compile(
    loss=crossentropy_with_rdrop,
    optimizer=Adam(2e-5)
)


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('best_model.weights')
        print(
            u'val_acc: %.5f, best_val_acc: %.5f\n' %
            (val_acc, self.best_val_acc)
        )


def predict_to_file(in_file, out_file):
    """输出预测结果到文件
    结果文件可以提交到 https://www.cluebenchmarks.com 评测。
    """
    fw = open(out_file, 'w')
    with open(in_file) as fr:
        for l in tqdm(fr):
            l = json.loads(l)
            text = l['sentence']
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            label = model.predict([[token_ids], [segment_ids]])[0].argmax()
            l = json.dumps({'id': str(l['id']), 'label': labels[label]})
            fw.write(l + '\n')
    fw.close()


if __name__ == '__main__':

    evaluator = Evaluator()
    checkpointer = keras.callbacks.ModelCheckpoint(os.path.join(path_model, 'model_{epoch:03d}.h5'),
                                   verbose=1, save_weights_only=True, period=1)
    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=50,
        callbacks=[checkpointer]
    )

else:

    model.load_weights('best_model.weights')
    # predict_to_file('/root/CLUE-master/baselines/CLUEdataset/tnews/test.json', 'tnews_predict.json')
