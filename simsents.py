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
maxlen = 128
batch_size = 32

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
        batch_token_ids, batch_segment_ids = [], []
        for is_end, d in self.sample(random):
            text, synonyms = d['text'], d['synonyms']
            synonyms = [text] + synonyms
            np.random.shuffle(synonyms)
            text, synonym = synonyms[:2]
            text, synonym = truncate(text), truncate(synonym)
            self.some_samples.append(text)
            if len(self.some_samples) > 1000:
                self.some_samples.pop(0)
            token_ids, segment_ids = tokenizer.encode(
                text, synonym, maxlen=maxlen * 2
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            token_ids, segment_ids = tokenizer.encode(
                synonym, text, maxlen=maxlen * 2
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []
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
output = Dense(
    units=num_classes,
    activation='softmax',
    kernel_initializer=bert.initializer
)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()


def crossentropy_with_rdrop(y_true, y_pred, alpha=4):
    """配合R-Drop的交叉熵损失
    """
    # 相似性loss
    idxs = K.arange(0, K.shape(y_pred)[0])
    idxs_1 = idxs[None, :]
    idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
    labels = K.equal(idxs_1, idxs_2)
    y_true = K.cast(labels, K.floatx())
    y_pred = K.l2_normalize(y_pred, axis=1)  # 句向量归一化
    similarities = K.dot(y_pred, K.transpose(y_pred))  # 相似度矩阵
    similarities = similarities - K.eye(K.shape(y_pred)[0]) * 1e12  # 排除对角线
    similarities = similarities * 30  # scale
    loss1 = K.categorical_crossentropy(
        y_true, similarities, from_logits=True
    )
    # K-L loss
    loss2 = kld(y_pred[::2], y_pred[1::2]) + kld(y_pred[1::2], y_pred[::2])
    return loss1 + K.mean(loss2) / 4 * alpha


model.compile(
    loss=crossentropy_with_rdrop,
    optimizer=Adam(2e-5),
    metrics=['sparse_categorical_accuracy'],
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
