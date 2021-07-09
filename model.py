#! -*- coding:utf-8 -*-
# 通过R-Drop增强模型的泛化性能
# 数据集：TNEWS 短文本分类 (https://github.com/CLUEbenchmark/CLUE)
# 博客：https://kexue.fm/archives/8496

import numpy as np
from bert4keras.models import build_transformer_model
from keras.layers import Lambda, Dense
import os
import keras
maxlen = 128
batch_size = 32
dim = 312

def get_model(bert_model):
    # BERT base
    config_path = '/search/odin/guobk/data/model/{}/bert_config.json'.format(bert_model)
    checkpoint_path = '/search/odin/guobk/data/model/{}/bert_model.ckpt'.format(bert_model)
    dict_path = '/search/odin/guobk/data/model/{}/vocab.txt'.format(bert_model)
    # 加载预训练模型
    bert = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        dropout_rate=0.3,
        return_keras_model=False,
    )
    output = Lambda(lambda x: x[:, 0])(bert.model.output)
    output = keras.layers.Dense(dim,activation='tanh')(output)
    model = keras.models.Model(bert.model.input, output)
    return model
