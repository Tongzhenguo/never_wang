from __future__ import print_function

import codecs
import json
import random
from collections import defaultdict

import numpy as np
from gensim import corpora
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from data_processing.dataprocessing import token_extract, batch_size, _build_vocabulary

np.random.seed(1337)  # for reproducibility
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D


def train_gen( batch_size=32,maxlen=200,drop=0 ):
    '''
    训练集（及验证集）批生成器
    :param batch_size: int,批次大小
    :param maxlen: int,最大填充序列，如果大于maxlen截断，小于则向后填充0
    :param drop: int,随机drop掉n个tokenid,如果是0则不删除
    :return: x_batch,y_batch
    '''
    doc_path = '../cache/doc_list.pkl'
    doc_list = pd.read_pickle(doc_path)
    num_doc = len(doc_list)
    ys = pd.read_pickle('../cache/penalty_list.pkl')
    dictionary_path = '../data/vocabulary_all.dict'
    dictionary = corpora.Dictionary.load(dictionary_path)
    dictionary.compactify()

    while True:
        x_batch,y_batch = [],[]
        while( len(x_batch)<batch_size ):
            randi = random.randint(0,num_doc-1)
            line = doc_list[randi]
            y = int(ys[randi])-1
            ids = [dictionary.token2id[token] for token in token_extract(line) if token in dictionary.token2id]

            x_batch.append( ids )
            y_batch.append( y )

            if drop>0:
                randi = random.randint(0, len(x_batch) - 1)
                ids = x_batch[randi]
                drop_list = []
                for _ in range(drop):
                    dropi = random.randint(0, len(ids) - 1)
                    drop_list.append( dropi )
                x_batch.append( [0 if id in drop_list else id for id in ids ] )
                y_batch.append(y)

        x_batch = sequence.pad_sequences(x_batch, maxlen=maxlen, padding='post',value=0)
        yield x_batch,y_batch

def test_gen( batch_size=32,maxlen=200):
    '''
    测试集批生成器
    :param batch_size: int,批次大小
    :param maxlen: int,最大填充序列，如果大于maxlen截断，小于则向后填充0
    :return: x_batch
    '''
    doc_path = '../cache/doc_list_te.pkl'
    doc_list = pd.read_pickle(doc_path)

    dictionary_path = '../data/vocabulary_all.dict'
    dictionary = corpora.Dictionary.load(dictionary_path)
    dictionary.compactify()

    x_batch = []
    for i in range(30000):
        line = doc_list[i]
        ids = [dictionary.token2id[token] for token in token_extract(line) if token in dictionary.token2id]
        x_batch.append( ids )

        if (i+1)%batch_size==0 or i==30000-1:
            X_batch = sequence.pad_sequences(x_batch, maxlen=maxlen, padding='post',value=0)
            x_batch = []
            yield X_batch

def sub_fasttext(model_path = '../model/fasttext.h5',res_name='../res/fasttext.txt'):

    clf = load_model(model_path)
    id_list_te = pd.read_pickle('../cache/id_list_te.pkl')
    y_pred = []
    for x_batch in test_gen(maxlen=maxlen):
        y_batch = clf.predict_classes(x_batch,batch_size=batch_size)
        for y in y_batch:
            y_pred.append( y )
    with codecs.open(res_name,encoding='utf-8',mode='w') as f:
        for i in range(len(id_list_te)):
            id = id_list_te[i]
            penalty = y_pred[i]
            data = json.dumps({'id': str(id), 'penalty': int(penalty), "laws": [1, 2, 3, 4]})
            f.write(data+'\n')
        f.flush()

if __name__ == "__main__":

    dictionary_path = '../data/vocabulary_all.dict'
    _build_vocabulary(dictionary_path)
    dictionary = corpora.Dictionary().load(dictionary_path)
    dictionary.compactify()

    # Set parameters: 设定参数
    max_features = len(dictionary.token2id)  # 词汇表大小
    maxlen = 10000  # 序列最大长度
    batch_size = 32  # 批数据量大小
    embedding_dims = 300  # 词向量维度
    nb_epoch = 30  # 迭代轮次
    ys = pd.read_pickle('../cache/penalty_list.pkl')
    class_weight = defaultdict(int)
    for y in ys:
        class_weight[y] = class_weight[y]+1
    class_weight = {cls:40000.0/cnt for cls,cnt in class_weight.items()  }

    # 构建模型
    print('Build model...')
    model = Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    # 先从一个高效的嵌入层开始，它将词汇表索引映射到 embedding_dim 维度的向量上
    model.add(Embedding(max_features,
                        embedding_dims,
                        input_length=maxlen))

    # we add a GlobalAveragePooling1D, which will average the embeddings
    # of all words in the document
    # 添加一个 GlobalAveragePooling1D 层，它将平均整个序列的词嵌入
    model.add(GlobalAveragePooling1D())

    model.add(Dense(16, activation='softmax'))

    model.summary()  # 概述

    model.compile(optimizer=optimizers.Adam(lr=0.0005)
                  ,loss='sparse_categorical_crossentropy'
                  ,metrics=['accuracy']
                  )

    # early_stopping = EarlyStopping(monitor='val_loss', patience=100, min_delta=0.0005, mode='min')  # loss最少下降0.0005才算一次提升
    # model_checkpoint = ModelCheckpoint('../model/fasttext_all_vocab_ngram_drop.h5', save_best_only=False, save_weights_only=True, mode='min')
    # 训练与验证
    model.fit_generator(train_gen(batch_size,maxlen=maxlen,drop=20)
                        ,samples_per_epoch=40000
                        ,nb_val_samples = batch_size
                        ,class_weight=class_weight
                        ,nb_epoch=nb_epoch
                        ,validation_data=train_gen(batch_size,maxlen=maxlen,drop=20)
                        ,nb_worker=8
                        # ,callbacks=[early_stopping,model_checkpoint]
                        )

    model.save('../model/fasttext_all_vocab_ngram_drop.h5')

    sub_fasttext(model_path='../model/fasttext_all_vocab_ngram_drop.h5')