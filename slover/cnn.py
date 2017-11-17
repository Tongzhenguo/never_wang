from __future__ import print_function

from data_processing.dataprocessing import _build_vocabulary

'''
python 3.5
keras 2
'''
import codecs
import json
from collections import defaultdict

import keras
import numpy as np
import pandas as pd
from gensim import corpora
from keras import Model
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Input

from slover.fasttext import train_gen, test_gen

np.random.seed(8888)
from keras.models import Sequential, load_model
from keras.layers import BatchNormalization
from keras.layers import Dense, Dropout, Activation, Conv1D
from keras.layers import Embedding
from keras.layers import GlobalMaxPooling1D


def cnn_model():
    ## keras functional api,the way to go for defining complex models
    print('Build model...')
    ## define input vector
    train_x = Input(shape=(maxlen,), dtype='int32')
    ## define embedding layer
    embed_x = Embedding(input_dim=max_features,output_dim=embedding_dims, input_length=maxlen)(train_x)
    ## define convolution layer,
    convolutions = []
    i = 0
    for n_gram in [1,2,3,4,5]:
        i += 1
        feature_vector = Conv1D(
            name="conv_" + str(n_gram) + '_' + str(i),
            filters=nb_filter,
            kernel_size=n_gram,
            activation=None
        )(embed_x)
        # add bn
        feature_vector = BatchNormalization()(feature_vector) ## scale to unit vector
        feature_vector = Activation('relu')(feature_vector)
        feature_vector = Dropout(0.2)(feature_vector)
        if n_gram != 1:
            i += 1
            feature_vector = Conv1D(
                name="conv_" + str(n_gram) + '_' + str(i),
                filters=nb_filter,
                kernel_size=n_gram,
                activation='relu'
            )(feature_vector)
        # i += 1
        # feature_vector = Conv1D(x
        #     name="conv_" + str(n_gram) + '_' + str(i),
        #     filters=nb_filter,
        #     kernel_size=n_gram,
        #     activation=None
        # )(feature_vector)
        # # add bn
        # feature_vector = BatchNormalization()(feature_vector)  ## scale to unit vector
        # feature_vector = Activation('relu')(feature_vector)
        # pooling layer
        one_max = GlobalMaxPooling1D(name='one_max_pooling_%s' %n_gram)(feature_vector)
        convolutions.append(one_max)
    ## concat all conv output vector into a long vector
    sentence_vector = keras.layers.concatenate(convolutions)  # hang on to this layer!
    # add first hidden layer
    sentence_vector = BatchNormalization()(sentence_vector)  ## scale to unit vector and
    sentence_vector = Dense(hidden_dims, activation='relu')(sentence_vector)

    # sentence_vector = Dropout(0.3)(sentence_vector)
    # add second hidden layer
    sentence_vector = BatchNormalization()(sentence_vector)  ## scale to unit vector
    sentence_vector = Dense(32, activation='relu')(sentence_vector)

    # sentence_vector = Dropout(0.3)(sentence_vector)
    ## define output vector
    pred = Dense(8, activation='softmax')(sentence_vector)

    model = Model(input=train_x, output=pred)
    model.summary()

    model.compile(optimizer=optimizers.Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def sub_cnn(model_path ='../model/cnn.h5', res_name='../res/cnn.txt'):
    #The predict_classes method is only available for the Sequential class
    clf = load_model(model_path)
    id_list_te = pd.read_pickle('../cache/id_list_te.pkl')
    y_pred = []
    for x_batch in test_gen(batch_size=batch_size,maxlen=maxlen):
        y_batch = clf.predict(x_batch,batch_size=batch_size)
        for pred_prob in y_batch:## get label
            y_pred.append( np.argmax(pred_prob)+1 ) #remapping to 1-8
    with codecs.open(res_name,encoding='utf-8',mode='w') as f:
        for i in range(len(id_list_te)):
            id = id_list_te[i]
            penalty = y_pred[i]
            data = json.dumps({'id': str(id), 'penalty': int(penalty), "laws": [1, 2, 3, 4]})
            f.write(data+'\n')
        f.flush()

def simple_cnn():
    # 构建模型
    print('Build model...')
    model = Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    # 先从一个高效的嵌入层开始，它将词汇的索引值映射为 embedding_dims 维度的词向量
    model.add(Embedding(max_features,
                        embedding_dims,
                        input_length=maxlen))
    # we add a Convolution1D, which will learn nb_filter
    # word group filters of size filter_length:
    # 添加一个 1D 卷积层，它将学习 nb_filter 个 filter_length 大小的词组卷积核
    model.add(Conv1D(nb_filter=nb_filter,
                     filter_length=filter_length,
                     border_mode='valid',
                     activation='relu',
                     init='glorot_uniform',
                     subsample_length=1))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))  # Dropout层
    # we use max pooling:
    # 使用最大池化
    model.add(GlobalMaxPooling1D())

    # We add a vanilla hidden layer:
    # 添加一个原始隐藏层
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))  ###

    # We project onto a single unit output layer, and squash it with a sigmoid:
    # 投影到一个单神经元的输出层，并且使用 sigmoid 压缩它
    model.add(Dense(8))
    model.add(Activation('softmax'))

    model.summary()  # 模型概述

    # 定义损失函数，优化器，评估矩阵
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=0.0005),
                  metrics=['accuracy'])

if __name__ == '__main__':
    dictionary_path = '../data/vocabulary_all.dict'
    _build_vocabulary(dictionary_path,ngram=None,filter=False)
    dictionary = corpora.Dictionary().load(dictionary_path)
    dictionary.compactify()

    # Set parameters: 设定参数
    max_features = len(dictionary.token2id)  # 词汇表大小
    maxlen = 10000  # 序列最大长度
    batch_size = 5  # 批数据量大小
    embedding_dims = 300  # 词向量维度
    nb_epoch = 10  # 迭代轮次
    nb_filter = 200  # 1维卷积核个数
    filter_length = 2  # 卷积核长度
    hidden_dims = 128  # 隐藏层维度
    ys = pd.read_pickle('../cache/penalty_list.pkl')
    class_weight = defaultdict(int)
    for y in ys:
        class_weight[int(y)-1] = class_weight[int(y)-1]+1
    class_weight = {cls:400000.0/cnt for cls,cnt in class_weight.items()  }

    model = cnn_model()

    # 训练与验证
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, min_delta=0.005, mode='min')  # loss最少下降0.0005才算一次提升
    model_checkpoint = ModelCheckpoint('../model/cnn.h5', save_best_only=False, save_weights_only=True, mode='min')
    model.fit_generator(train_gen(batch_size,maxlen=maxlen,drop=0)
                        ,steps_per_epoch=int(40000/batch_size)+1
                        ,class_weight=class_weight
                        ,epochs=nb_epoch
                        ,callbacks=[early_stopping,model_checkpoint]
                        ,validation_data=train_gen(batch_size,maxlen=maxlen,drop=0)
                        ,validation_steps=1
                        )

    model.save('../model/cnn.h5')

    sub_cnn(model_path='../model/cnn.h5', res_name='../res/cnn.txt')