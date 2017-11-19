from __future__ import print_function

import random

import gc
import jieba
import re
from keras.preprocessing import sequence

'''
python 3.5
keras 2
'''
import codecs
import json
from collections import defaultdict
import os
import keras
import numpy as np
import pandas as pd
from gensim import corpora
from keras import Model
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Input

np.random.seed(8888)
from keras.models import load_model
from keras.layers import BatchNormalization
from keras.layers import Dense, Dropout, Activation, Conv1D
from keras.layers import Embedding
from keras.layers import GlobalMaxPooling1D

i = 0
def token_extract(text,ngram=[]):
    global i
    words = []
    ## manual control
    #1.match person name or person mention
    # 1.match person name or person mention
    re_person = re.compile(u'被告人([^，。、]+)')  # eg.被告人杨某某
    re_person2 = re.compile(u'([\u4E00-\u9FD5]某{1,2})')  # eg.杨某某
    for name in re_person.findall(text):
        jieba.add_word(name)
        jieba.suggest_freq(name)
    for name in re_person2.findall(text):
        jieba.add_word(name)
        jieba.suggest_freq(name)
    word_list = list(jieba.cut(text, HMM=False))
    for i,word in enumerate(word_list):
        for n in ngram :
            if i+n<len(word_list):
                words.append(''.join(word_list[i:i+n]))
        words.append(word)
    i += 1
    if i%1000 ==0:print(words[:30])
    return words

def _build_vocabulary(dictionary_path ='../data/vocabulary.dict',ngram=[2,3]):
    '''
    词表是一个很重要的影响因素，不过滤构造的词矩阵会OOM
    '''
    ## add law vocabulary
    id2laws = pd.read_pickle('../cache/law_vocab.pkl')
    for id,laws in id2laws.items():
        for law in laws:
            jieba.add_word( law )
            jieba.suggest_freq( law )
    with codecs.open('../data/form-laws.txt', encoding='utf-8') as f:
        ls = f.readlines()
        for i, line in enumerate(ls):
            for law in re.findall('【(.*?)】',line):
                for word in law.split(';'):
                    jieba.add_word(word)
                    jieba.suggest_freq(word)

    if os.path.exists(dictionary_path):
        dictionary = corpora.Dictionary().load(dictionary_path)
    else:
        doc_list = pd.read_pickle('../cache/doc_list.pkl')
        doc_list_te = pd.read_pickle('../cache/doc_list_te.pkl')
        doc_list.extend(doc_list_te)

        with codecs.open('../data/form-laws.txt', encoding='utf-8') as f:
            ls = f.readlines()
        doc_list.extend(ls)

        cor = [token_extract(line,ngram=[]) for line in doc_list]
        dictionary = corpora.Dictionary(cor)

        if ngram:
            cor = [token_extract(line,ngram=[2,3]) for line in doc_list]
            dictionary2 = corpora.Dictionary(cor)
            once_ids = [tokenid for tokenid, docfreq in dictionary2.dfs.items() if docfreq < 100]
            dictionary2.filter_tokens( once_ids )
            dictionary2.compactify()

            print('len dictionary = %s' % len(dictionary))  # len dictionary = 125156
            dict2_to_dict1 = dictionary.merge_with(dictionary2)
        print('len dictionary = %s' % len(dictionary)) #len dictionary = 125156
        dictionary.save(dictionary_path)
        del doc_list,doc_list_te
        gc.collect()
    return dictionary

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
            ids = [dictionary.token2id[token] for token in token_extract(line,ngram=[2,3]) if token in dictionary.token2id]

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
        ids = [dictionary.token2id[token] for token in token_extract(line,ngram=[2,3]) if token in dictionary.token2id]
        x_batch.append( ids )

        if (i+1)%batch_size==0 or i==30000-1:
            X_batch = sequence.pad_sequences(x_batch, maxlen=maxlen, padding='post',value=0)
            x_batch = []
            yield X_batch


def cnn_model():
    ## keras functional api,the way to go for defining complex models
    print('Build model...')
    ## define input vector
    train_x = Input(shape=(maxlen,), dtype='int32')
    ## define embedding layer
    embed_x = Embedding(input_dim=max_features,output_dim=embedding_dims, input_length=maxlen)(train_x)
    ## Data Augmentation
    embed_x = Dropout(0.05)(embed_x)
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
        if n_gram != 1:
            i += 1
            feature_vector = Conv1D(
                name="conv_" + str(n_gram) + '_' + str(i),
                filters=nb_filter,
                kernel_size=n_gram,
                activation='relu'
            )(feature_vector)
            # add bn
            feature_vector = BatchNormalization()(feature_vector)  ## scale to unit vector
            feature_vector = Activation('relu')(feature_vector)
        # pooling layer
        one_max = GlobalMaxPooling1D(name='one_max_pooling_%s' %n_gram)(feature_vector)
        convolutions.append(one_max)
    ## concat all conv output vector into a long vector
    sentence_vector = keras.layers.concatenate(convolutions)  # hang on to this layer!
    # add first hidden layer
    sentence_vector = BatchNormalization()(sentence_vector)  ## scale to unit vector and
    sentence_vector = Dense(hidden_dims, activation='relu')(sentence_vector)
    sentence_vector = Dropout(0.35)(sentence_vector)
    # add second hidden layer
    sentence_vector = BatchNormalization()(sentence_vector)  ## scale to unit vector
    sentence_vector = Dense(32, activation='relu')(sentence_vector)
    sentence_vector = Dropout(0.35)(sentence_vector)

    ## define output vector
    pred = Dense(8, activation='softmax')(sentence_vector)

    model = Model(input=train_x, output=pred)
    model.summary()
    model.compile(optimizer=optimizers.Adam(lr=0.01,beta_1=0.95), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

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

if __name__ == '__main__':
    dictionary_path = '../data/vocabulary_all.dict'
    _build_vocabulary(dictionary_path,ngram=None)
    dictionary = corpora.Dictionary().load(dictionary_path)
    dictionary.compactify()

    # Set parameters: 设定参数
    max_features = len(dictionary.token2id)  # 词汇表大小
    maxlen = 10000  # 序列最大长度
    batch_size = 32  # 批数据量大小
    embedding_dims = 300  # 词向量维度
    nb_epoch = 12  # 迭代轮次
    nb_filter = 200  # 1维卷积核个数
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

    model.save('../model/cnn_dp.h5')

    sub_cnn(model_path='../model/cnn_dp.h5', res_name='../res/cnn.txt')