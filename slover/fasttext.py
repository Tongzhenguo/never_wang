'''
    使用fasttext同时训练罚金等级与涉及法律条款，并对每一个样本进行赋予如下权重：
        weight = 0.5*pe_label_weight+0.5*dot(law_array,weight_array),其中weight(label==k) = 1/COUNT(label==k)
'''
from __future__ import print_function
import codecs
import json
import random
from collections import defaultdict
import numpy as np
from data_processing.dataprocessing import token_extract, batch_size, _build_vocabulary
from gensim import corpora
from keras import Input
from keras import Model
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
np.random.seed(1337)  # for reproducibility
import pandas as pd
from keras.preprocessing import sequence
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D


def train_gen( batch_size=32,maxlen=200,drop=0 ):
    '''
    训练集（及验证集）批生成器
    :param batch_size: int,批次大小
    :param maxlen: int,最大填充序列，如果大于maxlen截断，小于则向后填充0
    :param drop: int,随机drop掉n个tokenid,如果是0则不删除
    :return: x_batch,y_batch,sample_weights
    '''
    doc_path = '../cache/doc_list.pkl'
    doc_list = pd.read_pickle(doc_path)
    num_doc = len(doc_list)
    ys = pd.read_pickle('../cache/penalty_list.pkl')
    laws = pd.read_pickle('../cache/laws_list.pkl')
    lawno2label = pd.read_pickle('../cache/lawno2label.pkl')
    dictionary_path = '../data/vocabulary_all.dict'
    dictionary = corpora.Dictionary.load(dictionary_path)
    dictionary.compactify()

    while True:
        x_batch = []
        pe_array = np.zeros(shape=(batch_size,8),dtype=int)
        law_array = np.zeros(shape=(batch_size, 316),dtype=int)
        k = 0
        while( len(x_batch)<batch_size ):
            randi = random.randint(0,num_doc-1)
            line = doc_list[randi]
            pe_array[k][int(ys[randi])-1] = 1
            for lno in laws[randi].split(','):
                law_array[k][lawno2label[int(lno)]] = 1
            ids = [dictionary.token2id[token] for token in token_extract(line) if token in dictionary.token2id]
            x_batch.append( ids )

            if drop>0:
                randi = random.randint(0, len(x_batch) - 1)
                ids = x_batch[randi]
                drop_list = []
                for _ in range(drop):
                    dropi = random.randint(0, len(ids) - 1)
                    drop_list.append( dropi )
                x_batch.append( [0 if id in drop_list else id for id in ids ] )

        x_batch = sequence.pad_sequences(x_batch, maxlen=maxlen, padding='post',value=0)
        yield x_batch,[pe_array,law_array],

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
    for i in range(90000):
        line = doc_list[i]
        ids = [dictionary.token2id[token] for token in token_extract(line) if token in dictionary.token2id]
        x_batch.append( ids )

        if (i+1)%batch_size==0 or i==90000-1:
            X_batch = sequence.pad_sequences(x_batch, maxlen=maxlen, padding='post',value=0)
            x_batch = []
            yield X_batch

def make_nn_model():
    ## keras functional_API
    # 构建模型
    print('Build model...')
    ## define input vector
    train_x = Input(shape=(maxlen,), dtype='int32')
    ## define embedding layer
    embed_x = Embedding(input_dim=max_features, output_dim=embedding_dims, input_length=maxlen)(train_x)
    avg_x = GlobalAveragePooling1D()(embed_x)
    pred_pe = Dense(8, activation='softmax')(avg_x)
    pred_law = Dense(316, activation='softmax')(avg_x)
    model = Model(input=train_x, output=[pred_pe,pred_law])
    model.summary()  # 概述
    model.compile(optimizer=optimizers.Adam(lr=0.0005)
                  , loss='categorical_crossentropy'
                  , metrics=['accuracy']
                  ,sample_weight_mode={}
                  )
    return model
def sub_fasttext(model_path = '../model/fasttext_12w.h5',res_name='../res/fasttext.txt'):

    clf = load_model(model_path)
    id_list_te = pd.read_pickle('../cache/id_list_te.pkl')
    label2lawno = pd.read_pickle('../cache/label2lawno.pkl')
    penalty_pred = []
    laws_pred = []
    for x_batch in test_gen(maxlen=maxlen):
        y_batch = clf.predict(x_batch,batch_size=batch_size)
        for pe_prob,law_prob in y_batch:
            penalty_pred.append(np.argmax(pe_prob) + 1)  # remapping to 1-8
            laws_pred.append(label2lawno[np.argmax(law_prob)])  # remapping to 1-417

    with codecs.open(res_name,encoding='utf-8',mode='w') as f:
        for i in range(len(id_list_te)):
            id = id_list_te[i]
            penalty = penalty_pred[i]
            laws = list(laws_pred[i])
            data = json.dumps({'id': str(id), 'penalty': int(penalty), "laws": laws})
            f.write(data+'\n')
        f.flush()

if __name__ == "__main__":

    dictionary_path = '../data/vocabulary_all.dict'
    _build_vocabulary(dictionary_path,ngram=None,filter=False)
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
        class_weight[y-1] = class_weight[y-1]+1 #remapping 0 to 7
    class_weight = {cls:120000.0/cnt for cls,cnt in class_weight.items()  }
    model = make_nn_model()

    early_stopping = EarlyStopping(monitor='val_loss', patience=100, min_delta=0.0005, mode='min')  # loss最少下降0.0005才算一次提升
    model_checkpoint = ModelCheckpoint('../model/fasttext_12w.h5', save_best_only=True, save_weights_only=True, mode='min')
    # 训练与验证
    model.fit_generator(train_gen(batch_size,maxlen=maxlen,drop=2)
                        ,steps_per_epoch=120000/batch_size
                        ,nb_val_samples = batch_size
                        ,class_weight=None
                        ,nb_epoch=nb_epoch
                        ,validation_data=train_gen(batch_size,maxlen=maxlen,drop=0)
                        ,nb_worker=1
                        ,callbacks=[early_stopping,model_checkpoint]
                        )

    model.save('../model/fasttext_12w.h5')

    sub_fasttext(model_path='../model/fasttext_12w.h5')