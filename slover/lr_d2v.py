'''train dbow/dm for education/age/gender'''
import codecs
import json
import os
from collections import namedtuple
from datetime import datetime

import jieba
import numpy as np
import pandas as pd
from gensim.models import Doc2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

tr_examples = 40000
te_examples = 30000

#-------------------------prepare to train--------------------------------------------
SentimentDocument = namedtuple('SentimentDocument', 'words tags')
class Doc_list(object):
    def __init__(self,pkl_path):
        self.f = pkl_path
    def __iter__(self):
        for i,line in enumerate(pd.read_pickle(self.f)[:tr_examples+te_examples]):
            words = list(jieba.cut(str(line),HMM=False))
            tags = [i]##id

            yield SentimentDocument( words,tags )


'''
-------------------train dbow doc2vec---------------------------------------------
输出分布式向量300维,
    window是预测词与上下文词的最远距离,       *****通过统计句子长度，分析给出参考值(4-9)
    alpha和min_alpha是学习速率,
    min_count最小词频，                      ***  >3
    sample是高频词的下采样比例，
    hs:hierarchical softmax loss
    negative,每一个样本使用5个噪声词作负样本  *
一万单词的字典大约占用1M,

默认值
dm=1
iter=5，推荐10-20
hs=1
dm_mean=0,sum(vector)
negative=5,推荐5-20
sample=1e-5
'''

def train_dbow_cv(dm=0, size=300, negative=5, hs=0, min_count=3, window=30,sample=1e-5,workers=8,alpha=0.025,iter=5):
    path = '../model/dbow_d2v_%s_%s.model' % (window,min_count)
    doc_list_path = '../cache/doc_list.pkl'
    penalty_list_path = '../cache/penalty_list.pkl'
    if os.path.exists( path ):
        d2v = Doc2Vec.load(path)
    else:
        d2v = Doc2Vec(dm=dm,size=size,negative=negative,hs=hs,min_count=min_count,window=window,sample=sample,workers=workers
                      ,alpha=alpha,iter=iter)
        doc_list = Doc_list(doc_list_path)
        d2v.build_vocab(doc_list)
        print('build_vocab done !')
        doc_list = Doc_list(doc_list_path)
        d2v.train(doc_list,total_examples=tr_examples)
        print('dbow train done !')
        d2v.save(path)
        print(datetime.now(), 'save done')
    print('----------------')
    X_d2v = np.array([d2v.docvecs[i] for i in range(tr_examples)])
    penalty_list = pd.read_pickle(penalty_list_path)
    scores = cross_val_score(LogisticRegression(C=3),X_d2v,penalty_list,cv=5)
    print('dbow',scores,np.mean(scores))
    return d2v

def train_dm_cv(dm=1, size=300, negative=5, hs=0, min_count=3, window=10,sample=1e-5,workers=8,alpha=0.05,iter=10):
    path = '../model/dm_d2v_%s_%s.model' % (window, min_count)
    doc_list_path = '../cache/doc_list.pkl'
    penalty_list_path = '../cache/penalty_list.pkl'

    if os.path.exists( path ):
        d2v = Doc2Vec.load(path)
    else:
        d2v = Doc2Vec(dm=dm,size=size,negative=negative,hs=hs,min_count=min_count,window=window,sample=sample,workers=workers
                          ,alpha=alpha,iter=iter)
        doc_list = Doc_list(doc_list_path)
        d2v.build_vocab(doc_list)
        print('build_vocab done !')
        doc_list = Doc_list(doc_list_path)
        d2v.train(doc_list,total_examples=tr_examples)
        print('dm train done !')
        d2v.save(path)
        print(datetime.now(), 'save done')
    X_d2v = np.array([d2v.docvecs[i] for i in range(tr_examples)])
    penalty_list = pd.read_pickle(penalty_list_path)
    scores = cross_val_score(LogisticRegression(C=3),X_d2v,penalty_list,cv=5)
    print('dm',scores,np.mean(scores))
    return d2v


# for w in [6, 9]:
#     train_dbow_cv(dm=0, size=300, negative=5, hs=0, min_count=3, window=w, sample=1e-5, workers=8, alpha=0.025)
#
#     # train_dm_cv(dm=1, size=300, negative=5, hs=0, min_count=c, window=w, sample=1e-5, workers=8, alpha=0.025)
'''
dbow [ 0.36661252  0.37228193  0.375625    0.36309077  0.35913468] 0.367348979604
dm [ 0.3240035   0.3324169   0.3385      0.32983246  0.33037389] 0.331025348557

dbow [ 0.36436336  0.36915771  0.369625    0.36371593  0.3560085 ] 0.364574101296

dbow [ 0.36248907  0.36903274  0.37075     0.36771693  0.35875953] 0.365749654494

dbow [ 0.36273897  0.36965759  0.369875    0.36684171  0.35938477] 0.365699607641
'''
def eval(w,c):
    train_dbow_cv(dm=0, size=300, negative=5, hs=0, min_count=c, window=w, sample=1e-5, workers=8, alpha=0.025)

def make_sub( res_name ):
    penalty_list_path = '../cache/penalty_list.pkl'
    d2v = Doc2Vec(dm=0, size=300, negative=5, hs=0, min_count=3, window=5, sample=1e-5, workers=8, alpha=0.025,iter=20)

    doc_list = list(pd.read_pickle('../cache/doc_list.pkl'))
    doc_list.extend(pd.read_pickle('../cache/doc_list_te.pkl'))
    pd.to_pickle(doc_list,'../cache/all_doc.pkl')
    doc_list = Doc_list('../cache/all_doc.pkl')
    d2v.build_vocab(doc_list)
    print('build_vocab done !')

    d2v.train(doc_list, total_examples=tr_examples+te_examples)
    print('dbow train done !')
    d2v.save('../model/dbow_d2v_alldoc_4_3.model')

    X_d2v = np.array([d2v.docvecs[i] for i in range(tr_examples)])
    pd.to_pickle(X_d2v,'../cache/tr_x_d2v_300.pkl')

    penalty_list = pd.read_pickle(penalty_list_path)[:tr_examples]
    lr = LogisticRegression(C=3)
    lr.fit( X_d2v,penalty_list )
    print('lr train done !')

    id_list_te = pd.read_pickle('../cache/id_list_te.pkl')
    X_d2v = np.array([d2v.docvecs[tr_examples+i] for i in range(te_examples)])
    pd.to_pickle(X_d2v,'../cache/te_x_d2v_300.pkl')
    y_pred = lr.predict(X_d2v)

    with codecs.open(res_name,encoding='utf-8',mode='w') as f:
        for i in range(len(id_list_te)):
            id = id_list_te[i]
            penalty = y_pred[i]
            data = json.dumps({'id': str(id), 'penalty': int(penalty), "laws": [1, 2, 3, 4]})
            f.write(data+'\n')
        f.flush()

if __name__ == '__main__':
    make_sub('../res/lr_d2v_window4_mincount3.txt')