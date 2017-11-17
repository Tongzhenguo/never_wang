import codecs
import json
import random

from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd

def make_train_test( path= '../cache/tfidf_matrix.pkl' ):
    tr_X = pd.read_pickle(path).toarray()
    tr_y = pd.read_pickle('../cache/penalty_list.pkl')
    tr_y = np.array( tr_y )
    return tr_X,tr_y

def make_test_data( path= '../cache/tfidf_matrix_te.pkl' ):
    te_X = pd.read_pickle(path).toarray()
    return te_X

def train(  path= '../cache/tfidf_matrix.pkl' ):
    tr_X, tr_y = make_train_test( path )
    clf = GaussianNB()
    clf.fit(tr_X, tr_y)
    return clf

def eval( path= '../cache/tfidf_matrix.pkl' ):
    tr_X, tr_y = make_train_test(path)
    ids = list(range(40000))
    random.shuffle(ids)
    tr_X, tr_y,te_X, te_y = tr_X[ids[:30000]],tr_y[ids[:30000]],tr_X[ids[30000:]],tr_y[ids[30000:]]
    clf = GaussianNB()
    clf.fit(tr_X, tr_y)
    y_pred = clf.predict(te_X)
    print( metrics.f1_score(y_pred=y_pred,y_true=te_y,average='micro',labels=[1,2,3,4,5,6,7,8]) )
def make_sub( tr_path= '../cache/tfidf_matrix.pkl',te_path= '../cache/tfidf_matrix_te.pkl',res_name='../res/res_gnb.txt' ):
    clf = train( tr_path )
    te_X = make_test_data( te_path )
    id_list_te = pd.read_pickle('../cache/id_list_te.pkl')
    y_pred = clf.predict(te_X)
    with codecs.open(res_name,encoding='utf-8',mode='w') as f:
        for i in range(len(id_list_te)):
            id = id_list_te[i]
            penalty = y_pred[i]
            data = json.dumps({'id': str(id), 'penalty': int(penalty), "laws": [1, 2, 3, 4]})
            f.write(data+'\n')
        f.flush()


if __name__ == "__main__":
    tr_path = '../cache/tfidf_matrix_tr_False_l2.pkl'
    te_path = '../cache/tfidf_matrix_te_False_l2.pkl'
    # eval( '../cache/tfidf_matrix.pkl') #0.2773
    # eval(tr_path)

    make_sub( tr_path,te_path,'../res/res_gnb_voc38000.txt' )