# coding=utf-8
import codecs
import gc
import json
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.metrics.scorer import metric
from sklearn.model_selection import train_test_split
import sys


def make_data(isTrain=True):
    print('开始构造训练集/测试集---------------------------------------------------')
    if isTrain:
        d2v = pd.read_pickle('../cache/tr_x_d2v_300.pkl')
        data = pd.read_csv('../data/data.csv')[
            ['money_sum', 'money_m', 'money_max', 'money_std', 'money_avg', 'money_cnt']]
        for i in range(300):
            data['d2v_%s' % i] = d2v[:,i]
        return data
    else:
        d2v = pd.read_pickle('../cache/te_x_d2v_300.pkl')
        data = pd.read_csv('../data/data_te.csv')[
            ['money_sum', 'money_m', 'money_max', 'money_std', 'money_avg', 'money_cnt']]
        for i in range(300):
            data['d2v_%s' % i] = d2v[:, i]
        return data

def xgb_eval(  ):
    tr_x = pd.read_csv('../data/data.csv')
    label = pd.read_pickle('../cache/penalty_list.pkl')
    y = np.array(label)
    y = y - 1 # start 0

    X_train, X_test, y_train, y_test = train_test_split( tr_x,y,test_size=0.3,random_state=20171024 )
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    for n in [ 1000,10000,20000]:
        for d in [ 3,5,7,9 ]:
            for w in [5,15]:
                    param = {'max_depth': d
                        , 'min_child_weight': w #以前是5
                        , 'gamma': 0.1
                        , 'subsample': 1.0
                        , 'colsample_bytree': 1.0
                        , 'eta': 0.01
                        , 'lambda': 100  # L2惩罚系数
                       # ,'scale_pos_weight':763820 / 81164 #处理正负样本不平衡,
                        , 'objective': 'multi:softmax'
                        , 'eval_metric': 'mlogloss'  # 注意目标函数和评分函数的对应
                        ,'num_class':8
                        , 'early_stopping_rounds': 100  # eval 得分没有继续优化 就停止了
                        , 'seed': 8888
                        , 'nthread': 8
                        , 'silent': 0
                        }
                    print('nums :{0},depth :{1}'.format(n, d))
                    evallist = [(dtest, 'eval'), (dtrain, 'train')]
                    bst = xgb.train(param, dtrain, num_boost_round=n, evals=evallist)
                    bst.save_model('../model/xgb_{0}_{1}.model'.format( n,d ))
                    # print( metrics.f1_score(y_pred=y_pred,y_true=te_y,average='micro',labels=[1,2,3,4,5,6,7,8]) )
                    print()
# xgb_eval()
def sub_xgb( n,d ):
    tr_x = make_data()
    label = pd.read_pickle('../cache/penalty_list.pkl')
    y = np.array(label)
    y = y - 1  # start 0
    dtrain = xgb.DMatrix(tr_x, label=y)
    param = {'max_depth': d
        , 'min_child_weight': 15  # 以前是5
        , 'gamma': 0.1
        , 'subsample': 1.0
        , 'colsample_bytree': 1.0
        , 'eta': 0.01
        , 'lambda': 100  # L2惩罚系数
             # ,'scale_pos_weight':763820 / 81164 #处理正负样本不平衡,
        , 'objective': 'multi:softmax'
        , 'eval_metric': 'mlogloss'  # 注意目标函数和评分函数的对应
        , 'num_class': 8
        , 'early_stopping_rounds': 100  # eval 得分没有继续优化 就停止了
        , 'seed': 8888
        , 'nthread': 8
        , 'silent': 0
             }
    clf = xgb.train(param, dtrain, num_boost_round=n)
    clf.save_model('../model/xgb_%s_%s_15.model'.format(n, d))

    clf = xgb.Booster({'nthread': 8})  # init model
    clf.load_model('../model/xgb_%s_%s_15.model'.format(n, d))

    te_x = make_data(False)
    feat = te_x.columns
    data = pd.DataFrame( feat,columns=['feature'] )
    data['col'] = data.index
    feature_score = clf.get_fscore()
    keys = []
    values = []
    for key in feature_score:
        keys.append( key )
        values.append( feature_score[key] )
    df = pd.DataFrame( keys,columns=['features'] )
    df['score'] = values
    df['col'] = df['features'].apply( lambda x:int(x[1:]) )
    s = pd.merge( df,data,on='col' )
    s = s.sort_values('score',ascending=False)[['feature','score']]
    s.to_csv('../data/feature_scores.csv',index=False)
    te_X = xgb.DMatrix(te_x.values)
    y_pred = clf.predict(te_X)

    id_list_te = pd.read_pickle('../cache/id_list_te.pkl')
    with codecs.open('../res/result_xgb_%s_%s.txt' % (n,d), encoding='utf-8', mode='w') as f:
        for i in range(len(id_list_te)):
            id = id_list_te[i]
            penalty = y_pred[i]
            data = json.dumps({'id': id, 'penalty': int(penalty)+1, "laws":  [351,67,72,73]})
            f.write(data + '\n')
        f.flush()

sub_xgb(1000,9)