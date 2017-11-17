import codecs
import json
from collections import Counter

import pandas as pd
import re

from data_processing.dataprocessing import remove_previous_convictions

data = pd.read_csv('../cache/law_penalty_cnt.csv')
tmp = data.sort_values('cnt',ascending=False).drop_duplicates('law')[['law','penalty']]
tmp.index = tmp['law']
law2pe = tmp['penalty'].to_dict()
# print(law2pe)

penalty_list = pd.read_pickle('../cache/penalty_list.pkl')
cnt = Counter(penalty_list)
sortcnt = list(sorted( cnt.items(),key=lambda p:-p[1]))
default_penalty = sortcnt[0][0]
# print(default_penalty)

id2law = pd.read_pickle('../cache/id2law.pkl')  # 可能含有分号和顿号
law2id = {law: id for id, law in id2law.items()}
newlaw2id = {}
##手工标注法律条文
idlaw = pd.read_csv('../data/idlaw.csv', encoding='utf-8')
for row in idlaw.values:
    id, law, new_law = row[0], row[1], row[2]
    for law in re.split(r';|、', str(new_law)):
        newlaw2id[law] = int(id)
re_law = re.compile(u'犯(.*?罪)')

doc_list_te = pd.read_pickle('../cache/doc_list_te.pkl')
y_pred = []
for i,doc in enumerate(doc_list_te):
    ##先把嫌疑人前科，上次公诉等去掉
    line = remove_previous_convictions(doc)
    line = line.replace('涉嫌', '犯')
    lawno_list = []
    for ll in '掩饰犯罪所得罪、隐瞒犯罪所得罪、掩饰犯罪所得收益罪、隐瞒犯罪所得收益罪'.split('、'):
        if ll in line:
            lawno_list.append(newlaw2id[ll])
    for k in law2id:
        if k in ['任务', '程序', '立法目的', '罪刑法定', '法律面前人人平等', '生效日期']: continue
        if k in line:
            lawno_list.append(law2id[k])

        for law in re_law.findall(line):
            # print(law)
            if law == k:
                lawno_list.append(law2id[k])
            if law in str(k).split(';'):
                # print(k)
                lawno_list.append(law2id[k])
            if law.replace('、', '') in newlaw2id:
                # print(law)
                law = law.replace('、', '')
                lawno_list.append(newlaw2id[law])
    penalty_list = [law2pe[lawno] for lawno in lawno_list if lawno in law2pe]
    cnt = Counter(penalty_list)
    sortcnt = list(sorted(cnt.items(), key=lambda p: -p[1]))
    if not lawno_list or not sortcnt:
        y_pred.append( int(default_penalty) )
    else:
        y_pred.append( int(sortcnt[0][0]) )
    if i % 100 == 0:
        print('num %s' % i, lawno_list)
        print('num %s' % i, penalty_list)
id_list_te = pd.read_pickle('../cache/id_list_te.pkl')
with codecs.open('../res/rule.txt', encoding='utf-8', mode='w') as f:
    for i in range(len(id_list_te)):
        id = id_list_te[i]
        penalty = y_pred[i]
        data = json.dumps({'id': str(id), 'penalty': int(penalty), "laws": [1, 2, 3, 4]})
        f.write(data + '\n')
    f.flush()