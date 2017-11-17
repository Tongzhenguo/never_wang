import codecs
import gc
import os
import re
from collections import defaultdict

import jieba
import numpy as np
import pandas as pd
from gensim import corpora

batch_size = 1000
stop_words = set()
i =0
with codecs.open('../data/stop_words.txt',encoding='utf-8') as f:
    ls = f.readlines()
    for line in ls:
        i += 1
        if i%batch_size ==0:
            print( line )
        stop_words.add( str(line).strip().replace('\r', '').replace('\n', '') )
stop_words = stop_words.union({'被告人', '犯', '本院', '提起公诉', '依法', '简易程序', '公诉', '机关','犯罪','异议'
                                  , '受理', '审理', '公开', '开庭审理', '本案', '指派','检察员', '出庭','辩护人'
                               ,'出庭', '诉讼', '现已', '审理', '终结','指控', '立案', '审判','合议庭','上诉人'
                               ,'甲','乙','丙','当事人','因涉嫌','支持', '到庭', '参加', '月份','程序', '代理'
                               , '实行', '开庭',})
def remove_previous_convictions( doc ):
    '''
    嫌疑犯前科文本识别
    :param doc: 法律文书
    :return: str
    '''
    return re.sub(r'.*人民检察院以.*?起诉书.?指控','',doc,count=1)# only replace once,'*?' user for shortest match

def bulid_law_vacab():
    law_set = []
    id2laws = {}
    with codecs.open('../data/form-laws.txt', encoding='utf-8') as f:
        ls = f.readlines()
        for i, line in enumerate(ls):
            laws = []
            for law in re.findall('【(.*?)】',line):
                laws.append(law)
                law_set.append( law )
            id2laws[i + 1] = laws
    print(id2laws)

    ##手工标注法律条文,如果遇到不可分割的顿号(和分号)则用空字符串替换顿号，支持手动修改复杂法律条文（比如法条253）
    with codecs.open('../data/idlaw.csv', encoding='utf-8') as f:
        rows = f.readlines()
        for i,line in enumerate(rows):
            if i == 0:continue #skip header
            row = line.replace('\r','').replace('\n','').split(',')
            id,law,new_law = int(row[0]),row[1],row[2]
            for law in re.split( r';|、',str(new_law) ):
                if law not in law_set:
                    law_set.append(law)
                    id2laws[id].append( law )

    for id,laws in id2laws.items():
        print(id,laws)
    pd.to_pickle(id2laws,'../cache/law_vocab.pkl')
# bulid_law_vacab()

i = 0
def token_extract(text,ngram=[]):
    global i
    words = []
    word_list = list(jieba.cut(text, HMM=False))
    re_area = re.compile(u'.*?[省市县区乡镇街]+|.*?公安局|.*?检察院|.*?看守所|.*?法院')
    for i,word in enumerate(word_list):
        # if word not in stop_words: #and not word.isdigit() :and not re_area.match(word) and len(word)>1
            # if pos not in ['nr',"nr1","nr2","nrj","nrf","ns",'m']:
        for n in ngram :
            if i+n<len(word_list):
                words.append(''.join(word_list[i:i+n]))
        words.append(word)
    i += 1
    if i%batch_size ==0:print(words)
    return words


def get_train_cache():
    id_list = []
    doc_list = []
    penalty_list = []
    laws_list = []

    with codecs.open( '../data/train.txt',encoding='utf-8' ) as f:
        ls = f.readlines()
    i = 0
    for line in ls:
        ss = str(line).strip().replace('\r','').replace('\n','').split('\t')
        id,doc,penalty,laws = int(ss[0]),str(ss[1]),int(ss[2]),str(ss[3])
        i += 1
        if i%10 == 0:
            print(id,doc,penalty,laws)
        id_list.append( id )
        doc_list.append(doc)
        penalty_list.append(penalty)
        laws_list.append(laws)

    pd.to_pickle(id_list,'../cache/id_list.pkl')
    pd.to_pickle(doc_list,'../cache/doc_list.pkl')
    pd.to_pickle(penalty_list,'../cache/penalty_list.pkl')
    pd.to_pickle(laws_list,'../cache/laws_list.pkl')

def get_test_cache():
    id_list = []
    doc_list = []

    with codecs.open( '../data/test.txt',encoding='utf-8' ) as f:
        ls = f.readlines()
    i = 0
    for line in ls:
        ss = str(line).strip().replace('\r','').replace('\n','').split('\t')
        id,doc = int(ss[0]),str(ss[1])
        i += 1
        if i%10 == 0:
            print(id,doc)
        id_list.append( id )
        doc_list.append(doc)

    pd.to_pickle(id_list,'../cache/id_list_te.pkl')
    pd.to_pickle(doc_list,'../cache/doc_list_te.pkl')

def _build_vocabulary(dictionary_path ='../data/vocabulary.dict',ngram=[2,3],filter=True):
    '''
    词表是一个很重要的影响因素，不过滤构造的词矩阵会OOM
    '''
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

        # cor = [ token_extract(remove_previous_convictions(line) ) for line in doc_list ]
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
        if filter:
            once_ids = set([tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq < 3])
            dictionary.filter_tokens(once_ids.union( stop_words ))
            dictionary.compactify()
        print('len dictionary = %s' % len(dictionary)) #len dictionary = 125156
        dictionary.save(dictionary_path)
        del doc_list,doc_list_te
        gc.collect()
    return dictionary


# def build_tfidf_matrix(tfidf_matrix_path='../cache/tfidf_matrix.pkl',doc_list_path='../cache/doc_list.pkl',token2id={},sublinear_tf=False,norm='l2'):
#     if os.path.exists(tfidf_matrix_path):
#         return pd.read_pickle(tfidf_matrix_path)
#     else:
#         class MyCorpus():
#             def __init__(self,doc_list):
#                 self.doc_list = doc_list
#             def __iter__(self):
#                 for line in doc_list:
#                     yield ' '.join( token_extract(line.lower()) )
#         doc_list = pd.read_pickle(doc_list_path)
#         corpus = MyCorpus(doc_list)
#
#
#         '''
#            sublinear_tf : boolean, default=False Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
#            norm : 'l1', 'l2' or None, optional Norm used to normalize term vectors. None for no normalization.
#        '''
#         vectorizer = TfidfVectorizer(stop_words=stop_words,vocabulary=token2id,sublinear_tf=sublinear_tf,norm=norm)
#         tfidf_matrix = vectorizer.fit_transform(corpus)
#         pd.to_pickle(tfidf_matrix,tfidf_matrix_path)
#         return tfidf_matrix

# def prepare_tfidf_matrix():
#     dic = _build_vocabulary(dictionary_path ='../data/vocabulary_all.dict')
#     id2token = { tokenid:token for (tokenid,token) in dic.items() }
#     id2df = dic.dfs
#     token2df = { id2token[tokenid]:df for (tokenid,df) in id2df.items() }
#     df = pd.DataFrame()
#     df['token'] = token2df.keys()
#     df['df'] = token2df.values()
#     # eda_token_df(df)
#
#     tokens = list(df[(df['df'] > 4) & (df['df'] < 28000)]['token'].values)
#     print('token length %s' % len(tokens))
#     dic = {}
#     for i,token in enumerate(tokens): ##使用迭代器直接产生词编号
#         dic[token] = i
#
#     ##############调参 词表（min_df,max_df）和计算公式（sublinear_tf，norm）
#     options = [(False,True),('l1','l2')]
#     sublinear_tf = options[0][random.randint(0, 1)]
#     norm  = options[1][random.randint(0, 1)]
#     tfidf_matrix_tr_path = '../cache/tfidf_matrix_tr_%s_%s.pkl' % (sublinear_tf, norm)
#     tfidf_matrix_te_path = '../cache/tfidf_matrix_te_%s_%s.pkl' % (sublinear_tf, norm)
#     build_tfidf_matrix(tfidf_matrix_path=tfidf_matrix_tr_path,sublinear_tf=sublinear_tf,norm=norm,token2id=dic)
#     build_tfidf_matrix(tfidf_matrix_path=tfidf_matrix_te_path, sublinear_tf=sublinear_tf, norm=norm, token2id=dic,doc_list_path='../cache/doc_list_te.pkl')

def cnt_law_penalty(  ):
    laws_list = pd.read_pickle('../cache/laws_list.pkl')
    penalty_list = pd.read_pickle('../cache/penalty_list.pkl')
    lawpe2cnt = defaultdict(int)
    df = pd.DataFrame()
    for law_seq,penalty_label in zip( laws_list,penalty_list ):
        for law in law_seq.split(','):
            lawpe2cnt[(law,penalty_label)] = lawpe2cnt[(law,penalty_label)]+1
    law_list = []
    penalty_labels = []
    cnt_list = []
    for key,cnt in lawpe2cnt.items():
        law,penalty_label = key
        law_list.append(law)
        penalty_labels.append(penalty_label)
        cnt_list.append(cnt)
    df['law'] = law_list
    df['penalty'] = penalty_labels
    df['cnt'] = cnt_list
    df.to_csv( '../cache/law_penalty_cnt.csv',index=False,encoding='utf-8' )



def extract_money_law( doc_list_path='../cache/doc_list.pkl',num_example=40000,path='../data/data.csv' ):

    re_money = re.compile('([0-9]+[.][0-9]+)([零十百万千亿余]*)元')
    re_law = re.compile(u'犯(.*?罪)')

    doc_list = pd.read_pickle(doc_list_path)
    id2law = pd.read_pickle('../cache/id2law.pkl') #可能含有分号和顿号
    law2id = { law:id for id,law in id2law.items() }

    print(id2law)
    newlaw2id = {}
    ##手工标注法律条文
    idlaw = pd.read_csv('../data/idlaw.csv',encoding='utf-8')
    for row in idlaw.values:
        id,law,new_law = row[0],row[1],row[2]
        for law in re.split( r';|、',str(new_law) ):
            newlaw2id[law] = int(id)

    money_stats_matrix = np.zeros(shape=(num_example, 7), dtype=float)
    law_sequence_list = []
    laws_list = pd.read_pickle( '../cache/laws_list.pkl' )
    acnt = 0
    for i, line in enumerate(doc_list):
        ##先把嫌疑人前科，上次公诉等去掉
        line = remove_previous_convictions(line)
        line = line.replace('涉嫌','犯')
        # if i !=113:continue
    #     money_set = set()
    #     money_set.add(0)
    #
    #     for digit, power in re_money.findall(line):
    #         digit = float(digit)
    #         if '十万' in power:
    #             digit = digit * 100000
    #         if '百万' in power:
    #             digit = digit * 1000000
    #         if '十' in power:
    #             digit = digit * 10
    #         if '百' in power:
    #             digit = digit * 100
    #         if '千' in power:
    #             digit = digit * 1000
    #         if '万' in power:
    #             digit = digit * 10000
    #         money_set.add(digit)
    #     money_set = np.array(list(map(float,money_set)), dtype=float)
    #     sum = np.sum(money_set)
    #     max = np.max(money_set)
    #     min = np.min(money_set)
    #     cnt = money_set.shape[0]
    #     avg = np.mean(money_set)
    #     m = sorted(money_set)[int(len(money_set) / 2)]
    #     std = np.std(money_set)
    #     money_stats_matrix[i] = np.array([sum, max, min, cnt, avg, m, std])
    #     print(line)
        law_list = []
        for ll in '掩饰犯罪所得罪、隐瞒犯罪所得罪、掩饰犯罪所得收益罪、隐瞒犯罪所得收益罪'.split('、'):
            if ll in line:
                law_list.append(id2law[newlaw2id[ll]])
        for k in law2id:
            if k in ['任务','程序','立法目的','罪刑法定','法律面前人人平等','生效日期']:continue
            if  k in line :
                # print(k)
                law_list.append( k )

            for law in re_law.findall(line):
                # print(law)
                if law == k:
                    law_list.append(k)
                if law in str(k).split(';'):
                    # print(k)
                    law_list.append(k)
                if law.replace('、','') in newlaw2id:
                    # print(law)
                    law = law.replace('、', '')
                    law_list.append( id2law[newlaw2id[law]] )
                # if '、' in k and len(set(law) & set(k)) / len(set(law) | set(k))>0.8:
                #     law_list.append(k)
        # print('example %s' % i, set(law_list), set([id2law[int(id)] for id in laws_list[i].split(',')]))
        # if len(set(law_list))==0 :
        #     print('example %s' % i ,set(law_list),set([id2law[int(id)] for id in laws_list[i].split(',') ]))
        # law_sequence_list.append( set(map(str,set(law_list))) )
        #
        # a = len( set(str(laws_list[i]).split(',')) & law_sequence_list[i] ) \
        #     / len( set(str(laws_list[i]).split(',')) | law_sequence_list[i] )
        # if a>0.6:
        #     acnt += 1
        # print(acnt)
    ###构造金额和触犯法规的特征
    # data = pd.DataFrame()
    # data['money_sum'] = money_stats_matrix[:, 0]
    # data['money_max'] = money_stats_matrix[:, 1]
    # data['money_min'] = money_stats_matrix[:, 2]
    # data['money_cnt'] = money_stats_matrix[:, 3]
    # data['money_avg'] = money_stats_matrix[:, 4]
    # data['money_m'] = money_stats_matrix[:, 5]
    # data['money_std'] = money_stats_matrix[:, 6]

    # for law, id in law2id.items():
    #     data['law_%s' % law] = tfidf_matrix[:,id]
    # data.to_csv(path, encoding='utf-8', index=False)

def extract_area_age( doc_list_path='../cache/doc_list.pkl',num_example=10,path='../data/data_area_age.csv' ):
    re_age = re.compile(u'([0-9]+)岁|([0-9]+)年.*出生',re.U)
    re_area = re.compile(u'公诉机关([\u4E00-\u9FD5]+)人民检察院',re.U)

    doc_list = pd.read_pickle(doc_list_path)[:num_example]

    age_stats_matrix = np.zeros(shape=(num_example, 7), dtype=float)
    for i, line in enumerate(doc_list):
        age_set = set()
        age_set.add(0)
        area = ''
        for a in re_area.findall(line):
            area = a
        for age,year in re_age.findall(line):
            if age and age != '':
                age_set.add(int(age))
            if year and year != '':
                age_set.add(2016-int(year))
        print( area,age_set )

if __name__ == '__main__':
    dictionary_path = '../data/vocabulary_all_.dict'
    _build_vocabulary(dictionary_path)

    # cnt_law_penalty()

    # extract_area_age(  )
    # extract_money_law(doc_list_path='../cache/doc_list.pkl',num_example=40000,path='../data/data_.csv')
    #
    # extract_money_law(doc_list_path='../cache/doc_list_te.pkl', num_example=30000, path='../data/data_te.csv')



