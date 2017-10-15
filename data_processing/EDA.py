import jieba
import pandas as pd
import gensim
import codecs
from gensim import corpora

bath_size = 100
dictionary_path = '../data/vocabulary.dict'


stop_words = set()
i =0
with codecs.open('../data/stop_words.txt',encoding='utf-8') as f:
    ls = f.readlines()
    for line in ls:
        i += 1
        if i%bath_size ==0:
            print( line )
        stop_words.add( str(line).strip().replace('\r', '').replace('\n', '') )
i = 0
def token_extract(text):
    words = list(jieba.cut(text, cut_all=False, HMM=False))
    global i
    i += 1
    if i%bath_size ==0:print(words)
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

def build_vocabulary():
    cor = [ token_extract(line.lower() ) for line in pd.read_pickle('../cache/doc_list.pkl') ]
    dictionary = corpora.Dictionary(cor)

    # remove stop words and words that appear only once
    stop_ids = [dictionary.token2id[stopword] for stopword in stop_words
                if stopword in dictionary.token2id]
    #document frequencies: tokenId -> in how many documents this token appeared
    once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq <= 3]
    dictionary.filter_tokens(stop_ids + once_ids)  # remove stop words and words that appear only once
    dictionary.compactify()  # remove gaps in id sequence after words that were removed
    print(dictionary)
    print(len(dictionary)) #44810
    dictionary.save(dictionary_path)

def build_token_id_matrix():
    doc_list = pd.read_pickle('../cache/doc_list.pkl')
    dictionary = corpora.Dictionary.load(dictionary_path)
    tokens = dictionary.values()

    cor = [ [token for token in token_extract(line.lower()) ]for line in  doc_list]
    pd.to_pickle(cor, '../cache/corpora_matrix.pkl')
    corpora_num_matrix = []
    for k in range(len(cor)):
        doc = cor[k]
        doc2ids = [dictionary.token2id[token] for token in doc if token in tokens]
        if k%bath_size == 0:
            print(doc2ids)
        corpora_num_matrix.append( doc2ids )
    pd.to_pickle(cor, '../cache/corpora_num_matrix.pkl')

if __name__ == "__main__":
    pass
