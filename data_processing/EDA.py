from collections import Counter
from operator import itemgetter
import codecs
import re
import snownlp
import pandas as pd
from pytagcloud import create_tag_image, make_tags, \
    LAYOUT_HORIZONTAL
from pytagcloud.colors import COLOR_SCHEMES

from data_processing.dataprocessing import _build_vocabulary

def show_token_df():
    dic = _build_vocabulary(dictionary_path='../data/vocabulary_all.dict')
    id2token = {tokenid: token for (tokenid, token) in dic.items()}
    id2df = dic.dfs
    token2df = {id2token[tokenid]: df for (tokenid, df) in id2df.items()}
    df = pd.DataFrame()
    df['token'] = token2df.keys()
    df['df'] = token2df.values()

    print(df['df'].describe())
    '''
    count    125156.000000
    mean         63.621824
    std         858.189270
    min           1.000000
    25%           1.000000
    50%           2.000000
    75%           7.000000
    max       39912.000000

    '''

    print({token: df for (token, df) in token2df.items() if df > 30000} )
    '''
    {'起诉书': 38442, '公诉': 39386, '现已': 39136, '参加': 38840, '检察员': 37974, '检': 37350, '机关': 39859, '元': 31317, '指控': 39265, '终结': 39468, '月': 39911, '证据': 37175, '年': 39912, '上述事实': 33553, '犯': 39459, '人民检察院': 39234, '号': 39814, '审理': 39629, '开庭审理': 35738, '到庭': 38301, '供述': 30093, '证实': 32083, '被告人': 39864, '提起公诉': 38118, '依法': 39123, '指派': 33070, '本案': 36616, '出庭': 34811, '支持': 35414, '公开': 38635, '中': 31875, '本院': 39852, '刑诉': 38329, '日': 39902, '诉讼': 38437} len 35
    '''
    print(df[(df['df'] > 3) & (df['df'] < 30000)].describe())

    filter_words = {token:df for  (token,df) in token2df.items() if df>5000 }
    print(filter_words,'len %s' % len(filter_words) )
    swd = sorted(filter_words.items(), key=itemgetter(1), reverse=True)
    tags = make_tags(swd, minsize=10, maxsize=50, colors=COLOR_SCHEMES['goldfish'])
    create_tag_image(tags, 'keyword_tag_cloud4.png',size=(2400, 1000), background=(240, 255, 255),
                     layout=LAYOUT_HORIZONTAL, fontname="SimHei")


def stats_sentence_len_freq(  ):
    dl = pd.read_pickle('../cache/doc_list.pkl')
    lens = []

    for line in dl:
        for i in line.split('，'):
            lens.append(len(i))

    cnt = Counter(lens)
    print(cnt)
    '''
    Counter({0: 158,
         1: 30896,
         2: 41542,
         3: 15492,
         4: 68405,
         5: 64413,
         6: 83588,
         7: 80843,
         8: 71987,
         9: 76409,
         10: 73825,
         11: 72276,
         12: 74226,
         13: 69081,
         14: 66797,
         15: 62649,

    '''
import numpy as np
import os
def stats_law_count():
    if os.path.exists('../data/law_stats.csv'):
        df = pd.read_csv('../data/law_stats.csv',encoding='utf-8')
    else:
        id2law = {}
        with codecs.open('../data/form-laws.txt',encoding='utf-8') as f:
            ls = f.readlines()
            for i,line in enumerate(ls):
                id2law[i+1] = line.split('】')[0].split('【')[1]

        pd.to_pickle(id2law,'../cache/id2law.pkl')
        laws_list = pd.read_pickle( '../cache/laws_list.pkl' )
        lawnos = np.zeros(shape=(40000,452))

        for i,laws_sequence in enumerate(laws_list):
            for lno in str(laws_sequence).split(','):
                if lno == '376':print(i,laws_sequence)
                lawnos[i,int(lno)-1] = lawnos[i,int(lno)-1] + 1

        df = pd.DataFrame()
        df['index'] = list(range(40000))
        for i in range(1,453):
            df['law_%s' % i] = lawnos[:,i-1]

        law_cnt = {}
        for i in range(1,453):
            law_cnt[i] = df['law_%s' % i].sum()
        law_cnt = {law: cnt for (law, cnt) in law_cnt.items() if cnt > 0}
        law_cnt = sorted(law_cnt.items(),key=lambda k:k[1],reverse=True)
        print(law_cnt[:10])
        # [(68, 29907.0), (53, 19852.0), (54, 17805.0), (265, 16570.0), (65, 15043.0), (73, 9888.0), (26, 9296.0), (74, 7673.0), (66, 5873.0), (134, 5056.0)]
        print(law_cnt)
        # (68, 29907.0), (53, 19852.0), (54, 17805.0), (265, 16570.0), (65, 15043.0), (73, 9888.0), (26, 9296.0), (74, 7673.0), (66, 5873.0), (134, 5056.0), (27, 3293.0), (348, 3140.0), (28, 3074.0), (62, 2936.0), (267, 2769.0), (70, 2586.0), (24, 1944.0), (304, 1835.0), (48, 1748.0), (264, 1556.0), (355, 1334.0), (69, 1196.0), (197, 1190.0), (43, 1111.0), (46, 1018.0), (313, 1004.0), (45, 660.0), (346, 617.0), (357, 606.0), (225, 558.0), (268, 536.0), (63, 524.0), (79, 505.0), (78, 491.0), (349, 459.0), (226, 433.0), (275, 424.0), (142, 415.0), (71, 394.0), (80, 388.0), (57, 343.0), (384, 328.0), (56, 320.0), (77, 282.0), (206, 239.0), (72, 235.0), (270, 225.0), (18, 224.0), (20, 216.0), (358, 200.0), (386, 194.0), (235, 193.0), (387, 190.0), (177, 189.0), (383, 170.0), (215, 166.0), (13, 154.0), (360, 154.0), (343, 139.0), (214, 136.0), (94, 126.0), (39, 125.0), (277, 123.0), (294, 122.0), (37, 119.0), (32, 117.0), (19, 115.0), (31, 115.0), (276, 110.0), (38, 101.0), (339, 94.0), (145, 86.0), (42, 85.0), (141, 83.0), (58, 81.0), (60, 80.0), (144, 79.0), (176, 79.0), (239, 75.0), (281, 75.0), (64, 72.0), (278, 69.0), (337, 65.0), (345, 63.0), (129, 62.0), (87, 61.0), (272, 59.0), (178, 57.0), (59, 56.0), (359, 56.0), (76, 47.0), (344, 47.0), (193, 46.0), (210, 46.0), (342, 45.0), (44, 44.0), (240, 42.0), (293, 42.0), (391, 41.0), (394, 41.0), (55, 40.0), (398, 40.0), (364, 38.0), (23, 37.0), (82, 37.0), (352, 35.0), (199, 33.0), (211, 33.0), (311, 33.0), (241, 32.0), (218, 29.0), (237, 29.0), (329, 29.0), (390, 29.0), (154, 28.0), (227, 28.0), (229, 27.0), (254, 26.0), (25, 23.0), (273, 23.0), (323, 23.0), (7, 22.0), (216, 22.0), (172, 21.0), (173, 21.0), (164, 20.0), (385, 18.0), (6, 17.0), (36, 17.0), (49, 17.0), (119, 17.0), (194, 17.0), (232, 17.0), (347, 17.0), (15, 16.0), (263, 16.0), (322, 15.0), (341, 15.0), (30, 14.0), (115, 14.0), (126, 14.0), (221, 14.0), (187, 13.0), (280, 13.0), (195, 12.0), (301, 12.0), (319, 12.0), (351, 12.0), (2, 11.0), (4, 11.0), (35, 11.0), (224, 10.0), (286, 10.0), (152, 8.0), (233, 8.0), (238, 8.0), (308, 8.0), (354, 8.0), (388, 8.0), (81, 7.0), (246, 7.0), (295, 7.0), (317, 7.0), (41, 6.0), (159, 6.0), (163, 6.0), (314, 6.0), (40, 5.0), (52, 5.0), (75, 5.0), (125, 5.0), (157, 5.0), (165, 5.0), (242, 5.0), (289, 5.0), (350, 5.0), (362, 5.0), (366, 5.0), (393, 5.0), (397, 5.0), (47, 4.0), (51, 4.0), (92, 4.0), (135, 4.0), (151, 4.0), (160, 4.0), (291, 4.0), (320, 4.0), (365, 4.0), (61, 3.0), (156, 3.0), (248, 3.0), (259, 3.0), (266, 3.0), (288, 3.0), (334, 3.0), (389, 3.0), (396, 3.0), (3, 2.0), (21, 2.0), (29, 2.0), (116, 2.0), (128, 2.0), (131, 2.0), (132, 2.0), (147, 2.0), (148, 2.0), (150, 2.0), (213, 2.0), (220, 2.0), (228, 2.0), (234, 2.0), (244, 2.0), (247, 2.0), (269, 2.0), (282, 2.0), (285, 2.0), (306, 2.0), (321, 2.0), (400, 2.0), (5, 1.0), (8, 1.0), (9, 1.0), (10, 1.0), (14, 1.0), (33, 1.0), (50, 1.0), (85, 1.0), (88, 1.0), (90, 1.0), (93, 1.0), (98, 1.0), (123, 1.0), (127, 1.0), (155, 1.0), (167, 1.0), (169, 1.0), (171, 1.0), (186, 1.0), (192, 1.0), (200, 1.0), (201, 1.0), (202, 1.0), (203, 1.0), (212, 1.0), (219, 1.0), (243, 1.0), (245, 1.0), (251, 1.0), (255, 1.0), (261, 1.0), (283, 1.0), (284, 1.0), (287, 1.0), (296, 1.0), (315, 1.0), (316, 1.0), (325, 1.0), (327, 1.0), (338, 1.0), (373, 1.0), (376, 1.0), (381, 1.0), (392, 1.0), (403, 1.0), (406, 1.0)]

        df = pd.DataFrame()
        df['id'] = [lno for (lno,cnt) in law_cnt ]
        df['law'] = [id2law[lno] for (lno, cnt) in law_cnt]
        df['cnt'] = [cnt for (lno, cnt) in law_cnt]
        df.to_csv('../data/law_stats.csv',encoding='utf-8',index=False)

    print( df.head(30) )
    '''
         id                law      cnt
0    68                 立功  29907.0
1    53           罚金的缴纳、减免  19852.0
2    54          剥夺政治权利的含义  17805.0
3   265                盗窃罪  16570.0
4    65               一般累犯  15043.0
5    73               考验期限   9888.0
6    26                 主犯   9296.0
7    74            累犯不适用缓刑   7673.0
8    66               特别累犯   5873.0
9   134  重大责任事故罪;强令违章冒险作业罪   5056.0
10   27                 从犯   3293.0
11  348            非法持有毒品罪   3140.0
12   28                胁从犯   3074.0
13   62          从重处罚与从轻处罚   2936.0
14  267            抢夺罪;抢劫罪   2769.0
15   70       判决宣告后发现漏罪的并罚   2586.0
16   24               犯罪中止   1944.0
17  304          故意延误投递邮件罪   1835.0
18   48    死刑、死缓的适用对象及核准程序   1748.0
19  264                盗窃罪   1556.0
20  355     非法提供麻醉药品、精神药品罪   1334.0
21   69      判决宣告前一人犯数罪的并罚   1196.0
22  197            有价证券诈骗罪   1190.0
23   43              拘役的执行   1111.0
24   46       有期徒刑与无期徒刑的执行   1018.0
25  313         拒不执行判决、裁定罪   1004.0
26   45            有期徒刑的期限    660.0
27  346  单位犯破坏环境资源保护罪的处罚规定    617.0
28  357    毒品的范围及毒品数量的计算原则    606.0
29  225              非法经营罪    558.0

    '''

def show_pe_cnt():
    import pandas as pd
    import matplotlib.pyplot as plt
    lawpe2cnt = pd.read_pickle('../cache/lawpe2cnt.pkl' )
    lawpe2cnt_list = sorted( lawpe2cnt.items(),key=lambda p:-p[1] )
    df = pd.DataFrame()
    df['law_pelabel'] = [label for label,cnt in lawpe2cnt_list]
    df['cnt'] = [cnt for label,cnt in lawpe2cnt_list]
    print(df)
    df.plot(kind='bar')
    plt.show()

def stats_laws_penalty_relationship():
    data = pd.read_csv('../cache/law_penalty_cnt.csv')
    data['freq'] = 1
    law_freq = data[data['cnt'] > 5][['law', 'freq']].groupby(['law'],as_index=False).count().sort_values(['freq'])
    top_laws = list(law_freq[law_freq['freq']==1]['law'].values)
    ##那些出现次数多，并且每次都是只对应同一罚金范围的是很有判别性的特征
    print(top_laws)#[198, 80, 226, 223, 220, 217, 215, 210, 205, 236, 194, 192, 186, 177, 172, 171, 163, 125, 118, 193, 48, 228, 262, 390, 389, 387, 384, 350, 346, 253, 318, 340, 307, 294, 272, 285]

    print( data[data['law'].isin( top_laws )].sort_values(['cnt'],ascending=False).drop_duplicates('law')[['law','penalty','cnt']] )

if __name__ == "__main__":
    stats_laws_penalty_relationship()
    pass


