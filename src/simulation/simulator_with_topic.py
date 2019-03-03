#coding:utf-8
'''
个人引用行为，引文网络形成过程仿真器

完成仿真基本过程

'''
import uuid
import random
import numpy as np
import json
import sys
sys.path.extend(['.','..'])
from tools.basic_config import *
import powerlaw
import time

def add_norm_item(mu,sigma,not_neg=True,must_incre=False):


    while True:
        nn = False
        mi = False

        margin = np.random.normal(mu,sigma,1)[0]

        if margin>0:
            nn = True

        if margin>mu:
            mi = True

        # print mu,margin,nn,mi,not_neg,must_incre
        if nn==not_neg and mi==must_incre:
            break

    return margin


## 对主题相关性的仿真,初始化100个主题
def simulate_topic_relevance(size=100):

    topic_rel_func = json.loads(open('topic_relevance.json').read())
    xs = topic_rel_func['x']
    ys = topic_rel_func['y']

    print ys[0]

    index_rel = {}
    for i,x in enumerate(xs):
        y = ys[i]
        index_rel[x] = y

    basic_rel_list = []
    for i in range(1,size+1):

        ## 根据顺序获得 各个主题与本主题的相关性
        rel = index_rel[i]

        basic_rel_list.append(rel)

    ## 比例的和需要归一化为1,第一项为本主题的相关性
    all_rel_list = []
    for i in range(1,size+1):
        print i
        ##每一个主题的每一项
        specific_rel_list = []
        for j,rel in enumerate(basic_rel_list):
            ## 加入随机项，
            mi=False
            if j==0:
                mi=True

            rel = add_norm_item(rel,rel*1,must_incre=mi)

            specific_rel_list.append(rel)

        specific_rel_list = specific_rel_list/np.sum(specific_rel_list)
        all_rel_list.append(specific_rel_list)


    ## 对于每一个list,第一项最大值为本主题相关性，其他随机分布
    # topics = ['T_{:}'.format(t) for t in range(1,size+1)]
    new_rels_list = []
    for i,rel_list in enumerate(all_rel_list):
        first_rel = rel_list[0]
        other_rels = rel_list[1:]

        shuffle_rels = np.random.shuffle(other_rels)

        new_rels = []

        new_rels.append(first_rel)
        new_rels.extend(other_rels)

        replace_rel = new_rels[i]

        ## 第一个和第i个进行交换，保证本主题的最大
        new_rels[0] = replace_rel
        new_rels[i] = first_rel

        new_rels_list.append(new_rels)

    ## 将矩阵进行对称
    asym_matrix = []
    for i,rel_list in enumerate(new_rels_list):

        asym_list = []
        for j,rel in enumerate(rel_list):

            avg = np.max([rel,new_rels_list[j][i]])

            asym_list.append(avg)

        asym_matrix.append(asym_list)

    ## 对每一行进行归一化
    t1_t2_rel = defaultdict(lambda:defaultdict(float))
    for i,rel_list in enumerate(asym_matrix):

        rel_list = rel_list/np.sum(rel_list)

        for j,rel in enumerate(rel_list):

            t1_t2_rel['T_{:}'.format(i+1)]['T_{:}'.format(j+1)] = rel

    ## 返回的是t2对于t1的相关性
    return t1_t2_rel


## 根据作者的选择主题的历史，主题分布概率，以及规则选择主题
def simulate_author_select_topic(author_topic_list,topics,props,rule):

    ##规则R1, 不根据历史，只根据主题分布进行选择
    if rule=='R1':
        topic = np.random.choice(topics,size=1,p=props)[0]
    elif rule =='R2':
        ## 规则2 如果没有选择，按概率选择，如果已经有选择，则按选择来
        if len(author_topic_list)==0:
            topic = np.random.choice(topics,size=1,p=props)[0]
        else:
            topic = author_topic_list[-1]
    elif rule=='R3':
        ## 规则3
        ## 如果没有选过
        if len(author_topic_list)==0:
            topic = np.random.choice(topics,size=1,p=props)[0]
        else:

            pass






if __name__ == '__main__':
    t1_t2_rel = simulate_topic_relevance(15)

    lines = ['t1,t2,rel']
    for t1 in sorted(t1_t2_rel.keys()):
        t2_rel = t1_t2_rel[t1]

        for t2 in sorted(t2_rel.keys()):

            rel = t1_t2_rel[t1][t2]
            lines.append('{:},{:},{:}'.format(t1,t2,rel))

    open('fig/test/test_topic_matrix.csv','w').write('\n'.join(lines))

    plot_heatmap('fig/test/test_topic_matrix.csv','主题相关性矩阵','主题','主题','fig/test/test_topic_rel_matrix.png')







