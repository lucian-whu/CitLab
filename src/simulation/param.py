#coding:utf-8
import uuid
import random
import numpy as np
import json
import sys
sys.path.extend(['.','..'])
from tools.basic_config import *
import powerlaw
import time
import argparse
from scipy.stats import lognorm
import copy


## 模型中所有的参数以及使用方法
class PARAM:

    def __init__(self,arg):

        ## 人数增加指数函数参数
        self.AUTHOR_INCREASE_FUNC_PARAS = [4.4786305,0.05994124]

        self.mode = arg.mode
        ## model分为三个部分，模型，阅读规则
        self.MD,self.PR = self.mode.split('-')
        ## 影响因素
        self.fator = arg.factor
        ## 超参数
        self.length = arg.length
        self.topic_number = arg.topic_number
        self.reference_number = arg.reference_number
        self.author_number = arg.author_number
        self.initial_value = arg.initial_value
        ##
        self.exclude_read = arg.exclude_read

        ## 研究周期分布函数，每个人的研究周期使用该函数进行随机抽样
        rs_dis = json.loads(open('rs_dis.json').read())
        self.rs_xs = [int(i) for i in rs_dis['x']]
        self.rs_probs = [float(y) for y in rs_dis['y']]

        ## 生产力拟合函数
        prod_dis = json.loads(open('prod_dis.json').read())
        self.prod_xs = [int(i) for i in prod_dis['x']]
        self.prod_probs = [float(y) for y in prod_dis['y']]
        ## 第一年进入的人没有0的抽样
        self.prod_new_xs = self.prod_xs[1:]
        self.prod_new_probs = np.array(self.prod_probs[1:])/np.sum(self.prod_probs[1:])

        ## 文章主题分布
        topic_dis = json.loads(open('topic_dis.json').read())
        self.topic_xs = [int(i) for i in topic_dis['x'][:self.topic_number]]
        topic_probs = [float(y) for y in topic_dis['y'][:self.topic_number]]
        self.topic_probs = np.array(topic_probs)/np.sum(topic_probs)

        ## 一个主题发表N篇论仍然继续发表该主题论文的概率
        topic_selection_dis = json.loads(open('topic_selection_dis.json').read())
        self.topic_cont_pro = {}
        for i,x in enumerate(topic_selection_dis['x']):
            y = topic_selection_dis['y'][i]
            self.topic_cont_pro[x] = y

        ## 主题
        self._topics = ['T_{:}'.format(i+1) for i in range(self.topic_number)]
        ## 主题相关性矩阵
        self.topic_relevance_matrix = self.simulate_topic_relevance()

        ## 各个主题的价值系数分布
        topic_lambda_dis = json.loads(open('topic_lambda_dis.json').read())
        topic_sigma_dis = json.loads(open('topic_sigma_dis.json').read())

        ## 为每个主题生成一个主题分布
        self._topic_lambda_dis = {}
        for topic in self._topics:

            ## 随机一个lambda 随机一个sigma

            _lambda = np.random.choice(topic_lambda_dis['x'],size=1,p=topic_lambda_dis['y'])[0]
            _sigma = np.random.choice(topic_sigma_dis['x'],size=1,p=topic_sigma_dis['y'])[0]



            ## 根据两个值计算出lognorm的
            _scale = (np.log(_lambda)+_sigma**2)**2

            print topic,_lambda,_sigma,_scale


            ## 使用 lognorm 计算x以及y值
            xs = np.linspace(0.0001,20,10000)
            ys = lognorm.pdf(xs, _sigma, loc=0, scale=_scale)
            ys = np.array(ys)/np.sum(ys)

            self._topic_lambda_dis[topic]=[xs,ys]

        ## 加载整体的lambda_dis
        lambda_dis = json.loads(open('lambda_dis.json').read())
        self._topic_lambda_dis['ST'] =[lambda_dis['x'],lambda_dis['y']]


    ## 价值转移函数参数,根据相关性计算的转化比
    def trans(self,rel):
        a = 0.23535594
        b = 0.1164405
        return a*rel**b


    ## 根据模式以及主题生成每篇文章的价值系数
    def knowledge_gain_coef(self,topics,model):

        if model=='ST':
            lambda_list = self._topic_lambda_dis['ST'][0]
            lambda_probs = self._topic_lambda_dis['ST'][1]
            return np.random.choice(lambda_list,size=len(topics),p=lambda_probs,replace=True)

        elif model=='MT':
            ## 对主题以及位置进行计算
            t_num = defaultdict(list)
            for i,t in enumerate(topics):
                t_num[t].append(i)

            ## 对每一个主题
            lambdas = []
            indexes = []
            for t in t_num.keys():
                ins = t_num[t]
                lambda_list = self._topic_lambda_dis[t][0]
                lambda_probs = self._topic_lambda_dis[t][1]
                t_ls = np.random.choice(lambda_list,size=len(ins),p=lambda_probs,replace=True)

                lambdas.extend(t_ls)
                indexes.extend(ins)

            ## 根据index对lambdas进行排序
            return [lambdas[i] for i in sorted(range(len(indexes)),key = lambda x:indexes[x])]

    ## 根据个人价值列表进行论文的引用
    def cit_based_on_prob(self,articles,kgs,kg_probs):

        ### 参考文献数量定义为30为均值，5为平均差的正态分布
        num_ref = self.N_REF()
        if len(articles)<=num_ref:
            return articles,kgs

        ref_indexes = np.random.choice(range(len(articles)), size=num_ref, replace=False, p=kg_probs)

        refs = []
        ref_kgs = []
        for i in ref_indexes:
            refs.append(articles[i])
            ref_kgs.append(kgs[i])

        return refs,ref_kgs

    ## 该主题向其他主题进行价值转移
    def trans_values(self,topic,kg):
        ## 该主题向其他主题相关性
        topic_kgs = {}
        for t2 in self.topic_relevance_matrix[topic].keys():
            if topic==t2:
                continue

            rel = self.topic_relevance_matrix[topic][t2]
            value_coef = self.trans(rel)

            topic_kgs[t2] = value_coef*kg

        topic_kgs[topic] = kg
        return topic_kgs


    ## 作者从未读的论文中按照mode进行阅读论文
    def read_papers(self,_ALL_articles_ids,_article_kgs,_article_kg_probs,_read_indexes,PR,exclude_read):

        tn = len(_ALL_articles_ids)
        ## 确定需要阅读的论文数量在150到450篇论文之间
        rn = int(np.random.normal(300,50,1)[0])
        ## totoal num
        article_indexes = range(tn)

        if tn<rn:
            return article_indexes,_ALL_articles_ids,_article_kgs,_article_kg_probs

        ## 阅读所有论文
        if PR=='ALL':
            return article_indexes,_ALL_articles_ids,_article_kgs,_article_kg_probs

        else:
            _article_kg_probs_copy =copy.copy(_article_kg_probs)
            _same_probs = [1]*tn
            ## 是否在未读文献中进行
            if exclude_read:
                for article_index in _read_indexes:
                    _article_kg_probs_copy[article_index]=0
                    _same_probs[article_index] = 0

            _article_kg_probs_copy = _article_kg_probs_copy/np.sum(_article_kg_probs_copy)
            _same_probs = np.array(_same_probs)/float(np.sum(_same_probs))

            # print _article_kg_probs_copy
            # print article_indexes
            # print rn
            if PR=='PROP':
                ## 根据概率随机选择
                selected_indexes = np.random.choice(article_indexes,size=rn,p=_article_kg_probs_copy,replace=False)

            elif PR =='TOP':
                selected_indexes = sorted(article_indexes,key=lambda x:_article_kg_probs_copy[x],reverse=True)[:rn]

            elif PR == 'RND':
                ## 概率相同的
                selected_indexes = np.random.choice(article_indexes,size=rn,p=_same_probs,replace=False)

            _read_indexes = list(set(_read_indexes)|set(selected_indexes))
            # _read_papers = [_ALL_articles_ids[p]]
            _read_papers = []
            _read_kgs = []
            for i in _read_indexes:
                _read_papers.append(_ALL_articles_ids[i])
                _read_kgs.append(_article_kgs[i])

            _read_kg_prbs = np.array(_read_kgs)/np.sum(_read_kgs)

            return _read_indexes,_read_papers,_read_kgs,_read_kg_prbs



    ## 每篇论文平均参考文献数量
    ## N_REF = 30
    def N_REF(self):
        return int(np.random.normal(self.reference_number, 5, 1)[0])

    def gen_id(self):
        return str(uuid.uuid1())

    def ids_of_articles(self,num):
        return ['A'+self.gen_id() for i in range(num)]

    def add_norm_item(self,mu,sigma,not_neg=True,must_incre=False):


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
    def simulate_topic_relevance(self):
        print 'initalize topic relevance matrix ...'
        size= self.topic_number
        topic_rel_func = json.loads(open('topic_relevance.json').read())
        xs = topic_rel_func['x']
        ys = topic_rel_func['y']

        # print ys[0]

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
            # print i
            ##每一个主题的每一项
            specific_rel_list = []
            for j,rel in enumerate(basic_rel_list):
                ## 加入随机项，
                mi=False
                if j==0:
                    mi=True

                rel = self.add_norm_item(rel,rel*1,must_incre=mi)

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


    ## 根据生产力分布，进行论文数量抽样
    def sample_author_paper_num(self,num_of_past,num_of_new):

        new_prods =  np.random.choice(list(self.prod_new_xs),size=num_of_new,p=self.prod_new_probs,replace=True)
        past_prods = np.random.choice(list(self.prod_xs),size=num_of_past,p=self.prod_probs,replace=True)

        ##整体生产力分布
        prods = []
        prods.extend(past_prods)
        prods.extend(new_prods)

        return prods


    ##根据现有人数，增加作者
    def add_authors(self,t,s0,isfirst=False):
        ## 根据拟合指数增长参数[4.4786305  0.05994124]
        a = self.AUTHOR_INCREASE_FUNC_PARAS[0]
        b = self.AUTHOR_INCREASE_FUNC_PARAS[1]
        num_add = int(s0*a*np.exp(b*t))

        ## 初始的s0人需要加上
        if isfirst:
            num_add+=s0

        ## 每年进入的人上下浮动 0%~20%
        margin = int(np.random.normal(0,int(num_add*0.2),1)[0])
        num_add +=margin

        ### 每个人的声明周期进行模拟
        rses = np.random.choice(list(self.rs_xs),size=num_add,p=self.rs_probs,replace=True)
        return ['S_'+self.gen_id() for i in range(num_add)],rses

    ### 每一个作者生产力的随机项
    ### prods[ia],author_year_articles,author,state
    def random_pn(self,prod,author_year_articles,author,state):

        mean = self.author_mean_pn(author_year_articles,author)
        ## 加上一个以mean为期望，以1位均差的正态随机项
        prod = prod+int(np.random.normal(mean,1,1)[0])

        if prod<0:
            prod=0

        ## 如果是作者今年离开，那么作者至少发表一篇论文
        if state==-1 and prod==0:
            prod=1

        return prod

    def author_mean_pn(self,author_year_articles,author):

        year_dict = author_year_articles.get(author,{})

        values = year_dict.values()

        if len(values)==0:
            return 0
        else:
            return int(np.mean([len(v) for v in values]))

    ## 作者主题选择
    def author_select_topic(self,author_year_articles,author,prod):
        topics = []
        year_dict = author_year_articles.get(author,{})

        # topic_history = []
        topic_dict = defaultdict(int)
        ## 获得该作者的论文历史
        for year in year_dict.keys():
            # print year_dict[year]
            papers,ts = year_dict[year]
            for t in ts:
                topic_dict[t]+=1

        ## 每一次写文章，根据历史选择主题
        for i in range(prod):

            ## 如果第一次写论文 按照主题文章数量分布进行选择
            if len(topic_dict.keys())==0:
                topic = "T_{:}".format(np.random.choice(self.topic_xs,size=1,p=self.topic_probs,replace=True)[0])
            else:
                topic = None
                ## 如果不是第一次,选择新主题的概率是所有主题都不再写论文的概率
                for t in topic_dict.keys():
                    num = topic_dict[t]

                    cont_prob = self.topic_cont_pro[num]

                    ## 0继续 1离开 该主题 继续下一个主题判断
                    if np.random.choice([0,1],size=1,p=[cont_prob,1-cont_prob],replace=True)[0]==0:
                        topic = t
                        break

                ## 如果都不再继续选择之前所有的主题，选择新的主题
                if topic is None:
                    topic = "T_{:}".format(np.random.choice(self.topic_xs,size=1,p=self.topic_probs,replace=True)[0])


            ## topic 加入该作者的文章历史
            topic_dict[topic]+=1

            topics.append(topic)

        return topics



















