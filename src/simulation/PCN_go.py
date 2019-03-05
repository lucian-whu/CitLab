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
from param import PARAM
import copy


## 所有仿真
def simulate_CN(arg):

    modes = ['ST-ALL','ST-PROP','ST-TOP','ST-RND','MT-ALL','MT-PROP','MT-TOP','MT-RND']

    ## 参数对象
    paramObj = PARAM(arg)
    paramObj.init_param()

    ## 对作者数量的仿真
    year_news,year_ends = simulate_authors(paramObj)

    ## 对作者写论文的仿真

    author_year_articles = simulate_author_writting_papers(year_news,year_ends,paramObj)


    if arg.all_mode:
        ## 对引用论文的仿真
        for mode in modes:
            paramObj.set_mode(mode)
            print '========simulate mode:',mode
            simulate_citations(year_news,year_ends,author_year_articles,paramObj)
    else:
        simulate_citations(year_news,year_ends,author_year_articles,paramObj)

### 对论文数量过程进行仿真按照不同的模式进行引用
## --------------
## -------------
def simulate_citations(year_news,year_ends,author_year_articles,paramObj):

    # mode e.g. ST-ALL, 用于命名
    mode = paramObj.mode
    print '-----',mode
    ## 模型名 有两个模型ST, MT
    model = paramObj.MD
    ## 个人阅读规则 ALL, RND, TOP, PROP
    PR = paramObj.PR
    ## 主题数量
    TN = paramObj.topic_number
    ## 长度
    length = paramObj.length

    ## 每一个元素是论文的属性,作者,年份,引用,价值
    article_list = []

    ## 所有作者集合
    totals = set([])
    total_num_authors = 0

    ## 全部论文列表
    _ALL_articles_ids = []

    ## 每个主题下 论文对应的价值，"ST"为单主题假设下的价值,'MT-T1'是多主题假设下T1的价值列表
    _ALL_topic_kgs = defaultdict(list)

    ## 个人论文库,作者在各个主题阅读的论文id以及kg
    _author_topic_papers = defaultdict(lambda:defaultdict(list))

    print 'simulate wirting papers ...'
    ## 从第一年开始
    json_file = open('data/simulation/articles_jsons_{:}_{:}_{:}.txt'.format(mode,length,TN),'w+')

    for i in sorted(year_news.keys()):

        print '------ In total, %d articles ...'%len(_ALL_articles_ids),'time:',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

        _year_articles = []
        _year_topic_kgs = defaultdict(list)

        ## 该年的新人数量
        news = year_news[i]
        num_new = len(news)

        ## 写完论文后离开一部分人
        ends = year_ends[i]
        num_end = len(ends)

        au_states_list = list(totals|news)
        num_au = len(au_states_list)

        ## 剩余人数
        totals = totals|news
        totals = totals-ends
        total_num_authors=total_num_authors+num_new-num_end

        print 'year %d, %d new authors, %d authors left, reserved total %d' % (i,num_new,num_end,total_num_authors)

        for ia,author in enumerate(au_states_list):

            ### 获得一个用户一年要写的论文，以及这些论文对应的主题
            article_ids,topics = author_year_articles[author][i]

            ### --------------------------------
            ### 根据所选模式的不同进行价值增益生成
            ### 单主题下主题生成，多主题下加载对应主题的论文进行生成
            ### --------------------------------
            lambdas = paramObj.knowledge_gain_coef(topics,model)

            ## 对于每一篇论文来讲
            for aindex,aid in enumerate(article_ids):

                topic = topics[aindex]



                ## 第一年没有论文可以看，所以只能按照默认设置
                if i==1:
                    ## 参考文献文献数量设置为[]
                    ref_list = []
                    ## 知识增益设置为100
                    kg= paramObj.initial_value
                    _lambda_coef = 1
                    refv = 0

                else:

                    ## 主题模型选择该主题下的论文列表以及价值列表,以及该作者已经阅读过的论文
                    if model =='ST':
                        model_topic = 'ST'
                    elif model == 'MT':
                        model_topic = 'MT-{:}'.format(topic)

                    # print i,_ALL_topic_kgs.keys()
                    _article_kgs,_article_kg_probs = _ALL_topic_kgs[model_topic]

                    ## 第一年可能为空 [][][][]
                    _read_indexes,_read_papers,_read_kgs,_read_kg_probs = _author_topic_papers[aid].get(model_topic,[[],[],[],[]])
                    ## ---------
                    ## 个人阅读论文
                    ## ---------
                    ## 作者每次撰写论文都要进行知识储备，写一篇论文阅读500篇相关文献
                    ##
                    ##      I.读新的500篇
                    ##      II. 不管是否看过
                    ## ---------

                    ## 第一年没有文章读,从去年
                    _read_indexes,_read_papers,_read_kgs,_read_kg_probs =  paramObj.read_papers(_ALL_articles_ids,_article_kgs,_article_kg_probs,_read_indexes,PR, paramObj.exclude_read)
                    _author_topic_papers[aid][model_topic] = [_read_indexes,_read_papers,_read_kgs,_read_kg_probs]

                    ##在个人论文库中，根据kgs选择参考文献
                    ### -------------------------
                    ### 根据本论文的主题进行以及阅读进行引用
                    ### -------------------------
                    # print np.sum(_personal_kg_probs)
                    ref_list,ref_kgs = paramObj.cit_based_on_prob(_read_papers,_read_kgs,_read_kg_probs)

                    _lambda_coef = lambdas[aindex]
                    # ris,rsv = zscore_outlier(ref_kgs,3)
                    refv = np.mean(ref_kgs)

                    # kg = np.mean(ref_kgs)*lambdas[aindex]
                    ### 在这里，将本主题价值转化为其他主题价值
                    ## 首先是本主题的价值
                    kg = refv*_lambda_coef


                ## 将这个价值根据转换矩阵以及转换函数转化为各个主题的价值，并存入各主题的列表内
                ## ST以及MT的价值计算不同
                if model=='MT':
                    topic_values = paramObj.trans_values(topic,kg)
                    for t2 in topic_values.keys():
                        _year_topic_kgs['MT-{:}'.format(t2)].append(topic_values[t2])
                elif model=='ST':
                    _year_topic_kgs['ST'].append(kg)


                ## 存储该文章
                articleObj = {}
                articleObj['id'] = aid
                articleObj['lambda'] = _lambda_coef
                articleObj['refv'] = refv
                articleObj['author'] = author
                articleObj['kg'] = kg
                articleObj['refs'] = ref_list
                articleObj['year'] = i
                articleObj['topic'] = topic

                article_list.append(articleObj)

                _year_articles.append(aid)
                # _year_kgs.append(kg)


        ## 论文的id以及论文的增益
        _ALL_articles_ids.extend(_year_articles)
        # _ALL_topic_kgs.extend(_year_kgs)
        for topic in _year_topic_kgs.keys():
            _t_kgs,_t_kg_probs = _ALL_topic_kgs.get(topic,[[],[]])
            _t_kgs.extend(_year_topic_kgs[topic])
            _t_kg_probs = np.array(_t_kgs)/float(np.sum(_t_kgs))
            _ALL_topic_kgs[topic]=[_t_kgs,_t_kg_probs]

        ## 存储article_list中的论文
        lines = [json.dumps(a) for a in article_list]
        article_list = []


        print 'year %d, %d articles saved.'%(i,len(lines))
        json_file.write('\n'.join(lines)+'\n')

    json_file.close()
    print 'simulation done, {:} articles are writen.'.format(len(_ALL_articles_ids))


### 仿真作者写论文数量变化
##-------------
## 1. 加入作者为论文选择主题的步骤
## 2. 对主题数量的仿真结果可视化
##--------------
def simulate_author_writting_papers(year_news,year_ends,paramObj):

    total_num_authors = 0
    ## 初始作者
    totals = set([])
    attrs = []

    ## 作者在各年份发表的论文
    author_year_articles = defaultdict(lambda:defaultdict(list))

    ##记录所有的论文数量
    _ALL_articles_ids = []
    _ALL_articles_topics = []

    print 'simulate wirting papers ...'
    ## 从第一年开始
    for i in sorted(year_news.keys()):

        print '------ In total, %d articles ...'%len(_ALL_articles_ids),'time:',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

        _year_articles = []
        _year_topics = []

        ## 该年的新人数量
        news = year_news[i]
        num_new = len(news)

        ## 作者生产力的抽样，分剩余作者以及新作者两种，新作者第一年必须是大于等于1的
        prods = paramObj.sample_author_paper_num(total_num_authors+num_new)

        ## 写完论文后离开一部分人
        ends = year_ends[i]
        num_end = len(ends)

        ## 今年所有作者的状态列表，将要离开的-1，新人0，不离开 1
        ## 三个状态的主要作用是控制作者每年的论文数量，离开的人和新人的论文数量不能为空
        au_states_list = []
        ## 对去年论文判断是否有人今年离开
        for pa in totals:
            ## 今年要离开的人
            if pa in ends:
                au_states_list.append([pa,-1])
            ## 其他人
            else:
                au_states_list.append([pa,1])

        for pa in news:
            au_states_list.append([pa,0])


        ## 剩余人数
        totals = totals|news
        totals = totals-ends
        total_num_authors=total_num_authors+num_new-num_end

        attrs.append([i,total_num_authors,num_new,num_end])
        print 'year %d, %d new authors, %d authors left, reserved total %d' % (i,num_new,num_end,total_num_authors)

        ## 对于au_state_list中的作者
        num_au = len(au_states_list)
        # print num_au,total_num_authors
        for ia,(author,state) in enumerate(au_states_list):

            ## 根据作者往年生产力平均水平进行随机项的添加
            num_of_papers =  paramObj.random_pn(prods[ia],author_year_articles,author,state)
            ## 确定论文的ID
            article_ids = paramObj.ids_of_articles(num_of_papers)
            ##------------
            ## 对于所有的论文进行主题选择
            ##------------
            topics = paramObj.author_select_topic(author_year_articles,author,num_of_papers)

            ## article id
            author_year_articles[author][i] = [article_ids,topics]

            _year_articles.extend(article_ids)
            _year_topics.extend(topics)


        _ALL_articles_ids.extend(_year_articles)
        _ALL_articles_topics.extend(_year_topics)


    print 'total authors:',len(author_year_articles.keys()),'; number of articles:',len(_ALL_articles_ids)
    ## 保存作者年论文的json
    open('data/simulation/simulate_author_year_papers_{:}.json'.format(paramObj.length),'w').write(json.dumps(author_year_articles))

    print 'author simulate data saved to data/simulation/simulate_author_year_papers_{:}.json'.format(paramObj.length)

    ## 领域内作者变化曲线
    plt.figure(figsize=(5,4))
    years,totals,news,ends = zip(*attrs)

    plt.plot(years,news,label=u'新作者')
    plt.plot(years,ends,'--',label=u'离开作者')
    plt.plot(years,totals,label=u'剩余作者总数')

    plt.xlabel(u'年份',fontproperties='SimHei')
    plt.ylabel(u'作者数量',fontproperties='SimHei')
    plt.yscale('log')
    plt.legend(prop={'family':'SimHei','size':8})
    plt.tight_layout()

    plt.savefig('fig/simulation/simulated_author_num_{:}.png'.format(paramObj.length),dpi=400)
    print 'author num simulation saved to fig/simulation/simulated_author_num_{:}.png'.format(paramObj.length)


    ## 主题论文数量分布
    plt.figure(figsize=(14,2.8))
    # years,totals,news,ends = zip(*attrs)

    topic_counter = Counter(_ALL_articles_topics)
    xs = []
    ys = []
    labels = []
    for i,p in enumerate(sorted(topic_counter.keys(),key=lambda x:topic_counter[x],reverse=True)):
        xs.append(i+1)
        labels.append(p)
        ys.append(topic_counter[p])

    plt.bar(xs,ys)
    plt.xticks(xs,labels,rotation=-90)
    plt.xlabel(u'主题',fontproperties='SimHei')
    plt.ylabel(u'文章数量',fontproperties='SimHei')
    plt.ylim(1,ys[0]+10000)
    plt.yscale('log')
    plt.legend(prop={'family':'SimHei','size':8})
    plt.tight_layout()

    plt.savefig('fig/simulation/simulated_topic_num_{:}.png'.format(paramObj.length),dpi=400)
    print 'author num simulation saved to fig/simulation/simulated_topic_num_{:}.png'.format(paramObj.length)


    return author_year_articles

### 仿真作者数量变化
def simulate_authors(paramObj):
    ## 每年新人的数量
    year_news = defaultdict(set)
    ## 每年离开人的数量
    year_ends = defaultdict(set)

    ## 需要对数据进行存储
    ## 初始人数
    s0 = paramObj.author_number
    LENGTH = paramObj.length

    print 'initail author num %d , simulation length %d ...' % (s0,LENGTH)

    print 'simulated authors and author research life ...'
    for i in range(1,LENGTH+1):
        print 'year %d ..'% i
        ## 根据模拟的函数进行增加人数，以及每个人对应的声明周期，最少一年
        authors,rses  = paramObj.add_authors(i,s0,i==1)
        for j,a in enumerate(authors):

            ## 这个人的研究周期， 从1开始，最长40年
            rs = rses[j]

            ## 开始年份为i
            start = i
            ## 结束年份为i+rs-1， 研究周期为1 代表当年结束
            end = start+rs-1

            year_news[start].add(a)
            year_ends[end].add(a)

    return year_news,year_ends

if __name__ == '__main__':

    parser = argparse.ArgumentParser(usage='python %(prog)s [options] \n  exp. python -m ST-NA-ALL -f impact')

    ## 模型选择
    parser.add_argument('-m','--mode',dest='mode',default='ST-ALL',type=str,choices=['ST-ALL','ST-PROP','ST-TOP','ST-RND','MT-ALL','MT-PROP','MT-TOP','MT-RND'],help='select model prefered to use.')
    ## 影响因素选择
    parser.add_argument('-f','--factor',dest='factor',default='NA',help='select impact factor.')
    ## 仿真年份长度
    parser.add_argument('-l','--length',dest='length',type=int,default=100,help='set length of simulation, default is 50.')
    ## 主题数量
    parser.add_argument('-t','--topic_number',dest='topic_number',type=int,default=50,help='set number of topics, default is 50.')
    ## 参考文献数量
    parser.add_argument('-r','--reference_number',dest='reference_number',type=int,default=30,help='set number of references, default is 30.')
    ## 作者数量
    parser.add_argument('-a','--author_number',dest='author_number',type=int,default=10,help='set initail number of authors, default is 10.')
    ## 初始文献熟练
    parser.add_argument('-i','--initial_value',dest='initial_value',type=int,default=100,help='set initail value of papers, default is 100.')
    ## 排除已读
    parser.add_argument('-x','--exclude_read',action='store_true',dest='exclude_read',default=True,help='whether exclude read papers.')

    ## 排除已读
    parser.add_argument('-A','--all_mode',action='store_true',dest='all_mode',default=False,help='all mode used to compare all modes')

    arg = parser.parse_args()

    simulate_CN(arg)



