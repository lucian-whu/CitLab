#coding:utf-8
'''
验证以及画图

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
from scipy.stats import pearsonr


### 根据生成的article，对仿真结果进行验证
def validate_all_simulations(mode,length,tn):

    ## 存数据
    outpath = 'fig/validation/{:}/{:}'.format(mode,length)
    isExists=os.path.exists(outpath)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        os.makedirs(outpath)

    print 'MODE:',mode,',Length:',length
    tit = mode

    ref_dict = defaultdict(int)
    ##文献在各年份被引次数
    ref_year_dict = defaultdict(lambda:defaultdict(int))

    progress = 0
    ## 论文发表年
    pid_year = {}
    ## 论文的真实价值
    pid_kg = {}
    ##kg的分布
    year_kgs = defaultdict(list)
    ##作者引用
    author_ref_num = defaultdict(lambda:defaultdict(int))
    ##作者数量
    author_paper_num = defaultdict(int)

    ## 价值增益的大小分布
    all_kgs = []
    all_years = []

    num_ref_relations = 0

    for line in open('data/simulation/articles_jsons_{:}_{:}_{:}.txt'.format(mode,length,tn)):
        progress+=1
        if progress%10000==0:
            print progress
        try:
            article = json.loads(line.strip())
        except:
            print line.strip()
            continue
        year = article.get('year',-1)
        pid = article['id']
        kg = article['kg']
        author_id = article['author']
        ref_list = article['refs']

        num_ref_relations+=len(ref_list)

        for ref in ref_list:
            ref_dict[ref]+=1
            author_ref_num[author_id][ref]+=1
            ref_year_dict[ref][year]+=1

        year_kgs[year].append(kg)
        author_paper_num[author_id]+=1
        all_kgs.append(kg)

        pid_kg[pid] = kg
        pid_year[pid] = year

    print 'total author num:',len(author_paper_num.keys())
    print 'number of papers:',len(pid_kg.keys())
    print 'total number of citation relations:',num_ref_relations

    '''
    ## =====================
    3.3.6.1 引用次数分布
    ## =====================

    '''

    ## 引文分布
    citation_nums = []

    high_cited_articles = []
    year_num = defaultdict(int)

    for ref in ref_dict.keys():

        cit_num = ref_dict[ref]

        if cit_num > 100:
            high_cited_articles.append(ref)

        year = pid_year[ref]
        year_num[year]+=1

        citation_nums.append(cit_num)

    print '%d articles has citations.'%len(citation_nums)

    fit = powerlaw.Fit(citation_nums)

    print fit.power_law.xmin
    print fit.power_law.alpha
    print '---------------','powerlaw vs. exponential:',fit.distribution_compare('power_law', 'exponential')

    ## citation nums
    total_num = len(citation_nums)

    low_num = len([v for v in citation_nums if v <10])
    mid_num = len([v for v in citation_nums if v>=10 and v <100])
    high_num = len([v for v in citation_nums if v>=100])
    print '========= MODE:',mode
    print 'low:',low_num/float(total_num)
    print 'midum:',mid_num/float(total_num)
    print 'high:',high_num/float(total_num)

    num_dict = Counter(citation_nums)
    xs = []
    ys = []
    for num in sorted(num_dict):
        xs.append(num)
        ys.append(num_dict[num])
    ys = np.array(ys)/float(np.sum(ys))

    plt.figure(figsize=(5,4))

    plt.plot(xs,ys)
    plt.xlabel('$\#(c_i)$')
    plt.ylabel('$p(\#(c_i))$')
    plt.title(tit,fontproperties='SimHei')
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()

    plt.savefig(outpath+'/simulated_citation_distribtuiotn_{:}_{:}.pdf'.format(mode,length),dpi=400)
    print 'citation distribution saved to simulated_citation_distribtuiotn.pdf'


    '''
    ======================
    3.3.6.3 学术文献的真实价值分析
    ======================

    '''

    ## ----------
    ## 1. 价值增益分布, 总体价值系数
    ## ----------

    _100_kgs = [kg for kg in all_kgs if kg>100]
    _200_kgs = [kg for kg in all_kgs if kg>200]

    print len(_100_kgs),len(_200_kgs),len(all_kgs)

    plt.figure(figsize=(5,4))

    plt.hist(all_kgs,bins=100,rwidth=0.5)
    plt.title(tit,fontproperties='SimHei')
    plt.plot([100]*10,np.linspace(0,10000,10),'--',c='r',linewidth=2, label=u'初始价值增益')
    plt.xlabel(u'价值增益',fontproperties='SimHei')
    plt.ylabel(u'论文数量',fontproperties='SimHei')
    # plt.xscale('log')
    plt.title(tit,fontproperties='SimHei')
    plt.yscale('log')
    plt.legend(prop={'family':'SimHei','size':8})

    plt.tight_layout()

    plt.savefig(outpath+'/simulated_kg_dis_{:}_{:}.pdf'.format(mode,length),dpi=400)

    print 'kg dis saved tooutpath+ /simulated_kg_dis.pdf'


    ## -------------
    ## 2. 随着年份的增加 学术论文的平均价值
    ## -------------
    xs = []
    ys = []
    for year in sorted(year_kgs.keys()):
        xs.append(year)
        ys.append(np.mean(year_kgs[year]))

    plt.figure(figsize=(5,4))
    plt.title(tit,fontproperties='SimHei')
    plt.plot(xs[1:],ys[1:])
    plt.title(tit,fontproperties='SimHei')
    plt.xlabel(u'年份',fontproperties='SimHei')
    plt.ylabel(u'价值',fontproperties='SimHei')
    plt.tight_layout()
    plt.savefig(outpath+'/simulated_year_average_kg_{:}_{:}.pdf'.format(mode,length),dpi=600)
    print 'fig saved tooutpath+ /simulated_year_average_kg_{:}_{:}.pdf'.format(mode,length)


    ## -------
    ## 3. 大于100的价值的文献的年份分布
    ## -------

    kg_years = []
    all_kgs = []
    all_ccs = []

    cut_kgs = []
    cut_ccs = []
    for pid in pid_kg.keys():
        kg = pid_kg[pid]

        cc = ref_dict[pid]
        # if cc>1:
        #     print cc
        ## 记录散点图关系的
        all_kgs.append(kg)
        all_ccs.append(cc)

        year = pid_year[pid]

        if year <length/2:
            cut_kgs.append(kg)
            cut_ccs.append(cc)

        if kg >=100:
            year = pid_year[pid]
            kg_years.append(year)

    # kg_year_counter = Counter(kg_years)
    # xs = []
    # ys = []
    # for year in sorted(kg_year_counter.keys()):
    #     xs.append(year)
    #     ys.append(kg_year_counter[year])

    # plt.figure(figsize=(5,4))

    # plt.plot(xs[1:],ys[1:],label=u'高价值论文')
    # plt.xlabel(u'年份',fontproperties='SimHei')
    # plt.ylabel(u'文章数量',fontproperties='SimHei')

    # plt.tight_layout()
    # plt.savefig(outpath+'/simulated_g100_kg_year_dis_{:}_{:}.pdf'.format(mode,length),dpi=600)
    # print 'fig saved tooutpath+ /simulated_g100_kg_year_dis_{:}_{:}.pdf'.format(mode,length)

    # ## -----
    # ## 4.高被引论文的分布
    # ## -----

    years = []
    for pid in high_cited_articles:
        years.append(pid_year[pid])

    year_dis = Counter(years)
    xs = []
    ys = []
    high_ys = []
    hr_ys = []
    for year in sorted(year_num.keys()):
        xs.append(year)
        ys.append(year_num[year])
        high_ys.append(year_dis[year])

        hr_ys.append(year_dis[year]/float(year_num[year]))

    # plt.figure(figsize=(5,4))

    fig,ax = plt.subplots(1,1,figsize=(5,4))

    ax.plot(xs,ys,label=u'文章数量',linewidth=1)
    print xs
    print ys
    print high_ys
    ax.plot(xs,high_ys,label=u'高被引论文',linewidth=1,c='g')
    ax.set_xlabel(u'发表年份',fontproperties='SimHei')
    ax.set_ylabel(u'文章数量',fontproperties='SimHei')
    ax.set_ylim(1,np.max(ys)+1000)
    ax.set_yscale('log')
    ax.legend(prop={'family':'SimHei','size':8})
    ax.set_title(tit)
    # plt.title(u'APS')

    ax2 = ax.twinx()
    ax2.plot(xs,hr_ys,'--',label=u'高被引比例',c='r')
    ax2.set_ylabel(u'高被引文文章比例',fontproperties='SimHei')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.legend(prop={'family':'SimHei','size':8})

    plt.tight_layout()
    plt.savefig(outpath+'/simulated_high_cited_paper_year_dis_{:}_{:}.pdf'.format(mode,length),dpi=400)

    print outpath+'/simulated_high_cited_paper_year_dis_{:}_{:}.pdf'.format(mode,length)

    ## -------
    ## 5.kg与引用次数的关系
    ## ------
    ps1 =  pearsonr(all_kgs,all_ccs)
    print '全部数据的pearson相关系数:',ps1
    ps2 =  pearsonr(cut_kgs,cut_ccs)
    print '节选数据的pearson相关系数:',ps2

    fig,axes = plt.subplots(1,2,figsize=(10,4))

    ax0 = axes[0]
    ax0.plot(all_kgs,all_ccs,'o',alpha=0.7,label=u'皮尔逊相关系数:{:.4f}'.format(ps1[0]))
    ax0.set_xlabel(u'真实价值',fontproperties='SimHei')
    ax0.set_ylabel(u'引用次数',fontproperties='SimHei')
    ax0.legend(prop={'family':'SimHei','size':16})

    ax1 = axes[1]
    ax1.plot(cut_kgs,cut_ccs,'o',alpha=0.7,label=u'皮尔逊相关系数:{:.4f}'.format(ps2[0]))
    ax1.set_xlabel(u'真实价值',fontproperties='SimHei')
    ax1.set_ylabel(u'引用次数',fontproperties='SimHei')
    ax1.legend(prop={'family':'SimHei','size':16})

    plt.tight_layout()
    plt.savefig(outpath+'/simulated_correlate_value_citations_{:}_{:}.png'.format(mode,length),dpi=800)
    print 'fig saved tooutpath+ /simulated_correlate_value_citations_{:}_{:}.png'.format(mode,length)

    # return

    '''
    ===============
    3.3.6.4 学术文献的声明周期分析
    ===============
    '''
    year_lls = defaultdict(list)
    year_lls_norm = defaultdict(list)

    year_hfs = defaultdict(list)
    year_hfs_norm = defaultdict(list)
    for ref in ref_year_dict.keys():

        year_dict = ref_year_dict[ref]

        years = year_dict.keys()

        lifelength = np.max(years)-np.min(years)+1

        year = pid_year[ref]

        year_lls[year].append(lifelength)

        year_lls_norm[year].append(lifelength/float(length-year-1))

        total = np.sum(year_dict.values())
        num=0
        hl = 0
        for c_year in sorted(year_dict.keys()):
            num+= year_dict[c_year]

            if num/float(total)>0.5:
                hl=c_year-year
                break

        year_hfs[year].append(hl)

        year_hfs_norm[year].append(hl/float(length-year))


    ## --------------------
    ## 1. 不同年份发表的学术文献平均生命周期长度
    ## --------------------
    xs = []
    ys = []
    norm_ys = []
    for year in year_lls.keys():
        xs.append(year)
        ys.append(np.mean(year_lls[year]))
        norm_ys.append(np.mean(year_lls_norm[year]))

        print year_lls[year][:10]
        print year_lls_norm[year][:10]

    fig,ax = plt.subplots(figsize=(5,4))

    ax.plot(xs,ys)

    ax.set_xlabel(u'年份',fontproperties='SimHei')
    ax.set_ylabel(u'$ML$',fontproperties='SimHei')
    ax.set_title(tit,fontproperties='SimHei')

    ax2 = ax.twinx()
    ax2.plot(xs,norm_ys,'--',c='r')
    ax2.set_xlabel(u'年份',fontproperties='SimHei')
    ax2.set_ylabel(u'$ML_{norm}$',fontproperties='SimHei')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_title(tit,fontproperties='SimHei')

    plt.tight_layout()
    plt.savefig(outpath+'/simulated_life_length_norm_dis_over_year_{:}_{:}.pdf'.format(mode,length),dpi=400)
    print 'fig saved tooutpath+ /simulated_life_length_norm_dis_over_year.pdf'

    ## --------------------
    ## 3. 不同年份发表的学术文献半衰期
    ## --------------------
    xs = []
    ys = []
    norm_ys = []
    for year in year_hfs.keys():
        xs.append(year)
        ys.append(np.mean(year_hfs[year]))
        norm_ys.append(np.mean(year_hfs_norm[year]))


    fig,ax = plt.subplots(figsize=(5,4))

    ax.plot(xs,ys)

    ax.set_xlabel(u'年份',fontproperties='SimHei')
    ax.set_ylabel(u'$HL$',fontproperties='SimHei')
    ax.set_title(tit,fontproperties='SimHei')

    ax2 = ax.twinx()
    ax2.plot(xs,norm_ys,'--',c='r')
    ax2.set_xlabel(u'年份',fontproperties='SimHei')
    ax2.set_ylabel(u'$HL_{norm}$',fontproperties='SimHei')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_title(tit,fontproperties='SimHei')

    plt.tight_layout()
    plt.savefig(outpath+'/simulated_hf_dis_over_year_{:}_{:}.pdf'.format(mode,length),dpi=400)
    print 'fig saved tooutpath+ /simulated_hf_dis_over_year.pdf'



    ## ----------------------
    ## 4. 随机选择20个高被引论文进行可视化
    ## ----------------------
    selected_highs=np.random.choice(high_cited_articles,size=20,replace=False)

    fig,axes = plt.subplots(4,5,figsize=(12.5,10))
    for hi,ref in enumerate(selected_highs):

        ax = axes[hi/5,hi%5]

        year_dict = ref_year_dict[ref]
        xs = []
        ys = []

        tn = 0
        print 'year:%d, life:%d-%d' %(pid_year[ref],year_dict.keys()[0],year_dict.keys()[-1])
        for i,year in enumerate(sorted(year_dict.keys())):

            num = year_dict[year]



            xs.append(i)
            ys.append(num)

            tn+=num

        ax.plot(xs,ys)
        ax.set_xlabel(u'发表年份',fontproperties='SimHei')
        ax.set_ylabel(u'引用次数',fontproperties='SimHei')
        ax.set_title('%s(%d)'%(pid_year[ref],tn))

    plt.tight_layout()
    plt.savefig(outpath+'/simulated_high_life_length_dis_over_year_{:}_{:}.pdf'.format(mode,length),dpi=800)
    print 'fig saved to outpath+ /simulated_high_life_length_dis_over_year.pdf'



## 对多主题模型中的分布进行主题层次分析
## 主要包括两个：
##  1。 主题层次的引用次数分布
##  2. 主题层次的价值变化，高被引
def valid_mt_distribution(mode,length,tn):
    ## 存数据
    outpath = 'fig/validation/{:}/{:}'.format(mode,length)
    isExists=os.path.exists(outpath)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        os.makedirs(outpath)

    print 'MODE:',mode,',Length:',length
    tit = mode

    ref_dict = defaultdict(int)
    ##文献在各年份被引次数
    ref_year_dict = defaultdict(lambda:defaultdict(int))

    progress = 0
    ## 论文发表年
    pid_year = {}
    ## 论文的真实价值
    pid_kg = {}
    ##kg的分布
    year_kgs = defaultdict(list)
    ##作者引用
    author_ref_num = defaultdict(lambda:defaultdict(int))
    ##作者数量
    author_paper_num = defaultdict(int)

    ## 论文的主题
    pid_topic = {}
    pid_topic_num = defaultdict(lambda:defaultdict(int))

    ## 主题数量
    topic_papers = defaultdict(list)

    ## 价值增益的大小分布
    all_kgs = []
    all_years = []

    num_ref_relations = 0

    for line in open('data/simulation/articles_jsons_{:}_{:}_{:}.txt'.format(mode,length,tn)):
        progress+=1
        if progress%10000==0:
            print progress
        try:
            article = json.loads(line.strip())
        except:
            print line.strip()
            continue
        year = article.get('year',-1)
        pid = article['id']
        kg = article['kg']
        author_id = article['author']
        ref_list = article['refs']
        topic = article['topic']

        pid_topic[pid] = topic

        topic_papers[topic].append(pid)


        num_ref_relations+=len(ref_list)

        for ref in ref_list:
            ref_dict[ref]+=1
            author_ref_num[author_id][ref]+=1
            ref_year_dict[ref][year]+=1

            pid_topic_num[pid][topic]+=1


        year_kgs[year].append(kg)
        author_paper_num[author_id]+=1
        all_kgs.append(kg)

        pid_kg[pid] = kg
        pid_year[pid] = year

    print 'total author num:',len(author_paper_num.keys())
    print 'number of papers:',len(pid_kg.keys())
    print 'total number of citation relations:',num_ref_relations

    # fig,axes = plt.subplots(2,5,figsize=(17,5.6))


    top_10_xys = {}
    top_10_year_xys = {}
    ## 对前十的主题的论文引用分布进行绘制
    for i,topic in enumerate(sorted(topic_papers.keys(),key= lambda x:len(topic_papers[x]),reverse=True)[:10]):

        ## 这个领域的论文
        papers =  topic_papers[topic]

        ## 该主题下每年发表的论文
        year_papers = defaultdict(list)

        ## 统计每一篇论文的引用次数
        num_list = []
        for pid in papers:
            year = pid_year[pid]
            topic = pid_topic[pid]
            num = ref_dict[pid]
            num_list.append(num)

            year_papers[year].append(pid)


        ## 引用次数分布
        num_counter = Counter(num_list)

        cs_xs = []
        cs_ys = []

        for num in sorted(num_counter.keys()):
            if num <=0:
                continue
            cs_xs.append(num)
            cs_ys.append(num_counter[num])

        top_10_xys[topic]=[cs_xs,cs_ys]


        year_xs = []
        year_tn = []
        year_hn = []
        year_hp = []
        ## 论文曲线，高被引曲线，高被比例
        for year in sorted(year_papers.keys()):
            ## 不要第一年
            if year==1:
                continue
            _num = len(year_papers[year])
            _high_num = len([p for p in year_papers[year] if ref_dict[p]>100])

            # print _high_num
            year_xs.append(year)
            year_tn.append(_num)
            year_hn.append(_high_num)
            year_hp.append(float(_high_num)/_num)


        top_10_year_xys[topic] = [year_xs,year_tn,year_hn,year_hp]

    fig,axes = plt.subplots(2,5,figsize=(14,5.6))

    for i,topic in enumerate(sorted(topic_papers.keys(),key= lambda x:len(topic_papers[x]),reverse=True)[:10]):

        ax = axes[i/5,i%5]

        cs_xs,cs_ys = top_10_xys[topic]

        ax.plot(cs_xs,cs_ys)

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlabel(u'$\#(c_i)$')
        ax.set_ylabel(u'$p(\#(c_i))$')

        ax.set_title('{:}({:})'.format(topic,len(topic_papers[topic])))

    plt.tight_layout()

    plt.savefig(outpath+'/top_10_topic_citation_dis_{:}_{:}.png'.format(mode,length),dpi=800)

    print 'saved to',outpath+'/top_10_topic_citation_dis_{:}_{:}.png'.format(mode,length)


    fig,axes = plt.subplots(2,5,figsize=(17,5.6))

    for i,topic in enumerate(sorted(topic_papers.keys(),key= lambda x:len(topic_papers[x]),reverse=True)[:10]):

        ax = axes[i/5,i%5]

        year_xs,year_tn,year_hn,year_hp = top_10_year_xys[topic]

        # print year_tn

        ax.plot(year_xs,year_tn,label=u'所有论文')
        ax.plot(year_xs,year_hn,label=u'高被引论文',c='g')
        # ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title('{:}({:})'.format(topic,len(topic_papers[topic])))
        ax.legend(prop={'family':'SimHei','size':5})
        ax.set_xlabel(u'年份',fontproperties='SimHei')
        ax.set_ylabel(u'文章数量',fontproperties='SimHei')

        ax2 = ax.twinx()
        ax2.plot(year_xs,year_hp,'--',label=u'高被引论文比例',c='r')
        ax2.set_ylabel(u'比例',fontproperties='SimHei')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.legend(prop={'family':'SimHei','size':5})

    plt.tight_layout()

    plt.savefig(outpath+'/top_10_topic_year_dis_{:}_{:}.png'.format(mode,length),dpi=800)

    print 'saved to',outpath+'/top_10_topic_year_dis_{:}_{:}.png'.format(mode,length)





if __name__ == '__main__':
    # simulated_data_viz()
    # modes =['ST-ALL','MT-ALL','ST-PROP','MT-PROP','ST-RND','MT-RND','ST-TOP','MT-TOP']
    modes =['ST-ALL']

    length = 100
    tn = 50
    for mode in modes:
        validate_all_simulations(mode,length,tn)

    # valid_mt_distribution('MT-ALL',100,50)


