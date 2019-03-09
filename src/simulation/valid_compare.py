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

DATA_PATH = '/Users/huangyong/Workspaces/Study/datasets/APS/aps-dataset-metadata-2016'
AUTHOR_JSON_PATH = 'data/author_year_articles.json'
NUM_AUTHOR_DIS_PATH = 'data/author_num_dis.json'

def compare_plots():
    author_year_articles = json.loads(open(AUTHOR_JSON_PATH).read())

    ## 所有作者的文章总数量
    tnas = []

    ## 领域内作者总数量
    year_an = defaultdict(int)

    ## 领域内文章总数量
    year_pn = defaultdict(list)

    ## 对于每一位作者来讲
    for author in author_year_articles.keys():

        total_num_of_articles = 0
        ## 每一年
        for i,year in enumerate(sorted(author_year_articles[author].keys(),key=lambda x:int(x))):

            ##第一年是作者进入的年
            if i==0:
                year_an[int(year)]+=1

            ## 文章数量
            num_of_articles = len(author_year_articles[author][year])

            total_num_of_articles+=num_of_articles

            year_pn[int(year)].append(num_of_articles)

        tnas.append(total_num_of_articles)

    ## 随着时间的增长领域内论文总数量
    xs = []
    ys = []
    an_ys = []
    total_pn = 0
    total_an = 0
    for year in sorted(year_pn.keys()):
        xs.append(year)
        total_pn+=np.sum(year_pn[year])

        total_an += year_an[year]
        an_ys.append(total_an)
        ys.append(total_pn)


    plt.figure(figsize=(5,4))

    plt.plot(xs,ys,label=u'文章总数')
    plt.plot(xs,an_ys,'--',label=u'作者总数')

    plt.xlabel(u'年份',fontproperties='SimHei')
    plt.ylabel(u'论文数量',fontproperties='SimHei')
    plt.title('APS')
    # plt.xscale('log')
    plt.yscale('log')
    plt.legend(prop={'family':'SimHei','size':8})

    plt.tight_layout()
    plt.savefig('fig/compare_pn.pdf',dpi=400)
    print 'simulation of total papers saved to fig/compare_pn.pdf'

    ## 画出作者的文章总数量分布
    tn_dict = Counter(tnas)
    xs = []
    ys = []
    for tn in sorted(tn_dict.keys()):

        xs.append(tn)
        ys.append(tn_dict[tn])

    print xs,ys
    plt.figure(figsize=(5,4))
    plt.plot(xs,ys,'o',fillstyle='none')
    plt.xlabel(u'作者文章总数量',fontproperties='SimHei')
    plt.ylabel(u'作者数',fontproperties='SimHei')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('APS')
    plt.tight_layout()
    plt.savefig('fig/compare_tn.pdf',dpi=400)
    print 'data saved to fig/compare_tn.pdf'


## 作者的被引次数分布
## 51万
def compare_citation_dis():

    pid_cn = json.loads(open('data/pid_all_cn.json').read())

    print len(pid_cn.keys())

    values = pid_cn.values()

    fit = powerlaw.Fit(values)

    print fit.power_law.xmin

    print fit.power_law.alpha

    print 'compare:',fit.distribution_compare('power_law', 'exponential')


    ## total num
    total_num = len(values)

    low_num = len([v for v in values if v <10])
    mid_num = len([v for v in values if v>=10 and v <100])
    high_num = len([v for v in values if v>=100])

    print 'low:',low_num/float(total_num)
    print 'midum:',mid_num/float(total_num)
    print 'high:',high_num/float(total_num)


    cn_counter = Counter(values)

    xs = []
    ys = []
    for cn in sorted(cn_counter.keys()):
        xs.append(cn)
        ys.append(cn_counter[cn])

    plt.figure(figsize=(5,4))

    ys = np.array(ys)/float(np.sum(ys))

    plt.plot(xs,ys)

    plt.xlabel('$\#(c_i)$')
    plt.ylabel('$p(\#(c_i))$')
    plt.title('APS')
    plt.xscale('log')

    plt.yscale('log')

    plt.tight_layout()

    plt.savefig('fig/validation/compare_citation_dis.pdf',dpi=800)

    print 'fig saved to fig/validation/compare_citation_dis.pdf.'

## 学术文献的平均价值的变化
def average_value_over_year():
    ## 文献ID C10
    pid_cn = json.loads(open('data/pid_cn.json').read())
    ##pid all cn
    pid_all_cn = json.loads(open('data/pid_all_cn.json').read())
    ## 文献发表年份
    pid_year = json.loads(open('data/paper_year.json').read())

    print 'start to ..'

    year_pns = defaultdict(list)
    progress = 0
    for pid in pid_all_cn.keys():

        year= pid_year.get(pid,-1)

        if year==-1:
            continue

        # if int(year)<1960:
        #     continue

        # if int(year)>2006:
        #     continue

        progress+=1

        if progress%10000==0:
            print progress

        # tn = pid_all_cn[tn]

        # cn = pid_cn.get(pid,0)

        year_pns[year].append(pid)

    xs = []
    ys = []
    high_ys = []
    hr_ys = []
    c10_ys = []
    for year in sorted(year_pns.keys(),key=lambda x:int(x)):
        xs.append(int(year))
        ys.append(len(year_pns[year]))

        dois = [pid_all_cn.get(doi,0) for doi in year_pns[year] if pid_all_cn.get(doi,0)>=100]

        c10s = [pid_cn.get(doi,0) for doi in year_pns[year]]

        avg_c10 = np.mean(c10s)

        c10_ys.append(avg_c10)

        high_num = len(dois)
        high_ys.append(high_num)
        hr = high_num/float(len(year_pns[year]))

        hr_ys.append(hr)

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
    ax.set_title('APS')
    # plt.title(u'APS')

    ax2 = ax.twinx()
    ax2.plot(xs,hr_ys,'--',label=u'高被引比例',c='r')
    ax2.set_ylabel(u'高被引文文章比例',fontproperties='SimHei')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.legend(prop={'family':'SimHei','size':8})

    plt.tight_layout()
    plt.savefig('fig/validation/compare_year_pn.pdf',dpi=800)

    print 'fig/validation/compare_year_pn.pdf'


    plt.figure(figsize=(5,4))
    plt.plot(xs[:-10],c10_ys[:-10])
    plt.xlabel(u'年份',fontproperties='SimHei')
    plt.ylabel(u'平均$c_{10}$', fontproperties='SimHei')
    plt.title('APS')
    plt.ylim(0,20)
    plt.plot(xs[:-10],[10]*len(xs[:-10]),'--',c='r',label='$y=10$')
    plt.legend(prop={'family':'SimHei','size':8})

    plt.tight_layout()

    plt.savefig('fig/validation/compare_average_c10.pdf',dpi=800)
    print 'fig saved to fig/validation/compare_average_c10.pdf'

## 生命周期分布
def compare_lifespan():

    pid_refs = json.loads(open('data/pid_all_refs.json').read())

    pid_all_cn = json.loads(open('data/pid_all_cn.json').read())

    pid_year = json.loads(open('data/paper_year.json').read())

    ref_year_num = defaultdict(lambda:defaultdict(int))

    high_papers = []
    for pid in pid_refs.keys():

        if pid_all_cn.get(pid,0)>1000:
            high_papers.append(pid)

        for ref in pid_refs[pid]:

            year = int(pid_year[pid])

            ref_year_num[ref][year]+=1

    year_lls = defaultdict(list)
    year_lls_norm = defaultdict(list)

    year_hfs = defaultdict(list)
    year_hfs_norm = defaultdict(list)
    for pid in ref_year_num.keys():
        year_num = ref_year_num[pid]
        years = year_num.keys()

        year = int(pid_year[pid])

        lifespan = np.max(years)-np.min(years)+1

        year_lls[year].append(lifespan)

        year_lls_norm[year].append(lifespan/float(2016-year+1))

        total = float(np.sum(year_num.values()))
        num = 0
        hf=0
        for y in sorted(year_num.keys(),key=lambda x:int(x)):
            num+=year_num[y]

            if num/total>0.5:
                hf=y-year+1
                break
        year_hfs[year].append(hf)
        year_hfs_norm[year].append(hf/((2016-year)+1))

    ## 随着时间生命周期长度的变化
    xs = []
    ys = []
    norm_ys = []
    for year in year_lls.keys():

        if year < 1960:
            continue

        xs.append(year)
        ys.append(np.mean(year_lls[year]))
        norm_ys.append(np.mean(year_lls_norm[year]))


    fig,ax = plt.subplots(figsize=(5,4))

    ax.plot(xs,ys)

    ax.set_xlabel(u'年份',fontproperties='SimHei')
    ax.set_ylabel(u'$ML$',fontproperties='SimHei')
    ax.set_title('APS',fontproperties='SimHei')

    ax2 = ax.twinx()
    ax2.plot(xs,norm_ys,'--',c='r')
    ax2.set_xlabel(u'年份',fontproperties='SimHei')
    ax2.set_ylabel(u'$ML_{norm}$',fontproperties='SimHei')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_title('APS',fontproperties='SimHei')

    plt.tight_layout()
    plt.savefig('fig/validation/compare_ll.pdf',dpi=800)
    print 'fig saved to fig/validation/compare_ll.pdf'

    xs = []
    ys = []
    norm_ys = []
    for year in year_hfs.keys():
        if year < 1960:
            continue

        xs.append(year)
        ys.append(np.mean(year_hfs[year]))
        norm_ys.append(np.mean(year_hfs_norm[year]))


    fig,ax = plt.subplots(figsize=(5,4))

    ax.plot(xs,ys)

    ax.set_xlabel(u'年份',fontproperties='SimHei')
    ax.set_ylabel(u'$HL$',fontproperties='SimHei')
    ax.set_title('APS',fontproperties='SimHei')

    ax2 = ax.twinx()
    ax2.plot(xs,norm_ys,'--',c='r')
    ax2.set_xlabel(u'年份',fontproperties='SimHei')
    ax2.set_ylabel(u'$HL_{norm}$',fontproperties='SimHei')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_title('APS',fontproperties='SimHei')

    plt.tight_layout()
    plt.savefig('fig/validation/compare_hf.pdf',dpi=800)
    print 'fig saved to fig/validation/compare_hf.pdf'




    # xs = []
    # ys = []
    # for year in sorted(year_lls.keys()):
    #     if int(year)<1960:
    #         continue
    #     xs.append(year)
    #     ys.append(np.mean(year_lls[year]))

    # print xs
    # print ys

    # plt.figure(figsize=(5,4))

    # plt.plot(xs,ys)
    # plt.title(u'APS',fontproperties='SimHei')

    # plt.xlabel(u'年份',fontproperties='SimHei')
    # plt.ylabel(u'平均生命周期',fontproperties='SimHei')

    # plt.tight_layout()

    # plt.savefig('fig/validation/compare_lifespan.png',dpi=800)
    # print 'fig saved to fig/validation/compare_lifespan.png'

    ## 随机抽取20片论文
    fig,axes = plt.subplots(4,5,figsize=(12.5,10))
    for i,pid in enumerate(sorted(ref_year_num.keys(),key=lambda x:np.sum(ref_year_num[x].values()),reverse=True)[:20]):

        year_num = ref_year_num[pid]
        tn = np.sum(year_num.values())

        xs = []
        ys = []
        for j,year in enumerate(sorted(year_num.keys())):
            xs.append(year)
            ys.append(year_num[year])


        ax = axes[i/5,i%5]
        ax.plot(xs,ys)
        ax.set_xlabel(u'年份',fontproperties='SimHei')
        ax.set_ylabel(u'被引次数',fontproperties='SimHei')
        ax.set_title("{:}({:})".format(pid_year[pid],tn))

    plt.tight_layout()
    plt.savefig('fig/validation/compare_20_life.pdf',dpi=800)
    print 'fig saved to fig/validation/compare_20_life.pdf'


if __name__ == '__main__':
    # compare_citation_dis()
    # average_value_over_year()
    compare_lifespan()
    # compare_plots()

