#coding:utf-8
'''
对主题的演化进行讨论

1. 主题文章数量的变化



'''
import sys
sys.path.extend(['.','..'])
from tools.basic_config import *
from scipy.stats import pearsonr
import powerlaw
# import scipy
from scipy.stats import norm


PCAS_PATH = '/Users/huangyong/Workspaces/Study/datasets/APS/PCAS.txt'

PCAS_CODES = []


def select_topic(topic_list):

    reserved_pcases = []
    for pcas in topic_list:

        if len(pcas)!=2:
            continue

        if pcas[0] not in ['0','1','2','3','4','5','6','7','8','9'] or pcas[1] not in ['0','1','2','3','4','5','6','7','8','9']:
            continue

        if pcas.endswith("0"):
            continue

        reserved_pcases.append(pcas)

    if len(reserved_pcases)==0:
        return None

    pcas_counter = Counter(reserved_pcases)
    v_pl = defaultdict(list)
    for pcas in pcas_counter.keys():
        v_pl[pcas_counter[pcas]].append(pcas)

    pl = v_pl[sorted(v_pl.keys(),reverse=True)[0]]

    return np.random.choice(pl,size=1)[0]



## 主题数量是如何随时间变化的
def topic_nums():

    ## 文章发表的年份
    paper_year = json.loads(open('data/paper_year.json').read())

    ## 主题每年的数量
    pcas_year_papers = defaultdict(lambda:defaultdict(list))
    ## 主题的数量
    pcas_nums = defaultdict(list)

    no_year_papers = 0
    has_year_papers = 0

    num_topic_list = []

    pid_topic = {}

    ## PCAS 文件
    for line in open(PCAS_PATH):

        line = line.strip()

        if line.startswith('DOI'):
            continue

        # print line
        try:
            doi,pcas1,pcas2,pcas3,pcas4,pcas5 = line.split(',')
        except:
            print line
            continue

        pcas1,pcas2,pcas3,pcas4,pcas5 = pcas1.strip().split('.')[0],pcas2.strip().split('.')[0],pcas3.strip().split('.')[0],pcas4.strip().split('.')[0],pcas5.strip().split('.')[0]

        pcas_list = [pcas1,pcas2,pcas3,pcas4,pcas5]

        tn = 0
        for pcas in list(set(pcas_list)):

            if len(pcas)!=2:
                continue

            if pcas[0] not in ['0','1','2','3','4','5','6','7','8','9'] or pcas[1] not in ['0','1','2','3','4','5','6','7','8','9']:
                continue

            if pcas.endswith("0"):
                continue

            tn+=1

        if tn>0:
            num_topic_list.append(tn)

        year = paper_year.get(doi,-1)
        pcas = select_topic(pcas_list)

        if pcas is None:
            continue

        pcas_nums[pcas].append(doi)
        pid_topic[doi] = pcas


        if year==-1:
            no_year_papers+=1

        if year !=-1:
            has_year_papers+=1
            pcas_year_papers[pcas][year].append(doi)


    open('data/pid_topic.json','w').write(json.dumps(pid_topic))

    print 'pid totpic saved to data/pid_topic.json.'


    print 'no year papers',no_year_papers,'has year papers',has_year_papers

    num_topic_counter = Counter(num_topic_list)

    plt.figure(figsize=(3.5,2.8))

    xs = []
    ys = []

    for num in sorted(num_topic_counter.keys()):
        tn = num_topic_counter[num]
        xs.append(num)
        ys.append(tn)

    plt.bar(range(len(xs)),ys)
    plt.xlabel(u'主题个数',fontproperties='SimHei')
    plt.ylabel(u'文章数量',fontproperties='SimHei')
    plt.xticks(range(len(xs)),xs)
    plt.yscale('log')
    plt.tight_layout()

    plt.savefig('fig/topic/topic_num_dis.png',dpi=800)
    print 'num of topic distribution saved fig/topic/topic_num_dis.png'

    open('data/topic_year_dois.json','w').write(json.dumps(pcas_year_papers))
    print 'data saved to data/topic_year_dois.json'

    ### 各个主题按照数量多少画图
    ys = []
    xs = []
    topic_nums = {}
    tn_list = []
    for pcas in sorted(pcas_nums.keys(),key=lambda x:len(pcas_nums[x]),reverse=True):

        tn = len(pcas_nums[pcas])
        if pcas=='' or tn<100:
            continue

        tn_list.append(tn)
        xs.append(pcas)
        ys.append(tn)
        topic_nums[pcas] = pcas_nums[pcas]

    open('data/topic_papers.json','w').write(json.dumps(topic_nums))
    print 'data saved to data/topic_papers.json'


    ### 分别输出多少个PCAS
    print 'Num of PCAS:',len(xs)
    ## 画出柱状图

    plt.figure(figsize=(10,2.8))
    plt.bar(range(len(xs)),ys)
    plt.xticks(range(len(xs)),xs,rotation=90)
    plt.ylim(1,100000)
    plt.yscale("log")
    plt.xlabel(u'主题',fontproperties='SimHei')
    plt.ylabel(u'数量',fontproperties='SimHei')
    plt.tight_layout()
    plt.savefig('fig/topic/topic_nums.jpg',dpi=800)
    print 'topic nums saved to fig/topic/topic_nums.jpg'

    ### 拟合曲线
    plt.figure(figsize=(3.5,2.8))
    expfunc = lambda t,a,b:a*np.exp(b*t)
    index_xs= np.arange(len(xs))+1
    fit_ys = np.array(ys)/float(np.sum(ys))
    popt,pcov = scipy.optimize.curve_fit(expfunc,index_xs,fit_ys,p0=(0.2,-2))
    plt.plot(np.array(index_xs),fit_ys,label=u'主题文献数量分布')
    plt.plot(index_xs,[expfunc(x,*popt) for x in index_xs],'--',label=u'拟合曲线$p(n)=%.2f*e^{%.2fn}$'%(popt[0],popt[1]),c='r')
    plt.xlabel(u'次序',fontproperties='SimHei')
    plt.ylabel(u'比例',fontproperties='SimHei')
    plt.yscale('log')
    plt.legend(prop={'family':'SimHei','size':8})
    plt.tight_layout()


    fitted_xs = range(1,101)
    fitted_ys = [expfunc(x,*popt) for x in fitted_xs]

    fitted_ys = list(np.array(fitted_ys)/np.sum(fitted_ys))

    topic_dis = {}
    topic_dis['x'] = fitted_xs
    topic_dis['y'] = fitted_ys

    open('topic_dis.json','w').write(json.dumps(topic_dis))
    print 'topic dis saved to topic_dis.json.'


    plt.savefig('fig/topic/topic_nums_fit.png',dpi=800)
    print 'fig saved to fig/topic/topic_nums_fit.png'

    ### 把前10的主题论文数量画出来
    plt.figure(figsize=(5,4))
    for pcas in xs[:10]:

        year_papers = pcas_year_papers[pcas]

        xs = []
        ys = []
        for year in sorted(year_papers.keys()):
            xs.append(year)
            ys.append(len(year_papers[year]))

        plt.plot(xs,ys,label='{:}'.format(pcas))

    plt.xlabel(u'年',fontproperties='SimHei')
    plt.ylabel(u'数量',fontproperties='SimHei')

    plt.legend()

    plt.tight_layout()

    plt.savefig("fig/topic/topic_year_num.jpg",dpi=400)
    print 'pcas year num saved to fig/topic_year_num.jpg'

## 主题相关性
def topic_relevance():

    pid_refs = json.loads(open('data/pid_all_refs.json').read())
    pid_topic = json.loads(open('data/pid_topic.json').read())

    topic_nums = json.loads(open('data/topic_nums.json').read())

    print len(pid_refs.keys())

    topics = sorted(topic_nums.keys(),key=lambda x:len(topic_nums[x]),reverse=True)[:15]
    all_topics = topic_nums.keys()
    t1_t2_num = defaultdict(lambda:defaultdict(int))
    t1_refnum = defaultdict(int)
    progress = 0
    for pid in pid_refs.keys():

        progress+=1

        if progress%1000==0:
            print progress

        topic = pid_topic.get(pid,'-1')

        if topic =='-1':
            continue

        refs = pid_refs[pid]

        for ref in refs:

            ref_topic = pid_topic.get(ref,'-1')

            if ref_topic=='-1':
                ref_topic=topic

            t1_t2_num[topic][ref_topic]+=1
            t1_refnum[topic]+=1

    t1_t2_rel = defaultdict(dict)
    for t1 in all_topics:
        ## 该主题引用总次数
        refnum = t1_refnum[t1]

        row = []
        for t2 in all_topics:
            num = t1_t2_num[t1].get(t2,0)
            ## 主题2对主题1的相关性
            rel_2_1 = num/float(refnum)
            t1_t2_rel[t1][t2] = rel_2_1

    open('data/topic_rel_matrix.json','w').write(json.dumps(t1_t2_rel))
    print 'topic relevance matrix saved to data/topic_rel_matrix.json.'



    rels  =['t1,t2,rel']
    for t1 in topics:
        ## 该主题引用总次数
        refnum = t1_refnum[t1]

        row = []
        ## 主题1引用主题2的次数
        for t2 in topics:
            num = t1_t2_num[t1].get(t2,0)

            ## 主题2对主题1的相关性
            rel_2_1 = num/float(refnum)

            rels.append('{:},{:},{:}'.format(t1,t2,rel_2_1))

    open('data/topic_relevance.csv','w').write('\n'.join(rels))
    print 'topic relevance saved to data/topic_relevance.csv'

    ## 画热力图
    plot_heatmap('data/topic_relevance.csv','主题相关性矩阵','主题','主题','fig/topic/topic_rel_matrix.png')

    ## 画出前15的排序相关性
    plt.figure(figsize=(5,4))
    all_topics=t1_t2_num.keys()
    all_num_list = []
    # all_rels =
    for t1 in all_topics:
        t2_num = t1_t2_num[t1]
        refnum = t1_refnum[t1]

        num_list = []
        for t2 in all_topics:

            num = t1_t2_num[t1].get(t2,0)
            num_list.append(num/float(refnum))

        if t1 in topics:
            plt.plot(range(1,len(all_topics)+1),sorted(num_list,reverse=True),alpha=0.6)

        all_num_list.append(sorted(num_list,reverse=True))

    all_avg = [np.mean([i for i in a if i>0]) for a in zip(*all_num_list)]


    plt.plot(range(1,len(all_topics)+1),all_avg,'--',linewidth=2,c='r',label=u'均值')

    xs = []
    ys = []
    for num_list in all_num_list:

        for i,num in enumerate(sorted(num_list,reverse=True)):
            if num >0:
                xs.append(i+1)
                ys.append(num)

    # xs = range(1,len(all_topics)+1)
    # ys = all_avg
    plaw = lambda t,a,b: a*t**b
    # expfunc = lambda t,a,b:a*np.exp(b*t)
    popt,pcov = scipy.optimize.curve_fit(plaw,xs,ys,p0=(0.2,-1))
    plt.plot(np.linspace(1,np.max(xs),10),[plaw(x+1,*popt) for x in np.linspace(1,np.max(xs),10)],'-^',label=u'拟合曲线',c='b')

    fit_xs = range(1,201)
    fit_ys = [plaw(x+1,*popt) for x in fit_xs]

    topic_relevance = {}

    topic_relevance['x'] = fit_xs
    topic_relevance['y'] = fit_ys

    open('topic_relevance.json','w').write(json.dumps(topic_relevance))
    print 'saved to topic relevance.json'


    plt.xlabel(u'次序',fontproperties='SimHei')
    plt.ylabel(u'相关系数',fontproperties='SimHei')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(prop={'family':'SimHei','size':8})

    plt.tight_layout()
    plt.savefig('fig/topic/topic_rel_dis.png',dpi=800)
    print 'fig saved to fig/topic/topic_rel_dis.png'


def gen_pid_topic_cits():
    pid_all_refs = json.loads(open('data/pid_all_refs.json').read())

    pid_year = json.loads(open('data/paper_year.json').read())

    pid_topic = json.loads(open('data/pid_topic.json').read())

    print 'paper with topic', len(pid_topic.keys())
    print 'paper with refs',len(pid_all_refs.keys())


    pid_topic_cits = defaultdict(lambda:defaultdict(int))
    progress= 0
    for pid in pid_all_refs.keys():
        citing_year = int(pid_year.get(pid,-1))
        topic = pid_topic.get(pid,'-1')

        if topic == '-1':
            continue

        progress+=1
        if progress%1000==0:
            print 'progress:',progress

        for ref in pid_all_refs[pid]:
            cited_year = int(pid_year.get(ref,-1))

            if cited_year >2006:
                continue

            if citing_year-cited_year<=10:
                pid_topic_cits[ref][topic]+=1

    print progress
    open('data/pid_topic_cits.json','w').write(json.dumps(pid_topic_cits))
    print '%d papers reserved, and saved to data/pid_topic_cits.json' % len(pid_topic_cits.keys())

###计算论文在各个主题的引用价值
def trans_rate():

    # pid_refs = json.loads(open('data/pid_refs.json').read())
    pid_topic = json.loads(open('data/pid_topic.json').read())
    pid_topic_cits = json.loads(open('data/pid_topic_cits.json').read())
    topic_rel_matrix = json.loads(open('data/topic_rel_matrix.json').read())

    topic_nums = json.loads(open('data/topic_nums.json').read())

    all_topics = set(sorted(topic_nums.keys(),key=lambda x:len(topic_nums[x]),reverse=True)[:68])
    ## 对于筛选的数据
    VRs = []
    rels = []

    progress=0
    bad_data = 0
    ## t1的值向t2转化
    t1_t2_vrs = defaultdict(lambda:defaultdict(list))
    for pid in pid_topic_cits.keys():

        t1 = pid_topic.get(pid,'-1')

        if t1=='-1':
            continue

        if t1 not in all_topics:
            continue

        cn0 = pid_topic_cits[pid].get(t1,0)

        if cn0<10:
            continue


        progress+=1

        for t2 in pid_topic_cits[pid].keys():

            if t2 not in all_topics:
                continue

            tc = pid_topic_cits[pid][t2]

            ## 计算转化率
            vr = tc/float(cn0)

            if vr>1:
                bad_data+=1
                # continue

            # print t1,t2
            ## 记录两个topic之间相关性
            # print topic_rel_matrix[t2]
            # m12 = topic_rel_matrix[t2][t1]/topic_rel_matrix[t2][t2]
            t1_t2_vrs[t1][t2].append(vr)
            # VRs.append(vr)
            # rels.append(m12)
    print progress
    plt.figure(figsize=(5,4))

    for t1 in t1_t2_vrs.keys():
        # VRs = []
        # rels = []
        for t2 in t1_t2_vrs[t1].keys():

            if t1==t2:
                continue

            vr = np.mean(t1_t2_vrs[t1][t2])
            # print t2,topic_rel_matrix[t2]
            m12 = topic_rel_matrix[t2][t1]
            VRs.append(vr)
            rels.append(m12)

        # indexes = sorted(range(len(rels)),key=lambda x:rels[x])
        # xs = [rels[i] for i in indexes]
        # ys = [VRs[i] for i in indexes]
        # plt.plot(xs,ys,alpha=0.6)

    rel_vrs = defaultdict(list)
    for i,rel in enumerate(rels):
        vr = VRs[i]
        rel = float('{:.4f}'.format(rel))

        if rel>0 and vr>0:


            rel_vrs[rel].append(vr)

    xs = []
    ys = []
    avgs = []
    for rel in sorted(rel_vrs.keys()):

        avr = np.mean(rel_vrs[rel])
        xs.append(rel)
        ys.append(avr)
        avgs.append(np.mean(ys))


    plt.plot(xs,ys,label=u'平均转化率',alpha=0.6)
    plt.plot(xs,avgs,'--',c='g',label=u'移动平均转化率',alpha=0.9)

    expfunc = lambda t,a,b:a*t**b
    popt,pcov = scipy.optimize.curve_fit(expfunc,xs,avgs)
    plt.plot(xs,[expfunc(x,*popt) for x in xs],'--',label=u'拟合曲线$Tr(z)=%.2f\\times z^{%.2f}$'%(popt[0],popt[1]),c='r')

    print 'trans param',popt

    plt.xlabel(u'相关性',fontproperties='SimHei')
    plt.ylabel(u'价值转化率',fontproperties='SimHei')
    plt.yscale('log')
    plt.xscale('log')
    # plt.ylim(0.05,0.99)
    plt.legend(prop={'family':'SimHei','size':8})
    plt.tight_layout()

    plt.savefig('fig/topic/topic_tr_relation.png',dpi=800)

    print 'fig saved to fig/topic/topic_tr_relation.png'


def topic_values():

    ## paper citaton number
    pid_cn = json.loads(open('data/pid_all_cn.json').read())
    pid_topic_cits = json.loads(open('data/pid_topic_cits.json').read())

    pid_refs = json.loads(open('data/pid_all_refs.json').read())

    ## paper c10
    # pid_c10 = json.loads(open('data/pid_cn.json').read())

    print len(pid_topic_cits.keys())
    # print len(pid_c10.keys())

    pid_set = set(pid_topic_cits.keys())

    ## topic year dois
    topic_year_papers = json.loads(open('data/topic_year_dois.json').read())

    ## topic nums
    topic_nums = json.loads(open('data/topic_papers.json').read())

    ## 主题分布
    top_10_citation_dis_xys = {}

    ##引用次数随时间的数量变化
    top_10_year_nums_xys = {}
    ## 高被引率数量变化
    top_10_hr_xys = {}

    ## 论文的c10
    top_10_c10_xys = {}

    ## 前10主题下，文章c_10及参考价值之间的关系
    top_10_c10_refvs_xys = {}
    ## lambda的样子
    top_10_lambda_xys = {}

    ## 画出前10的topic
    top_10_topics = sorted(topic_nums.keys(),key=lambda x:len(topic_nums[x]),reverse=True)[:40]

    # top_10_base_lambda={}

    _max_c10 = 0
    _max_refv = 0

    for topic in top_10_topics:

        ## 每一个topic下的引用次数分布
        papers = [p for p in topic_nums[topic] if p in pid_set]

        for pid in papers:

            c10 = pid_topic_cits.get(pid,{}).get(topic,-1)

            if c10==-1:
                continue

            refs = pid_refs.get(pid,[])

            # print refs

            if len(refs)<15:
                continue

            ## 必须保证ref都有主题
            has_t = True
            for ref in refs:
                if ref not in pid_set:
                    has_t=False
                    break

            if not has_t:
                continue

            ref_c10s = [pid_topic_cits.get(pid,{}).get(topic,0) for pid in refs]
            # print ref_c10s
            ris,res = zscore_outlier(ref_c10s,2.5)
            refv = np.mean(res)

            if refv>_max_refv:
                _max_refv = refv

            if c10>_max_c10:
                _max_c10=c10

    _base_lambda = _max_c10/_max_refv
    print 'max c10:',_max_c10,',max refv:',_max_refv,',base lambda:',_base_lambda

    for topic in top_10_topics:

        ## 每一个topic下的引用次数分布
        papers = [p for p in topic_nums[topic] if p in pid_set]

        cns = [pid_cn.get(p,0) for p in papers if pid_cn.get(p,0)!=0 ]

        cn_counter = Counter(cns)
        xs = []
        ys = []
        for cn in sorted(cn_counter.keys()):
            xs.append(cn)
            ys.append(cn_counter[cn])

        top_10_citation_dis_xys[topic] = [xs,ys]

        ## 每年的文章数量
        year_papers = topic_year_papers[topic]

        xs = []
        ys = []
        high_ys = []
        hr_ys = []
        c10_ys = []
        for year in sorted(year_papers.keys(),key=lambda x:int(x)):
            xs.append(int(year))
            ys.append(len(year_papers[year]))

            dois = [pid_cn.get(doi,0) for doi in year_papers[year] if pid_cn.get(doi,0)>=100]

            ## c10是该论文在本主题的引用次数
            c10s = [pid_topic_cits[doi].get(topic,0) for doi in year_papers[year] if doi in pid_set]

            avg_c10 = np.mean(c10s)

            c10_ys.append(avg_c10)

            high_num = len(dois)
            high_ys.append(high_num)
            hr = high_num/float(len(year_papers[year]))

            hr_ys.append(hr)

        top_10_year_nums_xys[topic]=[xs,ys,high_ys,hr_ys,c10_ys]


        ## 主题下 c_10与refvs之间的关系
        c10_list = []
        refvs = []
        for pid in papers:

            c10 = pid_topic_cits.get(pid,{}).get(topic,-1)

            if c10==-1:
                continue

            refs = pid_refs.get(pid,[])

            # print refs

            if len(refs)<15:
                continue

            ## 必须保证ref都有主题
            has_t = True
            for ref in refs:
                if ref not in pid_set:
                    has_t=False
                    break

            if not has_t:
                continue

            ref_c10s = [pid_topic_cits.get(pid,{}).get(topic,0) for pid in refs]
            # print ref_c10s
            ris,res = zscore_outlier(ref_c10s,2.5)

            refv = np.mean(res)

            c10_list.append(c10)
            refvs.append(refv)


        lambdas = []
        for j,c10 in enumerate(c10_list):
            refv = refvs[j]
            lambdas.append((c10/float(_max_c10))/(float(refv)/_max_refv))

        top_10_c10_refvs_xys[topic] = [c10_list,refvs]

        top_10_lambda_xys[topic] = lambdas

    top_10_topics = sorted(top_10_lambda_xys.keys(),key=lambda x:len(top_10_lambda_xys[x]),reverse=True)[:10]
    ## 各个主题下的引用次数分布
    fig,axes = plt.subplots(2,5,figsize=(14,5.6))
    for i,topic in enumerate(top_10_topics):
        ax = axes[i/5,i%5]
        xs,ys = top_10_citation_dis_xys[topic]
        ax.plot(xs,ys)
        ax.set_xlabel(u'引用次数',fontproperties='SimHei')
        ax.set_ylabel(u'文章数量',fontproperties='SimHei')

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_title(topic)
    plt.tight_layout()
    plt.savefig('fig/topic/topic_citation_dis.jpg',dpi=800)
    print 'fig saved to fig/topic/topic_citation_dis.jpg.'

    ## 各个主题下的不同年代数量分布
    fig,axes = plt.subplots(2,5,figsize=(17,5.6))
    for i,topic in enumerate(top_10_topics):
        ax = axes[i/5,i%5]
        xs,ys,high_ys,hr_ys,c10_ys = top_10_year_nums_xys[topic]

        ax.plot(xs,ys,label=u'文章数量',linewidth=2)
        ax.plot(xs,high_ys,label=u'高被引论文',c='g',linewidth=2)
        ax.set_xlabel(u'年份',fontproperties='SimHei')
        ax.set_ylabel(u'文章数量',fontproperties='SimHei')
        ax.set_yscale('log')
        ax.legend(prop={'family':'SimHei','size':5})
        ax.set_title(topic)

        ax2 = ax.twinx()
        ax2.plot(xs,hr_ys,'--',label=u'高被引比例',c='r')
        ax2.set_ylabel(u'高被引文文章比例',fontproperties='SimHei')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.legend(prop={'family':'SimHei','size':5})

    plt.tight_layout()
    plt.savefig('fig/topic/topic_year_pn.jpg',dpi=800)

    print 'fig saved to fig/topic/topic_year_pn.jpg.'

    ## 各个主题下的不同年代平均c10分布
    fig,axes = plt.subplots(2,5,figsize=(14,5.6))
    for i,topic in enumerate(top_10_topics):
        ax = axes[i/5,i%5]
        xs,ys,high_ys,hr_ys,c10_ys = top_10_year_nums_xys[topic]

        xl = []
        yl = []
        for j,x in enumerate(xs):
            if x <= 2006 and x>1990:
                xl.append(x)
                yl.append(c10_ys[j])

        ax.plot(xl,yl,linewidth=2)
        ax.set_xlabel(u'年份',fontproperties='SimHei')
        ax.set_ylabel(u'$c_{10}$',fontproperties='SimHei')
        # ax.set_yscale('log')
        ax.set_title(topic)
        ax.set_ylim(0,25)

    plt.tight_layout()
    plt.savefig('fig/topic/topic_year_c10.jpg',dpi=800)

    print 'fig saved to fig/topic/topic_year_c10.jpg.'

    ## 各主题下的相关性
    fig,axes = plt.subplots(2,5,figsize=(14,5.6))
    for i,topic in enumerate(top_10_topics):
        lambdas = top_10_lambda_xys[topic]

        # lambdas = [l for l in lambdas if l <1]
        size = len(lambdas)

        ax = axes[i/5,i%5]
        c10s,refvs = top_10_c10_refvs_xys[topic]

        prs = pearsonr(c10s,refvs)[0]
        ax.plot(c10s,refvs,'o',label=u'皮尔逊相关系数:{:.2f}'.format(prs))
        ax.set_xlabel(u'$c_{10}$',fontproperties='SimHei')
        ax.set_ylabel(u'$\langle v(refs) \\rangle $',fontproperties='SimHei')

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_title('%s (%d)' %(topic,size))
        # ax.set_ylim(0,25)
        ax.legend(prop={'family':'SimHei','size':5})

    plt.tight_layout()
    plt.savefig('fig/topic/topic_c10_rev_relation.jpg',dpi=800)

    print 'fig saved to fig/topic/topic_c10_rev_relation.jpg.'

    ## 各主题下的lambdas
    x0s = []
    sigmas = []
    fig,axes = plt.subplots(2,5,figsize=(14,5.6))
    for i,topic in enumerate(sorted(top_10_lambda_xys.keys(),key=lambda x:len(top_10_lambda_xys[x]),reverse=True)):
        lambdas = top_10_lambda_xys[topic]
        # lambdas = [l for l in lambdas if l <1]
        size = len(lambdas)
        print size

        xs = []
        ys = []

        l_counter = Counter([float('{:.2f}'.format(l)) for l in lambdas])
        for l in sorted(l_counter):
            xs.append(l)
            ys.append(l_counter[l]/float(size))

        print xs
        print ys
        # _base_lambda = top_10_base_lambda[topic]

        # ax.hist(lambdas,bins=100,density=True,log=True)

        # ax.set_ylim(0,25)
        # ax.legend(prop={'family':'SimHei','size':5})

        scale,loc,sigma,mu,mode = fit_lognorm(lambdas)

        pdf_fitted = scipy.stats.lognorm.pdf(xs, sigma, loc=0, scale=scale)
        pdf_fitted = np.array(pdf_fitted)/np.sum(pdf_fitted)
        x0s.append(mode*_base_lambda)
        sigmas.append(sigma)

        if i>=10:
            continue

        ax = axes[i/5,i%5]

        ax.plot(np.array(xs)*_base_lambda,ys,'o',fillstyle='none',label=u'价值系数散点图')
        ax.set_xlabel(u'$\lambda$',fontproperties='SimHei')
        ax.set_ylabel(u'$P(\lambda)$',fontproperties='SimHei')

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_title('%s (%d)' %(topic,size))
        ax.plot(np.array(xs)*_base_lambda,pdf_fitted,label=u'拟合曲线$scale=%.2f,\sigma=%.2f$' %(scale,sigma))
        ax.plot([mode*_base_lambda]*10,np.linspace(np.min(pdf_fitted),0.1,10),'--',label='$\lambda_0 = {:.2f}$'.format(mode*_base_lambda))

        ax.legend(prop={'family':'SimHei','size':5})

    plt.tight_layout()
    plt.savefig('fig/topic/topic_lambdas.jpg',dpi=800)

    print 'fig saved to fig/topic/topic_lambdas.jpg.'

    x0_counter = Counter(x0s)
    xs = []
    ys = []
    for x0 in sorted(x0_counter.keys()):
        xs.append(x0)
        ys.append(x0_counter[x0])

    mu,std = norm.fit(x0s)
    xp = np.linspace(np.min(xs), np.max(xs), 20)
    pp = norm.pdf(xp, mu, std)
    pp = np.array(pp)/np.sum(pp)



    plt.figure()
    n,bins,patches = plt.hist(x0s,bins=20)
    xs = [x for x in bins[:-1]]
    ys = [x for x in n]
    print xs
    print ys

    scale,loc,sigma,mu,mode = fit_lognorm(x0s)

    fit_x = np.linspace(0.05,0.2,100)
    fit_y = scipy.stats.lognorm.pdf(fit_x, sigma, loc=0, scale=scale)
    fit_y = np.array(fit_y)/np.sum(fit_y)
    lambda_dis = {}
    lambda_dis['x'] = list(fit_x)
    lambda_dis['y'] = list(fit_y)

    open('topic_lambda_dis.json','w').write(json.dumps(lambda_dis))


    pdf_fitted = scipy.stats.lognorm.pdf(xs, sigma, loc=0, scale=scale)
    pdf_fitted = np.array(pdf_fitted)/np.sum(pdf_fitted)

    ys = np.array(ys)/np.sum(ys)
    plt.figure(figsize=(3.2,2.8))
    plt.bar(xs,ys,width=0.01,label=u'$\lambda_0$分布',alpha=0.6)
    # plt.plot(xp,pp,'--',linewidth=2,c='r',label=u'$\Phi (%.2f,%.2f)$'%(mu,std))
    plt.plot(xs,pdf_fitted,'--',c='r',label=u'$scale=%.2f,\sigma=%.2f$'%(scale,sigma))
    plt.plot([mode]*10,np.linspace(0,np.max(pdf_fitted)*1.1,10),'--',c='g',label=u'$mode=%.2f$'%(mode))

    plt.xlabel(u'$\lambda_0$')
    plt.ylabel(u'主题比例',fontproperties='SimHei')
    plt.legend(prop={'family':'SimHei','size':5})
    plt.tight_layout()

    plt.savefig('fig/topic/topic_lambda0_dis.png',dpi=800)
    print 'fig saved to fig/topic/topic_lambda0_dis.png'

    plt.figure()
    n,bins,patches = plt.hist(sigmas,bins=20)
    xs = [x for x in bins[:-1]]
    ys = [x for x in n]
    print xs
    print ys

    scale,loc,sigma,mu,mode = fit_lognorm(sigmas)

    fit_x = np.linspace(0.5,1.5,100)
    fit_y = scipy.stats.lognorm.pdf(fit_x, sigma, loc=0, scale=scale)
    fit_y = np.array(fit_y)/np.sum(fit_y)
    sigma_dis = {}
    sigma_dis['x'] = list(fit_x)
    sigma_dis['y'] = list(fit_y)

    open('topic_sigma_dis.json','w').write(json.dumps(sigma_dis))


    pdf_fitted = scipy.stats.lognorm.pdf(xs, sigma, loc=0, scale=scale)
    pdf_fitted = np.array(pdf_fitted)/np.sum(pdf_fitted)

    ys = np.array(ys)/np.sum(ys)
    plt.figure(figsize=(3.2,2.8))
    plt.bar(xs,ys,width=0.01,label=u'$\sigma$分布',alpha=0.6)
    # plt.plot(xp,pp,'--',linewidth=2,c='r',label=u'$\Phi (%.2f,%.2f)$'%(mu,std))
    plt.plot(xs,pdf_fitted,'--',c='r',label=u'$scale=%.2f,\sigma=%.2f$'%(scale,sigma))
    plt.plot([mode]*10,np.linspace(0,np.max(pdf_fitted)*1.1,10),'--',c='g',label=u'$mode=%.2f$'%(mode))

    plt.xlabel(u'$\sigma$')
    plt.ylabel(u'主题比例',fontproperties='SimHei')
    plt.legend(prop={'family':'SimHei','size':5})
    plt.tight_layout()

    plt.savefig('fig/topic/topic_sigma_dis.png',dpi=800)
    print 'fig saved to fig/topic/topic_sigma_dis.png'


## 作者在选择一个主题后继续选择该主题的概率
def author_topic_selection():

    author_year_papers = json.loads(open('data/author_year_articles.json').read())

    pid_topic = json.loads(open('data/pid_topic.json').read())

    tn_num = defaultdict(int)
    for author in author_year_papers.keys():
        topics = []
        for year in author_year_papers[author].keys():

            for pid in author_year_papers[author][year]:

                topic = pid_topic.get(pid,'-1')
                if pid=='-1':
                    continue

                topics.append(topic)

        # if len(topics)<10:
            # continue
        topic_counter = Counter(topics)

        for v in topic_counter.values():

            tn_num[v]+=1
    xs = []
    ys = []
    for tn in sorted(tn_num.keys()):
        xs.append(tn)
        ys.append(tn_num[tn])

    plt.figure(figsize=(3.2,2.8))

    ys = np.array(ys)/float(np.sum(ys))
    plt.plot(xs,ys,alpha=0.7)

    plt.xlabel(u'$z$',fontproperties='SimHei')
    plt.ylabel(u'比例',fontproperties='SimHei')

    expfunc = lambda t,a,b:a*t**b
    popt,pcov = scipy.optimize.curve_fit(expfunc,xs,ys)
    fit_x = np.linspace(1,1000,1000)
    fit_y = [expfunc(x,*popt) for x in fit_x]
    plt.plot(fit_x[:100],fit_y[:100],'--',label='$f(z)=%.2f\\times z^{%.2f}$'%(popt[0],popt[1]))

    total = np.sum(fit_y)
    ccdf_y = []
    con_y = []
    last = 1
    for i,x in enumerate(fit_x):
        y = expfunc(x,*popt)
        ccdf = total-np.sum(fit_y[:i+1])
        ccdf_y.append(ccdf)
        con_y.append(ccdf/last)
        last = ccdf

    plt.plot(fit_x[:100],ccdf_y[:100],'-.',label='$F(x>z)$',linewidth=1,c='b',alpha=0.8)
    plt.plot(fit_x[:100],con_y[:100],'-.',label='$P_c(z|s_i,t_i)$',linewidth=2,c='r')

    selection_dis = {}
    selection_dis = {}
    selection_dis['x'] = list(fit_x)
    selection_dis['y'] = list(con_y)

    open('topic_selection_dis.json','w').write(json.dumps(selection_dis))

    plt.xscale('log')
    plt.yscale('log')

    plt.legend(prop={'size':5})
    plt.tight_layout()

    plt.savefig('fig/topic/topic_one_pn.png',dpi=800)
    print 'fig saved to fig/topic/topic_one_pn.png'


if __name__ == '__main__':
    ## 主题数量分布
    # topic_nums()

    ## 主题相关性
    # topic_relevance()

    ## 主题转化率
    # gen_pid_topic_cits()
    # trans_rate()

    ## 价值系数分布
    topic_values()

    ## 作者主题选择概率
    # author_topic_selection()





