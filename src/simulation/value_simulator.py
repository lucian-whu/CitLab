#coding:utf-8
'''
价值函数生成
1. 参考价值与引用次数相关性分析
2. 相同参考价值下的论文价值分布
3. 价值系数拟合以及仿真

'''
import sys
sys.path.extend(['.','..'])
from tools.basic_config import *

from scipy.stats import pearsonr

import statsmodels.api as sm
import powerlaw

### 首先生成APS的引用数据
CITATION_DATA_PATH = '/Users/huangyong/Workspaces/Study/datasets/APS/aps-dataset-citations-2016/aps-dataset-citations-2016.csv'
DATA_PATH = '/Users/huangyong/Workspaces/Study/datasets/APS/aps-dataset-metadata-2016'


def list_metadata():
    for journal in os.listdir(DATA_PATH):
        for issue in os.listdir(DATA_PATH+'/'+journal):

            for article in os.listdir(DATA_PATH+'/'+journal+'/'+issue):

                if not article.endswith('json'):
                    continue

                yield DATA_PATH+'/'+journal+'/'+issue+'/'+article


def extract_author_info(article_json):
    pid = article_json['id']
    authors = article_json.get('authors','-1')
    date = article_json.get('date','-1')
    atype = article_json.get('articleType','-1')
    affiliations = article_json.get('affiliations',[])

    return pid,authors,date,atype,affiliations

## 记录每篇论文的年份
def record_paper_year():
    pid_year = {}

    progress = 0
    for article_path in list_metadata():

        progress+=1

        if progress%10000==1:
            print 'progress,',progress

        article_json = json.loads(open(article_path).read())
        pid,authors,date,atype,affiliations = extract_author_info(article_json)

        if authors=='-1' or len(authors)==0 or date =='-1' or atype!='article' or len(authors)>10:
            continue

        year = int(date.split('-')[0])

        pid_year[pid] = year

    print '%d papers has year attr.' % len(pid_year)

    open('data/paper_year.json','w').write(json.dumps(pid_year))

    print 'data saved to data/paper_year.jon'




def gen_dataset():

    pid_year = json.loads(open('data/paper_year.json').read())

    pid_cn = defaultdict(int)

    pid_refs = defaultdict(list)

    pid_all_cn = defaultdict(int)

    for line in open(CITATION_DATA_PATH):
        line = line.strip()

        if line=='citing_doi,cited_doi':
            continue


        citing_pid,cited_pid = line.split(',')

        pid_all_cn[cited_pid]+=1


        citing_year = int(pid_year.get(citing_pid,-1))
        cited_year = int(pid_year.get(cited_pid,-1))

        if citing_year==-1 or cited_year==-1:
            continue

        if cited_year >2006:
            continue

        if citing_year-cited_year<=10:
            # print citing_year,cited_year
            pid_cn[cited_pid]+=1

        pid_refs[citing_pid].append(cited_pid)

    open('data/pid_all_cn.json','w').write(json.dumps(pid_all_cn))
    print 'data saved to data/pid_all_cn.json'

    open('data/pid_all_refs.json','w').write(json.dumps(pid_refs))
    print 'data saved to data/pid_all_refs.json'



    saved_pid_refs= {}
    for pid in pid_refs.keys():
        ref_num = len(pid_refs[pid])

        if pid_cn.get(pid,0) ==0:
            continue

        if ref_num<15:
            continue

        has_0 = False
        for ref in pid_refs[pid]:

            if pid_cn.get(ref,0)==0:
                has_0=True

                break

        if has_0:
            continue

        saved_pid_refs[pid] = pid_refs[pid]


    open('data/pid_cn.json','w').write(json.dumps(pid_cn))
    print '%d papers reserved, and saved to data/pid_cn.json' % len(pid_cn.keys())

    open('data/pid_refs.json','w').write(json.dumps(saved_pid_refs))
    print '%d papers reserved, and saved to data/pid_refs.json' % len(saved_pid_refs.keys())



def ref_cit_relations():

    pid_refs = json.loads(open('data/pid_refs.json').read())
    pid_cn = json.loads(open('data/pid_cn.json').read())

    refvs = []
    c10l = []

    ref_c10s = defaultdict(list)

    v_coef_list = []

    print '%d articles loaded' % len(pid_refs)


    ### 随机选择12篇论文可视化他们ref的c10
    fig,axes = plt.subplots(3,4,figsize=(20,12))
    for i,pid in enumerate(np.random.choice(pid_refs.keys(),size=12)):

        ref_vs = sorted([pid_cn[p] for p in pid_refs[pid]],reverse=True)

        ris,rvs = zscore_outlier(ref_vs,2.5)

        print np.mean(ref_vs),np.mean(rvs)

        xs = range(1,len(ref_vs)+1)

        ax = axes[i/4,i%4]

        ax.plot(xs,ref_vs,label='{:.2f}'.format(np.mean(ref_vs)))
        ax.plot(ris,rvs,'o',label='{:.2f}'.format(np.mean(rvs)))
        ax.set_xlabel(u'次序',fontproperties='SimHei')
        ax.set_ylabel('$c_{10}$')

        ax.legend()

    plt.tight_layout()

    plt.savefig('fig/_12_refv_dis.jpg',dpi=800)
    print 'ref c10 distribution saved to fig/_12_refv_dis.jpg'

    # return
    t_num = 0
    all_refvs = []
    for pid in pid_refs.keys():

        refs = pid_refs[pid]

        c10 = int(pid_cn.get(pid,0))

        if c10==0:
            continue

        t_num+=1

        ref_vs = sorted([pid_cn[p] for p in refs],reverse=True)

        ris,rvs = zscore_outlier(ref_vs,2.5)

        ref_v = np.mean(rvs)

        all_refvs.append(ref_v)


        # if ref_v>30:
        # v_coef_list.append(float('{:.3f}'.format(c10/float(ref_v))))

        ref_c10s[int(ref_v)].append(c10)

        c10l.append(c10)
        refvs.append(ref_v)

    print 'total num %d' % t_num

    _max_c10 = float(np.max(c10l))
    _max_refv = float(np.max(refvs))

    print 'base lambda',_max_c10,_max_refv,_max_c10/_max_refv
    _base_lambda = _max_c10/_max_refv


    ##
    for i,c10 in enumerate(c10l):
        refv = refvs[i]
        v_coef_list.append(float('{:.3f}'.format((c10/_max_c10)/(refv/_max_refv))))

    ### refv的数量分布
    # refv_counter = Counter(all_refvs)

    plt.figure(figsize=(4,3.2))
    # xs =[]
    # ys = []
    # for refv in sorted(refv_counter.keys()):
    #     xs.append(refv)
    #     ys.append(refv_counter[refv])


    # plt.plot(xs,ys)
    n,bins,patchs = plt.hist(all_refvs,bins=100,rwidth=0.8)

    mean = np.mean(all_refvs)
    median = np.median(all_refvs)

    plt.plot([mean]*10,np.linspace(0,np.max(n),10),'-.',label=u'均值={:.2f}'.format(mean))
    plt.plot([median]*10,np.linspace(0,np.max(n),10),'--',label=u'中位数={:.2f}'.format(median))


    plt.xlabel(u'参考价值',fontproperties='SimHei')
    plt.ylabel(u'文章数量',fontproperties='SimHei')

    # plt.xscale('log')
    plt.legend(prop={'family':'SimHei','size':8})
    plt.tight_layout()

    plt.savefig('fig/refv_dis.jpg',dpi=600)
    print 'refv distribution saved to fig/refv_dis.jpg.'

    ## 相同的参考价值之上的真实价值分布
    plt.figure(figsize=(4,3.2))
    # styles = ['-']
    for i,refv in enumerate([10,20,30,40,50]):

        c10_list = ref_c10s[refv]

        print len(c10_list)

        c10_dis = Counter(c10_list)

        xs = []
        ys = []

        for c10 in sorted(c10_dis.keys()):
            xs.append(c10)
            ys.append(c10_dis[c10])

        ys = np.array(ys)/float(np.sum(ys))

        print xs,ys

        # ax = axes[i/3,i%3]
        # xs = range(1,len(c10_list)+1)
        plt.plot(xs,ys,label=u'$\widehat{\langle v(refs) \\rangle}$=%d'%refv )

    plt.yscale('log')

    plt.xlabel(u'$c_{10}$',fontproperties='SimHei')
    plt.ylabel(u'比例',fontproperties='SimHei')
    plt.xscale('log')

    plt.legend(prop={'family':'SimHei','size':8})
    plt.tight_layout()

    plt.savefig('fig/refv_c10_dis.png',dpi=800)

    print 'fig saved to fig/refv_c10_dis.png'

    # return
    ##在同样基础之上的分布
    plt.figure(figsize=(4,3.2))
    # styles = ['-']
    for i,refv in enumerate([10,20,30,40,50]):

        c10_list = ref_c10s[refv]

        c10_dis = Counter(c10_list)

        xs = []
        ys = []

        for c10 in sorted(c10_dis.keys()):
            xs.append(c10/float(refv))
            ys.append(c10_dis[c10])

        ys = np.array(ys)/float(np.sum(ys))

        # expfunc = lambda t,a,b:b*np.exp(a*t)

        # popt,pcov = scipy.optimize.curve_fit(expfunc,xs,ys,p0=(0.1,-1))

        # print popt
        # ax = axes[i/3,i%3]
        # xs = range(1,len(c10_list)+1)
        plt.plot(xs,ys,label=u'$\widehat{\langle v(refs) \\rangle}$=%d'%refv )

    plt.yscale('log')

    plt.xlabel(u'价值系数$\hat{\lambda}$',fontproperties='SimHei')
    plt.ylabel(u'比例',fontproperties='SimHei')
    plt.xscale('log')

    plt.xlim(-1,7)
    # plt.xticks(range(len(xs)))
    # plt.set_xticklabels([int(x) for x in xs])
    # plt.spines['top'].set_visible(False)
    # plt.spines['right'].set_visible(False)
    # ax.set_xlim(-1,19)

    # ax.legend(loc=2)
    plt.legend(prop={'family':'SimHei','size':8})


    plt.tight_layout()

    plt.savefig('fig/lambda_dis.png',dpi=800)

    print 'fig saved to fig/lambda_dis.png'

    ### 归一化之后的价值系数图
    plt.figure(figsize=(4,3.2))
    # styles = ['-']
    for i,refv in enumerate([10,20,30,40,50]):

        c10_list = ref_c10s[refv]

        c10_dis = Counter(c10_list)

        xs = []
        ys = []

        for c10 in sorted(c10_dis.keys()):
            xs.append((c10/float(_max_c10))/(float(refv)/_max_refv))
            ys.append(c10_dis[c10])

        ys = np.array(ys)/float(np.sum(ys))

        # expfunc = lambda t,a,b:b*np.exp(a*t)

        # popt,pcov = scipy.optimize.curve_fit(expfunc,xs,ys,p0=(0.1,-1))

        # print popt
        # ax = axes[i/3,i%3]
        # xs = range(1,len(c10_list)+1)
        plt.plot(np.array(xs),ys,label=u'$\widehat{\langle v(refs) \\rangle}$=%d'%refv )

    plt.yscale('log')

    plt.xlabel(u'价值系数$\hat{\lambda}$',fontproperties='SimHei')
    plt.ylabel(u'比例',fontproperties='SimHei')
    plt.xscale('log')

    plt.xlim(-1,7)
    # plt.xticks(range(len(xs)))
    # plt.set_xticklabels([int(x) for x in xs])
    # plt.spines['top'].set_visible(False)
    # plt.spines['right'].set_visible(False)
    # ax.set_xlim(-1,19)

    # ax.legend(loc=2)
    plt.legend(prop={'family':'SimHei','size':8})


    plt.tight_layout()

    plt.savefig('fig/lambda_dis_normed.png',dpi=800)

    print 'fig saved to fig/lambda_dis_normed.png'

    # return

    ps =  pearsonr(c10l,refvs)[0]

    print ps

    X = sm.add_constant(c10l)
    model = sm.OLS(refvs,X)
    res = model.fit()
    print(res.summary())

    linear_func = lambda x,a,b:a*x+b
    popt,pcov = scipy.optimize.curve_fit(linear_func,c10l,refvs)
    print popt


    plt.figure(figsize=(4,3.2))
    plt.plot(c10l,refvs,'o',alpha=0.7,label='皮尔逊相关系数:%.4f' % float(ps))
    xs = range(1,2000)
    ys = [linear_func(x,*popt) for x in xs ]
    plt.plot(xs,ys,'--',linewidth=3,label=u'y=%.3fx+%.3f,$R^2=$%.3f'%(popt[0],popt[1],0.031))
    plt.xlabel('$c_{10}$')
    plt.ylabel('$\langle v(refs) \\rangle$')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(prop={'family':'SimHei','size':8})
    plt.tight_layout()

    plt.savefig('fig/c10_ref_relations.png',dpi=800)

    print 'relation fig saved to fig/c10_ref_relations.png'


    ## 对价值系数进行估计
    vc_dict = Counter(v_coef_list)
    xs = []
    ys = []
    for vc in sorted(vc_dict.keys()):
        xs.append(vc)
        ys.append(vc_dict[vc])

    fit=powerlaw.Fit(np.array(v_coef_list),discrete=True,xmin=1)

    print 'xmin:',fit.power_law.xmin
    print 'alpha:',fit.power_law.alpha
    print 'sigma:',fit.power_law.sigma

    print 'compare:',fit.distribution_compare('power_law', 'exponential')
    print 'compare:',fit.distribution_compare('truncated_power_law', 'exponential')
    print 'compare:',fit.distribution_compare('lognormal', 'exponential')


    ### 拟合lognorm
    ys = np.array(ys)/float(np.sum(ys))
    scale,loc,sigma,mu,mode = fit_lognorm(v_coef_list)

    ## 根据拟合的lognorm进行概率计算，存储到文件用于仿真
    lambda_dis = {}
    fitted_xs = np.linspace(0.001,25,10000)
    pdf_fitted = scipy.stats.lognorm.pdf(fitted_xs, sigma, loc=0, scale=scale)
    pdf_fitted_ys = np.array(pdf_fitted)/np.sum(pdf_fitted)
    lambda_dis['x'] = list(fitted_xs*(_max_c10/_max_refv))
    lambda_dis['y'] = list(pdf_fitted_ys)
    # print 'X:',fitted_xs[sorted(range(len(fitted_xs)),key=lambda x:pdf_fitted_ys[x],reverse=True)[0]]
    print 'length of lambdas',len(xs),min(xs),max(xs)
    open('data/lambda_dis.json','w').write(json.dumps(lambda_dis))
    print 'data saved to data/lambda_dis.json'


    plt.figure(figsize=(4,3.2))
    plt.plot(np.array(xs)*_base_lambda,ys,'o',fillstyle='none')

    ## 这里需要对lognorm得到的数据的和归一化为1
    pdf_fitted = scipy.stats.lognorm.pdf(np.array(xs), sigma, loc=0, scale=scale)
    pdf_fitted = np.array(pdf_fitted)/np.sum(pdf_fitted)

    ## 将X平移值真正的lambda
    plt.plot(np.array(xs)*_base_lambda,pdf_fitted,'--',linewidth=2,label=u'拟合曲线$scale=%.2f,\sigma=%.2f$' %(scale*_base_lambda,sigma))
    x0=mode*_base_lambda

    print 'mode:',mode
    plt.plot([x0]*10,np.linspace(0.000001,0.01,10),'-.',label='$\lambda_{0}=%.3f$'%x0,c='r')
    plt.yscale('log')
    plt.xscale('log')

    plt.xlabel('$\lambda$')
    plt.ylabel('$P(\lambda)$')
    plt.legend(prop={'family':'SimHei','size':8})

    plt.tight_layout()

    plt.savefig('fig/lambda_fit.png',dpi=800)

    print 'lambda fit figure saved to lambda_fit.png'


### 不同年份的论文 系数的变化
def year_lambdas():

    pid_refs = json.loads(open('data/pid_refs.json').read())
    pid_cn = json.loads(open('data/pid_cn.json').read())
    pid_year = json.loads(open('data/paper_year.json').read())

    year_c10_ref = defaultdict(list)
    year_nums = defaultdict(int)

    c10l = []
    refvs = []
    for pid in pid_refs.keys():

        year = pid_year.get(pid,-1)

        if year==-1:
            continue

        year_nums[year]+=1
        ## 参考价值
        refs = pid_refs[pid]
        ref_vs = sorted([pid_cn[p] for p in refs],reverse=True)
        ris,rvs = zscore_outlier(ref_vs,2.5)
        refv = np.mean(rvs)
        ## 真实价值
        c10 = pid_cn[pid]

        c10l.append(c10)
        refvs.append(refv)

        ## lambda
        # v_coef = float('{:.3f}'.format(c10/float(refv)))

        year_c10_ref[year].append([c10,refv])



    _max_c10 = float(np.max(c10l))
    _max_refv = np.max(refvs)
    _base_lambda = _max_c10/_max_refv

    print '_max_c10:',_max_c10,'_max_refv:',_max_refv,'_base_lambda:',_base_lambda


    year_vcoef_list = defaultdict(list)

    for year in year_c10_ref.keys():

        for c10,refv in year_c10_ref[year]:
            year_vcoef_list[year].append((c10/_max_c10)/(float(refv)/_max_refv))


    ## 不同的年份的文章数量
    xs = []
    ys = []
    zone_num = defaultdict(int)
    for year in sorted(year_nums.keys()):
        xs.append(year)
        ys.append(year_nums[year])
        zone = year_zone(year)
        zone_num[zone]+= year_nums[year]

    zone_xs = []
    zone_ys = []
    zone_labels = ['-1960','1960-1970','1970-1975','1976-1980','1981-1985','1986-1990','1991-1993','1994-1995','1996-1997','1998-1999','2000-2001','2002','2003','2004','2005','2006']

    for zone in sorted(zone_num.keys()):
        zone_xs.append(zone)
        zone_ys.append(zone_num[zone])

    fig,axes = plt.subplots(1,2,figsize=(8,4))
    ax0 = axes[0]
    ax0.plot(xs,ys)
    ax0.set_xlabel(u'年份\n(a)',fontproperties='SimHei')
    ax0.set_ylabel(u'文章数量',fontproperties='SimHei')

    ax1 = axes[1]

    ax1.bar(zone_xs,zone_ys,width=0.6)
    ax1.set_xlabel(u'年份区间\n(b)',fontproperties='SimHei')
    ax1.set_ylabel(u'文章数量',fontproperties='SimHei')
    ax1.set_xticks(zone_xs)
    ax1.set_xticklabels(zone_labels,rotation=-90)

    plt.tight_layout()

    plt.savefig('fig/lambda_year_num.jpg',dpi=800)
    print 'fig saved to fig/lambda_year_num.jpg'
    # return
    ##对于不同的年份画出lambda的分布以及拟合
    zone_lambdas = defaultdict(list)
    for year in  sorted(year_vcoef_list.keys()):

        zone = year_zone(year)
        zone_lambdas[zone].extend(year_vcoef_list[year])

    fig,axes = plt.subplots(4,4,figsize=(14,12.8))

    x0s = []
    ## 对每一个zone的lambda进行统计
    for i,zone in enumerate(sorted(zone_lambdas.keys())):

        ax = axes[i/4,i%4]

        lambdas = zone_lambdas[zone]
        size = len(lambdas)
        print 'zone:',zone,',size:',size

        xs=[]
        ys = []
        l_counter = Counter([float('{:.3f}'.format(l)) for l in lambdas])
        for l in sorted(l_counter):
            xs.append(l)
            ys.append(l_counter[l]/float(size))

        ax.plot(np.array(xs)*_base_lambda,ys,'o',fillstyle='none',label=u'价值系数散点图')
        ax.set_xlabel(u'$\lambda$',fontproperties='SimHei')
        ax.set_ylabel(u'$P(\lambda)$',fontproperties='SimHei')

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_title('%s (%d)' %(zone_labels[zone],size))
        # ax.set_ylim(0,25)
        # ax.legend(prop={'family':'SimHei','size':8})

        # param=scipy.stats.lognorm.fit(lambdas,floc=0)
        # fit_lognorm(lambdas)
        scale,loc,sigma,mu,mode = fit_lognorm(lambdas)

        pdf_fitted = scipy.stats.lognorm.pdf(xs, sigma, loc=0, scale=scale)
        pdf_fitted = np.array(pdf_fitted)/np.sum(pdf_fitted)
        x0s.append(mode*_base_lambda)

        ax.plot(np.array(xs)*_base_lambda,pdf_fitted,label=u'拟合曲线$scale=%.2f,\sigma=%.2f$' %(scale*_base_lambda,sigma))
        ax.plot([mode*_base_lambda]*10,np.linspace(np.min(pdf_fitted),0.1,10),'--',label='$\lambda_0 = {:.2f}$'.format(mode*_base_lambda))

        ax.legend(prop={'family':'SimHei','size':8})

    plt.tight_layout()

    plt.savefig('fig/lambda_zone_dis.jpg',dpi=800)

    print 'zone lambda dis saved to fig/lambda_zone_dis.jpg'

    plt.figure(figsize=(3.5,3.2))

    plt.plot(range(len(x0s)),x0s,label=u'各年份$\lambda_{0}$')
    plt.plot(range(len(x0s)),[0.091]*len(x0s),'--',c='r',label=u'整体$\lambda_{0}$')

    plt.xticks(range(len(x0s)),zone_labels,rotation=-90)
    plt.xlabel(u'年份区间',fontproperties='SimHei')
    plt.ylabel(u'$\lambda_0$',fontproperties='SimHei')
    plt.legend(prop={'family':'SimHei','size':8})

    plt.tight_layout()

    plt.savefig('fig/lambda_zone_x0_dis.jpg',dpi=800)
    print 'lambda zone dis saved to fig/lambda_zone_x0_dis.jpg'



def year_zone(year):

    if year <=1960:
        return 0
    elif year <=1970:
        return 1
    elif year <=1975:
        return 2
    elif year <=1980:
        return 3
    elif year <=1985:
        return 4
    elif year <=1990:
        return 5
    elif year <=1993:
        return 6
    elif year <=1995:
        return 7
    elif year <=1997:
        return 8
    elif year <=1999:
        return 9
    elif year <=2001:
        return 10
    elif year <=2002:
        return 11
    elif year <=2003:
        return 12
    elif year <=2004:
        return 13
    elif year <=2005:
        return 14
    else:
        return 15




if __name__ == '__main__':
    # record_paper_year()

    # gen_dataset()

    # ref_cit_relations()

    year_lambdas()

    # compare_citation_dis()





