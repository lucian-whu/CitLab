#coding:utf-8
import os
import sys
import json
from collections import defaultdict
from collections import Counter
import math
import numpy as np
import random
import logging
import networkx as nx
from networkx.algorithms import isomorphism
from collections import Counter
import scipy
from scipy.stats import zscore


'''
==================
## logging的设置，INFO
==================
'''
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

'''
==================
### 示意图的画图方法
==================
'''
from viz_graph import *



'''
==================
### 数据库
==================
'''
from database import *



'''
==================
### 路径
==================
'''
from paths import *


'''
==================
## pyplot的设置
==================
'''
from plot_config import *

## 根据阈值及zscore对数据进行循环去除，直到不改变为止
def zscore_outlier(alist,zv=2):

    reserved_indexes = range(len(alist))
    last_list = alist
    num = 0
    while True:
        zs = zscore(last_list)
        new_indexes = []
        for j,i in enumerate(reserved_indexes):
            if abs(zs[j])>zv:
                continue

            new_indexes.append(i)

        newlist = [alist[i] for i in new_indexes]

        num+=1
        # print num
        # print new_indexes,len(new_indexes)
        # print reserved_indexes,len(reserved_indexes)
        # print '---'

        if len(new_indexes)==len(reserved_indexes):
            break

        reserved_indexes = new_indexes
        last_list = newlist

    return  [i+1 for i in new_indexes],newlist



## 拟合lognorm函数
def fit_lognorm(data):
    sigma,loc,scale=scipy.stats.lognorm.fit(data,floc=0)
    mu = np.log(scale)
    mode = np.exp(mu-sigma**2)
    ## 返回参数 scale,loc,sigma,mode
    return scale,loc,sigma,mu,mode





