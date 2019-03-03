#coding:utf-8
'''
定义常用的函数

'''
import numpy as np

## 指数
def exp_func(t,a,b):

    return 9*a*np.exp(b*t)

def powlaw(t,a,b):

    return a*t**(-b)

if __name__ == '__main__':
    import copy

    a = range(10)

    b= copy.copy(a)

    print a,b
    b[-1]=10000

    print a,b




    from time import time
    lista=set([1,2,3,4,5,6,7,8,9,13,34,53,42,44])
    listb=set([2,4,6,9,23])
    # a = range(100)
    a = range(10000000)
    b = np.arange(10000000)
    p = np.array(a)/float(np.sum(a))


    print b[0]

    print b[:10]

    # b.append('a')
    d = set(range(10000))

    t1 = time()

    new_b=np.array(filter(lambda x:b[x] in d, b))
    t2 = time()
    print 1,t2-t1


    t1 = time()
    new_a = []
    for i in a:
        if i in d:
            continue
        new_a.append(i)
    t2 = time()
    print 2,t2-t1

    t1 = time()
    new_a = list(set(a)-d)
    t2 = time()
    print 3,t2-t1

    t1 = time()
    for i in d:
        index = a.index(i)
        a[index] = 0

    t2 = time()
    print 4,t2-t1


    # t1 = time()
    # for i in d:
    #     index = np.where(b==i)
    #     # b[index] = 0

    # t2 = time()
    # print 5,t2-t1

    t1 = time()
    sorted(a)

    t2 = time()
    print 6,t2-t1

    # np.random.shuffle(a[:1000])
    for i in range(100):

        a[10*i/5] = 0


    t1 = time()
    sorted(a,reverse=True)

    t2 = time()
    print 6.5,t2-t1

    iss = np.arange(len(a))
    t1 = time()
    sorted(iss,key=lambda x:a[x],reverse=True)

    t2 = time()
    print 7,t2-t1
    iss = range(len(a))
    t1 = time()
    sorted(iss,key=lambda x:a[x])

    t2 = time()
    print 8,t2-t1

    t1 = time()
    # zip(*a)
    np.random.choice(a,size=10,p=p)
    t2 = time()
    print t2-t1

    t1 = time()
    # zip(*a)
    np.random.choice(b,size=10,p=p)
    t2 = time()
    print t2-t1

    t1 = time()
    # zip(*a)
    np.random.choice(a,size=100,p=p)
    t2 = time()
    print t2-t1

    t1 = time()
    # zip(*a)
    np.random.choice(b,size=100,p=p)
    t2 = time()
    print t2-t1

    t1 = time()
    # zip(*a)
    np.random.choice(b,size=1000,p=p)
    t2 = time()
    print t2-t1

    # intersection=[]
    # t = time()
    # for i in range (100):
    #     # list(lista&listb)
    #     # a/np.sum(a)
    #     np.random.choice(a,size=10)

    # print "total run time:"
    # print time()-t2