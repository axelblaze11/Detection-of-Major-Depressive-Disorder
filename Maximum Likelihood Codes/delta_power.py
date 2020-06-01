# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 14:16:34 2019

@author: Axel Blaze
"""
import math
import numpy as np  
import pandas as pd  
#from pandas_datareader import data as wb  
import matplotlib.pyplot as plt  
from scipy.stats import norm
dataset = pd.read_excel('E:/Project/Machine Learning/Data Set/Depression Dataset/delta_pow.xlsx')  
x=dataset.loc[:,"ch1":"ch19"]
#calculation mean and standard deviation
mean=x.mean()
std=x.std()
#calculating normal distribution using cdf
normal=norm.cdf(x,loc=mean,scale=std)
#calculation likelihood values
df = pd.DataFrame(normal)
lh=np.product(df)
#calculating loglikelihood
llh=np.log(lh)
b=abs(llh)
m=np.median(b)
#graph plotting
a = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
plt.plot(a,lh, marker='o')
plt.xticks(np.arange(0, 20, 1))
plt.show()   
plt.plot(a, llh, marker='o')
plt.xticks(np.arange(0, 20, 1))
plt.show()   