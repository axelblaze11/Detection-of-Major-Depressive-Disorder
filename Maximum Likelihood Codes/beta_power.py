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
#importing Dataset
dataset = pd.read_excel('E:/Project/Machine Learning/Data Set/Depression Dataset/beta_power.xlsx')
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
#graph plotting
a = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
plt.plot(a,lh, marker='o')
plt.xticks(np.arange(0, 20, 1))
plt.show()   
plt.plot(a,llh, marker='o')
plt.xticks(np.arange(0, 20, 1))
plt.show()   
X=dataset.iloc[:,[0,1,10,11,16,18]]
#X=x;
y=dataset.iloc[:,19]
#X=dataset.iloc[:,[0,3,13]]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.svm import SVC
classifier=SVC(kernel='linear')
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
a=cm[0][0]
b=cm[0][1]
c=cm[1][0]
d=cm[1][1]
Err=(b+c)/(a+b+c+d)
Acc=(a+d)/(a+b+c+d)
SN=a/(a+c)
Prec=a/(a+b)
print(cm)
print(Acc)
print(Err)
print(SN)
print(Prec)
