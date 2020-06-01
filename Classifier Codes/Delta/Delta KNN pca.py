# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 12:55:47 2019

@author: Axel Blaze
"""

import math
import numpy as np  
import pandas as pd  
from pandas_datareader import data as wb  
import matplotlib.pyplot as plt  
from scipy.stats import norm
dataset = pd.read_excel('E:/Project/Machine Learning/Data Set/Depression Dataset/delta_pow.xlsx')  
X=dataset.iloc[:,0:19]
#X=x;
y=dataset.iloc[:,19]
#X=dataset.iloc[:,[0,3,13]]
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
from sklearn.decomposition import PCA
pca= PCA(n_components = 8)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5)
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
