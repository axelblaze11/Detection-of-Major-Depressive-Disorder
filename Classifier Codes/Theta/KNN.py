# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 02:48:51 2019

@author: Axel Blaze
"""

import math
import numpy as np  
import pandas as pd  
from pandas_datareader import data as wb  
import matplotlib.pyplot as plt  
from scipy.stats import norm
dataset = pd.read_excel('E:/Project/Machine Learning/Data Set/Depression Dataset/theta_power.xlsx') 
X=dataset.iloc[:,0:19]
#X=x;
y=dataset.iloc[:,19]
#PCA
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# KNN classifier to the Training set
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
print('Accuracy=',Acc*100,'%')
print('Error=',Err*100,'%')
print('Sensitivity=',SN*100,'%')
print('Prediction=',Prec*100,'%')
