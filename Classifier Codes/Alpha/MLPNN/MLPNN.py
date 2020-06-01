# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:51:53 2019

@author: Axel Blaze
"""
import numpy as np  
import pandas as pd  
#from pandas_datareader import data as wb   
#importing Dataset
dataset = pd.read_excel(r'E:\Project\Machine Learning\Project 1-Major Depressive Disorder\Dataset\alpha power.xlsx')
df=pd.read_excel(r'E:\Project\Machine Learning\Project 1-Major Depressive Disorder\Dataset\beta_power.xlsx')
#seeing the charts
X=dataset.iloc[:,0:19]
y=dataset.iloc[:,19]
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.01, random_state = False)
#applying MPLNN classifier
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(activation='relu',hidden_layer_sizes=(14,14,14,14,14),max_iter=500,learning_rate='constant')
mlp.fit(X_train,y_train)
y_pred = mlp.predict(X_test)
# Making the Confusion Matrix
'''from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
a=cm[0][0]
b=cm[0][1]
c=cm[1][0]
d=cm[1][1]
#calculation Error,Accuracy,Sensitivity and Prediction
Err=(b+c)/(a+b+c+d)
Acc=(a+d)/(a+b+c+d)
SN=a/(a+c)
Prec=a/(a+b)
print(cm)
print('Accuracy=',Acc*100,'%')
print('Error=',Err*100,'%')
print('Sensitivity=',SN*100,'%')
print('Prediction=',Prec*100,'%')'''