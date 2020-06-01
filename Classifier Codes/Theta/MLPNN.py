# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:51:53 2019

@author: Axel Blaze
"""
import pandas as pd  
dataset = pd.read_excel(r'F:\Project\Machine Learning & Artificial Intelligence\Project 1-Major Depressive Disorder\Dataset\theta_power.xlsx') 
X=dataset.iloc[:,0:19]
#X=x;
y=dataset.iloc[:,19]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

model= Sequential()
model.add(Dense(100, input_dim=19, kernel_initializer='normal' , activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(45, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(35, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mse','mae'])


model.fit(X_train,y_train, epochs=500, batch_size=2,  verbose=1, validation_split=0.2)

'''from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=400)
mlp.fit(X_train,y_train)
MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(13, 13, 13), learning_rate='constant',
       learning_rate_init=0.001, max_iter=500, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)'''
# Predicting the Test set results
eval_model=model.evaluate(X_train, y_train)
eval_model
y_pred = model.predict(X_test)
y_pred =(y_pred>0.5)


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