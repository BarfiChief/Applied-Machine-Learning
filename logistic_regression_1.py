# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 00:10:04 2020

@author: hp
"""
#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statistics as st
#importing training and test sets
#training set
train_set=pd.read_csv('thigh_training_final.csv')
train_set = train_set.drop("Unnamed: 0", axis=1)
#separating features and dependent variables
X_train=train_set.iloc[:,[1,2,3,5,6,7]]
Y_train=train_set.iloc[:,9]
#test set
test_set=pd.read_csv('thigh_testing_final.csv')
test_set = test_set.drop("Unnamed: 0", axis=1)
#separating features and dependent variables
X_test=test_set.iloc[:,[1,2,3,5,6,7]]
Y_test=test_set.iloc[:,9]
#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)
#fitting logistic regression
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,Y_train)
#predicting training set results and thus finding accuracy
y_pred=classifier.predict(X_train)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_train,y_pred)
correct_train=0
for i in range (0,8):
    correct_train=correct_train+cm[i,i]
accuracy_train=(correct_train/Y_train.size)*100
#predicting test set results and thus finding accuracy
y_pred=classifier.predict(X_test)
cm=confusion_matrix(Y_test,y_pred)
correct_test=0
for i in range (0,8):
    correct_test=correct_test+cm[i,i]
accuracy_test=(correct_test/Y_test.size)*100
