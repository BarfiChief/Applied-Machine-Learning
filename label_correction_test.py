# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:31:55 2020

@author: hp
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statistics as st
#getting the dataset
dataset=pd.read_csv('thigh_testing.csv')
dataset = dataset.drop("Unnamed: 0", axis=1)
y=[]
s=[]
    
for i in range(0,431600):
    
    y.append(dataset.attr_y_acc[i])
    if (i+1)%100==0:
        std_dev=st.stdev(y)
        for j in range(0,100):
            s.append(std_dev)
        y=[]
row_rem=[]    
dataset['STANDARD_DEVIATION']=s
thresholds=[1.2,1.2,3.5,0.5,5,1,0,1.5]
for i in range (0,431600):
    j=dataset.Label[i]
    if j!=6:
        if j!=3 and j!=5 :
            if dataset.STANDARD_DEVIATION[i]<thresholds[j]:
                    row_rem.append(i)
        else:
            if dataset.STANDARD_DEVIATION[i]>thresholds[j]:
                    row_rem.append(i)
dataset=dataset.drop(row_rem,axis=0)
dataset.to_csv('./thigh_testing_final.csv')