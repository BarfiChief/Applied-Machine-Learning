# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 14:05:34 2020

@author: hp
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statistics as st
#getting the dataset
dataset=pd.read_csv('thigh_training.csv')
dataset = dataset.drop("Unnamed", axis=1)
y=[]
s=[]
    
for i in range(0,902900):
    
    y.append(dataset.attr_y_acc[i])
    if (i+1)%100==0:
        std_dev=st.stdev(y)
        for j in range(0,100):
            s.append(std_dev)
        y=[]
row_rem=[]    
dataset['STANDARD_DEVIATION']=s
thresholds=[1.2,1.2,3.5,0.5,5,1,0,1.5]
for i in range (0,902900):
    j=dataset.Label[i]
    if j!=6:
        if j!=3 and j!=5 :
            if dataset.STANDARD_DEVIATION[i]<thresholds[j]:
                    row_rem.append(i)
        else:
            if dataset.STANDARD_DEVIATION[i]>thresholds[j]:
                    row_rem.append(i)
dataset=dataset.drop(row_rem,axis=0)
dataset.to_csv('./thigh_training_final.csv')


    
            
        
    