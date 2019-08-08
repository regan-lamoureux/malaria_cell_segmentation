#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 14:17:04 2019

@author: reganlamoureux
"""

import numpy as np
import os
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

def compare():
    folder = os.fsdecode('/Users/reganlamoureux/comparison data')
    for file in os.listdir(folder):
        print(file)
        if file == '.DS_Store':
            pass
        else:   
            data = pd.read_csv('/Users/reganlamoureux/comparison data'+'/'+file)
            cols = ['contrast', 'dissimilarity', 'homogeniety', 'asm', 'energy', 'correlation', 'ent']
            X = data[cols]
            y = data['y']

            #logit_model=sm.Logit(y,X)
            #result=logit_model.fit()
            #print(result.summary2())

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        
            logreg = LogisticRegression()
            logreg.fit(X_train, y_train)

            y_pred = logreg.predict(X_test)
            print('Accuracy of logistic regression: {}'.format(logreg.score(X_test, y_test)))

            from sklearn.metrics import confusion_matrix
            confusion_matrix = confusion_matrix(y_test, y_pred)
            print(confusion_matrix)

            from sklearn.metrics import classification_report
            print(classification_report(y_test, y_pred))
            print('------------------------------------------')
            
data = pd.read_csv('eccentricity.csv')
array = np.asanyarray(data['eccentricity'])
X = array.reshape(-1,1)
y = data['y']
        
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression: {}'.format(logreg.score(X_test, y_test)))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
print('------------------------------------------')