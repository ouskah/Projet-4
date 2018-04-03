# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 10:53:36 2018

@author: zakis
"""
#load data file
import pandas as pd
data = pd.read_csv('2016_preprocessed.csv')
data = data.drop('Unnamed: 0', 1)

#Enable the following line to remove outliers
#data = data[data[['ARR_DELAY']].values < 60]

#create x and y
x = data.drop('ARR_DELAY',1).values
y = data[['ARR_DELAY']].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#scaling the data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler().fit(X_train)
X_train = sc_X.transform(X_train)
X_test = sc_X.transform(X_test)

import numpy as np
# function to compute mae
def mae(x): 
    errors_lassos_cv = np.ones(len(x))
    for i in range(0, len(x)):
        errors_lassos_cv[i] = np.absolute(x[i] - y_test[i])
    baseline_lasso = np.mean(errors_lassos_cv)
    return baseline_lasso

from sklearn import linear_model

#SGD regressor with huber function as lost function, and a lasso penalization with alpha = 0.00046
SGD = linear_model.SGDRegressor(loss='huber', alpha=0.00046, l1_ratio=0)
SGD.fit(X_train, y_train)
y_pred = SGD.predict(X_test)
baseline_SGD = mae(y_pred)

#create file for api
import pickle
filehandler = open("fonction_std_scaling.pyc", 'wb')
pickle.dump(sc_X, filehandler)

filehandler = open("fonction_prediction.pyc", 'wb')
pickle.dump(SGD, filehandler)

"""
d = data.drop('ARR_DELAY', 1)
columns = np.array(d.columns).astype(str).reshape((len(d.columns), 1))
values = np.concatenate([SGD.coef_.reshape((len(SGD.coef_), 1)), columns], axis = 1)
coefficient = pd.DataFrame(values, columns = ['COEFS','VARIABLES'])
coefficient.to_csv('coefs.csv')
"""

