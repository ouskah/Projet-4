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
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

#compute linear regression
from sklearn import linear_model
lr = linear_model.LinearRegression()
lr.fit(X_train, y_train)

#compute error for linear regression
import numpy as np
baseline_error = np.mean(np.absolute(lr.predict(X_test) - y_test))

n_alphas = 50
alphas = np.logspace(-10, 10, n_alphas)

#compute auto Cross Validation For linear regression with ridge penalization
ridgeCV = linear_model.RidgeCV(alphas = alphas)
ridgeCV.fit(X_train, y_train)
baseline_ridge = np.mean(np.absolute(ridgeCV.predict(X_test) - y_test))

#compute Cross Validation For linear regression with ridge penalization
coefs = []
errors = []
scores = []
ridge = linear_model.Ridge()
for a in alphas:
    print('################## ALPHA = ' + str(a) + '##############')
    ridge.set_params(alpha=a)
    ridge.fit(X_train, y_train)
    coefs.append(ridge.coef_)
    errors.append([baseline_error, baseline_ridge, np.mean(np.absolute(ridge.predict(X_test) - y_test))])
  
# plot errors with linear regression and linear regression with ridge penalization  
labels = ['Baseline', 'Ridge Cross-Validation','Ridge']
errors = np.array(errors).astype(float)
import matplotlib.pyplot as plt
ax = plt.gca()
for i in range(0, 3): 
    ax.plot(alphas, errors[:,i], label = labels[i])
ax.legend()
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('errors')
plt.title('Ridge errors as a function of the regularization')
plt.axis('tight')
plt.show()
