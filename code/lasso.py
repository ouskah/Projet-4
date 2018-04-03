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

# function to compute mae
def compute_error_lasso(x): 
    errors_lassos_cv = np.ones(len(x))
    for i in range(0, len(x)):
        errors_lassos_cv[i] = np.absolute(x[i] - y_test[i])
    baseline_lasso = np.mean(errors_lassos_cv)
    return baseline_lasso


n_alphas = 50
alphas = np.logspace(-10, 20, n_alphas)

#compute auto Cross Validation For linear regression with lasso penalization
lassoCV = linear_model.LassoCV(alphas = alphas)
lassoCV.fit(X_train, y_train)
x = lassoCV.predict(X_test)
baseline_lasso = compute_error_lasso(x)


#compute auto Cross Validation For linear regression with lasso penalization
coefs = []
errors = []
scores = []
lasso = linear_model.Lasso()
for alpha in alphas : 
    print('################## ALPHA = ' + str(alpha) + '##############')
    lasso.set_params(alpha = alpha)
    lasso.fit(X_train, y_train)
    coefs.append(lasso.coef_)
    x = lasso.predict(X_test)
    errors.append([baseline_error, baseline_lasso, compute_error_lasso(x)])
   
# plot errors with linear regression and linear regression with lasso penalization   
labels = ['Baseline', 'Cross-Validation','Lasso']
import matplotlib.pyplot as plt
ax = plt.gca()
ax.plot(alphas, coefs)
ax.legend()
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('errors')
plt.title('Ridge Coefficients as a function of the regularization')
plt.axis('tight')
plt.show()
