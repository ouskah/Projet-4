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

#compute mae
def mae(x): 
    errors_lassos_cv = np.ones(len(x))
    for i in range(0, len(x)):
        errors_lassos_cv[i] = np.absolute(x[i] - y_test[i])
    baseline_lasso = np.mean(errors_lassos_cv)
    return baseline_lasso

## FINDING the right proportion between lasso and ridge
n_alphas = 50
alphas = np.logspace(-10, 20, n_alphas)
l1 = np.arange(0,1.1,0.1)
errors = []
best_alphas = []
for l in l1:
    elasticnetCV = linear_model.ElasticNetCV(alphas = alphas, l1_ratio=l1)
    elasticnetCV.fit(X_train, y_train)
    best_alphas.append(elasticnetCV.alpha_)
    errors.append(mae(elasticnetCV.predict(X_test)))

print('best alpha for elastic net = ' + str(alphas[errors.argmin()]))

#printing the correct proportion between lasso and ridge
print('Right proportion for this dataset = ' + l1[errors.argmin()]*100 +'%')
# => lasso is the best fit fot this dataset

