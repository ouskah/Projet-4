# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 19:00:48 2018

@author: zakis
"""


import pandas as pd
import numpy as np
from creation_origin_airport_class import creation_origin_airport_class, creation_dest_airport_class

def create_nb_flight_arr_same_hour(d):
    nb_flight_arr_same_hour = np.ones((len(d),1))
    for iden in np.unique(d[['DEST_AIRPORT_ID']].values):
        arg1 = np.argwhere(reshape(d[['DEST_AIRPORT_ID']].values == iden))
        arg1 = reshape(arg1)
        d1 = d.iloc[arg1,:]
        for month in np.unique(d1[['MONTH']].values):
            arg2 = np.argwhere(reshape(d1[['MONTH']].values == month))
            arg2 = reshape(arg2)
            d2 = d1.iloc[arg2,:]
            for day in np.unique(d2[['DAY_OF_MONTH']].values):
                arg3 = np.argwhere(reshape(d2[['DAY_OF_MONTH']].values == day))
                arg3 = reshape(arg3)
                d3 = d2.iloc[arg3,:]
                for dep_time in  np.unique(d3[['CRS_ARR_TIME']]):
                    arg4 = np.argwhere(reshape(d3[['CRS_ARR_TIME']].values == dep_time))
                    arg4 = reshape(arg4)
                    arg = arg1[arg2[arg3[arg4]]]
                    nb_flight_arr_same_hour[arg] = len(arg)
    nb_flight_arr_same_hour = np.array(nb_flight_arr_same_hour).astype(int)
    NB_FLIGHT_ARR_SAME_HOUR = pd.DataFrame(nb_flight_arr_same_hour, index = d.index, columns = ['NB_FLIGHT_ARR_SAME_HOUR'])
    return NB_FLIGHT_ARR_SAME_HOUR


def create_nb_flight_dep_same_hour(d):
    nb_flight_dep_same_hour = np.ones((len(d),1))
    for iden in np.unique(d[['ORIGIN_AIRPORT_ID']].values):
        arg1 = np.argwhere(reshape(d[['ORIGIN_AIRPORT_ID']].values == iden))
        arg1 = reshape(arg1)
        d1 = d.iloc[arg1,:]
        for month in np.unique(d1[['MONTH']].values):
            arg2 = np.argwhere(reshape(d1[['MONTH']].values == month))
            arg2 = reshape(arg2)
            d2 = d1.iloc[arg2,:]
            for day in np.unique(d2[['DAY_OF_MONTH']].values):
                arg3 = np.argwhere(reshape(d2[['DAY_OF_MONTH']].values == day))
                arg3 = reshape(arg3)
                d3 = d2.iloc[arg3,:]
                for dep_time in  np.unique(d3[['CRS_DEP_TIME']]):
                    arg4 = np.argwhere(reshape(d3[['CRS_DEP_TIME']].values == dep_time))
                    arg4 = reshape(arg4)
                    arg = arg1[arg2[arg3[arg4]]]
                    nb_flight_dep_same_hour[arg] = len(arg)
    nb_flight_dep_same_hour = np.array(nb_flight_dep_same_hour).astype(int)
    NB_FLIGHT_DEP_SAME_HOUR = pd.DataFrame(nb_flight_dep_same_hour, index = d.index, columns = ['NB_FLIGHT_DEP_SAME_HOUR'])
    return NB_FLIGHT_DEP_SAME_HOUR

def reshape(arg):
    return np.reshape(arg,(np.shape(arg)[0],))

name_file = '2016'
print('PREPROCESSING FILE -> '+name_file)        
dataset =   pd.read_csv(name_file+'.csv')
print('FILE RED -> '+name_file) 

dataset = dataset[['QUARTER', 'MONTH','DAY_OF_MONTH','DAY_OF_WEEK','UNIQUE_CARRIER','ORIGIN_AIRPORT_ID','DEST_AIRPORT_ID','CRS_DEP_TIME','CRS_ARR_TIME','ARR_DELAY','CANCELLED','DISTANCE']]

dataset = dataset[(~dataset[['ARR_DELAY']].isnull()).values]
dataset = dataset[dataset[['CANCELLED']].values ==0]
print('UNUSEFULL DATA DELETED -> '+name_file) 

dataset[['CRS_DEP_TIME']] = (dataset[['CRS_DEP_TIME']]/100).astype(int)
dataset[['CRS_ARR_TIME']] = (dataset[['CRS_ARR_TIME']]/100).astype(int)


NB_FLIGHT_DEP_SAME_HOUR = create_nb_flight_dep_same_hour(dataset)
print('NB_FLIGHT_DEP_SAME_HOUR CREATED') 

NB_FLIGHT_ARR_SAME_HOUR = create_nb_flight_arr_same_hour(dataset)
print('NB_FLIGHT_ARR_SAME_HOUR CREATED')


ORIGIN_AIRPORT_CLASS = creation_origin_airport_class(dataset)
DEST_AIRPORT_CLASS = creation_dest_airport_class(dataset)
print('AIRPORT CLASS VARIABLE CREATED') 

values = np.concatenate([dataset.values, NB_FLIGHT_DEP_SAME_HOUR.values, NB_FLIGHT_ARR_SAME_HOUR.values, ORIGIN_AIRPORT_CLASS.values, DEST_AIRPORT_CLASS.values], axis = 1)
columns = np.concatenate([dataset.columns, NB_FLIGHT_DEP_SAME_HOUR.columns, NB_FLIGHT_ARR_SAME_HOUR.columns, ORIGIN_AIRPORT_CLASS.columns, DEST_AIRPORT_CLASS.columns])
dataset = pd.DataFrame(values, index = dataset.index, columns = columns)

print('CATEGORICAL VARIABLE CREATION') 
QUARTER = pd.get_dummies(dataset[['QUARTER']])
MONTH = pd.get_dummies(dataset[['MONTH']])
DAY_OF_MONTH = pd.get_dummies(dataset[['DAY_OF_MONTH']])
DAY_OF_WEEK = pd.get_dummies(dataset[['DAY_OF_WEEK']])
UNIQUE_CARRIER = pd.get_dummies(dataset[['UNIQUE_CARRIER']])
CRS_ARR_TIME = pd.get_dummies(dataset[['CRS_ARR_TIME']])
CRS_DEP_TIME = pd.get_dummies(dataset[['CRS_DEP_TIME']])
ORIGIN_AIRPORT_CLASS = pd.get_dummies(dataset[['ORIGIN_AIRPORT_CLASS']])
DEST_AIRPORT_CLASS = pd.get_dummies(dataset[['DEST_AIRPORT_CLASS']])

print('PRINTING...') 

values = np.concatenate([QUARTER.values,
                         MONTH.values,
                         DAY_OF_MONTH.values,
                         DAY_OF_WEEK.values,
                         UNIQUE_CARRIER.values,
                         CRS_ARR_TIME.values,
                         CRS_DEP_TIME.values,
                         ORIGIN_AIRPORT_CLASS.values,
                         DEST_AIRPORT_CLASS.values,
                         dataset[['ARR_DELAY','DISTANCE','NB_FLIGHT_ARR_SAME_HOUR', 'NB_FLIGHT_DEP_SAME_HOUR']].values],
    axis = 1)
columns = np.concatenate([QUARTER.columns,
                         MONTH.columns,
                         DAY_OF_MONTH.columns,
                         DAY_OF_WEEK.columns,
                         UNIQUE_CARRIER.columns,
                         CRS_ARR_TIME.columns,
                         CRS_DEP_TIME.columns,
                         ORIGIN_AIRPORT_CLASS.columns,
                         DEST_AIRPORT_CLASS.columns,
                         dataset[['ARR_DELAY','DISTANCE','NB_FLIGHT_ARR_SAME_HOUR', 'NB_FLIGHT_DEP_SAME_HOUR']].columns])
dataset = pd.DataFrame(values, index = dataset.index, columns = columns)

dataset.to_csv(name_file+'_preprocessed.csv')
print('FINISHED...') 
dataset = dataset.sample(frac=0.1)
dataset.to_csv('2016_preprocessed_sampled.csv')


    
    

"""
X = dataset.drop('ARR_DELAY',1).values
y = dataset[['ARR_DELAY']].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

import statsmodels.api as sm
X_opt = sm.add_constant(X_train)
est = sm.OLS(y_train, X_opt).fit()
print(est.summary())
"""