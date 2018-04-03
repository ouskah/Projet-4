 # -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 14:36:05 2018

@author: zakis
"""

#This file is designed to compute the class or every airport corresponding to the mean value of every airport

def creation_origin_airport_class(dataset):
    import pandas as pd
    
    airport_id = dataset[['ORIGIN_AIRPORT_ID']]
    
    import numpy as np
    unique_airport_id = np.unique(airport_id)
    mean = []
    for iden in unique_airport_id:
        d = dataset[['ARR_DELAY']].values[airport_id == iden]
        mean.append(np.mean(d))
        
    mean = np.array(mean).astype(float)
    airport_class = np.ones((len(unique_airport_id), 2))*-1
    airport_class[:,0] = unique_airport_id
    airport_class[mean < -10, 1] = 0
    airport_class[np.logical_and(mean >= -10, mean< 0), 1] = 1
    airport_class[np.logical_and(mean >= 0, mean< 10), 1] = 2
    airport_class[np.logical_and(mean >= 10, mean< 20), 1] = 3
    airport_class[mean >= 20, 1] = 4
    
    values = np.ones((len(dataset),))
    for c in airport_class:
        values[np.argwhere(airport_class == c[0])] = c[1]
        
    ORIGIN_AIRPORT_CLASS = pd.DataFrame(values, index = dataset.index, columns = ['ORIGIN_AIRPORT_CLASS'])
    
    return ORIGIN_AIRPORT_CLASS


def creation_dest_airport_class(dataset):
    import pandas as pd
    
    dest_airport_id = dataset[['DEST_AIRPORT_ID']]
    
    import numpy as np
    unique_airport_id = np.unique(dest_airport_id)
    mean = []
    for iden in unique_airport_id:
        d = dataset[['ARR_DELAY']].values[dest_airport_id == iden]
        mean.append(np.mean(d))
        
    mean = np.array(mean).astype(float)
    dest_airport_class = np.ones((len(unique_airport_id), 2))*-1
    dest_airport_class[:,0] = unique_airport_id
    dest_airport_class[mean < -5, 1] = 0
    dest_airport_class[np.logical_and(mean >= -5, mean< 0), 1] = 1
    dest_airport_class[np.logical_and(mean >= 0, mean< 5), 1] = 2
    dest_airport_class[np.logical_and(mean >= 5, mean< 10), 1] = 3
    dest_airport_class[np.logical_and(mean >= 10, mean< 15), 1] = 4
    dest_airport_class[mean >= 15, 1] = 5
    
    values = np.ones((len(dataset),))
    for c in dest_airport_class:
        values[np.argwhere(dest_airport_id.values == c[0])] = c[1]
        
    DEST_AIRPORT_CLASS = pd.DataFrame(values, index = dataset.index, columns = ['DEST_AIRPORT_CLASS'])
    
    return DEST_AIRPORT_CLASS


import pandas as pd
dataset = pd.read_csv('2016_exploration.csv')
dataset = dataset[(~dataset[['ARR_DELAY']].isnull()).values]
dataset = dataset[dataset[['CANCELLED']].values ==0]
ORIGIN_AIRPORT_CLASS =  (dataset)
DEST_AIRPORT_CLASS = creation_dest_airport_class(dataset)


