
import pandas as pd
import pickle

unique_carrier = pd.read_csv('unique_carrier.csv')
unique_carrier = unique_carrier[['UNIQUE_CARRIER']]

variables = pd.read_csv('variables.csv')
variables = variables[['VARIABLES']]

origin_airport_class  = pd.read_csv('origin_airport_class.csv')
origin_airport_class = origin_airport_class[['ORIGIN','AIRPORT_CLASS']]

dest_airport_class  = pd.read_csv('dest_airport_class.csv')
dest_airport_class = dest_airport_class[['DEST','AIRPORT_CLASS']]


infile = open("fonction_std_scaling.pyc",'rb')
std_scale = pickle.load(infile)
infile.close()

infile = open("fonction_prediction.pyc",'rb')
sgd = pickle.load(infile)
infile.close()