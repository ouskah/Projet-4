# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 13:34:54 2018

@author: zakis
"""

from flask import Flask, request, render_template
from .datas import unique_carrier, variables, origin_airport_class, dest_airport_class, std_scale, sgd
from datetime import datetime
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/prediction',methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        flight_date = request.form['flight_date']
        departure_hour = request.form['departure_hour']
        
        
        arrival_hour = request.form['arrival_hour']
        flight_company = request.form['flight_company']
        flight_origin = request.form['flight_origin']
        flight_dest = request.form['flight_dest']
        distance = request.form['distance']
        nb_flights_departure = request.form['nb_flights_departure']
        nb_flights_arrival = request.form['nb_flights_arrival']
        
        DELAY = get_delay(flight_date, departure_hour, arrival_hour, flight_company, flight_origin, flight_dest, distance, nb_flights_departure, nb_flights_arrival)
        
        if DELAY < 0:
            return "Votre vol sera en avance de " + "%.2f"%np.absolute(DELAY) + " min."
        else :
            return "Votre vol sera en retard de " + "%.2f"%DELAY + " min."
                
    else :
        return render_template('index.html', airport_origin = list(origin_airport_class["ORIGIN"]),
	airport_destination = list(dest_airport_class["DEST"]),company_list = list(unique_carrier['UNIQUE_CARRIER']))
 

@app.route('/')
def blabla():
    return 'hello'
    
 
def get_delay(flight_date, 
              departure_hour, 
              arrival_hour, 
              flight_company, 
              flight_origin, 
              flight_dest, 
              distance,
              nb_flights_departure, 
              nb_flights_arrival):
    
    modelization = pd.DataFrame(dtype=int)
    
    annee, mois, jour = flight_date.split("-")
    annee = int(annee)
    mois = int(mois)
    jour = int(jour)
    heure_arrivee = int(arrival_hour.split(":")[0])
    heure_depart = int(departure_hour.split(":")[0])
      
    d = datetime.strptime(flight_date, '%Y-%m-%d')
    weekday = d.weekday()
    
    for i in range(0, len(variables)):
        modelization.at[0,variables.iloc[i,0]] = 0
    
    if mois <= 3 : quarter = 1
    elif mois <= 6 :  quarter = 2
    elif mois <= 9 :  quarter = 3
    else : quarter = 4
    
    
    modelization[['QUARTER_'+str(quarter)]] = 1
    modelization[['MONTH_'+str(mois)]] = 1
    modelization[['DAY_OF_MONTH_'+str(jour)]] = 1
    modelization[['DAY_OF_WEEK_'+str(weekday)]] = 1
    modelization[['UNIQUE_CARRIER_'+str(flight_company)]] = 1
    modelization[['CRS_ARR_TIME_'+str(heure_arrivee)]] = 1
    modelization[['CRS_DEP_TIME_'+str(heure_depart)]] = 1
    
    origin_airport_class_flight = origin_airport_class.iloc[np.argwhere(origin_airport_class.iloc[:,0].values.astype(str) == flight_origin)[0][0],1]
    dest_airport_class_flight = dest_airport_class.iloc[np.argwhere(dest_airport_class.iloc[:,0].values.astype(str) == flight_dest)[0][0],1]
    
    modelization[['ORIGIN_AIRPORT_CLASS_'+str(origin_airport_class_flight)]] = 1
    modelization[['DEST_AIRPORT_CLASS_'+str(dest_airport_class_flight)]] = 1
    modelization[['DISTANCE']] = int(distance)
    if nb_flights_departure == '' : nb_flights_departure = '0'
    if nb_flights_arrival == '' : nb_flights_arrival = '0'
    modelization[['NB_FLIGHT_DEP_SAME_HOUR']] = int(nb_flights_departure)
    modelization[['NB_FLIGHT_ARR_SAME_HOUR']] = int(nb_flights_arrival)
    
    x = std_scale.transform(modelization.values)
    y = sgd.predict(x)
    return y[0]
