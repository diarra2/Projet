#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 16:48:48 2019

@author: klanan2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def nan_traitement(data, cols):

    temp = data[data[cols[0]].isna() == False]
    temp_prime = temp[['mean_national_temp', 'consumption_secondary_1','consumption_secondary_2', 'consumption_secondary_3']]
    temp_sec = temp[cols]


    from sklearn.neighbors import NearestNeighbors
    classifier = NearestNeighbors(n_neighbors=5)
    classifier.fit(temp_prime)

    n = len(data[data[cols[0]].isna() == True][['mean_national_temp']])
    temp = list()
    humidity = list()

    for i in range(n):
        o = np.array(data[data[cols[0]].isna() == True][['mean_national_temp', 'consumption_secondary_1',
                             'consumption_secondary_2', 'consumption_secondary_3']].iloc[i,:])
        t = classifier.kneighbors([o])
        #dist = t[0]
        nears = t[1][0,:]
        z = np.array(temp_sec.iloc[nears,:])
        temp.append(np.mean(z, axis=0)[0])
        humidity.append(np.mean(z, axis=0)[1])
    return temp, humidity




def replace_nan(data):

    temp_2, humidity_2 = nan_traitement(data, ['temp_2','humidity_2'])
    data.temp_2[data.temp_2.isna() == True] = temp_2 #automne.temp_2[automne.temp_2.isna() == True]
    data.humidity_2[data.humidity_2.isna() == True] = humidity_2

    temp_1, humidity_1 = nan_traitement(data, ['temp_1','humidity_1'])
    data.temp_1[data.temp_1.isna() == True] = temp_1 #automne.temp_2[automne.temp_2.isna() == True]
    data.humidity_1[data.humidity_1.isna() == True] = humidity_1

    
def replace_nan_process(data):
    ret = pd.DataFrame()
    for s in ['automne', 'ete', 'hiver', 'printemps']:
        X = data[data.saison == s]
        replace_nan(X)
        ret = pd.concat([ret,X])
    return ret


def sub_data(data, cols, saison):
    

    X = data[data.saison == saison]
    X_week = X[X.periode == 'travaille']
    X_week = X_week[X_week.days == 'week'][cols]
    X_week_end_1 = X[X.periode == 'vacances']
    X_week_end_2 = X[X.periode == 'travaille']
    X_week_end_2 = X_week_end_2[X_week_end_2.days == 'weekend']
    X_week_end = pd.concat([X_week_end_1, X_week_end_2], axis = 0)
    X_week_end = X_week_end.sort_values(by='timestamp')
    X_week_end = X_week_end[cols]
    
    return X_week, X_week_end




def weighted_mean_absolute_error(dataframe):
    
    from sklearn import metrics

    true_site_1 = dataframe.consumption_1
    pred_site_1 = dataframe.pred_1
    mae_site_1 = metrics.mean_absolute_error(true_site_1, pred_site_1)
    print("site 1 {}".format(mae_site_1))

    true_site_2 = dataframe.consumption_2
    pred_site_2 = dataframe.pred_2
    mae_site_2 = metrics.mean_absolute_error(true_site_2, pred_site_2)
    print("site 2 {}".format(mae_site_2))

    total_conso = list(dataframe[['consumption_1', 'consumption_2']].sum(axis = 0))
    print(total_conso)
    
    return (mae_site_1*total_conso[0] + mae_site_2*total_conso[1])/np.sum(total_conso)




def saisonal_decomp(input_train, data_saisons):
    
    plt.figure()
    ax = plt.gca()
    input_train.plot(x='timestamp', y='consumption_secondary_1', ax=ax, legend = False, title = 'consumption_secondary_1')
    ################################### AUTOMNE 2016 #############################

    automne = input_train[input_train.timestamp <= data_saisons['Automne'][1]]
    n = len(automne['temp_1'])
    automne['saison'] = n*['automne']
    #replace_nan(automne)
    automne.plot(x='timestamp', y='consumption_secondary_1', c = 'yellow', ax=ax, label = 'automne')
    
    ##############################################################################
    ########################################## HIVER ##############################
    
    hiver = input_train[input_train.timestamp <= data_saisons['Hiver'][1]]
    hiver = hiver[hiver.timestamp > data_saisons['Hiver'][0]]
    hiver.temp_2[hiver.humidity_2.isna() == True] = [np.nan]
    n = len(hiver['temp_1'])
    hiver['saison'] = n*['hiver']
    #replace_nan(hiver)
    hiver.plot(x='timestamp', y='consumption_secondary_1', c = 'blue', ax=ax, label = 'hiver')
    
    #############################################################################
    ################################# PRINTEMPS #################################
    
    printemps = input_train[input_train.timestamp <= data_saisons['Printemps'][1]]
    printemps = printemps[printemps.timestamp > data_saisons['Printemps'][0]]
    n = len(printemps['temp_1'])
    printemps['saison'] = n*['printemps']
    #replace_nan(printemps)
    printemps.plot(x='timestamp', y='consumption_secondary_1', c = 'green', ax=ax, label = 'printemps')
    
    ###############################################################################
    ########################## ETE #################################################
    
    ete = input_train[input_train.timestamp <= data_saisons['Ete'][1]]
    ete = ete[ete.timestamp > data_saisons['Ete'][0]]
    n = len(ete['temp_1'])
    ete['saison'] = n*['ete']
    #replace_nan(ete)
    ete.plot(x='timestamp', y='consumption_secondary_1', c = 'red', ax=ax, label = 'ete')
    
    ##################################################################################
    ##################################### AUTOMNE 2017 ###############################
    
    aut = input_train[input_train.timestamp <= data_saisons['Aut_2017'][1]]
    aut = aut[aut.timestamp > data_saisons['Aut_2017'][0]]
    n = len(aut['temp_1'])
    aut['saison'] = n*['automne']
    #replace_nan(aut)
    aut.plot(x='timestamp', y='consumption_secondary_1', c = 'yellow', ax=ax, legend=False)
    
    ####################################################################################
    
    data = pd.concat([automne, hiver, printemps, ete, aut], axis = 0)
    
    return data



def vac_tr(data, vacances, jours_ouvres):

    X = data[data.timestamp < vacances[0][1]]
    y = data[data.timestamp >= jours_ouvres[0][0]]
    y = y[y.timestamp <= jours_ouvres[0][1]]
    
    print(y.head())
    
    n = len(X['temp_1'])
    X['periode'] = n*['vacances']
    n = len(y['temp_1']) 
    y['periode'] = n*['travaille']
    #X.plot(x='timestamp', y='consumption_secondary_1', ax=ax,c='red')
    
    
    X1 = data[data.timestamp < vacances[1][1]]
    X1 = X1[X1.timestamp >= vacances[1][0]]
    y1 = data[data.timestamp >= jours_ouvres[1][0]]
    y1 = y1[y1.timestamp < jours_ouvres[1][1]]
    
    n = len(X1['temp_1'])
    X1['periode'] = n*['vacances']
    n = len(y1['temp_1'])
    y1['periode'] = n*['travaille']
    #X.plot(x='timestamp', y='consumption_secondary_1', ax=ax,c='red')
    
    
    X2 = data[data.timestamp < vacances[2][1]]
    X2 = X2[X2.timestamp >= vacances[2][0]]
    y2 = data[data.timestamp > jours_ouvres[2][0]]

    n = len(X2['temp_1'])
    X2['periode'] = n*['vacances']
    n = len(y2['temp_1'])
    y2['periode'] = n*['travaille']
    
    week = pd.concat([y, y1, y2], axis = 0)
    hollydays = pd.concat([X, X1, X2], axis = 0)
    
    data_train = pd.concat([X, y, X1, y1, X2, y2], axis = 0)
    return data_train, week, hollydays



def days_labels(input_train):
    
    days = ["{:%A}".format(x) for x in input_train['timestamp']]

    for i in range(len(days)):
        if days[i] == 'samedi' or days[i] == 'dimanche':
            days[i] = 'weekend'
        else:
            days[i] = 'week'
    return days
