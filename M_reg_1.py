#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 16:44:27 2019

@author: klanan2
"""

import pandas as pd
from functions_utiles import *


    

def reg_ml_aut(data, out, site, saison):
    
    
    if site == 1:
        cols = ['temp_1', 'mean_national_temp','humidity_1', 'consumption_secondary_1',
            'consumption_secondary_2', 'consumption_secondary_3']
        output_col = ['consumption_1']
    else:
        cols = ['temp_2', 'mean_national_temp','humidity_2', 'consumption_secondary_1',
            'consumption_secondary_2', 'consumption_secondary_3']
        output_col = ['consumption_2']

    X_week, X_week_end = sub_data(data, cols, saison)
                       
    Y_week, Y_week_end = sub_data(out, output_col, saison)
    print("week : {} {}".format(X_week.shape, Y_week.shape))
    print("week_end : {} {}".format(X_week_end.shape, Y_week_end.shape))
    
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor

    if saison == 'ete':
        clf_week = RandomForestRegressor(n_estimators=100, criterion='mae')
        clf_week.fit(X_week, Y_week)

        clf_week_end =  SVR(kernel ='rbf', gamma = 'scale', tol = 10e-5)
        clf_week_end.fit(X_week_end, Y_week_end)
    else:
        clf_week = SVR(kernel ='rbf', gamma = 'scale', tol = 10e-5)
        clf_week.fit(X_week, Y_week)

        clf_week_end =  RandomForestRegressor(n_estimators=100, criterion='mae')
        clf_week_end.fit(X_week_end, Y_week_end)
    print("training score {} : {}".format(saison, (clf_week.score(X_week, Y_week), clf_week_end.score(X_week_end, Y_week_end))))
    return (clf_week, clf_week_end)




def training_process(data_train, output_train):

    S = dict({'automne':list(), 'ete':list(), 'hiver':list(), 'printemps':list()})
    
    for s in ['automne', 'ete', 'hiver', 'printemps']:
        
        S[s].append(reg_ml_aut(data_train, output_train, site = 1, saison = s))
        S[s].append(reg_ml_aut(data_train, output_train, site = 2, saison = s))

    return S




def pred(data_test, saison,  S):

    cols = ['temp_1', 'mean_national_temp','humidity_1', 'consumption_secondary_1',
            'consumption_secondary_2', 'consumption_secondary_3']
    #output_col = ['consumption_1']
    X_week, X_week_end = sub_data(data_test, cols, saison)
    #Y_week, Y_week_end = sub_data(output_test, output_col, saison)
    t_week, t_week_end = sub_data(data_test, ['timestamp'], saison)
    t_1 = pd.DataFrame({'timestamp' : list(t_week.timestamp), 'pred_1' : list(S[0][0].predict(X_week))})
    t_2 = pd.DataFrame({'timestamp' : list(t_week_end.timestamp), 'pred_1' : list(S[0][1].predict(X_week_end))})
    prime = pd.concat([t_1, t_2], axis = 0)
    

    cols = ['temp_2', 'mean_national_temp','humidity_2', 'consumption_secondary_1',
            'consumption_secondary_2', 'consumption_secondary_3']
    #output_col = ['consumption_2']
    X_week, X_week_end = sub_data(data_test, cols, saison) 
    #Y_week, Y_week_end = sub_data(output_test, output_col, saison)
    pred_2 = list(S[1][0].predict(X_week)) + list(S[1][1].predict(X_week_end))

    prime['pred_2'] = pred_2
    return prime





def test_process(data_test, S):


    result = pd.DataFrame({'timestamp' : list(), 'pred_1' : list(), 'pred_2' : list()})
    
    for s in ['automne', 'ete', 'hiver', 'printemps']:
        result = pd.concat([result, pred(data_test, s,  S[s])])
        
    return result.sort_values(by='timestamp')


