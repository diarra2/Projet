#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 16:46:16 2019

@author: klanan2
"""

import pandas as pd
from functions_utiles import *




def multi_reg(data, out, saison):
    
    cols = ['temp_1', 'temp_2', 'mean_national_temp','humidity_1', 'humidity_2', 'consumption_secondary_1',
            'consumption_secondary_2', 'consumption_secondary_3']
    
    output_col = ['consumption_1','consumption_2']
    
    X_week, X_week_end = sub_data(data, cols, saison)                   
    Y_week, Y_week_end = sub_data(out, output_col, saison)
    
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR

    if saison == 'ete':
        clf_week = RandomForestRegressor(n_estimators=100, criterion='mae', random_state = 0)
        clf_week.fit(X_week, Y_week)

        clf_week_end =  MultiOutputRegressor(RandomForestRegressor(n_estimators=100, criterion='mae', random_state = 0))  #SVR(kernel ='rbf', gamma = 'scale', tol = 10e-5))
        clf_week_end.fit(X_week_end, Y_week_end)
    else:
        clf_week = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, criterion='mae', random_state = 0)) #SVR(kernel ='rbf', gamma = 'scale', tol = 10e-5))
        clf_week.fit(X_week, Y_week)

        clf_week_end =  RandomForestRegressor(n_estimators=100, criterion='mae', random_state = 0)
        clf_week_end.fit(X_week_end, Y_week_end)
    print("training score {} : {}".format(saison, (clf_week.score(X_week, Y_week), clf_week_end.score(X_week_end, Y_week_end))))
    return (clf_week, clf_week_end)





def Multi_training_process(data_train, output_train):

    S = dict({'automne':list(), 'ete':list(), 'hiver':list(), 'printemps':list()})
    
    for s in ['automne', 'ete', 'hiver', 'printemps']:
        S[s].append(multi_reg(data_train, output_train, saison = s))

    return S






def multi_pred(data_test, saison,  S):

    cols = ['temp_1', 'temp_2', 'mean_national_temp','humidity_1', 'humidity_2', 'consumption_secondary_1',
            'consumption_secondary_2', 'consumption_secondary_3']
    #output_col = ['consumption_1']
    X_week, X_week_end = sub_data(data_test, cols, saison)
    #Y_week, Y_week_end = sub_data(output_test, output_col, saison)
    t_week, t_week_end = sub_data(data_test, ['timestamp'], saison)
    t_1 = pd.DataFrame({'timestamp' : list(t_week.timestamp), 'pred_1' : list(S[0][0].predict(X_week)[:,0]), 'pred_2' : list(S[0][0].predict(X_week)[:,1])})
    t_2 = pd.DataFrame({'timestamp' : list(t_week_end.timestamp), 'pred_1' : list(S[0][1].predict(X_week_end)[:,0]), 'pred_2' : list(S[0][1].predict(X_week_end)[:,1])})

    prime = pd.concat([t_1, t_2], axis = 0)

    return prime




def multi_test_process(data_test, S):


    result = pd.DataFrame({'timestamp' : list(), 'pred_1' : list(), 'pred_2' : list()})
    
    for s in ['automne', 'ete', 'hiver', 'printemps']:
        result = pd.concat([result, multi_pred(data_test, s,  S[s])])
        
    return result.sort_values(by='timestamp')




