import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import math
import datetime
#from M_reg_1 import *
from functions_utiles import *
from Multi_output import *





keys = ['ID', 'timestamp', 'temp_1', 'temp_2', 'mean_national_temp',
       'humidity_1', 'humidity_2', 'loc_1', 'loc_2', 'loc_secondary_1',
       'loc_secondary_2', 'loc_secondary_3', 'consumption_secondary_1',
       'consumption_secondary_2', 'consumption_secondary_3']

data_saisons = {'Automne' : (datetime.datetime(2016,9,22),datetime.datetime(2016,12,21)), 'Hiver' : (datetime.datetime(2016,12,21),datetime.datetime(2017,3,20)), 'Printemps' : (datetime.datetime(2017,3,20),datetime.datetime(2017,6,21)), 'Ete' : (datetime.datetime(2017,6,21),datetime.datetime(2017,9,22)), 'Aut_2017' : (datetime.datetime(2017,9,22),datetime.datetime(2017,12,21))}
vacances = [(datetime.datetime(2016,10,19),datetime.datetime(2016,11,4)), (datetime.datetime(2016,12,17),datetime.datetime(2017,1,4)), (datetime.datetime(2017,7,4),datetime.datetime(2017,9,4))]
jours_ouvres = [(datetime.datetime(2016,11,4),datetime.datetime(2016,12,17)),(datetime.datetime(2017,1,4),datetime.datetime(2017,7,4)), (datetime.datetime(2017,9,4),)]


input_train = pd.read_csv('input_training_ssnsrY0.csv')
input_train['timestamp'] = pd.to_datetime(input_train['timestamp'])


input_test = pd.read_csv('input_test_cdKcI0e.csv')
input_test['timestamp'] = pd.to_datetime(input_train['timestamp'])

output_train = pd.read_csv('output_training_Uf11I9I.csv')
output_train['timestamp'] = pd.to_datetime(input_train['timestamp'])

output_train.plot(x='timestamp', y='consumption_1')
output_train.plot(x='timestamp', y='consumption_2')
plt.show()

################################# Lignes avec nan ###################################

print("########################## Poportion de lignes avec nan ########################### \n")
N = pd.DataFrame(input_train.count(axis='columns'))
N.columns = ['numbers']
print("{} %".format(100*float((len(N.numbers) - len(N[N.numbers == 15].numbers)))/float(len(N.numbers))))

###########################################################################################################
###########################################################################################################

input_train.hist(column = ['humidity_1','humidity_2'])
input_train.plot(x='timestamp', y='humidity_1')
input_train.plot(x='timestamp', y='temp_1')
input_train.plot(x='timestamp', y='humidity_2')

################################################# Labelisation des données #####################################################

data = saisonal_decomp(input_train, data_saisons)
days = days_labels(data)
data['days'] = days


import seaborn as sns

plt.figure()
data.boxplot(by = 'days', column = ['consumption_secondary_1'])
    
plt.figure()
ax = plt.subplot(2,2,1)
data.boxplot(column = ['consumption_secondary_1'], by = 'saison', showfliers=True, positions=range(data.saison.unique().shape[0]), ax=ax)
sns.pointplot(x='saison', y='consumption_secondary_1', data=data.groupby('saison', as_index=False).mean(), ax=ax)
ax = plt.subplot(2,2,2)
data.boxplot(column = ['consumption_secondary_2'], by = 'saison', showfliers=True, positions=range(data.saison.unique().shape[0]), ax=ax)
sns.pointplot(x='saison', y='consumption_secondary_2', data=data.groupby('saison', as_index=False).mean(), ax=ax)
ax = plt.subplot(2,2,3)
data.boxplot(column = ['consumption_secondary_3'], by = 'saison', showfliers=True, positions=range(data.saison.unique().shape[0]), ax=ax)
sns.pointplot(x='saison', y='consumption_secondary_3', data=data.groupby('saison', as_index=False).mean(), ax=ax)
ax = plt.subplot(2,2,4)
data.boxplot(column = ['mean_national_temp'], by = 'saison', showfliers=True, positions=range(data.saison.unique().shape[0]), ax=ax)
sns.pointplot(x='saison', y='mean_national_temp', data=data.groupby('saison', as_index=False).mean(), ax=ax)

plt.figure()
ax = plt.subplot(1,2,1)
data.boxplot(column = ['humidity_1'], by = 'saison', showfliers=True, positions=range(data.saison.unique().shape[0]), ax=ax)
sns.pointplot(x='saison', y='humidity_1', data=data.groupby('saison', as_index=False).mean(), ax=ax)
ax = plt.subplot(1,2,2)
data.boxplot(column = ['humidity_2'], by = 'saison', showfliers=True, positions=range(data.saison.unique().shape[0]), ax=ax)
sns.pointplot(x='saison', y='humidity_2', data=data.groupby('saison', as_index=False).mean(), ax=ax)

data_train, week, hollydays = vac_tr(data, vacances, jours_ouvres)

print(len(data_train.temp_1))
output_train['saison'] = list(data_train.saison)
output_train['days'] = list(data_train.days)
output_train['periode'] = list(data_train.periode)
print(output_train.head())

plt.figure()
ax = plt.gca()
input_train.plot(x='timestamp', y='consumption_secondary_1', ax=ax)
hollydays.plot(x='timestamp', y='consumption_secondary_1', ax=ax, c='red', label = 'vacances')
week.plot(x='timestamp', y='consumption_secondary_1', ax=ax, c='blue', label = 'travaille')

plt.figure()
week['consumption_secondary_2'].hist(bins=20, alpha=.4, density=1, label = 'travaille')
#plt.figure()
hollydays['consumption_secondary_2'].hist(bins=20, alpha=.4, density=1,label = 'vacances')
plt.legend()
plt.show()


############################################ remplacement des nan #######################################################

data_train = replace_nan_process(data_train)
data_train = data_train.sort_values(by='timestamp')


################################################# semaine et weekend #####################################################

week_vs_week_end = data_train[data_train.periode == 'travaille']
week_vs_week_end = week_vs_week_end[week_vs_week_end.days == 'weekend']
week_vs_week_end.boxplot(column = ['consumption_secondary_1'], by = 'saison')


############################################## Entrainement, Test et Prediction ##########################################

########################################## M_reg_1 test ##########################################################
"""print("######################################### M_reg_1 test #################################################### \n")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_train, output_train, test_size=0.2)
y_test = y_test.sort_values(by='timestamp')



S = training_process(X_train, y_train)
result = test_process(X_test, S)
result.plot(x='timestamp', y='pred_1')
plt.figure()
result.plot(x='timestamp', y='pred_2')
plt.show()

print(result.head())
y_test['pred_1'] = list(result.pred_1)
y_test['pred_2'] = list(result.pred_2)

m = weighted_mean_absolute_error(y_test)
print("measure of validation : {}".format(m))

####################### Prediction on the data_test using M_reg_1 modele ##################################

print("####################### Prediction on the data_test using M_reg_1 modele ################################## \n")


data_test = saisonal_decomp(input_test, data_saisons)
days_test = days_labels(data_test)
data_test['days'] = days_test
data_test = replace_nan_process(data_test)
data_test = data_test.sort_values(by='timestamp')
data_test_train, week, hollydays = vac_tr(data_test, vacances, jours_ouvres)
result = test_process(data_test_train, S)
print(result.head())
result.plot(x='timestamp', y='pred_1')
plt.figure()
result.plot(x='timestamp', y='pred_2')
plt.show()"""

################################## Multi_output modele ###################################
print("################################## Multi_output modele ################################### \n")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_train, output_train, test_size=0.2)
y_test = y_test.sort_values(by='timestamp')
      
S = Multi_training_process(X_train, y_train)
result = multi_test_process(X_test, S)
print(result.head())
y_test['pred_1'] = list(result.pred_1)
y_test['pred_2'] = list(result.pred_2)

m = weighted_mean_absolute_error(y_test)
print("measure of validation : {}".format(m))
plt.figure()
ax = plt.subplot(2,2,1)
y_test.plot(x='timestamp', y='consumption_1', ax=ax)
ax=plt.subplot(2,2,2)
result.plot(x='timestamp', y='pred_1',ax=ax)
ax=plt.subplot(2,2,3)
y_test.plot(x='timestamp', y='consumption_2',ax=ax)
ax=plt.subplot(2,2,4)
result.plot(x='timestamp', y='pred_2',ax=ax)
plt.show()

####################### Prediction on the data_test using Multi_output modele ##################################
print("####################### Prediction on the data_test using Multi_output modele ################################## \n")
data_test = saisonal_decomp(input_test, data_saisons)
days_test = days_labels(data_test)
data_test['days'] = days_test
data_test = replace_nan_process(data_test)
data_test = data_test.sort_values(by='timestamp')
data_test_train, week, hollydays = vac_tr(data_test, vacances, jours_ouvres)
result = multi_test_process(data_test_train, S)
print(result.head())
result.plot(x='timestamp', y='pred_1')
plt.figure()
result.plot(x='timestamp', y='pred_2')
plt.show()
