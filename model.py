
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import xgboost as xgb
import os
from sklearn.preprocessing import LabelEncoder

os.chdir('A:/Flask_Python/primer_modelo')

dataset = pd.read_csv('datos_alquiler_CABA.csv')

dataset['barrio']=dataset['barrio'].astype(str)
labelencoder = LabelEncoder()
dataset['barrio_cat'] = labelencoder.fit_transform(dataset['barrio'])

dataset=dataset.loc[:,['barrio_cat','ambientes','precio_prom']]

X = dataset.loc[:,['barrio_cat','ambientes']]

y = dataset.loc[:,['precio_prom']]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

'''
classifier=xgb.XGBRegressor()
              
classifier_xgb=classifier.fit(X_train ,y_train)
              
XGB_Alquiler_reg = classifier_xgb.predict(X_test)
'''
       
classifier=xgb.XGBRegressor()
              
classifier_xgb=classifier.fit(X ,y)

filename2 = 'XGB_Alquiler.sav'
pickle.dump(classifier_xgb, open(filename2, 'wb'))   

d = {'barrio_cat': [10], 'ambientes': [300]}
df = pd.DataFrame(data=d)


model = pickle.load(open('XGB_Alquiler.sav','rb'))
print(model.predict(df))
