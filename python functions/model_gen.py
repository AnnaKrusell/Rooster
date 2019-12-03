"""
Created on Tue Oct 4  8:32:21 2019

@author: Kristine
"""

import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
import sklearn
from sklearn import model_selection
from numba import jit, prange
# Prints model parameters based on distance from window and timestep


@jit
def model_gen(path, time, dist):

    
   os.chdir(str(path))

   df = pd.read_csv("Livestock_Data.csv", sep = ";", low_memory = False)

   rounded_data = df.round({'Distance': 1})
    
   chosen_points = rounded_data.loc[rounded_data['Distance'] == dist]
    
   data_train, data_test = sklearn.model_selection.train_test_split(chosen_points, test_size = 0.05)
   
   for _ in range(100):
       time = str(time)
       form =str('T_'+ time + '~ HEIGHT + Open_area + v_out + T_out + W + D')
       linear_model = sm.ols(form, data = data_train).fit()
                    
       acc = linear_model.rsquared
       result = 0
       best = 0
                    
       if acc > best:
           result = np.array(linear_model.params)
           best = acc
        
   #print(linear_model.summary())
   print(dist)
   print(acc)

#paralel computing

   return result
    
    #'C:\\Users\\Kristine\\Desktop\\Bachelor\\Data'