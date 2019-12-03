__author__ = "Christian Kongsgaard, modified by Anna Krusell and Kristine Marburger"
__license__ = "MIT"

# -------------------------------------------------------------------------------------------------------------------- #
# Imports

# Module imports
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import statsmodels.formula.api as sm
from sklearn.ensemble import GradientBoostingRegressor

# Livestock imports

# Grasshopper imports

# -------------------------------------------------------------------------------------------------------------------- #
# Livestock Functions

def Rooster_v_boost_function(folder):

    file = open('C:\livestock3d\data' + '\data_file.txt', 'r')
    params = file.readlines()
    file.close()
        
    os.chdir('C:\livestock3d\data')
    
    time = int(params[0])
    dist = float(params[1])
    HEIGHT = float(params[2])
    D = float(params[3])
    W = float(params[4])
    Open_area = float(params[5])
    v_out = float(params[6])
    T_out = float(params[7])
    
    df = pd.read_csv("Livestock_V_Data.csv", sep = ",", low_memory = False)
    rounded_data = df.round({'Distance': 1})
  
    Distance = int((dist-0.2)*10)
    v_list =[]
   
    for x in range(Distance): 
       
        intervals = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,
                    2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0,4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8,4.9,5.0,5.1,5.2,5.3,5.4,5.5,
                    5.6,5.7,5.8,5.9,6.0,6.1,6.2,6.3,6.4,6.5,6.6,6.7,6.8,6.9,7.0,7.1]

        chosen_points = rounded_data.loc[rounded_data['Distance'] == intervals[x]]
        time = str(time)
                   
        regressor = GradientBoostingRegressor(learning_rate=0.1, random_state=None, n_estimators = 100)
                   
        X = chosen_points[['Height','D','W','Open_area','v_out','T_out']]
       
        Y = chosen_points[str('T_'+ time)]
       
        fit = regressor.fit(X, Y)
           
        result_1 = GradientBoostingRegressor.predict(fit,[[HEIGHT, D, W, Open_area, v_out, T_out]])

        result = result_1[0]
        
        result2 = round(result,3)
          
        v_list.append(result2)

    with open(folder + r'\result2.txt', 'w') as result_file:        
        for number in v_list:
            result_file.write(str(number) + '\n')
    
    return None


def Rooster_boost_model_gen_function(folder):
    file = open('C:\livestock3d\data' + '\data_file.txt', 'r')
    params = file.readlines()
    file.close()
        
    os.chdir('C:\livestock3d\data')
    
    time = int(params[0])
    dist = float(params[1])
    HEIGHT = float(params[2])
    D = float(params[3])
    W = float(params[4])
    Open_area = float(params[5])
    v_out = float(params[6])
    T_out = float(params[7])
    
    df = pd.read_csv("Livestock_Data.csv", sep = ";", low_memory = False)
    rounded_data = df.round({'Distance': 1})
  
    Distance = int((dist-0.2)*10)
    temp_list =[]
   
    for x in range(Distance): 
       
        intervals = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,
                    2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0,4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8,4.9,5.0,5.1,5.2,5.3,5.4,5.5,
                    5.6,5.7,5.8,5.9,6.0,6.1,6.2,6.3,6.4,6.5,6.6,6.7,6.8,6.9,7.0,7.1]

        chosen_points = rounded_data.loc[rounded_data['Distance'] == intervals[x]]
        time = str(time)
                   
        regressor = GradientBoostingRegressor(learning_rate=0.1, random_state=None, n_estimators = 100)
                   
        X = chosen_points[['HEIGHT','D','W','Open_area','v_out','T_out']]
       
        Y = chosen_points[str('T_'+ time)]
       
        fit = regressor.fit(X, Y)
           
        result_1 = GradientBoostingRegressor.predict(fit,[[HEIGHT, D, W, Open_area, v_out, T_out]])

        result = result_1[0]-273.15
           
        temp_list.append(result)



    with open(folder + r'\result2.txt', 'w') as result_file:        
        for number in temp_list:
            result_file.write(str(number) + '\n')
    

    return None


def Rooster_datagen_function(folder):
    
    file = open('C:\livestock3d\data' + '/data_file.txt', 'r')
    dist = file.readlines()
    file.close()
        
    os.chdir('C:\livestock3d\data')
    
    df = pd.read_csv("Livestock_Data.csv", sep = ";", low_memory = False)
    
    rounded_data = df.round({'Distance': 1})
    
    df1 = rounded_data.loc[rounded_data['Distance'] == float(dist[0])]
    
    df1.to_csv(r"C:\livestock3d\data\result.txt", sep = ',', mode = 'w')
    
    return None

def Rooster_modelgen_function(folder):
    
    file = open('C:\livestock3d\data' + '\data_file.txt', 'r')
    params = file.readlines()
    file.close()
        
    os.chdir('C:\livestock3d\data')
    
    time = float(params[0])
    dist = float(params[1])
    
    df = pd.read_csv("Livestock_Data.csv", sep = ";", low_memory = False)
    
    rounded_data = df.round({'Distance': 1})
    
    df1 = rounded_data.loc[rounded_data['Distance'] == float(dist[0])]
   
    if dist >= 5:
       
       for _ in range(100):
           form = str('T_'+ time + '~ HEIGHT + D + Open_area + v_out + T_out')
           linear_model = sm.ols(form, data = df1).fit()
           
           acc = linear_model.rsquared
           result = 0
           best = 0
                        
           if acc > best:
               result = np.array(linear_model.params)
               best = acc
               
    elif dist < 5:

       for _ in range(100):
           form = str('T_'+ time + '~ HEIGHT + W + D + Open_area + v_out + T_out')
           linear_model = sm.ols(form, data = df1).fit()
           
           acc = linear_model.rsquared
           result = 0
           best = 0
                        
           if acc > best:
              result = np.array(linear_model.params)
              best = acc

           
    result_file = open(folder + '\result.txt', 'w')        
    
    result_file.write(best + '\n')

    result_file.close()
    
    return None




def my_function(folder):

    file = open(folder + '/data_file.txt', 'r')
    my_lines = [line.strip()
                for line in file.readlines()]
    file.close()

    repeat = int(my_lines[1].strip())
    line_to_write = my_lines[0].strip()

    result_file = open(folder + '/result.txt', 'w')

    for i in range(repeat):
        result_file.write(line_to_write + '\n')

    result_file.close()

    return None


def plot_graph():
    y_values = np.loadtxt('data_file.txt')
    x_values = np.linspace(0, len(y_values), len(y_values))

    plt.figure()
    plt.plot(x_values, y_values)
    plt.savefig('plot.png')

    return None