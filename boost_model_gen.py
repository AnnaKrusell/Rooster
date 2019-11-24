
import os
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor



def boost_model_gen(path, time, dist, HEIGHT, D, W, Open_area, v_out, T_out ):
    
   os.chdir(str(path))

   temp_list = []

   df = pd.read_csv("Livestock_Data.csv", sep = ";", low_memory = False)
    
   rounded_data = df.round({'Distance': 1})
  
   Distance = int((dist-0.2)*10)
   
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
       
   return temp_list

    

