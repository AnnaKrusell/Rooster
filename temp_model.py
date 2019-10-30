# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 11:16:33 2019

@author: s1638
"""

from model_gen import model_gen
#from multi_model_gen import multi_model_gen
import numpy as np

def temp_model(path, time, dist, HEIGHT, Open_area, v_out, T_out, W, D):
    

    result =  (model_gen(str(path),time,dist))
    Temperature =  ((result[0]) + (result[1]*HEIGHT) + (result[2]*Open_area) + (result[3]*v_out) + (result[4]*T_out) + (result[5]*W) + (result[6]*D)) - 273
    print("Model Parameters:", result)
        
    return Temperature

           
   