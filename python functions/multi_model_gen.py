# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 09:21:26 2019

@author: Kristine
"""

from model_gen import model_gen
import numpy as np


def multiple_model_gen(path, time, dist, HEIGHT, Open_area, v_out, T_out, W, D):

    global Temperature, result
    
    path = str(path)

    
    for x in np.arange(0.4, dist, 0.1):
        print("Distance:",x)
        result =  (model_gen(str(path),time,dist))
        Temperature =  ((result[0] + result[1]*HEIGHT + result[2]*Open_area + result[3]*v_out + result[4]*T_out + result[5]*W + result[6]*D) - 273)
        print("Temperature",Temperature)

        
    return Temperature
    