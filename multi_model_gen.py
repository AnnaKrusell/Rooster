# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 09:21:26 2019

@author: Kristine
"""

from model_gen import model_gen
import numpy as np


def multiple_model_gen(path,time,dist):

    path = str(path)
    
    D = np.array([0.2, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 
                  4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0])
    
    x1 = int(dist) * 10 
    
    for x in range(x1):
        dist = D[x]
        result =  (model_gen(str(path),time,dist))
        
    return result
    
#'C:\\Users\\Kristine\\Desktop\\Bachelor\\Data'  

