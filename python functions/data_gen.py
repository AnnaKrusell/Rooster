# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 08:32:41 2019

@author: Kristine
"""
import os
import pandas as pd

def data_gen(path,dist):
    
    os.chdir(str(path))

    df = pd.read_csv("Livestock_Data.csv", sep = ";", low_memory = False)
    
    rounded_data = df.round({'Distance': 1})
    chosen_points = rounded_data.loc[rounded_data['Distance'] == dist]
    
    return chosen_points