# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 17:11:46 2024

@author: vermaa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from scipy.optimize import curve_fit
import re #for splitting strings

file = open(r"<INSERT FILE PATH>",'r')
file_lines = file.readlines()
last_line = file_lines[-1]
for line in file_lines:
    if "DATA" in line:
        index_data_start = file_lines.index(line)+2
        
file_data_lines = file_lines[index_data_start:] #FILE DATA LINES EXTRACTED


parameters_var = file_lines[index_data_start-1]
parameters_to_be_extracted = parameters_var[:-1] 

###############
parameters = parameters_to_be_extracted.split('\t') #PARAMETERS EXTRACTED
###############
c=0

####### Creating a Dictionary
dop = {} #Dictionary of Parameters

for parameter in parameters:
    dop[parameter] = ''

for data_line in file_data_lines:
    data_line_split = data_line.split()
    for parameter in parameters:
        index_of_parameter = parameters.index(parameter)
        dop[parameter] = dop[parameter]+data_line_split[index_of_parameter]+" "

### CREATING A DATAFRAME

print(dop[parameters[0]])
  
for parameter in parameters:   #Creating a list for each key
    new_var = dop[parameter].split()
    dop[parameter] = new_var
    
for parameter in parameters: #Converting the String List to Float Variables
    appender=[]
    for var_string in dop[parameter]:
        appender.append(float(var_string))
    dop[parameter] = appender

df = pd.DataFrame.from_dict(dop) 

Voltages = df['Bias calc (V)']

Currents = df['Current (A)']

Lock_in_Xs = df['LIX 1 omega (A) [bwd] [filt]']

df.iloc[1]
plt.plot(Voltages,Lock_in_Xs, color = 'red')
plt.xlabel('V')
plt.ylabel('dI/dV')
plt.title('dI/dV vs V')
