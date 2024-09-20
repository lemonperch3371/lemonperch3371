# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 11:00:20 2024

@author: vermaa
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from scipy.optimize import curve_fit as cf
file = open(r"C:\Users\vermaa\Desktop\Python Codes\Anodic Bonding Data\AB_MoS2_Pre2_BS_4_620240916_145743.txt","r")
file_lines = file.readlines()


    # for trial_line in file_lines_3:
    #     if "ASpikey" in trial_line:
    #         print("Line Found")  
    #         print(file_lines_3.index(trial_line))
        

data_string_list = file_lines[24:] #Change the File Name Here

Temperature_Array = [] #degree C; This array stores the temperature
Current_Array = [] #nA; This array stores the Current
Index_Array = [] #Time and Index
Time_Array = []
Measured_V_Array = []
Programmed_V_Array = []

for data_string in data_string_list:
    split_data = data_string.split()
    
    Index_Array.append((split_data[0]))
    Time_Array.append(split_data[1])
    Measured_V_Array.append(split_data[2])
    Programmed_V_Array.append(split_data[3])
    Current_Array.append(split_data[4])
    Temperature_Array.append(split_data[5])
    

Temperature = [] #Temperature
Time = [] #Time
Measured_V = [] #Measure_V 
Current = [] #Current
Programmed_V = [] #Programmed View
Index = [] #Index


for i in range(0,len(Index_Array)):
    Temperature.append(float(Temperature_Array[i].replace(',','.')))
    Time.append(float(Time_Array[i].replace(',','.')))
    Measured_V.append(float(Measured_V_Array[i].replace(',','.')))
    Current.append(float(Current_Array[i].replace(',','.')))
    Programmed_V.append(float(Programmed_V_Array[i].replace(',','.')))
    Index.append(float(Index_Array[i].replace(',','.')))
    
Current = np.array(Current)
Time = np.array(Time)

def model(x,a,b,c):
    return(a*x**2+b*x+c)

popt, pcov = cf(model, Time, Current, p0 = [0.01,0.2,2])

plt.scatter(Time, Current, marker = 'x')
plt.xlabel('Time (s)')
plt.ylabel('Current (nA)')
plt.show()

Current_Mean = np.mean(Current)
Current_std = np.std(Current)
Current_var = np.var(Current)

a, b, c = popt
x_model = np.linspace(min(Time), max(Time), 1000)
y_model = model(x_model,a,b,c)

plt.scatter(Time, Current, marker = 'x')
plt.plot(x_model, y_model)
plt.xlabel('Time (s)')
plt.ylabel('Current (nA)')
plt.show()
