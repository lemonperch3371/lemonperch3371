# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 00:18:03 2024

@author: VERMA Anshuman - Department of Quantum Matter Physics, University of Geneva

This Python Code opens all the Raman Spectra Data in txt format in a given folder and performs some elementary analysis on the separation between the peaks
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 13:42:32 2024

@author: VERMA Anshuman
"""

"""
Class for Reading the Data and Plotting it 
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import glob
import os

  
file_list = list(os.listdir(r"<<Path for the Folder>>")) #Replace the Path Here
    
    

txt_files = []
for file in file_list:
    if (file[-3:]=='txt'):
        txt_files.append(file)
        
Raman_Peak_Difference_Data =[] #Array to Store the Peak Difference for Each File
File_Numbers_With_Good_Raman = []
for index_file,txt_file in enumerate(txt_files):
    file = open(r"<Path for the Folder>\F_Name_{}".format(txt_file))
    file_lines = file.readlines()
    Raman_Shift = []
    Intensities = []
    for line in file_lines:
        Raman_Shift.append(line.split()[0])
        Intensities.append(line.split()[1])
    
    Raman_Shift = [float(Raman_Shift_Element) for Raman_Shift_Element in Raman_Shift]
    Intensities = [float(Intensity_Element) for Intensity_Element in Intensities]
    
    plt.plot(Raman_Shift,Intensities)
    plt.xlabel("Raman Shift")
    plt.ylabel("Intensity")
    plt.title("Intensity vs Raman Shift [for File: {}]".format(txt_file[-6:-4]))
    plt.savefig(r"<Path for the Folder>\F_Name_{}".format(txt_file[-6:-4]))
    
    if (index_file!=(len(txt_files)-1)):
        plt.cla()
    
    flag = 0
    df = pd.DataFrame({"Intensities":Intensities,"Raman Shifts": Raman_Shift})
    df = df.sort_values(by = "Intensities", ascending = False)
    
    #### Since the Array is already sorted in Descending Order, The "top-most" element has the highest intensity
    Max_Raman_Shift_1 = df["Raman Shifts"].iloc[0]
    Max_Intensity_1 = df["Intensities"].iloc[0] 
    ### Choosing Max Intensity and the corresponding Raman Shift
    
    for index,intensity in enumerate(df['Intensities']):
        if (abs(df['Raman Shifts'].iloc[index] - Max_Raman_Shift_1) > 10):
            if( intensity > 1.5*np.mean(Intensities) ):
                Max_Raman_Shift_2 = df['Raman Shifts'].iloc[index]
                Max_Intensity_2 = df["Intensities"].iloc[index]
                flag=1
                break
            
    if (flag==1): #Good Raman Peaks
        Shift_Between_Peaks = abs(Max_Raman_Shift_1-Max_Raman_Shift_2)
        # print(Shift_Between_Peaks)
        Raman_Peak_Difference_Data.append(Shift_Between_Peaks)
        File_Numbers_With_Good_Raman.append(txt_file[-6:-4])
    else: #Bad Raman Peaks
        pass
        # print("Bad Raman!")
        
R = Raman_Peak_Difference_Data
FN = File_Numbers_With_Good_Raman
FN_Set = set(FN)
FN = FN_Set
print("Mean Peak Shift (from the Good Raman Spectra) = {}".format(np.mean(Raman_Peak_Difference_Data)))
print("Number of Good Raman Spectra = {}".format(np.size(Raman_Peak_Difference_Data)))
print("Percentage (%) of Good Raman Spectra Taken = {} %".format(np.size(Raman_Peak_Difference_Data)*100/np.size(txt_files)))

