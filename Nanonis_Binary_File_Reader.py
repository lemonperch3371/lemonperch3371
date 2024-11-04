# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 17:48:19 2024

@author: VERMA Anshuman
Ph.D. Student
Scanning Probe Microscopy (Renner Group)
Department of Quantum Matter Physics
Université de Genève
24 Quai Ernest-Ansermet, Genève-1205, Suisse
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
import cv2
from PIL import Image
import moviepy.video.io.ImageSequenceClip



# file_path = r"Z:\IvanMaggio-Aprile\NbPt-2020\Nb_Pt film S2\Tip 3 - Ir-Au coated\NbPt-S2_IMA_Aur_2020-02-06_15-37-34_20mV-400pA-100nm-5pix-0.400K-0.25T-ZFC-001.3ds"
file_path = r"C:\Users\vermaa\Desktop\NANONIS BINARY FILE DATA\Au111.3ds"
# file_path = r"Z:\TimGazdic\Good maps\52KOD\Grid Spectroscopy(3x3nm,256px,B=0T,50pA, atomic scale small gap modulation)001.3ds"
# file_path = r"C:\Users\vermaa\Desktop\NANONIS BINARY FILE DATA\Au111.3ds"
# file_path = r"Z:\TimGazdic\Good maps\71KUD\Grid Spectroscopy(5nm d density form factor)002.3ds"
### HEADER STRING EXTRACTOR
header_string_file = open(file_path,"rb")
long_header_string = header_string_file.read()
header_string = long_header_string[0:5000]
header_end_finder = long_header_string.find(b"HEADER_END")
byte_shift = header_end_finder + len("HEADER_END:\r\n")
unarranged_header = long_header_string[:header_end_finder + len("HEADER_END:\r\n")].decode("utf-8")
arranged_header= unarranged_header.split("\r\n")[:-2] #HEADER_END is not present in this string array

# temporary variable
header = arranged_header
# temporary variable

### HEADER STRING EXTRACTOR

#### BINARY FILE READER
long_file = np.fromfile(file_path,dtype = ">f4",offset = byte_shift)
file = long_file[0:5000]
#### BINARY FILE READER


#### HEADER Parameters Extractor

####HEADER Parameters Extractor
    
#This File is Now a Numpy Array
#We need to understand how 

# 12 Parameters; 366 for Each Parameters; Iterate over 377 (378 values)


    


def contains(variable = "not found",array = header):
    index_array = []
    final_array = []
    for index,element in enumerate(array):
        if variable.lower() in element.lower():
            print("Found!")
            index_array.append(index)
            final_array.append(element)
            print(index)
        
        # if variable in element:
        #     flag = True
        #     break
        # break #maybe it needs to be removed
    if (index_array == [] and final_array == []):
        print("bhak madarchod")
    else: 
        return(index_array,final_array)


# Finds the Number of Parameters
def number_of_parameters():
    index, nop = contains("Experiment Parameters", header)
    nop_string = nop[0]
    equals_finder = nop_string.find("=")
    nop1 = nop_string[equals_finder+1:]
    nop2 = nop1.replace('"','')
    number_of_parameters = len(nop2.split(";"))
    return(number_of_parameters)

def parameters():
    index, p = contains("Experiment parameters", header)
    p_string = p[0]
    equals_finder = p_string.find("=")
    p1 = (p_string[equals_finder+1:])
    p2 = p1.replace('"','') 
    parameters = p2.split(";")
    return(parameters)


def number_of_fixed_parameters():
    index, nofp = contains("Parameters", header)
    nofp_string = nofp[0]
    equals_index = nofp_string.find("=")
    nofp_1 = (nofp_string[equals_index+1:]) #Actual No. of Parameters in an integer form
    nofp_2 = nofp_1.replace('"','')
    count = 0 #For Counting the Number of Parameters
    for letter in nofp_2:
        if(letter==";"):
            count = count+1
    count = count+1
    return(count)

def fixed_parameters():
    index, fp = contains("Fixed parameters", header)
    fp_string = fp[0]
    equals_index = fp_string.find("=")
    fp_1 = (fp_string[equals_index+1:]) 
    fp_2 = fp_1.replace('"','')
    fixed_parameters = fp_2.split(";")
    return(fixed_parameters)



def total_number_of_parameters():
    total_number_of_parameters = number_of_parameters() + number_of_fixed_parameters()
    return(total_number_of_parameters)


def all_parameters():
    all_parameters = fixed_parameters() + parameters()
    return(all_parameters)


def grid_dimensions():
    index,dim = contains("Grid dim")
    dim_string = dim[0]
    dim1 = dim_string.replace('"','')
    dim2 = dim1.replace(" ","")
    equals_index = dim2.find("=")
    x_index = dim2.find("x")
    dim_x = dim2[equals_index+1:x_index]
    dim_y = dim2[x_index+1:]
    dim_x,dim_y = int(dim_x),int(dim_y)
    return(dim_x,dim_y)

def points():
    index,p = contains("Points")
    points = int((p[0].split("="))[1])
    return(points)
    

def channels():
    index,c = contains("Channels")
    c1 = c[0]
    equals_index = c1.find("=")
    c2 = c1[equals_index+1:]
    c3 = c2.replace('"',"")
    channels = c3.split(";")
    return(channels)

def number_of_channels():
    number_of_channels = len(channels())
    return(number_of_channels)


###############
###############
def byte_calculator():
    total_bytes = (grid_dimensions()[0]*grid_dimensions()[1])*( (number_of_fixed_parameters() + number_of_parameters() ) + ( points() * number_of_channels()))
    if (total_bytes==len(long_file)):
        print("Yes, the total number of bytes match with the Formula!")
    return(total_bytes)



###############

def partitions():
    tnop = total_number_of_parameters()
    p = points()
    noc = number_of_channels()
    data_block_size = tnop+p*noc
    return(data_block_size)


def long_data_blocks(): #RETURNS THE ENTIRE FILE
    data_block_size = partitions()
    # long_file_data = np.reshape(long_file,)
    grid_size = grid_dimensions()[0]*grid_dimensions()[1]

    long_file_data_blocks = long_file.reshape(grid_size,data_block_size) 
    return(long_file_data_blocks)

def data_matrix():
    data_block_size = partitions()
    # long_file_data = np.reshape(long_file,)
    grid_size = grid_dimensions()[0]*grid_dimensions()[1]

    long_file_data_blocks = long_file.reshape(grid_dimensions()[0],grid_dimensions()[1],data_block_size) 
    return(long_file_data_blocks)

#ATTEMPT 1
def parameter_dictionary():
    grid_x = grid_dimensions()[0]
    grid_y = grid_dimensions()[1]
    all_params = all_parameters()
    nop = len(all_params)
    data = data_matrix()
    
    parameter_dictionary = {param: np.zeros((grid_x,grid_y)) for param in all_params}
    print(parameter_dictionary.keys())
    print(len(parameter_dictionary.keys()))
    for index,parameter in enumerate(parameter_dictionary.keys()):
        for j in range(0, grid_x):
            for k in range(0, grid_y):
        
                parameter_dictionary[parameter][j][k] = data[j][k][index]
                
    return(parameter_dictionary)

def bias_sweep_array():
    parameter_dict = parameter_dictionary()
    ss = parameter_dict['Sweep Start']
    se = parameter_dict['Sweep End']
    pts1 = points()
    
    if np.all(ss == ss[0,0]) and np.all(se == se[0,0]):
        print(f"The Bias Sweep Start and Sweep End for all {grid_dimensions()[0]} x {grid_dimensions()[1]} points in the grid are the same!")
        sweep_start = ss[0,0]
        sweep_end = se[0,0]
        Bias_Array = np.linspace(sweep_start,sweep_end,pts1)
        return(Bias_Array)
               
        
    else:
        print("ERROR! THE BIAS SWEEP AND SWEEP END FOR ALL GRID POINTS ARE NOT THE SAME!")
        ## MAYBE DESIGN ANOTHER FUNCTION???
        return()
    


def channel_dictionary():
    raw_data = data_matrix()
    g_x = grid_dimensions()[0]
    g_y = grid_dimensions()[1]
    parts = partitions()
    pts = points()
    tonp = total_number_of_parameters() #total number of parameters (fixed parameters + parameters)
    channel_dictionary = {channel: np.zeros(((g_x,g_y,pts))) for channel in channels()}
    
    for i in range(g_x):
        for j in range(g_y):
            for index,channel in enumerate(channel_dictionary.keys()):
                channel_dictionary[channel][i,j] = raw_data[i,j][tonp+pts*index:tonp+(index+1)*pts]
    return(channel_dictionary)


def I_V_Plotter(x = 0,y = 0):
    i = channel_dictionary()["Current (A)"]
    v = bias_sweep_array()
    gx,gy = grid_dimensions()
    plt.plot(v,i[x,y])
    plt.xlabel("Bias (V)")
    plt.ylabel("Current (A)")
    plt.title("I (A) vs Bias (V) for ({},{})".format(x+1,y+1))
    return()

def dIdV_V_Plotter(x = 0,y = 0):
    i = channel_dictionary()["LIX 1 omega (A)"]
    v = bias_sweep_array()
    plt.plot(v,i[x-1,y-1])
    plt.xlabel("Bias (V)")
    plt.ylabel("dI/dV")
    plt.title("dI/dV vs Bias (V) for ({},{})".format(x,y))
    return()

def dIdV_Map(pt = 20, vmin_factor = 0.5,vmax_factor = 1.5):
    cd = channel_dictionary()
    # cd_key_list = list(cd.keys())
    lix = cd['LIX 1 omega (A)']
    # i = cd['Current (A)']
    print(bias_sweep_array()[pt])
    dat  = lix[:,:,pt]
    figure,axes = plt.subplots(figsize = (13,13))
    plt.subplots_adjust(top = 0.8,bottom = 0.1)
    img = axes.imshow(dat,vmin = vmin_factor*np.average(dat),vmax = vmax_factor*np.average(dat))
    plt.colorbar(img,ax = axes)
    plt.title("dIdV Map: Bias = {}".format(bias_sweep_array()[pt]))
    plt.savefig(r"C:\Users\vermaa\Desktop\Python Codes\Random Images\{}".format(pt))
    
    return(0)




def dIdV_Maps(point = 41, vmin_factor = 0.7,vmax_factor = 1.2, input_directory = r"C:\Users\vermaa\Desktop\Python Codes\Random Images", output_gif_path = r"C:\Users\vermaa\Desktop\Python Codes\Random Gifs\Gif2.gif"):
    cd = channel_dictionary()
    # cd_key_list = list(cd.keys())
    lix = cd['LIX 1 omega (A)']
    # i = cd['Current (A)']
    print(bias_sweep_array()[point])
    
    for pt in range(points()):
        dat  = lix[:,:,pt]
        
        figure,axes = plt.subplots(figsize = (11,11))
        plt.subplots_adjust(left = 0.05, right = 0.95,top = 0.95,bottom = 0.05)
        img = axes.imshow(dat,vmin = vmin_factor*np.average(dat),vmax = vmax_factor*np.average(dat))
        plt.colorbar(img,ax = axes)
        plt.title("dIdV Map: Bias = {}".format(bias_sweep_array()[pt]))
        plt.savefig(r"C:\Users\vermaa\Desktop\Python Codes\Random Images\{}".format(pt))
    return ()

#This function sorts the names of the Files given, according to the integers assigned
def file_name_sorter(file_names = ['0.png','23.png','4.png','12.png','1.png']):
    int_array = []
    new_file_names = []
    for file in file_names:
        new_file_names.append(file[:-4])
        
    for name in new_file_names:
        int_array.append(int(name))
    
    array = np.sort(int_array)
    
    returning_string_array = []
    
    for element in array:
        returning_string_array.append(str(element)+".png")
    return(returning_string_array)


def create_gif_from_images(input_directory, output_gif_path, duration=1.0):
    images = []
    file_names = os.listdir(input_directory)
    sorted_file_names = file_name_sorter(file_names)
    for file in sorted_file_names:
        if  file.endswith((".png",'jpg','png')):
            file_path = os.path.join(input_directory, file)
            images.append(imageio.imread(file_path))
    
    # Check if images were found
    if len(images)>1:
        # Save the images as a GIF
        imageio.mimsave(output_gif_path, images, duration=duration)
        print(f"GIF saved successfully at: {output_gif_path}")
    else:
        print("Cannot create a GIF for 1 images or les!")
    return(file_names)

    
def image_pdf_generator(input_directory,output_pdf_path):
    images = []
    file_names = os.listdir(input_directory)
    sorted_file_names = file_name_sorter(file_names)
    for file in sorted_file_names:
        if file.endswith(('png','jpg')):
            path = os.path.join(input_directory,file)
            image = Image.open(path).convert('RGB')
            images.append(image)
    images[0].save(output_pdf_path,save_all=True,append_images = images[1:])   
    print(f"PDF with the path {output_pdf_path} was saved")
    return()

def video_generator(input_directory,output_video_path):
    images = []
    file_names = os.listdir(input_directory)
    
    frame = cv2.imread(os.path.join(input_directory,file_names[0]))
    height,width,layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'H246')
    
    video = cv2.VideoWriter(output_video_path,fourcc,1,(width,height))
    
    for image in images:
        video.write(cv2.imread(os.path.join(input_directory,image)))
    
    cv2.destroyAllWindows()
    video.release()
    print(f"Video successfully saves to: {output_video_path}")  
    return()


def new_video_generator(input_directory,output_video_path):
    video_name = 'mov_1.avi'
    sorted_file_names = file_name_sorter(os.listdir(input_directory))
    image_paths = [os.path.join(input_directory,image) for image in sorted_file_names if image.endswith('.png')]
    fps = 2
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_paths, fps = fps)
    clip.write_videofile(os.path.join(output_video_path,video_name),codec = 'libx264')
    #os.path.join(output_video_path,video_name)
    return()

if __name__=="__main__":
    
    #Input Directory for Images
    input_directory = r"C:\Users\vermaa\Desktop\Python Codes\Random Images"
    
    
    output_gif_path = r"C:\Users\vermaa\Desktop\Python Codes\Random Gifs\GDC_GIF.gif"
    output_pdf_path = r"C:\Users\vermaa\Desktop\Python Codes\Random PDFs\PDF5.pdf"
    output_video_path = r"C:\Users\vermaa\Desktop\Python Codes\Random Videos"
    dIdV_Maps()
    create_gif_from_images(input_directory, output_gif_path)  
    image_pdf_generator(input_directory, output_pdf_path)
    new_video_generator(input_directory,output_video_path)
    # dIdV_Map()

       
        
##################  
   

#### OUTPUT FOR REFERENCE 366 DATA POINTS, 12 PARAMETERS
# for n in range(0,10):
#     print(n,file[378*n])
# 0 -0.6
# 1 -0.6
# 2 -0.6
# 3 -0.6
# 4 -0.6
# 5 -0.6
# 6 -0.6
# 7 -0.6
# 8 -0.6
# 9 -0.6


# file[number_of_parameters() + number_of_fixed_parameters() + number_of_channels()*points()]
# Found!
# 5
# Found!
# 4
# Found!
# 5
# Found!
# 6
# Found!
# 9
# Found!
# 33
# Found!
# 79
# Found!
# 96
# Found!
# 8
# Out[16]: -0.6     

### FOR GETTING THE COLOR MAP
# lix = channel_dictionary()['LIX 1 omega (A)']
# plt.imshow(lix[:,:,45],vmin = 0.60*np.average(lix[:,:,45]),vmax = 1.7*np.average(lix[:,:,45]))    
    
    
    
    
    
