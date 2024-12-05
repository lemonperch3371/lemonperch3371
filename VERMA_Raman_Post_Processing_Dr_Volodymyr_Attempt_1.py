# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 17:48:19 2024
Last Update: 5 December 2024

@author: VERMA Anshuman
Ph.D. Student
Scanning Probe Microscopy (Renner Group)
Department of Quantum Matter Physics
Université de Genève
24 Quai Ernest-Ansermet, Genève-1205, Suisse

Module Name: RaMoSVoPy
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
import pandas as pd
import os
import moviepy.video.io.ImageSequenceClip
import imageio.v2 as imageio
import shutil
import stat

data_path = r"F:\RAMAN DATA\2024\MoS2_Borosilicate_01_13_November_2024\BS15-AB-MoS2-hole-RCP-DP_map1.exdir\ScanND\data20241113-192831\data.npy"
data = np.load(data_path)

coordinates = data['position']['Stage3D:XYZ']

xs = []
ys = []
zs = []

for i in range(len(coordinates)):
    xs.append(coordinates[i][0])
    ys.append(coordinates[i][1])
    zs.append(coordinates[i][2])

source_image_path = r"F:\RAMAN DATA\2024\MoS2_Borosilicate_01_13_November_2024\Maybe_Map_1_Flake.png"
image = Image.open(source_image_path)

xmin = (np.min(xs))
ymin = (np.min(ys))
xmax = (np.max(xs))
ymax = (np.max(ys))

image_width ,image_height = image.size

xs = (np.array(xs))
ys = np.flip(np.array(ys))

xs_norm = np.abs(((xs-xmin)/(xmax - xmin))*image_width)
ys_norm = np.abs(((ys-ymin)/(ymax-ymin))*image_height)

#Flipped
# xs_norm = np.flip(xs_norm)
# ys_norm = np.flip(ys_norm)


# plt.imshow(image)
# plt.scatter(xs_norm,ys_norm)
# image.save("image_with_points.jpg")
###########

#data.dtype.names
Intensities = data['data']['LabRAM']['raw']['intensity']
Raman_Shifts = data['data']['LabRAM']['raw']['freq_shift']

total_points = np.size(data['iteration'].tolist())
points = total_points


pt = 1750
limits = 150

##############
################
################ POST PROCESSING PATH 
post_processing_path = r"F:\RAMAN DATA\2024\MoS2_Borosilicate_01_13_November_2024\VERMA_Post_Processing_new"
###############
################
##################



# output_folder_plots = r"C:\Users\vermaa\Desktop\RAMAN DATA\2024\MoS2_Borosilicate_01_13_November_2024\VERMA_Post_Processing\Plots"
# output_folder_images = r"C:\Users\vermaa\Desktop\RAMAN DATA\2024\MoS2_Borosilicate_01_13_November_2024\VERMA_Post_Processing\Image_Exact_Locations"
# output_folder_combined =  r"C:\Users\vermaa\Desktop\RAMAN DATA\2024\MoS2_Borosilicate_01_13_November_2024\VERMA_Post_Processing\Combined_Plots_Images"
# output_pdf_path = r"C:\Users\vermaa\Desktop\RAMAN DATA\2024\MoS2_Borosilicate_01_13_November_2024\VERMA_Post_Processing\Combined_Images_PDF"
# output_video_path = r"C:\Users\vermaa\Desktop\RAMAN DATA\2024\MoS2_Borosilicate_01_13_November_2024\VERMA_Post_Processing\Combined_Images_VIDEO_1"
# output_gif_path = r"C:\Users\vermaa\Desktop\RAMAN DATA\2024\MoS2_Borosilicate_01_13_November_2024\VERMA_Post_Processing\Combined_Images_GIF"

output_folder_plots = os.path.join(post_processing_path,"Plots")
output_folder_images = os.path.join(post_processing_path,"Image_Exact_Locations")
output_folder_combined = os.path.join(post_processing_path,"Combined_Plots_Images")
output_pdf_path = os.path.join(post_processing_path,"Combined_Images_PDF")
output_video_path = os.path.join(post_processing_path,"Combined_Images_VIDEO")
raman_laser_path = os.path.join(post_processing_path,"Raman_Laser_Path")

os.makedirs(raman_laser_path,exist_ok = True)
os.makedirs(output_folder_plots,exist_ok = True)
os.makedirs(output_folder_images,exist_ok = True)
os.makedirs(output_folder_combined,exist_ok = True)
os.makedirs(output_pdf_path,exist_ok = True)
os.makedirs(output_video_path,exist_ok = True)
# os.makedirs(output_gif_path,exist_ok = True)

for pt in range(0,points):
    fig1, ax1 = plt.subplots(figsize = (6,6))
    ax1.plot(Raman_Shifts[pt][limits:],Intensities[pt][limits:])
    plot1_path = os.path.join(output_folder_plots,f"{pt}")
    ax1.set_xlabel("Raman Shift (cm-1)")
    ax1.set_ylabel("Intensity (arb.)")
    ax1.set_title(f"Raman Spectrum: {pt}")
    fig1.savefig(plot1_path, bbox_inches='tight')
    
    plt.close(fig1)
    
    fig2,ax2 = plt.subplots(figsize = (7,7))
    ax2.imshow(image)
    ax2.scatter(xs_norm[pt],ys_norm[pt],color = 'red')
    
    ax2.set_title(f"Image: {pt}")
    
    plot2_path = os.path.join(output_folder_images, f"{pt}")
    fig2.savefig(plot2_path, bbox_inches='tight')
    
    plt.close(fig2)
    
    
    ##Combined Image
    img1 = Image.open(plot1_path+".png")
    img2 = Image.open(plot2_path+".png")
    
    total_width = img1.width + img2.width
    total_height = max(img1.height,img2.height)
    
    combined_image = Image.new("RGB",(total_width,total_height))
    combined_image.paste(img1,(0,0))
    combined_image.paste(img2,(img1.width,40))
    
    combined_image.save(os.path.join(output_folder_combined, f"{pt}.png"))
    
    img1.close()
    img2.close()
    
    
##Sorting File_Names
combined_unsorted_files = os.listdir(output_folder_combined)
int_array = []
new_files_names_temp = []
combined_sorted_files = []
for file in combined_unsorted_files:
    new_files_names_temp.append(file[:-4])
    
for name in new_files_names_temp:
    int_array.append(int(name))
    
array = np.sort(int_array)
    
for element in array:
    combined_sorted_files.append(str(element)+".png")
    
    


# generating PDF 
output_pdf = os.path.join(output_pdf_path,"PDF.pdf")
temp_pdf_images = []
for file in combined_sorted_files:
    file_path = os.path.join(output_folder_combined,file)
    temp_img = Image.open(file_path).convert('RGB')
    temp_pdf_images.append(temp_img)
temp_pdf_images[0].save(output_pdf,save_all=True,append_images = temp_pdf_images[1:]) 
print(f"Combined Image and Plot generated at: {output_pdf}")


# GIF Generator
# image_paths = [os.path.join(output_folder_combined,image) for image in combined_sorted_files if image.endswith('.png')]    
# images = [imageio.imread(img_path) for img_path in image_paths]  # Read each image into memory
# output_gif = os.path.join(output_gif_path, "CombinedImages.gif")
# imageio.mimsave(output_gif, images, duration=0.01)


image_paths = [os.path.join(output_folder_combined,image) for image in combined_sorted_files if image.endswith('.png')]

#generating a Video
# temp_video_images = []
# video_name = "movie.mp4"
# fps = 1
# clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_paths, fps = fps)
# clip.write_videofile(os.path.join(output_video_path,video_name),codec = 'libx264',preset = 'slow',verbose = True,logger = 'bar')

output_size = (1920,1080)


# Resize all images and add them to image_list
image_list = []
for img_path in image_paths:
    img = Image.open(img_path)
    img_resized = img.resize(output_size)  # Resize image
    image_list.append(np.array(img_resized))              

# Save video
fps = 25
output_video = os.path.join(output_video_path, "MOVIE.mp4")
imageio.mimsave(output_video, image_list, fps=fps)




filtered_raman_shifts_matrix = []

filtered_intensities_matrix = []


##Filtered
for index,n in enumerate(Intensities[0:points]):
    raman_shifts = Raman_Shifts[index]
    intensities = Intensities[index]
    
    filtered_intensities = []
    filtered_raman_shifts = []
    for index_rs, raman_shift in enumerate(raman_shifts):
        if(raman_shift>300):
           filtered_intensities.append(intensities[index_rs])
           filtered_raman_shifts.append(raman_shifts[index_rs])
           
    filtered_raman_shifts_matrix.append(filtered_raman_shifts)
    filtered_intensities_matrix.append(filtered_intensities)

    #and intensities[index_rs]<3*np.mean(intensities)
frsm = filtered_raman_shifts_matrix
fim = filtered_intensities_matrix
         
filtered_data_length = len(frsm)
fdl = filtered_data_length


#### Filtered Plots
output_filtered_plots_path = os.path.join(post_processing_path,"filtered_Plots")
os.makedirs(output_filtered_plots_path,exist_ok = True)

for index in range(fdl):
    fig1, ax1 = plt.subplots(figsize = (8,8))
    ax1.plot(frsm[index],fim[index])
    plot1_path = os.path.join(output_filtered_plots_path,f"{index}")
    ax1.set_xlabel("Raman Shift (cm-1)")
    ax1.set_ylabel("Intensity (arb.)")
    ax1.set_title(f"Raman Spectrum: {index}")
    fig1.savefig(plot1_path, bbox_inches='tight')
    
    plt.close(fig1)
    

### Filtered Plots PDF File (Just removed the zero frequency)
# output_filtered_plots_PDF_path = r"C:\Users\vermaa\Desktop\RAMAN DATA\2024\MoS2_Borosilicate_01_13_November_2024\VERMA_Post_Processing\filtered_Plots_PDF"
output_filtered_plots_PDF_path = os.path.join(post_processing_path,"filtered_Plots_PDF")
os.makedirs(output_filtered_plots_PDF_path,exist_ok = True)

filtered_plots_files = os.listdir(output_filtered_plots_path)
new_filterd_plots_files_temp = []
filtered_int_array = []
sorted_filtered_plots = []


for file in filtered_plots_files:
    new_filterd_plots_files_temp.append(file[:-4])

for name in new_filterd_plots_files_temp:
    filtered_int_array.append(int(name))  

array = np.sort(filtered_int_array)


for element in array:
    sorted_filtered_plots.append(str(element)+".png")

filtered_temp_pdf_images = []
output_filtered_plots_PDF = os.path.join(output_filtered_plots_PDF_path,"filtered_Plots_PDF.pdf")
for file in sorted_filtered_plots:
    file_path = os.path.join(output_filtered_plots_path,file)
    temp_img = Image.open(file_path).convert('RGB')
    filtered_temp_pdf_images.append(temp_img)
filtered_temp_pdf_images[0].save(output_filtered_plots_PDF,save_all = True,append_images = filtered_temp_pdf_images[1:])
    

    
### 
def Raman_Peak_Finder(Raman_Spectras = frsm, Intensities = fim):
    l = len(frsm)

    xy_index_array = []
    xy_maximas_array = []
    good_spectra = []
    xy_Raman_Peak_Shift=[]
    xy_rs_array = []
    
    for x in range(l):
        raman_spectras = Raman_Spectras[x]
        intensities = Intensities[x]

        intensity_maximas = []
        index_maximas = []
        raman_shift_maximas = []

        sorted_intensities = []
        sorted_indices = []


        for index, i in enumerate(intensities):
            if(raman_spectras[index]>370 and raman_spectras[index]<420):
                if (intensities[index]>1.5*np.mean(intensities)):
                    intensity_maximas.append(intensities[index])
                    raman_shift_maximas.append(raman_shifts[index])
                    index_maximas.append(index)

        if (len(intensity_maximas)>1):
            
            
            index_for_sorting = np.argsort(-np.array(intensity_maximas))
            
            intensity_maximas = np.array(intensity_maximas)
            index_maximas = np.array(index_maximas)
            raman_shift_maximas = np.array(raman_shift_maximas)
            
            sorted_intensities = intensity_maximas[index_for_sorting]
            
            sorted_indices = index_maximas[index_for_sorting]
            
            sorted_raman_shifts = raman_shift_maximas[index_for_sorting]
            
            xy_index_array.append((sorted_indices[0],sorted_indices[1]))
            xy_rs_array.append((sorted_raman_shifts[0],sorted_raman_shifts[1]))

            xy_maximas_array.append((sorted_intensities[0],sorted_intensities[1]))
            good_spectra.append(x)
            
            lrps = abs(sorted_raman_shifts[0]-sorted_raman_shifts[1])
            xy_Raman_Peak_Shift.append(lrps)
        
    df = pd.DataFrame({
    "Intensity Maximas": xy_maximas_array,
    "Index_Array": xy_index_array,
    "Good_Spectras": good_spectra,
    "Raman_Peak_Shifts": xy_Raman_Peak_Shift,
    "Raman_Shift_Pairs": xy_rs_array})       
    return(xy_maximas_array,xy_index_array,good_spectra,xy_Raman_Peak_Shift,xy_rs_array,df)

def Spot_Plotter():
    RS1,(Xs1,Ys1),AIA1 = df_Raman_PeakShift_Finder()
    plt.imshow(image)
    plt.scatter(Xs1,Ys1)
    os.makedirs(os.path.join(post_processing_path,"SpotPlotter_NoCmap"),exist_ok = True)
    Spot_Plotter_Path = os.path.join(post_processing_path,"SpotPlotter_NoCmap")
    plt.plot(xs_norm,ys_norm,color = 'black')
    plt.savefig(os.path.join(Spot_Plotter_Path,"Spot_Plotted_Map_All_Good_Raman_NoCmap"))
    return()


def Raman_DataFrame_Creator(Raman_Spectras = frsm, Intensities = fim):
    #Picking out each spectra from the DDA of Spectras
    rs_i_df = pd.DataFrame()
    
    for index_of_spectra_intensity,raman_spectra in enumerate(Raman_Spectras):
        r_s = Raman_Spectras[index_of_spectra_intensity]
        i_s = Intensities[index_of_spectra_intensity]
        #Extracted each Raman Spectra and Intensity Array 
        rs_i_df[f"r{index_of_spectra_intensity}"] = r_s
        rs_i_df[f"i{index_of_spectra_intensity}"] = i_s
    
    
    return(rs_i_df)


#Finds the Raman Peaks for the DataFrame Returned by Raman_DataFrame_Creator
def df_Raman_PeakShift_Finder():
    df = Raman_DataFrame_Creator()
    
    intensity_maxima_1_array = []
    raman_maxima_1_array = []
    index_maxima_1_array = []
   
    intensity_maxima_2_array = []
    raman_maxima_2_array = []
    index_maxima_2_array = []
    
    Raman_Shifts = []
   
    Absolute_Indices_Array = []
    Xs = []
    Ys = []
    for x1 in range(len(df.columns)//2):
        c = 0
        r = df[f'r{x1}']
        i = df[f'i{x1}']
        # print(r,i)
        intensity_maximas = []
        raman_maximas = []
        index_maximas = []
    
    ### Loop
        for index1,element in enumerate(r):
            if(r[index1]>360 and r[index1]<480 and i[index1]>1.5*np.mean(i[200:])):
               
                intensity_maximas.append(i[index1])
                raman_maximas.append(r[index1])
                index_maximas.append(index1)
        
        sorted_intensity_indices_array = np.argsort(np.multiply(-1,intensity_maximas))
        i_maxs = [intensity_maximas[a] for a in sorted_intensity_indices_array]
        r_maxs = [raman_maximas[a] for a in sorted_intensity_indices_array]
        if (len(i_maxs)>1):
            i_max_1 = i_maxs[0]
            r_max_1 = r_maxs[0]
            index_max_1 = sorted_intensity_indices_array[0]
            
            intensity_maxima_1_array.append(i_max_1)
            raman_maxima_1_array.append(r_max_1)
            index_maxima_1_array.append(index_max_1)
            
            for index3,i2 in enumerate(i_maxs[1:]):
                # if (abs(r_maxs[index3]-r_max_1)>10 and abs(r_maxs[index3]-r_max_1)<29):
                if (abs(r_maxs[index3]-r_max_1)>15 and abs(r_maxs[index3]-r_max_1)<26):
                    i_max_2 = i_maxs[index3]
                    r_max_2 = r_maxs[index3]
                    index_max_2 = index3
                    break
                else:
                    i_max_2 = None
                    r_max_2 = None
                    index_max_2 = None
                    
            
            intensity_maxima_2_array.append(i_max_2)
            raman_maxima_2_array.append(r_max_2)
            index_maxima_2_array.append(index_max_2) 
        else:
            i_max_1 = None
            r_max_1 = None
            index_max_1 = None
            i_max_2 = None
            r_max_2 = None
            index_max_2 = None
        
        if(i_max_1!=None and r_max_1!=None and i_max_2!=None and r_max_2!=None):
            Xs.append(xs_norm[x1])
            Ys.append(ys_norm[x1])
            Absolute_Indices_Array.append(x1)
    
    for index4, intensity in enumerate(intensity_maxima_1_array):
        if (intensity_maxima_1_array[index4]!=None and intensity_maxima_2_array[index4]!=None):
            Raman_Shifts.append(abs(raman_maxima_2_array[index4]-raman_maxima_1_array[index4]))
            

            #Raman Shifts with Coordinates
        

    return(Raman_Shifts,(Xs,Ys),Absolute_Indices_Array)






# To delete the Folder
def d():
    path = post_processing_path
    os.chmod(path,stat.S_IWRITE)
    if os.path.exists(path):
        shutil.rmtree(path)
    
    return()                        
    
    
def Raman_Laser_Path(raman_laser_path = raman_laser_path, image = image, xs_norm = xs_norm, ys_norm = ys_norm):
    
    rlp = os.path.join(raman_laser_path, "Raman_Laser_Path.png")
    
    
    fig3, ax3 = plt.subplots(figsize=(30, 30))
    ax3.imshow(image)
    ax3.plot(xs_norm, ys_norm,color = 'red')
    ax3.set_title("Raman Laser Path")
    
    print(f"Saving figure to {rlp}")

    fig3.savefig(rlp, bbox_inches='tight')
    
    
    plt.close(fig3)
    
    print(f"Closed Figure {rlp}")

    return
    
def Plot_ColorMapped_Raman_Shifts():
    Raman_Shifts,(Xs,Ys),AIA = df_Raman_PeakShift_Finder()
    # f_Xs = np.flip(Xs)
    # f_Ys = np.flip(Ys)
    plt.imshow(image)
    
    max_shift_between_peaks = 24
    min_shift_between_peaks = 15
    scatter = plt.scatter(Xs,Ys,c = Raman_Shifts,vmin  = min_shift_between_peaks,vmax = max_shift_between_peaks,cmap = 'magma')
    plt.xlabel("X Coordinates")
    plt.ylabel("Y Coordinates")
    plt.title(f"Shift between the E2g(1) and A1g peaks (Y flipped);\n Plotted Max Shift = {max_shift_between_peaks};\n Plotted Min Shift = {min_shift_between_peaks}")
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Shift between the E2g(1) \n and A1g peaks (in cm-1) ')
    color_map_folder = os.path.join(post_processing_path,"CMap_Peak_Shift")
    os.makedirs(color_map_folder,exist_ok = True)
    plt.savefig(os.path.join(color_map_folder,"CMAP_Peak_Shift.png"))
    
    return()


def Plot_ColorMapped_Raman_Shifts_New():
    Raman_Shifts, (Xs, Ys), AIA = df_Raman_PeakShift_Finder()
    # Ensure Xs and Ys are flipped correctly if necessary
    # f_Xs = np.flip(Xs)
    # f_Ys = np.flip(Ys)
    plt.figure(figsize = (8,8))
    plt.imshow(image)  # Fix subplots return variables (fig, ax order)
    max_shift_between_peaks = 24
    min_shift_between_peaks = 17
    
    scatter = plt.scatter(Xs, Ys, c=Raman_Shifts, vmin=min_shift_between_peaks, vmax=max_shift_between_peaks, cmap='rainbow',s = 40)
    plt.xlabel("X Coordinates")  # Correct method for setting X-axis title
    plt.ylabel("Y Coordinates")  # Correct method for setting Y-axis title
    plt.title(f"Shift between the E2g(1) and A1g peaks (Y flipped);\nPlotted Max Shift = {max_shift_between_peaks};\nPlotted Min Shift = {min_shift_between_peaks}")
    
    cbar = plt.colorbar(scatter)  # Specify the correct axis for the colorbar
    cbar.set_label('Shift between the E2g(1) \n and A1g peaks (in cm-1)')
    
    color_map_folder = os.path.join(post_processing_path, "CMap_Peak_Shift")
    os.makedirs(color_map_folder, exist_ok=True)
    plt.savefig(os.path.join(color_map_folder, "CMAP_Peak_Shift.png"))
    
    return ()

    
if __name__=="__main__":
    Raman_Laser_Path()
    Plot_ColorMapped_Raman_Shifts_New()
    
    
