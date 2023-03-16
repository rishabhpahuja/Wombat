import vtk
import os
import numpy as np
import cv2
from vtk.util.numpy_support import vtk_to_numpy
import sys
import glob
import os
# insert at 1, 0 is the script path (or '' in REPL)
from vtk import*

import pandas as pd

import dicom_with_vtk 

def generate_units_to_device_dict(catalogue_filename = "/media/biorobotics/BirobTejasDisk23/tejas_desktop_backup/Apple-2Dto3D/data/data_from_apple_2/phones/Unit Catalog with Scan Parameters.xlsx"):
    

    df = pd.read_excel(catalogue_filename)

    units = df['Unit No']

    devices = df['Product Type']

    units = units[~pd.isnull(units)]
    devices = devices[~pd.isnull(devices)]


    units = units.to_numpy()
    devices = devices.to_numpy()

    assert len(units)==len(devices)

    units_to_device_dict = {}
    for unit, device in zip(units,devices):
        key = "Unit " + '{0:02d}'.format(unit.astype(np.int8))
        units_to_device_dict[key] = device
        print ("key:", key, "\t device:", device)


    print("units_to_device_dict is generated")
    return units_to_device_dict


def generate_classification_data(CT_folders_location ='../data/data from apple/Initial Units CT DICOM (Examples)/' ):
    
    units_to_device_dict = generate_units_to_device_dict()
    for name in np.sort(glob.glob(CT_folders_location)): 
 
        nmFolders = name.split( os.path.sep )
        
        unit = nmFolders[-2]
        unit_dir = name + "*/"

        j = 0
        image_set_complete = []
        for dir_CT in glob.glob(unit_dir):
            print(j,"j")
            if j==0 or j==1:
                j+=1
                continue

            device = units_to_device_dict[unit]
            device_folder = os.path.join("./data_for_classification", device )
            pre_string = unit+"_{0:04d}".format(j)
            j += 1

            if not os.path.exists(device_folder):   
                os.mkdir(device_folder)
            
            # while len(image_set_complete) < 20:
            dicom_with_vtk.sim_xray_set_from_CT_folder(dir_CT,\
                                                      perturb_scale=1, \
                                                      debug=False,
                                                      num_images_per_unit=1000,\
                                                      device_folder = device_folder,
                                                      pre_string = pre_string)
                
                # image_set_complete.extend(image_set)
                # print(len(image_set_complete),"len(image_set_complete)")
            
            # for i in range(len(image_set_complete)):
            #     cv2.imwrite("./data_for_superglue/{}_{}.jpeg".format(unit,i), image_set_complete[i])


            # for i in range(len(image_set_complete)):
            #     cv2.imwrite(device_folder + "/{}_{}.jpeg".format(unit,i), image_set_complete[i])                

            print(pre_string, "done")

def generate_classification_data_single(CT_folders_location ='../data/data from apple/Initial Units CT DICOM (Examples)/',catalogue_filename
                                 apple_device='iPhone 6 Plus' ):
    units_to_device_dict = generate_units_to_device_dict(catalogue_filename)


    for name in np.sort(glob.glob(CT_folders_location)): 
        nmFolders = name.split( os.path.sep )
        
        unit = nmFolders[-2]
        unit_dir = name + "*/"

        j = 0
        image_set_complete = []
        for dir_CT in glob.glob(unit_dir):
            
            if apple_device == units_to_device_dict[unit]:
                print(apple_device,"apple_device")
                device = units_to_device_dict[unit]
                device_folder = os.path.join("./data_for_classification", device )
                pre_string = unit+"_{0:04d}".format(j)
                j += 1

                if not os.path.exists(device_folder):   
                    os.mkdir(device_folder)
                
                # while len(image_set_complete) < 20:
                dicom_with_vtk.sim_xray_set_from_CT_folder(dir_CT,\
                                                          perturb_scale=1, \
                                                          debug=False,
                                                          num_images_per_unit=1000,\
                                                          device_folder = device_folder,
                                                          pre_string = pre_string)


                    



def main():
    # CT_folders_location = "/media/biorobotics/BirobTejasDisk23/tejas_desktop_backup/Apple-2Dto3D/data/data_from_apple_2/phones/*/"
    # CT_folders_location = "/media/biorobotics/BirobTejasDisk23/tejas_desktop_backup/Apple-2Dto3D/data/data_from_apple_2/watches/*/"
    # catalogue_filename = "/media/biorobotics/BirobTejasDisk23/tejas_desktop_backup/Apple-2Dto3D/data/data_from_apple_2/phones/Unit Catalog with Scan Parameters.xlsx"
    CT_folders_location ="./Unit_01/FACT-6383 Project Daisy POC_Unit 1_5min_1_DICOM"
    catalogue_filename ="./Unit Catalog with Scan Parameters.xlsx"
    
    # units_to_device_dict = generate_units_to_device_dict(catalogue_filename)
    # print(units_to_device_dict['Unit 52'],"units_to_device_dict['Unit 52']")
    # generate_classification_data_singlessification_data(CT_folders_location)
    # generate_classification_data_single(CT_folders_location, apple_device='iPhone 6 Plus')
    generate_classification_data_single(CT_folders_location,catalogue_filename, apple_device='iPhone 5')
    # generate_classification_data_single(CT_folders_location, apple_device='iPhone 5SE')
    # generate_classification_data_single(CT_folders_location, apple_device='iPhone 6')
    # generate_classification_data_single(CT_folders_location, apple_device='iPhone 7')
    # generate_classification_data_single(CT_folders_location, apple_device='iPhone 7 Plus')
    # generate_classification_data_single(CT_folders_location, apple_device='iPhone 8 Plus')
    # generate_classification_data_single(CT_folders_location, apple_device='iPhone XR')
    # generate_classification_data_single(CT_folders_location, apple_device='iPhone X')
    # generate_classification_data_single(CT_folders_location, apple_device='Watch Series 3')
    # generate_classification_data_single(CT_folders_location, apple_device='Watch Series 4')
    # generate_classification_data_single(CT_folders_location, apple_device='Watch Series 5')
    # generate_classification_data_single(CT_folders_location, apple_device=units_to_device_dict['Unit 52'])

    # dir_CT = '../data/data from apple/Initial Units CT DICOM (Examples)/FACT6383-Project Daisy POC_Unit 40_5min_1_DICOM'


if __name__ == '__main__':
    main()