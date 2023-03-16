
# Wombat
This repository has two branches:

1. Dataset_generation: Used to generate a labelled X-ray data to perform 2D segmentation of electronic devices usign CT data.
2. Non-rigid_egistration: Used to find changing screw locations due to deforamtions experienced by electronic devices while punching screws on an automated punch station.

# Data Generation

This branch is reponsible for generating data to train 2D unet segmentation pipeline to segment out different components in X-ray scans of electronic devices. This is done in two steps:

1. Generating X-rays from CT scans
2. 'Semi-automatically' labelling the X-rays generated

Both these steps can be carried out by the same python file, called **test_bayes.py**, by typing python3 test_bayes.py with the necessary arguments explained below.

## Generating X-rays from CT scans

The speciality of this module is that it can automatically tune paramters to generate good quality X-ray images from CT scans using *Bayesian Optimization*. A range of values for the paramters is already stored in this code. If this piece of code does not give you good quality images, either the parameter value range needs to be updated in line 21 of test_bayes.py or the exact values can be manually entered in line 69 & 73 of test_bayes.py.

The arguments required for this section are:

- **CT_data:** Location of the CT data saved on to the system
- **image_name_real:** Location and name of the real X-ray image whose X-rays have to be generated from CT data. In case the real X-ray image of the device is not available, insert address and location of the real X-ray image of a similar electronic device.
- **num_images_to_create:** Number of X-ray images to be created from CT scan
- **synth_image_path:** Location where the X-rays generated have to be stored
- **device:** This variable indicates whether the X-rays generated are of a smart watch, smart phone or a tablet. Input 0,1 or 2 respectively
- **x_ray:** This varibale indicates if x-rays have to be generated from CT scans. Set this value to True to run this section of code.
-**automatic:** This variable indicates if the tuning of the paramters to generate X-rays has to be done automatically or manually. If selected, False (manually), the values of the parameters *opacity_points* and *scalar_color_mapping_points* are found using a 3D visualizer software. We used an open source software called 3D Slicer (https://www.slicer.org/).

## 'Semi-automatically' labelling the X-rays generated

The speciality of this module is that it can be used to label all the images generated in part 1 by labelling only one single image of each type of electronic device.

The arguments required for this section are: 

- **label:** Set this True for generating labels. This is otherwise set to False while generating X-rays from CT data.
-  **ref_image:** Stores the path and location of the one file which is labelled. This is taken as the reference image to label all the other images of the same electronic device.
-  **label_ref:** The label of the reference image stored in *.png* or *.jpg* format

This section works by extracting features from the reference image and the image on which the labels have to be transferred to. The correspondig keypoints are matched using an algorithm called *superglue algorithm* and the homography matrix is found between the two sets of keypoints. The refernce label is multiplied by the homography matrix to find the label of the new image.

**NOTE:** 
1. The reference image was labelled using an open source software called CVAT (https://cvat.org/) and downloaded in *.xml* format. This file is stored in the base directory, the location where *test_bayes.py* is stored.
2. Both the sections of this branch, use two deep learning models, *superpoint* and *superglue*. The weights of these models are stored in this link (https://drive.google.com/drive/folders/1izhpOcy6N5rSCzjYZVevcpTqaFOxjI_N?usp=sharing). They have to be stored at the root location.

# Non-rigid registration

This repository is used to find the new co-ordinates of screws in case they change while punching them out of the electronic device. This codebase is based on **Demon's algorithm** which is a non-rigid registration algorithm. It maps the flow in the pixels between the 'fixed image' and 'moving image', i.e. the image of the electronic device in pristine condition and the image of the device when deformed, respectively. It generates a flow field w.r.t moving image, which when applied to the fixed image, moves the pixels in fixed image to make it similar to moving image or vice versa. 

This repository has two components:
1. Arrange images in same frames to get good results (optional)
2. Demon's registration to register same components 

The folder Arrange_images contains the codebase to align images in similar frames. It uses superglue algorithm to align images and crop the unwated information from the images, which is selected by the user when the code is run.

## Arrange Images

This section of the code is run by directly running the python file *Arrange_images.py*. It has two parameters:
- reference_path: Path and name of the image, other images are aligned to 
- align_image_path: Path to the directory consisting of the images which have to be aligned like the reference image

It uses two deep learning models, *superglue* and *superpoint* whose weights can be found on this link (https://drive.google.com/drive/folders/1izhpOcy6N5rSCzjYZVevcpTqaFOxjI_N?usp=sharing)

## Demon's registration

This component can be run by running Demons.py <fixedImageFile> <movingImageFile> <fixedImageMask> <movingImageMask>, where fixedImageMask and movingImageMask, are the screw masks in image format of the respective image. The labelling is done usign an open sourse software called CVAT (https://cvat.org/). The code transforms the movingImageMask to fixedImageMask. This tranformation is done by using the respective images, fixed and moving and the input of fixedImageMask is just to test the effectiveness of the algorithm and is not used anywhere else in the code.