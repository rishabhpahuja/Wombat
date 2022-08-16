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









