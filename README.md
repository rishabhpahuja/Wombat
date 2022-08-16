# Data Generation

This branch is reponsible for generating data to train 2D unet segmentation pipeline to segment out different components in X-ray scans of electronic devices. This is done in two steps:

1. Generating X-rays from CT scans
2. 'Semi-automatically' labelling the X-rays generated

Both these steps can be carried out by the same python file, called **test_bayes.py**

## Generating X-rays from CT scans

The arguments required for this section are:

- **CT_data:** Location of the CT data saved on to the system
- **image_name_real:** Location and name of the real X-ray image whose X-rays have to be generated from CT data. In case the real X-ray image of the device is not available, insert address and location of the real X-ray image of a similar electronic device.
- **num_images_to_create:** Number of X-ray images to be created from CT scan
- **synth_image_path:** Location where the X-rays generated have to be stored
- **device:** This variable indicates whether the X-rays generated are of a smart watch, smart phone or a tablet. Input 0,1 or 2 respectively
-**automatic:** This variable indicates if the tuning of the paramters to generate X-rays has to be done automatically or manually. If selected, False (manually), the values of the parameters *opacity_points* and *scalar_color_mapping_points* are found using a 3D visualizer software. We used an open source software called 3D Slicer (https://www.slicer.org/).

The speciality of this module is that it can automatically tune paramters to generate good quality X-ray images from CT scans using *Bayesian Optimization*. A range of values for the paramters is already stored in this code. If this piece of code does not give you good quality images, either the parameter value range needs to be updated in line 21 of test_bayes.py or the exact values can be manually entered in line 69 & 73 of test_bayes.py.

## 'Semi-automatically' labelling the X-rays generated







