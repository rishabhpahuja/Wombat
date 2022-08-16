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

