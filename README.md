# Non-rigid registration

This repository is used to find the new co-ordinates of screws in case they change while punching them out of the electronic device. This codebase is based on **Demon's algorithm** which is a non-rigid registration algorithm. It maps the flow in the pixels between the 'fixed image' and 'moving image', i.e. the image of the electronic device in pristine condition and the image of the device when deformed, respectively. It generates a flow field w.r.t moving image, which when applied to the fixed image, moves the pixels in fixed image to make it similar to moving image or vice versa. 

This repository has two components:
1. Arrange images in same frames to get good results (optional)
2. Demon's registration to register same components 

The folder Arrange_images contains the codebase to align images in similar frames. It uses superglue algorithm to align images and crop the unwated information from the images, which is selected by the user when the code is run.
