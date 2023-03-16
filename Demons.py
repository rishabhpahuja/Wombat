import SimpleITK as sitk
import sys
import os
# import helper
# from IPython import embed    
import numpy as np
import cv2
from skimage import exposure
from skimage import io
import matplotlib.pyplot as plt
from skimage.transform import resize
import scipy.ndimage as ndi
import time

def command_iteration(filter):
    print(f"{filter.GetElapsedIterations():3} = {filter.GetMetric():10.5f}")

def smooth_and_resample(image,shrink_factor,smoothing_sigma):

    smoothed_image=sitk.SmoothingRecursiveGaussian(image,smoothing_sigma)

    orignal_spacing=image.GetSpacing()
    orignal_size=image.GetSize()
    new_size=[int(sz/float(shrink_factor)+0.5) for sz in orignal_size]
    new_spacing=[((orignal_sz-1)*orignal_spc)/(new_sz-1)for orignal_sz, orignal_spc, new_sz in zip(orignal_size,orignal_spacing,new_size)]

    return (sitk.Resample(smoothed_image, new_size, sitk.Transform(),sitk.sitkLinear,image.GetOrigin(),new_spacing,image.GetDirection(),0.0,image.GetPixelID()))

def multiscale_demons(registration_algorithm, fixed_image, moving_image, initial_transform=None, shrink_factors=None, smoothing_sigmas=None):

    fixed_images=[fixed_image]
    moving_images=[moving_image]

    if shrink_factors:

        for shrink_factor, smoothing_sigma in reversed(list(zip(shrink_factors,smoothing_sigmas))):

            fixed_images.append(smooth_and_resample(fixed_images[0],shrink_factor,smoothing_sigma))
            moving_images.append(smooth_and_resample(moving_images[0],shrink_factor,smoothing_sigma))
    

    if initial_transform:
        initial_displacement_field=sitk.TransformToDisplacementField(initial_transform,sitk.sitkVectorFloat64,fixed_images[-1].GetSize(),\
                                                                    fixed_images[-1].GetOrigin(), fixed_images[-1].GetSpacing(),
                                                                    fixed_images[-1].GetDirection())


    for f_image, m_image in reversed(list(zip(fixed_images[0:-1],moving_images[0:-1]))):


        initial_displacement_field=sitk.Resample(initial_displacement_field,f_image) #changing initial dispalcement image 
        initial_displacement_field=registration_algorithm.Execute(f_image,m_image, initial_displacement_field) #executing registration
    
    return sitk.DisplacementFieldTransform(initial_displacement_field)

    

def demons_regis(fixed_image,moving_image,mask_fixed,mask_moving):


    mask_moving = sitk.GetImageFromArray(mask_moving,sitk.sitkInt8)

    demon_filter=sitk.SymmetricForcesDemonsRegistrationFilter()
    demon_filter.SetNumberOfIterations(50)
    demon_filter.SetStandardDeviations(5.5)

    initial_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                      moving_image, 
                                                      sitk.Euler2DTransform(), 
                                                      sitk.CenteredTransformInitializerFilter.MOMENTS)
    
    start_time=time.time()

    tx=multiscale_demons(registration_algorithm=demon_filter,fixed_image=fixed_image, moving_image=moving_image,initial_transform=initial_transform ,
                        shrink_factors=[8,4,2], smoothing_sigmas=[30,20,4])

    end_time=time.time()

    print("Time taken:",(end_time-start_time))
    
    transformed_moving_image=sitk.Resample(moving_image,fixed_image,tx,sitk.sitkNearestNeighbor,0.0,moving_image.GetPixelID())
    transformed_moving_mask=sitk.Resample(mask_moving,fixed_image,tx,sitk.sitkNearestNeighbor,0.0,moving_image.GetPixelID())

    deformed_moving = sitk.GetArrayFromImage(transformed_moving_image)
    fixed_image=sitk.GetArrayFromImage(fixed_image)
    moving_image=sitk.GetArrayFromImage(moving_image)
    deformed_mask=sitk.GetArrayFromImage(transformed_moving_mask)
    overlay=cv2.addWeighted((fixed_image),1,(deformed_mask),0.3,0.0)
    overlay_test=cv2.addWeighted((fixed_image),1,(mask_fixed),0.3,0.0)

    cv2.namedWindow("Ground truth deformed mask with deformed image",cv2.WINDOW_NORMAL)
    cv2.namedWindow("Transformed_image",cv2.WINDOW_NORMAL)
    cv2.namedWindow("fixed",cv2.WINDOW_NORMAL)
    cv2.namedWindow("Orignal_image",cv2.WINDOW_NORMAL)
    cv2.namedWindow("Transformed_mask",cv2.WINDOW_NORMAL)
    cv2.namedWindow("overlay mask with image",cv2.WINDOW_NORMAL)
    cv2.namedWindow("Orignal_mask",cv2.WINDOW_NORMAL)
    # cv2.namedWindow("Deformed_mask",cv2.WINDOW_NORMAL)    

    cv2.imshow("Ground truth deformed mask with deformed image",overlay_test)
    cv2.imshow("Transformed_image",deformed_moving)
    cv2.imshow("fixed",fixed_image)
    cv2.imshow("Orignal_image",moving_image)
    cv2.imshow("Transformed_mask",deformed_mask)
    cv2.imshow("overlay mask with image",overlay)
    cv2.imshow("Orignal_mask",fixed_image)

    # print(deformed_mask)
    print(np.max(deformed_mask),np.min(deformed_mask))

    deformed_mask = cv2.normalize(deformed_mask, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    deformed_mask = deformed_mask.astype(np.uint8)
    cv2.imwrite('iPhone_6_deformed_mask.png',deformed_mask)
    # deformed_mask=deformed_mask*255
    print(np.max(deformed_mask),np.min(deformed_mask))
    # print(len(np.where(deformed_mask==255)[0]))

    fixed_image = cv2.normalize(fixed_image, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    fixed_image = fixed_image.astype(np.uint8)

    overlay = cv2.normalize(overlay, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    overlay = overlay.astype(np.uint8)

    cv2.imwrite('iPhone_6_with_pristine_mask.png',overlay)
    cv2.waitKey()    
    cv2.destroyAllWindows()


    print(f" RMS: {demon_filter.GetRMSChange()}")

from skimage.exposure import match_histograms

def equal_intensity(image1,image2, debug=False):

    matched = match_histograms(image2, image1)

    if debug:
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1,ncols=3,figsize=(8, 3),sharex=True,sharey=True)

        for aa in (ax1, ax2, ax3):
            aa.set_axis_off()

        ax1.imshow(image1)
        ax1.set_title('Source image')
        ax2.imshow(image2)
        ax2.set_title('Reference image')
        ax3.imshow(matched)
        ax3.set_title('Matched image')
        
        plt.tight_layout()
        plt.show()


    return matched

if __name__ == '__main__':
    
    if len(sys.argv) < 4:
        print(
            f"Usage: {sys.argv[0]} <fixedImageFile> <movingImageFile> <fixedImageMask> <movingImageMask>")
        sys.exit(1)

    fixed_CV = io.imread(sys.argv[1], as_gray=True)
    mask_fixed=io.imread(sys.argv[3], as_gray=True)
    mask_fixed = cv2.normalize(mask_fixed, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

    moving_CV = io.imread(sys.argv[2], as_gray=True)
    mask_moving=io.imread(sys.argv[4], as_gray=True)
    mask_moving = mask_moving.astype(np.uint8)

    if False:
        cv2.namedWindow("fixed", cv2.WINDOW_NORMAL)
        cv2.namedWindow("moving", cv2.WINDOW_NORMAL)
        cv2.namedWindow("overlay mask with image fixed",cv2.WINDOW_NORMAL)
        cv2.namedWindow("overlay mask with image moving",cv2.WINDOW_NORMAL)

        overlay_fixed=cv2.addWeighted(fixed_CV,1,mask_fixed,1,0.0)
        overlay_moving=cv2.addWeighted(moving_CV,1,mask_moving,1,0.0)
        cv2.imshow("fixed",fixed_CV)
        cv2.imshow("moving",moving_CV)
        cv2.imshow("overlay mask with image fixed",overlay_fixed)
        cv2.imshow("overlay mask with image moving",overlay_moving)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    

    fixed_CV = sitk.GetImageFromArray(fixed_CV,sitk.sitkInt8)
    moving_CV = sitk.GetImageFromArray(moving_CV,sitk.sitkInt8)
    demons_regis(fixed_CV,moving_CV,mask_fixed,mask_moving)



    
    
