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


    # mask_fixed = sitk.GetImageFromArray(mask_fixed,sitk.sitkInt8)
    mask_moving = sitk.GetImageFromArray(mask_moving,sitk.sitkInt8)

    demon_filter=sitk.SymmetricForcesDemonsRegistrationFilter()
    demon_filter.SetNumberOfIterations(50)
    # demon_filter.SetSmoothDisplacementField(True)
    demon_filter.SetStandardDeviations(5.5)
    # demon_filter.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(demon_filter))

    initial_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                      moving_image, 
                                                      sitk.Euler2DTransform(), 
                                                      sitk.CenteredTransformInitializerFilter.MOMENTS)
    
    start_time=time.time()

    tx=multiscale_demons(registration_algorithm=demon_filter,fixed_image=fixed_image, moving_image=moving_image,initial_transform=initial_transform ,
                        shrink_factors=[8,4,2], smoothing_sigmas=[30,20,4])

    # tx=multiscale_demons(registration_algorithm=demon_filter,fixed_image=fixed_image, moving_image=moving_image,initial_transform=initial_transform)

    end_time=time.time()

    print("Time taken:",(end_time-start_time))
    
    transformed_moving_image=sitk.Resample(moving_image,fixed_image,tx,sitk.sitkNearestNeighbor,0.0,moving_image.GetPixelID())
    transformed_moving_mask=sitk.Resample(mask_moving,fixed_image,tx,sitk.sitkNearestNeighbor,0.0,moving_image.GetPixelID())

    deformed_moving = sitk.GetArrayFromImage(transformed_moving_image)
    fixed_image=sitk.GetArrayFromImage(fixed_image)
    moving_image=sitk.GetArrayFromImage(moving_image)
    deformed_mask=sitk.GetArrayFromImage(transformed_moving_mask)
    # print(type(deformed_mask),type(moving_image))
    # print(deformed_mask.shape,deformed_moving.shape)
    # overlay=cv2.addWeighted(np.float32(fixed_image),1,np.float32(deformed_mask),1,0.0)
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



    # cv2.imwrite("Deformed_image.jpeg",deformed_moving)
    # cv2.imwrite("fixed.png",fixed_image)
    # cv2.imshow("Deformed_mask",deformed_moving)
    # cv2.imshow("Ground_truth_mask")
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
            f"Usage: {sys.argv[0]} <fixedImageFilter> <movingImageFile> <outputTransformFile>")
        sys.exit(1)

    mask_moving=io.imread('./im12_.png',as_gray=True)
    mask_moving = cv2.normalize(mask_moving, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    mask_moving = mask_moving.astype(np.uint8)
    # mask_moving=np.asarray(mask_moving,np.uint8)
    cv2.imshow('mask',mask_moving)
    # mask_moving = np.asarray(mask_moving, np.float64)
    # mask_moving=mask_moving[200:-200,200:-200]
    # mask_moving=cv2.resize(mask_moving,(int(mask_moving.shape[1]/7),int(mask_moving.shape[0]/7)))
    # print(mask_moving.shape)

    mask_fixed=io.imread('./im13_.png',as_gray=True)
    mask_fixed = cv2.normalize(mask_fixed, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    mask_fixed = mask_fixed.astype(np.uint8)
    # mask_fixed = np.asarray(mask_fixed, np.float64)
    # mask_fixed=mask_fixed[200:-200,200:-200]
    # mask_fixed=cv2.resize(mask_fixed,(int(mask_fixed.shape[1]/7),int(mask_fixed.shape[0]/7)))
    # mask_moving=resize(mask_moving,(mask_fixed.shape[0],mask_fixed.shape[1]))
    # print(mask_fixed.shape)

    fixed_CV = io.imread(sys.argv[1], as_gray=True)
    # print(type(fixed_CV[0][0]))
    # fixed_CV = np.asarray(fixed_CV, np.float64)
    # fixed_CV=fixed_CV[200:-200,200:-200]
    # fixed_CV=cv2.resize(fixed_CV,(int(fixed_CV.shape[1]/7),int(fixed_CV.shape[0]/7)))
    # print(fixed_CV.shape)

    moving_CV = io.imread(sys.argv[2], as_gray=True)
    # moving_CV = np.asarray(moving_CV, np.float64)
    # moving_CV=moving_CV[200:-200,200:-200]
    # moving_CV=cv2.resize(moving_CV,(640,800))
    # moving_CV=resize(moving_CV,(int(moving_CV.shape[0]/7),int(moving_CV.shape[1]/7)))
    # print(moving_CV.shape)
    print(type(fixed_CV), type(moving_CV))
    # moving_CV=equal_intensity(fixed_CV,moving_CV)
    print(type(moving_CV))

    #Blurring images to reduce intensity diference even further

    # fixed_CV= ndi.uniform_filter(fixed_CV,size=2)
    # moving_CV= ndi.uniform_filter(moving_CV,size=2)

    if True:
        cv2.namedWindow("fixed", cv2.WINDOW_NORMAL)
        cv2.namedWindow("moving", cv2.WINDOW_NORMAL)
        cv2.namedWindow("overlay mask with image fixed",cv2.WINDOW_NORMAL)
        cv2.namedWindow("overlay mask with image moving",cv2.WINDOW_NORMAL)

        # print(fixed_CV.shape, mask_fixed.shape)
        # print(fixed_CV.shape)
        # print(mask_fixed.shape)
        overlay_fixed=cv2.addWeighted(fixed_CV,1,mask_fixed,1,0.0)
        # print(type(moving_CV), type(mask_moving))
        # moving_cv=np.asarray(moving_CV, np.uint8)
        overlay_moving=cv2.addWeighted(moving_CV,1,mask_moving,1,0.0)
        # cv2.namedWindow("matched", cv2.WINDOW_NORMAL)
        cv2.imshow("fixed",fixed_CV)
        cv2.imshow("moving",moving_CV)
        cv2.imshow("overlay mask with image fixed",overlay_fixed)
        cv2.imshow("overlay mask with image moving",overlay_moving)
        # overlay_fixed = cv2.normalize(overlay_fixed, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        # overlay_fixed = overlay_fixed.astype(np.uint8)  
        cv2.imwrite('pristine_phone.jpeg',overlay_moving)
        # cv2.imshow("matched",matched)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    

    fixed_CV = sitk.GetImageFromArray(fixed_CV,sitk.sitkInt8)
    moving_CV = sitk.GetImageFromArray(moving_CV,sitk.sitkInt8)
    demons_regis(fixed_CV,moving_CV,mask_fixed,mask_moving)



    
    
