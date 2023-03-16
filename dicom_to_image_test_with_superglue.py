from super_matching import SuperMatching
import time
from scipy.stats import gamma
from models.superpoint import SuperPoint
import torch
from scipy import ndimage
import matplotlib.pyplot as plt

import vtk
from vtk import vtkTransform
import os
import numpy as np
import cv2
from vtk.util.numpy_support import vtk_to_numpy
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../2Dto2D/')
# import FeatureMatching_Tejas


config = {}
superpoint = SuperPoint(config.get('superpoint', {}))
MIN_NUM_KPS = 50  # for low res images


def frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)


def superpoint_pred_from_cv_image(image):
    data_torch = frame2tensor(image, device='cpu')
    pred = superpoint({'image': data_torch})
    kps = pred['keypoints'][0]
    scores = pred['scores'][0]
    descriptors = pred['descriptors'][0]
    return kps, scores, descriptors


def gamma_distribution_criterian(y1, nbins=100,  debug=False):
    '''
    requires a flattened image with  background filtered as input 

    '''

    y1 = y1[np.where(y1 < 250)]
    try:
        shape1, loc1, scale1 = gamma.fit(y1)
    except:
        print("here")
        return False, -10000
    '''
    shape1 is the k in gamma distribution parametrization
    '''
    k_min = 0.9
    # k_max = 10
    if (shape1 > k_min):  # and (shape1 > k_min):
        is_good_image = 1
        title = 'k_min = {:.2f}, k={:.2f} \n Result: DOES satisfy dist criteria'.format(
            k_min, shape1)
    else:
        title = 'k_min = {:.2f}, k={:.2f} \n Result: DOES NOT satisfy dist criteria'.format(
            k_min, shape1)
        is_good_image = 0

    if debug:
        print("shape1: {}, loc1: {}, scale1: {}".format(shape1, loc1, scale1))
        out = np.histogram(y1, density=True, bins=nbins)

        freq = out[0]
        bins = out[1]
        error = 0

        x_vals = []

        for i in range(nbins):
            bin_avg = (bins[i]+bins[i+1])/2
            x_vals.append(bin_avg)

            pdf_val = gamma.pdf(x=bin_avg,
                                a=shape1, loc=loc1, scale=scale1)
        plt.figure()

        plt.hist(y1, label='pixel distribution', alpha=0.5,
                 density=True, bins=nbins, color='g')

        x_vals = np.array(x_vals)

        bin_size = bins[1]-bins[0]
        x_vals = np.insert(x_vals, 0, bins[0] - bin_size)

        pdf_vals = gamma.pdf(x=x_vals, a=shape1, loc=loc1, scale=scale1)

        plt.plot(x_vals, pdf_vals, color='g', label='fitted distribution')
        plt.title(title, fontsize=12)
        plt.show(block=False)

    return is_good_image, shape1


def render_CT_image(imageData,
                    opacity_points=np.array([[2360.0,  0.0],
                                             [10632.0, 0.0],
                                             [23429.0, 1.0],
                                             [68735.0, 1.0]]),

                    scalar_color_mapping_points=np.array([[2360, 0.5],
                                                          [10632, 0.5],
                                                          [17055, 0.5],
                                                          [20267, 0.5],
                                                          [23429, 0.5],
                                                          [67895, 0.5]]),

                    camera_position=None,
                    image_shape=(510, 510),
                    Background=(0, 0, 0)):

    volumeMapper = vtk.vtkSmartVolumeMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        volumeMapper.SetInputConnection(imageData.GetProducerPort())
    else:
        volumeMapper.SetInputData(imageData)

    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.ShadeOff()
    volumeProperty.SetInterpolationType(vtk.VTK_LINEAR_INTERPOLATION)

    compositeOpacity = vtk.vtkPiecewiseFunction()

    for opacity_point in opacity_points:
        compositeOpacity.AddPoint(opacity_point[0], opacity_point[1])
    volumeProperty.SetScalarOpacity(compositeOpacity)
    scalarColorMapping = vtk.vtkPiecewiseFunction()

    for scalar_color_mapping_point in scalar_color_mapping_points:
        scalarColorMapping.AddPoint(
            scalar_color_mapping_point[0], scalar_color_mapping_point[1])
    # volumeProperty.SetColor(color)
    volumeProperty.SetColor(scalarColorMapping)

    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    
    volume.SetProperty(volumeProperty)
 

    renWin = vtk.vtkRenderWindow()
    ren1 = vtk.vtkRenderer()
    ren1.SetBackground(Background[0], Background[1], Background[2])
    # ren1.SetBackground(0.5, 0.5, 0.5);

    renWin.AddRenderer(ren1)
    # renWin.SetSize(800, 800);
    renWin.SetSize(image_shape[0], image_shape[1])

    ren1.AddViewProp(volume)
   
    if camera_position is None:
        ren1.ResetCamera()
    else:
        camera = vtk.vtkCamera()
        camera.SetPosition(
            camera_position[0], camera_position[1], camera_position[2])
        # camera.SetFocalPoint(0, 0, 0);
        boundingBox = imageData.GetBounds()
        camera.SetFocalPoint(
            boundingBox[1]/2, boundingBox[3]/2, boundingBox[5]/2)
        ren1.SetActiveCamera(camera)
    # Render composite. In default mode. For coverage.
    renWin.ShowWindowOff()
    renWin.Render()

    # 3D texture mode. For coverage.
    # volumeMapper.SetRequestedRenderModeToRayCastAndTexture()
    # renWin.Render()

    # Software mode, for coverage. It also makes sure we will get the same
    # regression image on all platforms.
    # volumeMapper.SetRequestedRenderModeToRayCast()
    # renWin.Render()

    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(renWin)
    w2if.Update()

    return vtk_img_to_np(w2if)

    # iren.Start()


def num_keypoints_in_image(img1, debug=False):

    ref_keypoints, _, _ = superpoint_pred_from_cv_image(img1)
    title = 'num keypoints = {}, \n min required={} \n'.format(
        len(ref_keypoints), MIN_NUM_KPS)

    if len(ref_keypoints) > MIN_NUM_KPS:
        title = title + "Result: DOES satisfy num_keypoints_criteria"
    else:
        title = title + "Result: DOES NOT satisfy num_keypoints_criteria"

    if debug:
        print(len(ref_keypoints), "len(ref_keypoints)")
        plt.figure()
        plt.imshow(img1, cmap='gray')
        plt.scatter(ref_keypoints[:, 0], ref_keypoints[:, 1], s=10, c='w')
        plt.scatter(ref_keypoints[:, 0], ref_keypoints[:, 1], s=5, c='k')

        plt.title(title)
        # cv2.namedWindow("img1", cv2.WINDOW_NORMAL)
        # cv2.imshow("img1",img1)
        # cv2.waitKey(0)
        # plt.show()

    try:
        return len(ref_keypoints)
    except:
        return 0


def num_keypoints_criteria(img1, debug=False, min_num_kps=MIN_NUM_KPS):
    
    
    num_keypoints = num_keypoints_in_image(img1, debug=debug)

    if num_keypoints > min_num_kps:
        return 1, num_keypoints
    else:
        return 0, num_keypoints


def vtk_img_to_np(w2if):
    '''
    reference: https://stackoverflow.com/questions/25230541/how-to-convert-a-vtkimage-into-a-numpy-array
    '''

    im = w2if.GetOutput()
    rows, cols, _ = im.GetDimensions()
    sc = im.GetPointData().GetScalars()
    a = vtk_to_numpy(sc)
    a = a.reshape(rows, cols, -1)
    return a


GAMMA = 2.0


def adjust_gamma(image, gamma=1.0):
    # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # Apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def change_imageData_datatype(imageData):
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

    imageData_copy = vtk.vtkImageData()
    imageData_copy.DeepCopy(imageData)

    dim = imageData_copy.GetDimensions()  # dimension is in 3D
    sc = imageData_copy.GetPointData().GetScalars()
    a = vtk_to_numpy(sc)  # dimension is 1D
    voccel_np = a.reshape(dim, order='F')
    depthArray = numpy_to_vtk(voccel_np.flatten(
        order='F'), deep=True, array_type=vtk.VTK_SHORT)
    imageData_copy.GetPointData().SetScalars(depthArray)
    return imageData_copy


def CT_data_from_CT_folder(dir_CT):
    reader = vtk.vtkDICOMImageReader()
    reader.SetDirectoryName(dir_CT)
    reader.Update()
    imageData = reader.GetOutput()

    imageData = change_imageData_datatype(imageData)

    return imageData


def get_params():
    '''
    parameters manually tuned using 3DSlicer
    '''

    params = []

    # Unit 40 0
    opacity_points = np.array([[2360.0,  0.0],
                               [10632.0, 0.0],
                               [23429.0, 1.0],
                               [68735.0, 1.0]])
    # opacity_points= np.array([[0.0,  0.0],
    #                           [20028.68, 0.00],
    #                           [56266.11, 0.5],
    #                           [65535.0, 1.0]])

    scalar_color_mapping_points = np.array([[2360, 0.5],
                                            [10632, 0.5],
                                            [17055, 0.5],
                                            [20267, 0.5],
                                            [23429, 0.5],
                                            [67895, 0.5]])

    params_1 = [opacity_points, scalar_color_mapping_points]
    params.append(params_1)

    # Unit 02 1
    # opacity_points= np.array([[1800.0,  0.0],
    #                           [20655.0, 0.0],
    #                           [30196.0, 1.0],
    #                           [67335.0, 0.05]])

    # scalar_color_mapping_points= np.array([[1800,0.5],
    #                               [7111,0.5],
    #                               [16639,0.5],
    #                               [22543,0.5],
    #                               [28045,0.5],
    #                               [29045,0.5]])
    opacity_points = np.array([[1800.0,  0.0],
                               [20300.0, 0.0],
                               [36814.0, 1.0],
                               [67335.0, 0.05]])

    scalar_color_mapping_points = np.array([[1800, 0.5],
                                            [19793.81, 0.5],
                                            [23049, 0.5],
                                            [25444, 0.5],
                                            [27838, 0.5],
                                            [30196, 0.5]])

    params_2 = [opacity_points, scalar_color_mapping_points]
    params.append(params_2)

    # Unit 03 2
    opacity_points = np.array([[1800.0,  0.0],
                               [19792.0, 0.0],
                               [36814.0, 1.0],
                               [67335.0, 0.05]])

    scalar_color_mapping_points = np.array([[0, 0.5],
                                            [5613.56, 0.5],
                                            [18146.14, 0.5],
                                            [25848, 0.5],
                                            [35770, 0.5],
                                            [67335, 0.5]])
    params_2 = [opacity_points, scalar_color_mapping_points]
    params.append(params_2)

    # Unit 06 3
    opacity_points = np.array([[1800.0,  0.0],
                               [20317.0, 0.0],
                               [38500.0, 1.0],
                               [67335.0, 1.0]])

    scalar_color_mapping_points = np.array([[0, 0.5],
                                            [6443.56, 0.5],
                                            [11944.14, 0.5],
                                            [21216, 0.5],
                                            [31903, 0.5],
                                            [67335, 0.5]])
    params_2 = [opacity_points, scalar_color_mapping_points]
    params.append(params_2)

    # Unit 11 4
    opacity_points = np.array([[1800.0,  0.0],
                               [18550, 0.0],
                               [38500.0, 1.0],
                               [67335.0, 1.0]])

    scalar_color_mapping_points = np.array([[0, 0.5],
                                            [11574, 0.5],
                                            [21897, 0.5],
                                            [30812, 0.5],
                                            [41291, 0.5],
                                            [67335, 0.5]])
    params_2 = [opacity_points, scalar_color_mapping_points]
    params.append(params_2)

    # Unit 12 exp1 5
    opacity_points = np.array([[1800.0,  0.0],
                               [25794, 0.0],
                               [40000.0, 1.0],
                               [67335.0, 1.0]])

    scalar_color_mapping_points = np.array([[0, 0.5],
                                            [10166, 0.5],
                                            [21000, 0.5],
                                            [33142, 0.5],
                                            [48091, 0.5],
                                            [67335, 0.5]])
    params_2 = [opacity_points, scalar_color_mapping_points]
    params.append(params_2)

    # Unit 12 exp2 6

    opacity_points = np.array([[1800.0,  0.0],
                               [25767, 0.0],
                               [5000.0, 1.0],
                               [67335.0, 1.0]])

    scalar_color_mapping_points = np.array([[0, 0.5],
                                            [7664, 0.5],
                                            [15328, 0.5],
                                            [23120, 0.5],
                                            [39755, 0.5],
                                            [67335, 0.5]])
    params_2 = [opacity_points, scalar_color_mapping_points]
    params.append(params_2)

    # Unit 16 exp1 7

    # opacity_points= np.array([[1800.0,  0.0],
    #                         [19703, 0.0],
    #                         [32000.0, 1.0],
    #                         [67335.0, 1.0]])

    # scalar_color_mapping_points= np.array([[0,0.5],
    #                             [7605,0.5],
    #                             [15363,0.5],
    #                             [23301,0.5],
    #                             [30362,0.5],
    #                             [57335,0.5]])

    opacity_points = np.array([[1800.0,  0.0],
                               [19176.43416335569, 0.0],
                               [36024.36628505861, 1.0],
                               [67335.0, 0.05]])
    scalar_color_mapping_points = np.array([[1800, 0.5],
                                            [7974.775892200420, 0.5],
                                            [16611.626876277034, 0.5],
                                            [20548.9044331412043, 0.5],
                                            [34309.774124902404, 0.5],
                                            [39196, 0.5]])
    params_2 = [opacity_points, scalar_color_mapping_points]
    params.append(params_2)

    # Unit 20 8
    opacity_points = np.array([[1800.0,  0.0],
                               [19778.0, 0.0],
                               [28000.0, 1.0],
                               [67335.0, 1.0]])

    scalar_color_mapping_points = np.array([[0, 0.5],
                                            [6774, 0.5],
                                            [12898.0, 0.5],
                                            [22279.0, 0.5],
                                            [39216.0, 0.5],
                                            [67335.0, 0.5]])
    params_2 = [opacity_points, scalar_color_mapping_points]
    params.append(params_2)

    # Unit 52 9
    # opacity_points = np.array([[1800.0,  0.0],
    #                            [10961.0, 0.0],
    #                            [28000.0, 1.0],
    #                            [67335.0, 1.0]])

    scalar_color_mapping_points = np.array([[0, 0.5],
                                            [3295.76, 0.5],
                                            [6718.29, 0.5],
                                            [11788.69, 0.5],
                                            [17366.14, 0.5],
                                            [67335, 0.5]])
    opacity_points = np.array([[1800.0,  0.0],
                               [10202.946172182774, 0.0],
                               [39133.743306303164, 1.0],
                               [67335.0, 1.0]])

    # scalar_color_mapping_points = np.array([[0, 0.5],
    #                                         [10138.605876719907, 0.5],
    #                                         [14687.714484706197, 0.5],
    #                                         [19680.723373078534, 0.5],
    #                                         [34778.3805813349, 0.5],
    #                                         [67335, 0.5]])
    params_2 = [opacity_points, scalar_color_mapping_points]
    params.append(params_2)

    # Unit 13
    opacity_points = np.array([[1800.0,  0.0],
                               [26089.0, 0.0],
                               [38534.0, 0.97],
                               [67335.0, 1.0]])

    scalar_color_mapping_points = np.array([[1800, 0.5],
                                            [26089, 0.5],
                                            [32335, 0.5],
                                            [35459, 0.5],
                                            [38534, 0.5],
                                            [67335, 0.5]])
    params_3 = [opacity_points, scalar_color_mapping_points]
    params.append(params_3)

    # Unit 30
    opacity_points = np.array([[3800.0,  0.0],
                               [28089.0, 0.0],
                               [40534.0, 0.97],
                               [69335.0, 1.0]])

    scalar_color_mapping_points = np.array([[3800, 0.5],
                                            [28089, 0.5],
                                            [31212, 0.5],
                                            [37459, 0.5],
                                            [40534, 0.5],
                                            [69335, 0.5]])
    params_4 = [opacity_points, scalar_color_mapping_points]
    params.append(params_4)

    return params


def perturb_params(params, perturb_scale=0.0):
    n = len(params)
    params_bef = np.copy(params)
    params[:, 0] = params[:, 0] + (np.random.rand(n)-0.5)*perturb_scale*1000
    return params


def perturb_params_opacity(params, perturb_scale=0.0):
    n = len(params)
    params_bef = np.copy(params)
    params[1, 0] = params[1, 0] + (np.random.rand()-0.5)*perturb_scale*50
    params[2, 0] = params[2, 0] + (np.random.rand()-0.5)*perturb_scale*1000
    return params


def perturb_params_color_map(params, perturb_scale=0.0):
    n = len(params)
    params_bef = np.copy(params)
    params[:, 0] = params[:, 0] + (np.random.rand(n)-0.3)*perturb_scale*1000
    return params


def sharpness_criteria(img):
    img_smooth = cv2.GaussianBlur(img, (5, 5), 0)

    array = np.asarray(img_smooth, dtype=np.int32)

    gy, gx = np.gradient(array)
    gnorm = np.sqrt(gx**2 + gy**2)
    sharpness = np.average(gnorm)
    print(sharpness, "SHARPNESS\n")


def setup_sg_class(superglue_weights_path):
    sg_matching = SuperMatching()
    sg_matching.weights = 'custom'
    sg_matching.weights_path = superglue_weights_path
    sg_matching.set_weights()

    return sg_matching


def num_matches_criteria(img1,image_real, debug=False):

    sg_matching = setup_sg_class('global_registration_sg.pth')
    image_real=str(image_real)
    # print('aaaaaaaaaa',image_real)
    # cv2.imshow('img',image_real)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    ref = cv2.imread(image_real, 0)
    conf, kp1, kp2, matches1, matches2 = sg_matching.detectAndMatch(ref, img1)

    if debug:
        rgb1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        rgb2 = cv2.cvtColor(ref, cv2.COLOR_GRAY2RGB)
        # Show keypoints
        for x, y in kp2(np.int64):
            cv2.circle(rgb1, (x, y), 100, (255, 0, 0), -1)
        for x, y in kp2.astype(np.int64):
            cv2.circle(rgb2, (x, y), 2, (0, 0, 255), -1)
        # displayImages(rgb1, 'Reference Image Keypoints',
    #               rgb2, 'Aligned Image Keypoints', 'results/superpoint_detection.jpg')

    # Show matches

        sg_matching.plot_matches(img1, ref, kp1, kp2, matches1, matches2)
    if len(np.where(conf >= 0.5)[0]) >= 4:
        # print("Yeeeees",conf)
        return 1, len(np.where(conf >= 0.5)[0])
    else:
        # print("Noooo",conf)
        return 0, len(np.where(conf >= 0.5)[0])


def flaten_foreground(img):

    
    img_background=int(np.average(img[0:2,0:2]))
    background_bool=np.isclose(img,img_background,atol=1.0,rtol=0.0)
    foreground_mask=np.where(background_bool==False)
    foreground=img[foreground_mask]
    return (foreground)

def image_score(xray_image_simulated, Background_255,real_img,workbook):
    debug = False
    # background_bool = np.isclose(
    #     xray_image_simulated, Background_255*np.ones(xray_image_simulated.shape), atol=1.0, rtol=0.0)
    # foreground_mask = np.where(background_bool == False)
    # foreground = xray_image_simulated[foreground_mask]
    # foreground=xray_image_simulated[]

    # cv2.imwrite("a.jpg", foreground)
    
    foreground_synthetic=flaten_foreground(xray_image_simulated)
    gamma_dist_synthetic=gamma_distribution_criterian(foreground_synthetic.ravel(), debug=debug)[1]
    real_image=cv2.imread(real_img,0)
    foreground_real=flaten_foreground(real_image)
    gamma_dist_real=gamma_distribution_criterian(foreground_real.ravel())[1]

    # well_distributed_img_intensities = gamma_distribution_criterian(
        # foreground.ravel(), debug=debug)[1]
    # if watch:
    well_distributed_img_intensities=1500-abs(gamma_dist_real-gamma_dist_synthetic)
    if well_distributed_img_intensities>=1500 or well_distributed_img_intensities<0:
        well_distributed_img_intensities=0
    
    # else:
    #     well_distributed_img_intensities=abs(gamma_dist_real-gamma_dist_synthetic)
    #     if well_distributed_img_intensities>=1500:
    #         well_distributed_img_intensities=0

    non_saturate_img = num_keypoints_criteria(
        xray_image_simulated, debug=debug)[1]

    clean_image = num_matches_criteria(xray_image_simulated, real_img,False)[1]

    # if  watch:
    score = well_distributed_img_intensities*0.2+non_saturate_img+clean_image
    # else:
    #     score= well_distributed_img_intensities+non_saturate_img*0.5+clean_image
    data=[well_distributed_img_intensities,non_saturate_img,clean_image,score]
    workbook.writerow(data)
    
    return(score)


def image_validity(xray_image_simulated, Background_255,real_img,debug,device=None):

    if device==0:
        h,w=xray_image_simulated.shape
        h,w=h//2,w//2
        
        xray_image_simulated=xray_image_simulated[w-int(w//2):w+int(w//2),h-int(h//2):h+int(h//2)]  
        xray_image_simulated=cv2.resize(xray_image_simulated,(504,504))

    background_bool = np.isclose(
        xray_image_simulated, Background_255*np.ones(xray_image_simulated.shape), atol=1.0, rtol=0.0)
    foreground_mask = np.where(background_bool == False)
    foreground = xray_image_simulated[foreground_mask]

    well_distributed_img_intensities = gamma_distribution_criterian(
        foreground.ravel(), debug=debug)[0]
    non_saturate_img = num_keypoints_criteria(
        xray_image_simulated, debug=debug)[0]
    clean_image = num_matches_criteria(xray_image_simulated, real_img, False)[0]

    # sharpness_criteria(xray_image_simulated)
    conclusion = well_distributed_img_intensities and non_saturate_img and clean_image

    if debug:
        title = 'simulated x ray\n'
        if conclusion:
            title += 'GOOD Image'
        else:
            title += 'BAD Image'

        xray_image_simulated_temp = xray_image_simulated
        xray_image_simulated_temp[0, 0] = 0
        plt.figure()
        plt.title(title)
        plt.imshow(xray_image_simulated_temp, cmap='gray')
        plt.show()

    return conclusion


def randomize_camera_pose(camera_position, high_res_shape):
    x, y, z = camera_position
    h, w = high_res_shape
    
    x += h*0.1*np.random.uniform(low=-0.05, high=0.05)
    y += w*0.1*np.random.uniform(low=-0.05, high=0.05)
    z += w*0.1*np.random.uniform(low=-0.05, high=0.05)

    camera_position = np.array([x, y, z])
    return camera_position




def make_image_one(real_img,CT_data,workbook=None,debug=True,device=None,**params):
    # dir_CT = '../iPhone_XS/Synthetic/Unit_52/FACT6383-Project Daisy POC_Unit 52_5min_1_DICOM'
    # imageData = CT_data_from_CT_folder(dir_CT)
    imageData = CT_data_from_CT_folder(CT_data)
    opacity_points = np.array([[1800.0,  0.0],
                               [params['op1'], 0.0],
                               [params['op2'], 1.0],
                               [67335.0, 0.05]])
    scalar_color_mapping_points = np.array([[1800, 0.5],
                                            [params['col1'], 0.5],
                                            [params['col2'], 0.5],
                                            [params['col3'], 0.5],
                                            [params['col4'], 0.5],
                                            [40000, 0.5]])
    
    Background = 0.27856470641658765  #random value taken to generate a background
    Background_255 = int(Background*255)
    camera_position = np.array([0, 0, -300])

    xray_image_simulated_lowres = render_CT_image(imageData,
                                                  opacity_points=opacity_points,
                                                  scalar_color_mapping_points=scalar_color_mapping_points,
                                                  Background=(
                                                      Background, Background, Background),
                                                  camera_position=camera_position,
                                                  image_shape=(800, 800))
    xray_image_simulated_lowres = cv2.cvtColor(
        xray_image_simulated_lowres, cv2.COLOR_BGR2GRAY)
    xray_image_simulated_lowres = 255-xray_image_simulated_lowres
    
    if device==0:
        h,w=xray_image_simulated_lowres.shape
        h,w=h//2,w//2
        
        xray_image_simulated_lowres=xray_image_simulated_lowres[w-int(w//2):w+int(w//2),h-int(h//2):h+int(h//2)]  
        xray_image_simulated_lowres=cv2.resize(xray_image_simulated_lowres,(524,550))
       
    score = -1*image_score(xray_image_simulated_lowres, Background_255,real_img,workbook)
    # if not watch:
    #     if score < -50:
    #         score = 0
    
    if debug:
        cv2.imwrite('./'+str(score)+'_'+str(image_validity(xray_image_simulated_lowres,
                                                       Background_255, real_img,debug=False,device=device))+'.jpg', xray_image_simulated_lowres)



    # return(score*(image_validity(xray_image_simulated_lowres, Background_255, real_img,debug=False)))
    return(score)


def sim_xray_set_from_CT_folder(dir_CT, device,ref_img,num_images_per_unit=3000,
                                debug=False,
                                perturb_scale=1.0,opacity_points=None,scalar_color_mapping_points=None,
                                device_folder="./Test2/",
                                pre_string="pre_string"):
    print("loading CT data from {}\n".format(dir_CT))
    


    imageData = CT_data_from_CT_folder(dir_CT)

    high_res_shape = (1600, 1600)
    count = 0

  
    while True:

        for k in range(num_images_per_unit):

            opacity_points = perturb_params_opacity(
                opacity_points, perturb_scale)
            scalar_color_mapping_points = perturb_params_color_map(
                scalar_color_mapping_points, perturb_scale)

            Background = np.random.rand()*0.5
            Background_255 = int(Background*255)
            if device==2:
                camera_position = np.array([0, 0, -500])
            else:
                camera_position=np.array([0, 0, -300])

            xray_image_simulated_lowres = render_CT_image(imageData,
                                                          opacity_points=opacity_points,
                                                          scalar_color_mapping_points=scalar_color_mapping_points,
                                                          Background=(
                                                              Background, Background, Background),
                                                          camera_position=camera_position,
                                                          image_shape=(800, 800))

            xray_image_simulated_lowres = cv2.cvtColor(
                xray_image_simulated_lowres, cv2.COLOR_BGR2GRAY)

            is_good_image = image_validity(
                xray_image_simulated_lowres, Background_255,ref_img, debug,device=device)
            
            if is_good_image:
                print("Might save images")
                scalar_color_mapping_points_good = scalar_color_mapping_points
                opacity_points_good = opacity_points
                Background += (np.random.rand()-0.5)*0.1
                Background_255 = int(Background*255)
                for i in range(5):

                    camera_position = randomize_camera_pose(
                        camera_position, high_res_shape)

                    opacity_points = perturb_params_opacity(
                        opacity_points_good, perturb_scale/4)
                    scalar_color_mapping_points = perturb_params_color_map(
                        scalar_color_mapping_points_good, perturb_scale/4)

                    tic = time.time()
                    xray_image_simulated_highres = render_CT_image(imageData,
                                                                   opacity_points=opacity_points,
                                                                   scalar_color_mapping_points=scalar_color_mapping_points,
                                                                   Background=(
                                                                       Background, Background, Background),
                                                                   camera_position=camera_position,
                                                                   image_shape=(high_res_shape[0], high_res_shape[1]))
                    del_t = time.time()-tic

                    xray_image_simulated_highres = cv2.cvtColor(
                        xray_image_simulated_highres, cv2.COLOR_BGR2GRAY)

                    is_good_image_highres = image_validity(
                        xray_image_simulated_highres, Background_255, ref_img,debug=debug,device=device)

                    if is_good_image_highres:
                        # file_name='../iPhone 6 Plus/Unit_16/X_ray_1_exp3_superglue/'+'Unit_16_0000_'+"{0:06d}".format(count)+'.jpg'
                        # file_name = device_folder +ref_img.split('_')[0]+'_0000_'+"{0:06d}".format(count)+'.jpg'
                        file_name = device_folder +'Unit116'+'_0000_'+"{0:06d}".format(count)+'.jpg'
                        print(file_name)

                        xray_image_simulated_highres = np.uint8(255-xray_image_simulated_highres)
                        if(np.random.rand() > 0.5):
                            xray_image_simulated_highres = cv2.rotate(
                                xray_image_simulated_highres, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        cv2.imwrite(file_name, xray_image_simulated_highres)
                        count += 1
                        print("{} / {} images done".format(count,
                                                           num_images_per_unit))

                        

                        if count >= num_images_per_unit:
                            print("broke loop 1")
                            break

            if count >= num_images_per_unit:
                print("broke loop 2")
                break

        if count >= num_images_per_unit:
            print("broke loop 3")
            break


def main():

    
    dir_CT = '../CT Data/Apple_watches/Watch Series 2 GPS/Unit 102/Unit102_23m_1_DICOM_/'

    sim_xray_set_from_CT_folder(
        dir_CT, debug=False, perturb_scale=1, device_folder=dir_CT,ref_img='./Unit52_top_avg10_skip1.jpg')    


if __name__ == '__main__':
    main()
