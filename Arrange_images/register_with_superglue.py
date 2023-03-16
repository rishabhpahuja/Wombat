import argparse
import numpy as np
import numpy.random as random
import cv2,cv
from super_matching import SuperMatching
import matplotlib.pyplot as plt
import imutils
# from IPython import embed

def transf_matrix(theta=0, translation=[0,0]):
    assert len(translation) == 2
    tx, ty  = translation

    # First two columns correspond to the rotation b/t images
    M = np.zeros((2,3))
    M[:,0:2] = np.array([[np.cos(theta), np.sin(theta)],\
                         [ -np.sin(theta), np.cos(theta)]])

    # Last column corresponds to the translation b/t images
    M[0,2] = tx
    M[1,2] = ty
    
    return M

""" Convert the 2x3 rot/trans matrices to a 3x3 matrix """
def transf_mat_3x3(M):
    M_out = np.eye(3)
    M_out[0:2,0:3] = M
    return M_out

def transf_pntcld(M_est, pt_cld):
    '''
    M_est 2x3
    pt_cld nx2
    '''
    R = M_est[:,0:2]
    t = M_est[:,-1].reshape(2,-1)
    pt_cld_transf = (R@pt_cld.T + t).T 

    return pt_cld_transf


def setup_sg_class(superglue_weights_path):
    sg_matching = SuperMatching()
    sg_matching.weights = 'custom'
    sg_matching.weights_path = superglue_weights_path
    sg_matching.set_weights()
    
    return sg_matching
 
def SuperGlueDetection(img1, img2, sg_matching, debug=False):
 
    mconf, kp1, kp2, matches1, matches2 = sg_matching.detectAndMatch(img1, img2)
    
    
    rgb1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    rgb2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    # Show keypoints
    for x,y in kp1.astype(np.int64):
        cv2.circle(rgb1, (x,y), 2, (255,0,0), -1)
    for x,y in kp2.astype(np.int64):
        cv2.circle(rgb2, (x,y), 2, (0,0,255), -1)
    # displayImages(rgb1, 'Reference Image Keypoints',
    #               rgb2, 'Aligned Image Keypoints', 'results/superpoint_detection.jpg')
    
    # Show matches
    if debug:
        sg_matching.plot_matches(img1, img2, kp1, kp2, matches1, matches2)

    return mconf,kp1, kp2, matches1, matches2


"""
Overlay the transformed noisy image with the original and estimate the affine
transformation between the two
# """
# def generateComposite(ref_keypoints, align_keypoints, ref_cloud, align_cloud,
#                       matches, rows, cols):
def get_warp_results(ref,align,M_est, debug=False):
    # Converts vector of keypoints to vector of points or the reverse, where each keypoint is assigned the same size and the same orientation.
    # ref_keypoints_np = cv2.KeyPoint_convert(ref_keypoints)
    # align_keypoints_np = cv2.KeyPoint_convert(align_keypoints)

    # reordered_ref = np.zeros((len(matches), 2))
    # reordered_align = np.zeros((len(matches), 2))
    
    rows, cols = ref.shape

    # debug = False
    # for i, m in enumerate(matches):
    #     # I had to adjust the indices for m here too
    #     reordered_ref[i,:] = ref_keypoints_np[m.queryIdx,:]
    #     reordered_align[i,:] = align_keypoints_np[m.trainIdx,:]

    # M_est = cv2.estimateAffinePartial2D(reordered_ref, reordered_align)[0]

    # # M to go from reference image to the aligned image; should be = to OG M up above
    M_est_inv = np.linalg.inv(transf_mat_3x3(M_est))[0:2,:]
    # from IPython import embed; embed()
    align_warped = cv2.warpAffine(align, M_est_inv, (cols, rows))

    alpha_img = np.copy(ref)
    alpha = 0.5
    composed_img = cv2.addWeighted(alpha_img, alpha, align_warped, 1-alpha, 0.0)
    if debug:
        displayImages(composed_img, 'Composite Image')
    return ref, align_warped, composed_img
    # return 

"""
Compute the translation/rotation pixel error between the estimated RANSAC
transformation and the true transformation done on the image.
"""
def computeError(M, M_est, M_est_inv):
    print('\nEstimated M\n', M_est)
    print('\nTrue M\n', M)

    # Add error
    error = M @ transf_mat_3x3(M_est_inv)
    R_del = error[0:2,0:2]
    t_del = error[0:2,2]

    print('\nTranslation Pixel Error: ', np.linalg.norm(t_del))
    print('Rotation Pixel Error: ', np.linalg.norm(R_del))
    
"""
Display a single image or display two images conatenated together for comparison
Specifying a path will save whichever image is displayed (the single or the
composite).
"""
def displayImages(img1, name1='Image 1', img2=None, name2='Image2', path=None):
    if img2 is None:
        # ASSERT: Display only 1 image
        output = img1
        cv2.namedWindow(name1, cv2.WINDOW_NORMAL)
        cv2.imshow(name1, img1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        # Display both images concatenated
        output = np.concatenate((img1, img2), axis=1)
        cv2.namedWindow(name1 + ' and ' + name2, cv2.WINDOW_NORMAL)
        cv2.imshow(name1 + ' and ' + name2, output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if path is None:
        # Save the image at the current path
        print("")
    else:
        cv2.imwrite(path, output)

"""
Test feature detection, feature matching, and pose estimation on an image.
"""
def cv_kp_to_np(cv_keypoints):
    list_kp_np = []
    for idx in range(0, len(cv_keypoints)):
        list_kp_np.append(cv_keypoints[idx].pt)
    
    return np.array(list_kp_np).astype(np.int64)        
    # ref_cloud = np.float([cv_keypoints[idx].pt for idx in range(0, len(cv_keypoints))]).reshape(-1, 1, 2)


# def find_transformation_keypoints_image(ref, align, debug=False):
#     # try:
    
#     if debug:
#         displayImages(ref, 'Reference Image', align, 'Aligned Image')
#         # displayImages(ref, 'Reference Image', align, 'Aligned Image','results/TIF Image Transformation.jpg' )
        
        
#         # Apply Feature Detection and Feature Matching
#     ref_keypoints, align_keypoints, matches1, matches2 = SuperPointDetection(ref, align)

#     # ref_keypoints, ref_descriptors, align_keypoints, align_descriptors, \
#     # ref_cloud_overlay, align_cloud_overlay = TraditionalDetection(ref, align)

#     ref_kp_image = np.zeros((np.shape(ref)),  dtype=np.uint8)
#     align_kp_image = np.zeros((np.shape(align)),  dtype=np.uint8)
 
#     # ref_cloud = np.float([ref_keypoints[idx].pt for idx in range(0, len(ref_keypoints))]).reshape(-1, 1, 2)
#     # ref_cloud = np.float([ref_keypoints[idx].pt for idx in range(0, len(ref_keypoints))]).reshape(-1, 1, 2)
#     ref_cloud = cv_kp_to_np(ref_keypoints)
#     align_cloud = cv_kp_to_np(align_keypoints)

#     ref_kp_image[ref_cloud[:,0], ref_cloud[:,1]] = 255    
#     align_kp_image[align_cloud[:,0], align_cloud[:,1]] = 255    

#     # import ipdb; ipdb.set_trace()    

#     cv2.imshow("ref_kp_image",ref_kp_image)
#     cv2.imshow("align_kp_image",align_kp_image)
#     cv2.waitKey(0)
#     return np.zeros((3,3)), 0
#     # except:
#     #     return np.zeros((3,3)), 0



# def find_transformation_ORB(ref, align, debug=False):

#     try:
#         if debug:
#             displayImages(ref, 'Reference Image', align, 'Aligned Image')
#         # displayImages(ref, 'Reference Image', align, 'Aligned Image','results/TIF Image Transformation.jpg' )
        
#         # ref_keypoints, align_keypoints, matches1, matches2 = SuperPointDetection(ref, align)
        
#         # Apply Feature Detection and Feature Matching
#         ref_keypoints, ref_descriptors, align_keypoints, align_descriptors, \
#             ref_cloud, align_cloud = TraditionalDetection(ref, align)


#         # cv2.imshow("ref_cloud",ref_cloud)
#         # cv2.waitKey(0)

#         # print(ref_cloud,"ref_cloud")
#         # import ipdb; ipdb.set_trace()
#         matches = TraditionalMatching(ref_descriptors, align_descriptors, bf=False)

#         ref_keypoints_np = cv2.KeyPoint_convert(ref_keypoints)
#         align_keypoints_np = cv2.KeyPoint_convert(align_keypoints)

#         reordered_ref = np.zeros((len(matches), 2))
#         reordered_align = np.zeros((len(matches), 2))

#         for (i, m) in enumerate(matches):
#             # I had to adjust the indices for m here too
#             reordered_ref[i,:] = ref_keypoints_np[m.queryIdx,:]
#             reordered_align[i,:] = align_keypoints_np[m.trainIdx,:]

#         M_est = cv2.estimateAffinePartial2D(reordered_ref, reordered_align)[0]


#         # Draw matches
#         img3 = cv2.drawMatches(ref, ref_keypoints,
#                                align, align_keypoints,
#                                matches, outImg=None, flags=2)
#         if debug:
#             displayImages(img3, 'Draw Matches', path='results/Test Image Feature Matching.jpg')

#         print(len(matches),"num of matches")
#         return M_est, len(matches)

#     except:
#         return np.zeros((3,3)), 0


def find_transformation_SuperGlue(ref, align, sg_matching, debug=False):

    # if debug:
    #     displayImages(ref, 'Reference Image', align, 'Aligned Image')
    # displayImages(ref, 'Reference Image', align, 'Aligned Image','results/TIF Image Transformation.jpg' )
    
    # sg_matching = setup_sg_class(args)

    mconf, ref_keypoints, align_keypoints, matches1, matches2 = SuperGlueDetection(ref, align, sg_matching, debug)
    
    
    try :
        M_est = cv2.estimateAffinePartial2D(matches1, matches2)[0]
        # M_est, _ = cv2.findHomography(matches1, matches2)
        # embed()
        # matches1_transf = transf_pntcld(M_est,matches1)
        # avg_dist_corr_pnts = np.linalg.norm(matches1_transf-matches2)/len(matches1)        

        # plt.imshow(align, cmap='gray')
        # # plt.scatter(matches1_transf[:,0], matches1_transf[:,1], c='r', label='matches1_transf')
        # plt.scatter(matches2[:,0], matches2[:,1], c='b', label='matches2')
        # # plt.scatter(matches1[:,0], matches1[:,1], c='g', label='matches1')

        # plt.show()

  
    except:
        print("could not find matches")
        M_est = np.array([[1,0,0],
                          [0,1,0]])

    # Draw matches
    # if debug:
    #     img3 = cv2.drawMatches(ref, ref_keypoints,
    #                            align, align_keypoints,
    #                            matches1, outImg=None, flags=2)
    #     displayImages(img3, 'Draw Matches', path='results/Test Image Feature Matching.jpg')

    # print(len(matches1),"num of matches")
    return mconf, M_est, ref_keypoints, align_keypoints, matches1, matches2 

    # except:
    #     return np.zeros((3,3)), 0



# def test_single(image_path):
#     ref = cv2.imread(image_path, 0)
#     print(ref,"ref")
#     # Transformations are proportional to the scale of the image
#     # M1 = transf_matrix(random.uniform(low=-np.pi/4, high=np.pi/4),
#     #                    [random.randint(low=-ref.shape[0]/20, high=ref.shape[0]/20),
#     #                     random.randint(low=-ref.shape[1]/20, high=ref.shape[1]/20)])

#     # M = transf_mat_3x3(M1)
#     theta = -np.pi/3

#     M1 = transf_matrix(0, [-ref.shape[0]/2,-ref.shape[1]/2])
#     M2 = transf_matrix(theta , [0,0])
#     M3 = transf_matrix(0, [ref.shape[0]/2,ref.shape[1]/2])

#     M = transf_mat_3x3(M3)@transf_mat_3x3(M2)@transf_mat_3x3(M1)

#     # Perform affine transform and add noise to the original image
#     rows, cols = ref.shape
#     align = cv2.warpAffine(ref, M[0:2,:], (cols, rows))
#     # align += np.random.randint(20, size=align.shape, dtype=align.dtype)

#     # M_est,_ = find_transformation_ORB(ref, align, debug=True)
#     M_est, num_matches = find_transformation_SuperGlue(ref, align, debug=True)

#     get_warp_results(ref,align,M_est)

 

def put_at_center(fg_img, bg_img):
    h, w = fg_img.shape
    hh, ww = bg_img.shape

    yoff = round((hh-h)/2)
    xoff = round((ww-w)/2)

    result = bg_img.copy()
    result[yoff:yoff+h, xoff:xoff+w] = fg_img

    return result

def rescale_images(img1,img2):
    img1 = img1.copy()
    img2 = img2.copy()
    
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]    

    if h1/h2 < 1: # img1 is small
        scale = h1/h2
        img2 = cv2.resize(img2,None,fx=scale,fy=scale)

    else : # img2 is small
        scale = h2/h1
        img1 = cv2.resize(img1,None,fx=scale,fy=scale)

    return img1, img2


def resize_images(ref,align):
    h1,w1 = ref.shape[:2]
    h2,w2 = align.shape[:2]

    h = max([h1,w1,h2,w2])

    bg_img = np.zeros((h,h), dtype=ref.dtype)
    ref_padded = put_at_center(ref, bg_img)
    align_padded = put_at_center(align, bg_img)
    # ref_padded[0:h1,0:w1] = ref

    # align_padded = np.zeros((h,h), dtype=align.dtype)
    # align_padded[0:h2,0:w2] = align

    return ref_padded, align_padded

def crop_image(img,rect,debug=False,save=False,name=None):

    img=img[rect[1]:rect[3]+rect[1],rect[0]:rect[0]+rect[2]]

    if debug:

        cv2.namedWindow("Cropped Image",cv2.WINDOW_NORMAL)
        cv2.imshow("Cropped Image",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    if save:
        print('aaaaaaaaaaa',name)
        cv2.imwrite(name,img)
    
    return img

def test_two(align,ref,rect,name, debug=False):

    if debug:
        cv2.imshow("Align",align)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    wpath='./global_registration_sg.pth'

    align_image_cropped=crop_image(align,rect=rect)

    sg_matching = setup_sg_class(wpath)
    
    # Finds matching points and registers the two images 
    #! make debig=True to plot matches
    mconf,M_est, ref_keypoints, align_keypoints, matches1, matches2  = find_transformation_SuperGlue(align_image_cropped, ref, sg_matching, debug=False)
  
    #finding homography matrix
    H, _ = cv2.findHomography(matches1, matches2, method=cv2.RANSAC,)

    moving_rigid_transf = cv2.warpPerspective(align_image_cropped, H, (align_image_cropped.shape[1], align_image_cropped.shape[0])) #rotating the reference image
    # label_transfer=cv2.warpPerspective(label, H, (align.shape[1], align.shape[0]))  #rotating the reference labels

    #Naming windows so that resizing is easy
    print(name)
    cv2.imwrite(name,moving_rigid_transf)

    


# Main Code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Image pair pose estimation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # angle=[5,10,15]
    # for i in angle:
    #     img1=cv2.imread("2.png")
    #     img2=imutils.rotate_bound(img1,i)
        

    parser.add_argument('-ref', '--reference_path',
                        type=str, default='img_10.png',
                        help='Reference Image')
    parser.add_argument('-align', '--align_path',
                        type=str, default='img_11.png',
                        help='Image to align')
    # parser.add_argument('--superglue', choices={'indoor', 'outdoor', 'custom'}, 
    #                     default='custom',
    #                     help='SuperGlue weights')

    # parser.add_argument('-weights', '--superglue_weights_path', default='./models/weights/superglue_indoor.pth',
    #                     help='SuperGlue weights path')
    
    parser.add_argument('-weights', '--superglue_weights_path', default='./global_registration_sg.pth',
                        help='SuperGlue weights path')

    
    args = parser.parse_args()
    # test_single(args.reference_path)
    # test_two(args.reference_path, args.align_path)
    test_two(args)
    # test_two_iterative(args.reference_path, args.align_path)
    
    