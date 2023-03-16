#! /usr/bin/env python3
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
# from IPython import embed

from models.matching import Matching
from models.utils import (frame2tensor, compute_epipolar_error,
                          estimate_pose, make_matching_plot_fast,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)

torch.set_grad_enabled(False)
import cv2


def pointInRect(point,rect):
    x1, y1, w, h = rect
    x2, y2 = x1+w, y1+h
    x, y = point
    if (x1 < x and x < x2):
        if (y1 < y and y < y2):
            return True
    return False


def save_kps_image(im,points, save_name):
    im = im.copy()

    for i,point in enumerate(points):
        im = cv2.circle(im, (int(point[0]), int(point[1])), 3, (255,255,255), 3)
        im = cv2.circle(im, (int(point[0]), int(point[1])), 2, (0,0,0), 2)

    cv2.imwrite(save_name,im)

def points_in_rect(im,points):
    im = im.copy()

    for i,point in enumerate(points):
        im = cv2.circle(im, (int(point[0]), int(point[1])), 2, (0,0,0), 1)
        im = cv2.circle(im, (int(point[0]), int(point[1])), 1, (255,255,255), 1)

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    rect = cv2.selectROI("Image", im)
    print("in points_in_rect")
    cv2.destroyAllWindows()

    inside_pts =[] 

    for i,point in enumerate(points):
        if pointInRect(point,rect):
            inside_pts.append(i)

    inside_pts = np.array(inside_pts)

    # plt.scatter(a[:,0],a[:,1])
    # plt.scatter(a[inside_pts,0],a[inside_pts,1]);plt.show()

    return inside_pts


class SuperMatching():

    def __init__(self, force_cpu=False):
        self.max_length = -1    # Maximum number of pairs to evaluate
        self.max_keypoints = 1024 # Maximum number of keypoints detected by Superpoint (-1=keeps all keypoints)
        self.keypoint_threshold = 0.005
        self.nms_radius = 4     # SuperPoint Non Maximum Suppression (NMS) radius (Must be positive)'
        self.weights = 'custom'
        self.weights_path = ''
        self.sinkhorn_iterations = 20 # Number of Sinkhorn iterations performed by SuperGlue
        self.match_threshold = 0.4
        self.force_cpu = force_cpu

        # self.max_length = -1    # Maximum number of pairs to evaluate
        # self.max_keypoints = 1024 # Maximum number of keypoints detected by Superpoint (-1=keeps all keypoints)
        # self.keypoint_threshold = 0.005
        # self.nms_radius = 4     # SuperPoint Non Maximum Suppression (NMS) radius (Must be positive)'
        # self.weights = 'custom'
        # self.weights_path = ''
        # self.sinkhorn_iterations = 20 # Number of Sinkhorn iterations performed by SuperGlue
        # self.match_threshold = 0.6
        # self.force_cpu = force_cpu

    def set_weights(self, weights=None, weights_path =None):

        if weights is not None: 
            self.weights = weights
        if weights_path is not None: 
            self.weights_path = weights_path

        self.config = {
            'superpoint': {
            'nms_radius': self.nms_radius,
            'keypoint_threshold': self.keypoint_threshold,
            'max_keypoints': self.max_keypoints
            },
            'superglue': {
            'weights': self.weights,
            'sinkhorn_iterations': self.sinkhorn_iterations,
            'match_threshold': self.match_threshold,
            'weights_path':self.weights_path
            }
        }
        # self.device = 'cuda' if torch.cuda.is_available() and not self.force_cpu else 'cpu'
        self.device = 'cpu'

        self.matching = Matching(self.config).eval().to(self.device)


    def detectAndMatch(self, img1, img2):
        # Convert the image pair.
        inp1 = frame2tensor(img1, self.device)
        inp2 = frame2tensor(img2, self.device)
                
        # Perform the matching.
        pred = self.matching({'image0': inp1, 'image1': inp2})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']
        
        # Write the matches to disk.
        out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                       'matches': matches, 'match_confidence': conf}
        
        # np.savez(str(matches_path), **out_matches)
        
        # Keep the matching keypoints.
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]    
        
        return mconf,kpts0, kpts1, mkpts0, mkpts1
    
    
    def plot_matches(self, image0, image1, kpts0, kpts1, mkpts0,mkpts1, save_path='results/superglue_matching.jpg'):
        # idx = np.random.randint(low=0, high=len(mkpts0), size=(50,))
        # embed()
        idx_sort_h = np.argsort(mkpts0[:,0])
        mkpts0 = mkpts0[idx_sort_h,:]
        mkpts1 = mkpts1[idx_sort_h,:]
        print("match after sort",mkpts0)
        # save_kps_image(image0, mkpts0, './results/ref_kps.png')
        # save_kps_image(image1, mkpts1, './results/align_kps.png')

        idx = points_in_rect(image0,mkpts0)
        # idx = np.arange(50,60)
        mkpts0 = mkpts0[idx,:]
        mkpts1 = mkpts1[idx,:]

        #random colors
        # color = np.random.randint(low=50, high=255, size=(len(mkpts0), 3))
        
        #Uniform red
        # color = np.random.randint(low=50, high=255, size=(len(mkpts0), 3)) * 0 
        # color[:,1] += 255 

        #Uniform gradient
        color_chan = np.linspace(0, 255, len(mkpts0))/255
        color = cm.jet(color_chan)[:,0:3]*255

        title = 'Test'
        # path ='results/superglue_matching.jpg'
        path = save_path
        make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0,
                            mkpts1, color, '', path, True, 0, True, title)
        # make_matching_plot(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
        #                color, text, path, show_keypoints=False,
        #                fast_viz=False, opencv_display=False,
        #                opencv_title='matches', small_text=[])