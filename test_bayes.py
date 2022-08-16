import hyperopt as hp
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from time import time
from super_matching import SuperMatching
import dicom_to_image_test_with_superglue as dc
import matplotlib.pyplot as plt
import cv2
import numpy as np
import argparse
import generate_polygon_images as gp
import csv

npy_file=open('./sample.txt', 'w')
def param_space(device):
    if device==0: #Apple Watch
        pass

    if device==1: #Iphone

        params_space={
            'op1':hp.uniform('op1',10000,30000),
            'op2':hp.uniform('op2',30000,42000),
            'col1':hp.uniform('col1',6000,12000),
            'col2':hp.uniform('col2',12000,19000),
            'col3':hp.uniform('col3',19010,26000),
            'col4':hp.uniform('col4',26001,38000),
            }

    if device==2: #Ipad
        pass
    return(params_space)
    
def parameter_opt(name,CT_data,workbook,device=None):
    num_val=15
    def objective_function(params):
        score=dc.make_image_one(name,CT_data,workbook,**params,device=device)
        return {'loss':score, 'status':STATUS_OK}
                
    trials=Trials()
    params_space=param_space(device)
    best_param=fmin(objective_function,params_space,algo=tpe.suggest,
    max_evals=num_val,trials=trials,rstate=np.random.RandomState(1))
    
    print(best_param)
    return(best_param)

def main(name,CT_data,num_image_to_create,synth_image_path,ref_image,label_ref,device=None,labels=True):
    
    
    if False: #Use bayesian optimisation
        header=["gamma", "keypoint", "superglue","file name"]
        with open('mycsv.csv','w',newline='') as f:
            workbook=csv.writer(f)
            workbook.writerow(header)
            
            params=parameter_opt(name,CT_data,workbook,device)
            opacity_points = np.array([[1800.0,  0.0],
                                    [params['op1'], 0.0],
                                    [params['op2'], 1.0],
                                    [67335.0, 1.0]])
            scalar_color_mapping_points = np.array([[1800, 0.5],
                                                    [params['col1'], 0.5],
                                                    [params['col2'], 0.5],
                                                    [params['col3'], 0.5],
                                                    [params['col4'], 0.5],
                                                    [40000, 0.5]])
    
    opacity_points = np.array([[1800.0,  0.0],
                            [22901.24, 0.0],
                            [33703.71, 1.0],
                            [67335.0, 1.0]])
    scalar_color_mapping_points = np.array([[161.81, 0.5],
                                            [18724.29, 0.5],
                                            [26214.00, 0.5],
                                            [39032.94, 0.5],
                                            [46954.75, 0.5],
                                            [65535.00, 0.5]]) 

    dc.sim_xray_set_from_CT_folder(CT_data, device,opacity_points=opacity_points,scalar_color_mapping_points=scalar_color_mapping_points,ref_img=name,
                            num_images_per_unit=num_image_to_create,debug=False,
                            perturb_scale=0.0,
                            device_folder=synth_image_path,
                            pre_string="pre_string")
    
    if labels:
        gp.main(ref_image,label_ref,synth_image_path,name)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Generate synthetic images and then transfer label using superglue')
    
    parser.add_argument('-c','--CT_data',type=str,help='Address for CT data',
                        default='../CT Data/iPhones/iPhone 12 Pro Max/Unit 171/Unit 171_5min_DICOM/')
        
    parser.add_argument('-ir','--image_name_real',type=str,help='Image name with location for real x-ray image to generate synthetic x-rays',
                        default='./Unit126_0000_000019.jpg')
    
    parser.add_argument('-n','--num_images_to_create',default=500,help='Number of synthetic images to create')

    parser.add_argument('-p','--synth_image_path', type=str,
                        default='./Test/iphone/iPhone 12 Pro Max/',
                        help='Path to save synthetic images')
    parser.add_argument('-b','--label',type=bool,default=False, help='True if labels have to generated else False')
    parser.add_argument('-l','--ref_image',default='./seg_label_mask/JPEGImages/motorola2_resized.png',type=str,help='Path for reference image to generate labels')
    parser.add_argument('-r','--label_ref',default='./seg_label_mask/SegmentationClass/motorola2_resized.png',type=str,help='Path for reference label image')
    parser.add_argument('-d','--device',default=1,type=bool,help='Enter 0 for Watch, 1 for iPhone, 2 for iPad')
    args=parser.parse_args()
    
    main(args.image_name_real,args.CT_data,args.num_images_to_create,args.synth_image_path,args.ref_image,args.label_ref,labels=args.label,device=args.device)
