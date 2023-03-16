import cv2
import glob
import numpy as np
import os
from tqdm import tqdm 
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from IPython     import embed
import argparse

# def adjust_gamma(image, gamma=1.0):
#     # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
#     invGamma = 1.0 / gamma
#     table = np.array([((i / 255.0) ** invGamma) * 255
#         for i in np.arange(0, 256)]).astype("uint8")
#     # Apply gamma correction using the lookup table
#     return cv2.LUT(image, table)

# def random_smooth_func(image, debug = False  ):
#     x = np.linspace(0, 255, num=5, endpoint=True)
#     # y = np.random.uniform(low = 0, high=255, size=(len(x),))
#     # y = np.random.uniform(low = image.min(), high=image.max(), size=(len(x),))
#     y = np.random.uniform(low = 50, high=220, size=(len(x),))

#     f2 = interp1d(x, y, kind='cubic')

#     table = f2(np.arange(0, 256)).astype("uint8")
#     # embed()   
#     if debug:
#         plt.figure()
#         plt.plot(table, label='smooth func')
#         plt.scatter(x,y, label='kps')
#         plt.legend()
#         # plt.show()

#     # table = np.array([f2(i) for i in np.arange(0, 256)]).astype("uint8")
#     # table = np.array([f2(i) for i in np.arange(0, 256)]).astype("uint8")
#     # Apply gamma correction using the lookup table
#     return cv2.LUT(image, table)

def add_noise(img, sigma=1):
    img_out = img.copy()
    noise_image = (np.random.rand(*img_out.shape) - 0.5)*sigma
    img_out = img_out + noise_image
    img_out = img_out.clip(0,255)
    return img_out.astype(np.uint8)

def get_augmeted_dir_name(dir_clean, suffix='noisy'):
    path_list = dir_clean.split(os.sep)
    path_list[-1] = path_list[-1] + '_' + suffix 

    return os.path.join(*path_list)

n = 3
gamma_low = 0.5
gamma_high = 1.5

# def augment_images(folder, save_loc = save_loc, debug=False):
#     for file in tqdm(sorted(glob.glob(folder))):
#         img = cv2.imread(file)
        
#         filename, file_extension = os.path.splitext(file)
        
#         if debug:
#             print(filename,"filename")
#             print(file_extension,"file_extension")
#             # cv2.imshow('img_orig', img)
#             # cv2.waitKey(0)
#             plt.imshow(img)
#             plt.show()        
#         for i in range(1):

#             # if np.random.rand() >0.5:
#             #     gam = np.random.uniform(gamma_low, gamma_high)
#             #     img_out = adjust_gamma(img, gam)
#             # else:
#             #     img_out = random_smooth_func(img, debug)

#             img_out = add_noise(img,sigma=2)
#             img_out_name = filename + '_{0:03d}'.format(i) + file_extension

#             if debug:
#                 plt.figure()
#                 plt.title('original')
#                 plt.imshow(img)
#                 # plt.show()        
#                 plt.figure()
#                 plt.title('augmented')
#                 plt.imshow(img_out)
#                 plt.show()
#                 # cv2.imshow(img_out_name, img_out)
#                 # cv2.waitKey(0)
#                 # cv2.destroyAllWindows()
#             else:
#                 cv2.imwrite(img_out_name, img_out)



def save_noisy_images(folder, save_loc = None, debug=False,sigma=100):
    for file in tqdm(sorted(glob.glob(folder))):
        img = cv2.imread(file)
        
        path_list = file.split(os.sep)
        img_name = path_list[-1]

        img_out = add_noise(img,sigma=sigma)
        img_out_name = os.path.join(save_loc, img_name)
        
        if debug:
            print(img_out_name,"img_out_name")
            plt.figure()
            plt.title('original')
            plt.imshow(img)
            # plt.show()        
            plt.figure()
            plt.title('augmented')
            plt.imshow(img_out)
            plt.show()
            # cv2.imshow(img_out_name, img_out)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        else:
            cv2.imwrite(img_out_name, img_out)

def generate_noisy_image_dataset(args):
    folder = args.folder
    dirs = [x[0] for x in os.walk(folder)] # get all root directories

    for dir_classes in dirs:
        noisy_dir = get_augmeted_dir_name(dir_classes, suffix='noisy')
        if not os.path.exists(noisy_dir):
            os.mkdir(noisy_dir)
        dir_classes = os.path.join(dir_classes, '*'+ args.ext) 
        save_noisy_images(dir_classes, save_loc = noisy_dir, debug=args.debug, sigma=args.sigma)


def save_same_bckgrnd_images(folder, save_loc = None, debug=False):
    for file in tqdm(sorted(glob.glob(folder))):
        img = cv2.imread(file)
        
        path_list = file.split(os.sep)
        img_name = path_list[-1]

        img_out = detect_backg(img,sigma=100)
        img_out_name = os.path.join(save_loc, img_name)
        
        if debug:
            print(img_out_name,"img_out_name")
            plt.figure()
            plt.title('original')
            plt.imshow(img)
            # plt.show()        
            plt.figure()
            plt.title('augmented')
            plt.imshow(img_out)
            plt.show()
            # cv2.imshow(img_out_name, img_out)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        else:
            cv2.imwrite(img_out_name, img_out)

def generate_same_background_dataset(args):
    folder = args.folder
    dirs = [x[0] for x in os.walk(folder)] # get all root directories

    for dir_classes in dirs:
        same_bckgrnd_dir = get_augmeted_dir_name(dir_classes, suffix='same_bckgrnd')
        if not os.path.exists(same_bckgrnd_dir):
            os.mkdir(same_bckgrnd_dir)
        dir_classes = os.path.join(dir_classes, '*'+ args.ext) 
        save_same_backg_images(dir_classes, save_loc = same_bckgrnd_dir, debug=args.debug)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='data augmentation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

 
    parser.add_argument('-folder', '--folder',
                        type=str, default='./sample_data',
                        help='folder containing set of images or folders of images')
    parser.add_argument('-ext', '--ext',
                        type=str, default='.jpg',
                        help='File extension')
    parser.add_argument('-debug', '--debug',
                        type=int, default=1,
                        help='debug')

    parser.add_argument('-sigma', '--sigma',
                        type=int, default=0,
                        help='sigma for noisy images')
    
    args = parser.parse_args()


    generate_noisy_image_dataset(args)

    # folder = './data_for_classification'
    # folder = './temp'
    # folder = args.folder
    # dirs = [x[0] for x in os.walk(folder)] # get all root directories

    # print(dirs,"dirs")
    # for dir_classes in dirs:
    #     print(dir_classes,"dir_classes")
    #     # dir_classes += '/*'+ args.ext 
    #     noisy_dir = get_noisy_dir_name(dir_classes)
    #     dir_classes = os.path.join(dir_classes, '*'+ args.ext) 
    #     print(noisy_dir,"noisy_dir")

    #     # augment_images(dir_classes, debug=False)
    #     # augment_images(dir_classes, debug=args.debug)
    #     # print(dir_classes,"dir_classes")