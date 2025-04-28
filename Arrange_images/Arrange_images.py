import cv2
import argparse
import register_with_superglue as rg
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Align all the images to a referenec image',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        

    parser.add_argument('-ref', '--reference_path',
                        type=str, default='img1.png',
                        help='Reference Image')
    
    parser.add_argument('-align_images', '--align_image_path', default='./iPhone13',
                        help='Path to images to be aligned')

    args = parser.parse_args()

    reference_image=cv2.imread(args.reference_path,0)
    cv2.namedWindow("Image",cv2.WINDOW_NORMAL)
    rect = cv2.selectROI("Image", reference_image)
    
    image_align_list=os.listdir(args.align_image_path)

    reference_image_cropped=reference_image[rect[1]:rect[3]+rect[1],rect[0]:rect[0]+rect[2]]
    cv2.imshow('Cropped',reference_image_cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    for image in image_align_list:

        align_img_path=args.align_image_path+'/'+image
        align_img=cv2.imread(align_img_path,0)
        rg.test_two(align_img,reference_image_cropped,rect,align_img_path)

