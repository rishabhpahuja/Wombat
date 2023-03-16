import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

def downsize():
    ref= cv2.imread('../Real Data/iPads/iPad_Pro_9.7.png',0)
    ref_=cv2.resize(ref, (1245,876))
    # ref_=cv2.flip(ref,1)
    # ref_=cv2.rotate(ref_,cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite('Unit174_top_av_40.jpg',ref_)

def crop():
    names=os.listdir('./Test/Test2/Watches/Apple_watch_5_GPS_LTE')
    # names=os.listdir('./a')
    for i in tqdm(names):
        path='./Test/Test2/Watches/Apple_watch_5_GPS_LTE/'+i
        img=cv2.imread(path,0)
        (w,l)=img.shape
        w,l=w//2,l//2
        img=img[w-w//2:w+w//2,l-l//2:l+l//2]
        cv2.imwrite(path,img)
def main():
    downsize()

if __name__=='__main__':
    main()
# a,b=ref.shape()
# a,b=a//2,b//2
# ref_=ref[]


# _,ref_thresh=cv2.threshold(ref,190,255,cv2.THRESH_BINARY)
# cont_ref, cont_hier = cv2.findContours(ref_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# ref_ = cv2.drawContours(ref, cont_ref, -1, (0,255,0),5)
# cv2.imshow("a",ref_)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Background_255=0
# a=cv2.imread("test.jpg",0)
# background_bool = np.isclose(
#         a, Background_255*np.ones(a.shape), atol=1.0, rtol=0.0)
# foreground_mask=np.where(background_bool==False)
# # a[a!=0] = a[a!=0]
# foreground=a[foreground_mask]
# # print(background_bool)
# cv2.imwrite("a.jpg",foreground)
# plt.show()

# print(ref.size())

