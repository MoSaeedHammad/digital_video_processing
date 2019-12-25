'''
Created on Dec 25, 2019

@author: msaeedha
'''
import time

import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image

def Convert_RGB_YUV(RGB_img):
    
    RGB_img = np.float32(np.array(RGB_img))

    RGB_YUV = np.float32(np.array([[0.299, 0.587, 0.114]
                                   ,[-0.14713, -0.28886, 0.436]
                                   ,[0.615, -0.51499, -0.10001]]))
    
    YUV = np.dot(RGB_img, RGB_YUV.T)

    return YUV

def Convert_YUV_RGB(YUV_img, ROUND_FLAG = False):  
    YUV_RGB = np.float32(np.array([[1.0, 0.0, 1.13983]
                    ,[1.0, -0.39465, -0.58060]
                    ,[1.0, 2.03211, 0.0]]))

    Reconstructed_RGB = np.dot(YUV_img, YUV_RGB.T)

    Reconstructed_RGB = np.clip(Reconstructed_RGB, 0, 255)

    if ROUND_FLAG == True:
        Reconstructed_RGB = np.round(Reconstructed_RGB) #Improves PSNR as it rounds floating point to nearest dec number

    return Reconstructed_RGB

def COLOR_Compression(YUV_img, mode = None):
    start_time = time.time()
    if mode == '4:2:2':
        for i in range(0,YUV_img.shape[0]):
            for j in range(0, YUV_img.shape[1]):
                if j%2 != 0:
                    YUV_img[i,j,1] = YUV_img[i,j-1,1]
                    YUV_img[i,j,2] = YUV_img[i,j-1,2]
    elif mode == '4:2:0':
        for i in range(0,YUV_img.shape[0]):
            for j in range(0, YUV_img.shape[1]):
                if i%2 == 0:
                    if j%2 != 0:
                        YUV_img[i,j,1] = YUV_img[i,j-1,1]
                        YUV_img[i,j,2] = YUV_img[i,j-1,2]
                else:
                    if j%2 != 0:
                        YUV_img[i,j,1] = YUV_img[i-1,j-1,1]
                        YUV_img[i,j,2] = YUV_img[i-1,j-1,2]
                    else:
                        YUV_img[i,j,1] = YUV_img[i-1,j,1]
                        YUV_img[i,j,2] = YUV_img[i-1,j,2]                    
    elif mode == '4:1:1':
        for i in range(0,YUV_img.shape[0]):
            for j in range(0, YUV_img.shape[1]):
                if j%4 != 0:
                    YUV_img[i,j,1] = YUV_img[i,j-1,1]
                    YUV_img[i,j,2] = YUV_img[i,j-1,2]
    elif mode == '4:4:0':
        for i in range(0,YUV_img.shape[0]):
            for j in range(0, YUV_img.shape[1]):
                if i%2 == 0:
                    break
                else:
                    YUV_img[i,j,1] = YUV_img[i-1,j,1]
                    YUV_img[i,j,1] = YUV_img[i-1,j,1]          
    elif mode == '4:2:0_Average':
        for i in range(0,YUV_img.shape[0]):
            for j in range(0, YUV_img.shape[1]):
                if i%2 == 0:
                    if j%2 == 0:
                        YUV_img[i,j,1] = np.mean(YUV_img[i:i+1,j:j+1,1])
                        YUV_img[i,j,2] = np.mean(YUV_img[i:i+1,j:j+1,2])
                    else:
                        YUV_img[i,j,1] = YUV_img[i,j-1,1]
                        YUV_img[i,j,2] = YUV_img[i,j-1,2]
                else:
                    YUV_img[i,j,1] = YUV_img[i-1,j,1]
                    YUV_img[i,j,2] = YUV_img[i-1,j,2]

    elif mode == '4:2:0_Median':
        for i in range(0,YUV_img.shape[0]):
            for j in range(0, YUV_img.shape[1]):
                if i%2 == 0:
                    if j%2 == 0:
                        YUV_img[i,j,1] = np.median(YUV_img[i:i+1,j:j+1,1])
                        YUV_img[i,j,2] = np.median(YUV_img[i:i+1,j:j+1,2])
                    else:
                        YUV_img[i,j,1] = YUV_img[i,j-1,1]
                        YUV_img[i,j,2] = YUV_img[i,j-1,2]
                else:
                    YUV_img[i,j,1] = YUV_img[i-1,j,1]
                    YUV_img[i,j,2] = YUV_img[i-1,j,2]
    elif mode == '4:2:2_Average':
        for i in range(0,YUV_img.shape[0]):
            for j in range(0, YUV_img.shape[1]):
                if j%2 == 0:
                    YUV_img[i,j,1] = np.mean(YUV_img[i,j:j+1,1])
                    YUV_img[i,j,2] = np.mean(YUV_img[i,j:j+1,2])
                else:
                    YUV_img[i,j,1] = YUV_img[i,j-1,1]
                    YUV_img[i,j,2] = YUV_img[i,j-1,2]
    elif mode == '4:1:1_Average':
        for i in range(0,YUV_img.shape[0]):
            for j in range(0, YUV_img.shape[1]):
                if j%4 == 0:
                    YUV_img[i,j,1] = np.mean( (YUV_img[i,j:j+3,1]) )
                    YUV_img[i,j,2] = np.mean( (YUV_img[i,j:j+3,2]) )
                else:
                    YUV_img[i,j,1] = YUV_img[i,j-1,1]
                    YUV_img[i,j,2] = YUV_img[i,j-1,2]
    elif mode == '4:1:1_Median':
        for i in range(0,YUV_img.shape[0]):
            for j in range(0, YUV_img.shape[1]):
                if j%4 == 0:
                    YUV_img[i,j,1] = np.median( (YUV_img[i,j:j+3,1]) )
                    YUV_img[i,j,2] = np.median( (YUV_img[i,j:j+3,2]) )
                else:
                    YUV_img[i,j,1] = YUV_img[i,j-1,1]
                    YUV_img[i,j,2] = YUV_img[i,j-1,2]
    print("{0:<15} Compression Execution Time : {1} seconds".format(mode,(time.time() - start_time)))
    return YUV

if __name__ == "__main__":

    mode_list = ['4:4:4','4:2:2', '4:2:0', '4:1:1', '4:4:0','4:2:0_Average', '4:2:0_Median', '4:2:2_Average', '4:1:1_Average', '4:1:1_Median']
    img_list = ['f0.bmp']
    fig, (ax1, ax2) = plt.subplots(2,2, figsize=(20,20))
    for image_name in img_list:
        RGB_img = Image.open("video_input/"+image_name)
        start_time=time.time()
        YUV = Convert_RGB_YUV(RGB_img)
        print("RGB->YUV444 Conversion Execution Time : {0} seconds".format((time.time() - start_time)))
        for Mode in mode_list:
            
            Compressed_YUV = COLOR_Compression(YUV,Mode)
            
            start_time=time.time()
            Reconstructed_RGB = Convert_YUV_RGB(Compressed_YUV, True)
            #print("Reconstructed_RGB Conversion Execution Time : {} seconds".format((time.time() - start_time)))


            MSE = np.mean( (RGB_img - Reconstructed_RGB)**2 )

            if MSE == 0:
                PSNR = np.Infinity
            else:
                PSNR = 20*math.log10(255.0/math.sqrt(MSE))

            diff_mat = RGB_img-Reconstructed_RGB

            plt.ion()
            fig.suptitle('Color Compression mode: '+str(Mode)+', PSNR: '+str(PSNR)+' ,MSE: '+str(MSE), fontsize = 16)
            ax1[0].imshow(np.uint8(RGB_img))
            ax1[0].title.set_text('Original RGB image')
            ax1[1].imshow(np.uint8(Compressed_YUV))
            ax1[1].title.set_text('YUV Image')
            ax2[0].imshow(np.uint8(Reconstructed_RGB))
            ax2[0].title.set_text('Reconstructed RGB')
            ax2[1].imshow(np.uint8(diff_mat))
            ax2[1].title.set_text('Difference')
#             plt.show()
#             plt.pause(0.05)
            nimage_name = image_name.split('.')[0]
            Mode = Mode.replace(':','_')
            plt.savefig("csc_output/"+str(Mode)+'_'+nimage_name+'.png', format='png')