import time
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image
from numba import jit, cuda, float32

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
    img_width = YUV_img.shape[0]
    img_hight = YUV_img.shape[1]
    
    if mode == '4:2:2':
        YUV_img[:,1::2,1:] = YUV_img[:,0::2,1:]

    elif mode == '4:2:0':
        YUV_img[1::2,1::2,1:] = YUV_img[1::2,0::2,1:] = YUV_img[0::2,1::2,1:] = YUV_img[0::2,0::2,1:]

    elif mode == '4:1:1':
        YUV_img[:,1::4,1:] = YUV_img[:,0::4,1:]

    elif mode == '4:4:0':
        YUV_img[1::2,:,1:] = YUV_img[0::2,:,1:]

    elif mode == '4:2:0_Average':
        for i in range(0,img_width,2):
            for j in range(0, img_hight,2):
                YUV_img[i,j,1:] = np.mean(YUV_img[i:i+1,j:j+1,1:])
          
        YUV_img[1::2,1::2,1:] = YUV_img[0::2,1::2,1:]
        YUV_img[:,1::2,1:] = YUV_img[:,0::2,1:]

    elif mode == '4:2:0_Median':
        for i in range(0,img_width,2):
            for j in range(0, img_hight,2):
                YUV_img[i,j,1:] = np.median(YUV_img[i:i+1,j:j+1,1:])

        YUV_img[1::2,1::2,1:] = YUV_img[0::2,1::2,1:]
        YUV_img[:,1::2,1:] = YUV_img[:,0::2,1:]
        
    elif mode == '4:2:2_Average':
        for i in range(0,img_width):
            for j in range(0,img_hight,2):
                    YUV_img[i,j,1:] = np.mean(YUV_img[i,j:j+1,1:])
        YUV_img[:,1::2,1:] = YUV_img[:,0::2,1:]
    
    elif mode == '4:1:1_Average':
        for i in range(0,img_width):
            for j in range(0, img_hight,4):
                YUV_img[i,j,1:] = np.mean(YUV_img[i,j:j+3,1:])
        YUV_img[:,1::2,1:] = YUV_img[:,0::2,1:]

    elif mode == '4:1:1_Median':
        for i in range(0,img_width):
            for j in range(0, img_hight,4):
                    YUV_img[i,j,1:] = np.median(YUV_img[i,j:j+3,1:])
        YUV_img[:,1::2,1:2] = YUV_img[:,0::2,1:2]
    print("{0:<15} Compression Execution Time : {1} seconds".format(mode,(time.time() - start_time)))
    return YUV_img