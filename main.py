import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
import pickle
from ME.HASM import MeshMotionEstimation, MeshMotionCompensation
from CCS.ColorSpaceConversion import Convert_RGB_YUV, Convert_YUV_RGB, COLOR_Compression
from DCT.DCT import DCT_Encode

if __name__ == "__main__":

    """ Read Image """
    data_path = 'data'#TODO: Argument
    seq_list = glob.glob(data_path+'/*.bmp')#TODO: Argument
    #TODO: PSNR Threshold
    PSNR_Threshold = 10
    """ Encoder """
    Optimize_Motion = False

    for img_name in seq_list:

        RGB_img = Image.open(img_name)

        YUV = Convert_RGB_YUV(RGB_img)

        Compressed_YUV = COLOR_Compression(YUV,'4:4:4')#TODO: Argument

        img_name = img_name.split('.bmp')[0]+'.pickle'

        if Optimize_Motion == True:

            PSNR, mv = MeshMotionEstimation(init_frame, Compressed_YUV)

            if PSNR >= PSNR_Threshold:
                init_frame = Compressed_YUV
                Quantized_img = DCT_Encode(Compressed_YUV)#TODO: Refactor and optimize
                pickle.dump(Quantized_img, open(img_name,"wb"))
                print("DCT path")
            else:
                pickle.dump(mv, open(img_name,"wb"))
                print("Motion Estimation path")
        else:
            Optimize_Motion = True
            init_frame = Compressed_YUV
            Quantized_img = DCT_Encode(Compressed_YUV)#TODO: Refactor and optimze
            #TODO: Integrate VLC
            print("DCT path")
            pickle.dump(Quantized_img, open(img_name,"wb"))

    """ Decoder """
    seq_list = glob.glob(data_path+'/*.pickle')
    
    #for img_name in seq_list:
