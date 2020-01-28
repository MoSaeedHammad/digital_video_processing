import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os
import pickle
from ME.HASM import MeshMotionEstimation, MeshMotionCompensation
from CCS.ColorSpaceConversion import Convert_RGB_YUV, Convert_YUV_RGB, COLOR_Compression
from Encoder import MESH_Encoder
from Decoder import MESH_Decoder

if __name__ == "__main__":

    """ Read Image """
    data_path = 'data'#TODO: Argument
    seq_list = glob.glob(data_path+'/*.bmp')#TODO: Argument
    
    
    MEncoder = MESH_Encoder()
    MDecoder = MESH_Decoder()

    """ Encoder """
    
    for img_name in seq_list:

        RGB_img = cv2.imread(img_name)

        MEncoder.Encode(RGB_img)
    
    """ Decoder """
    seq_list = glob.glob(data_path+'/Output/*.mesh_obj')

    for obj_pth in seq_list:

        MDecoder.Decode_Pickle(obj_pth)
