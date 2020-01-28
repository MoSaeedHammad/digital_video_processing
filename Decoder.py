import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
import pickle
import cv2
from ME.HASM import MeshMotionEstimation, MeshMotionCompensation
from CCS.ColorSpaceConversion import Convert_RGB_YUV, Convert_YUV_RGB, COLOR_Compression
from DCT.DCT import DCTQ_Decode

class MESH_Decoder():
    def __init__(self, PSNR_Threshold = 40, compression_mode='4:4:4', Dump_objs = True, block_size = 256):
        self.Optimize_Motion = False
        self.PSNR_Threshold = PSNR_Threshold
        self.compression_mode = compression_mode
        self.seq_counter = 0
        self.Dump_objs = Dump_objs
        self.ME_counter = 0
        self.block_size = block_size
    
    def new_sequence(self):
        self.Optimize_Motion = False
        self.seq_counter = 0
        self.f0dash_frame = np.empty((1,1))
        self.ME_counter = 0
    
    def Decode_obj(self, obj, path):
        #TODO: add VLC

        if path == 'mv':
            
            f1_frame = MeshMotionCompensation(obj,obj[1] ,obj[2], self.block_size)

            f1_frame = Convert_YUV_RGB(f1_frame)

            if self.Dump_objs == True:
                pickle.dump(f1_frame,open('data/im_'+str(pickle_file.split('_')[1])+'.mesh_obj'))
                
            return f1_frame
        
        elif path == 'DCTQ':
            
            self.f0dash_frame = DCT_Decode(obj)

            #TODO: inverse color compression

            self.f0dash_frame = Convert_YUV_RGB(self.f0dash_frame)

            if self.Dump_objs == True:
                pickle.dump(self.f0dash_frame,open('data/im_'+str(pickle_file.split('_')[1])+'.mesh_obj'))

            return self.f0dash_frame

        else:
            return None

    def Decode_Pickle(self, pickle_file):

        obj = pickle.load(open(pickle_file,"rb"))

        #TODO: add VLC

        path = pickle_file.split('_')[2].split('.mesh_obj')[0].split('.mesh')[0]

        if path == 'ME':
            
            t1 = time.time()
            f1_frame = MeshMotionCompensation(obj,obj[1] ,obj[2], self.block_size)
            time_f = (time.time()-t1)
            print("Mesh Compensation Time for CPU Not accelerated {} s".format(time_f))#TODO: Logger report with time

            f1_frame = Convert_YUV_RGB(f1_frame)

            if self.Dump_objs == True:
                pickle.dump(f1_frame,open('data/Output/im_'+str(pickle_file.split('_')[1])+'.mesh_obj',"wb"))
                cv2.imwrite('data/Output/im_'+str(pickle_file.split('_')[1])+'.jpg',np.uint8(self.f0dash_frame))

            return f1_frame
        
        elif path == 'DCTQ':
            
            t1 = time.time()
            
            self.f0dash_frame = DCTQ_Decode(obj)
            time_f = (time.time()-t1)
            print("Inverse-Quantization Time for CPU Accelerated {} s".format(time_f))#TODO: Logger report with time

            #TODO: inverse color compression

            self.f0dash_frame = Convert_YUV_RGB(self.f0dash_frame)

            if self.Dump_objs == True:
                
                pickle.dump(self.f0dash_frame,open('data/Output/im_'+str(pickle_file.split('_')[1])+'.mesh_obj',"wb"))
                cv2.imwrite('data/Output/im_'+str(pickle_file.split('_')[1])+'.jpg',np.uint8(self.f0dash_frame))

            return self.f0dash_frame

        else:
            return None
