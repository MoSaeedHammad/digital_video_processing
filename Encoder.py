import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
import pickle
from ME.HASM import MeshMotionEstimation, MeshMotionCompensation
from CCS.ColorSpaceConversion import Convert_RGB_YUV, Convert_YUV_RGB, COLOR_Compression
from DCT.DCT import DCTQ_Encode
from VLC.vlc import encode,init_encoder
import VLC.configuration as cfg

class MESH_Encoder():
    def __init__(self, PSNR_Threshold = 40, compression_mode='4:4:4', Dump_objs = True, ME_FramesThreshold = 7):
        self.Optimize_Motion = False
        self.PSNR_Threshold = PSNR_Threshold
        self.compression_mode = compression_mode
        self.seq_counter = 0
        self.ME_FramesThreshold = ME_FramesThreshold
        self.Dump_objs = Dump_objs
        self.ME_counter = 0
    
    def new_sequence(self):
        self.Optimize_Motion = False
        self.seq_counter = 0
        self.init_frame = np.empty((1,1))
        self.ME_counter = 0

    def Encode(self, RGB_img):

        YUV = Convert_RGB_YUV(RGB_img)

        Compressed_YUV = COLOR_Compression(YUV,'4:4:4')#TODO: Argument
        init_encoder("TestingOutput/", frame_resolution=[1280, 720], yuv_config=[4, 4, 4], encoder_mode=0)

        if self.Optimize_Motion == True:

            t1 = time.time()
            PSNR, mv = MeshMotionEstimation(self.init_frame, Compressed_YUV)
            if (mv[0] == -1):
                encode(cfg.MOTION_VECTORS_FRAME, mv)
            else:
                encode(cfg.MESH_FRAME, mv)
                
            #DCT_ip = [Quantized_img[:,:,0],Quantized_img[:,:,1],Quantized_img[:,:,2]]
            encode(cfg.DCT_FRAME, DCT_ip)
            time_f = (time.time()-t1)
            print("Mesh estimation Time for CPU Not accelerated {} s".format(time_f))#TODO: Logger report with time
            print("MESH PSNR: {}".format(PSNR))

            if PSNR >= self.PSNR_Threshold or self.ME_counter >= self.ME_FramesThreshold:

                self.ME_counter = 0
                
                self.init_frame = Compressed_YUV
                
                t1 = time.time()
                Quantized_img = DCTQ_Encode(np.float32(Compressed_YUV))#TODO: Refactor and optimize
                time_f = (time.time()-t1)
                print("Quantization Time for CPU Accelerated {} s".format(time_f))#TODO: Logger report with time
                #TODO: Integrate VLC
                init_encoder("TestingOutput/", frame_resolution=[1280, 720], yuv_config=[4, 4, 4], encoder_mode=0)
                DCT_ip = [Quantized_img[:,:,0],Quantized_img[:,:,1],Quantized_img[:,:,2]]
                encode(cfg.DCT_FRAME, DCT_ip)
                if self.Dump_objs == True:
                    pickle.dump(Quantized_img, open('data/Output/obj_'+str(self.seq_counter)+'_DCTQ.mesh_obj',"wb"))
                    self.seq_counter+=1
                return Quantized_img
            else:
                self.ME_counter += 1
                if self.Dump_objs == True:
                    pickle.dump(mv, open('data/Output/obj_'+str(self.seq_counter)+'_ME.mesh_obj',"wb"))
                    self.seq_counter+=1
                print("Motion Estimation path")#TODO: Logger report with time
                return mv
                
        else:
            self.ME_counter = 0
            self.Optimize_Motion = True
            self.init_frame = Compressed_YUV
            t1 = time.time()
            Quantized_img = DCTQ_Encode(Compressed_YUV)#TODO: Refactor and optimze
            
            time_f = (time.time()-t1)
            print("Quantization Time for CPU Accelerated {} s".format(time_f))#TODO: Logger report with time
            #TODO: Integrate VLC
            
            if self.Dump_objs == True:
                pickle.dump(Quantized_img, open('data/Output/obj_'+str(self.seq_counter)+'_DCTQ.mesh_obj',"wb"))
                self.seq_counter+=1
            return Quantized_img