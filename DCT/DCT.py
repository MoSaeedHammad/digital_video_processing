from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy
from numpy import pi
from numpy import r_
from scipy import signal
import matplotlib.pylab as pylab
import  cv2
import math
import numba
from numba import jit, cuda, float32, int64
import time

Q50 = np.float32(np.array( [
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
]))

Qunatization_mode = 'JPEG_STD'
if Qunatization_mode == 'JPEG_STD':
    Q = Q50
elif Qunatization_mode == 'Q90':
    Q = ((1.0/5.0) * Q50)
elif Qunatization_mode == 'Q70':
    Q = ((3.0/5.0)*Q50)
elif Qunatization_mode == 'Q30':
    Q = ((5.0/3.0)*Q50)
elif Qunatization_mode == 'Q10':
    Q = ((5.0/1.0)*Q50)
Q = np.float32(Q)
pi_rad = math.pi

zigzag_matrix = np.array( [
    [0, 1, 5, 6, 14, 15, 27, 28],
    [2, 4, 7, 13, 16, 26, 29, 42],
    [3, 8, 12, 17, 25, 30, 41, 43],
    [9, 11, 18, 24, 31, 40, 44, 53],
    [10, 19, 23, 32, 39, 45, 52, 54],
    [20, 22, 33, 38, 46, 51, 55, 60],
    [21, 34, 37, 47, 50, 56, 59, 61],
    [32, 36, 48, 49, 57, 58, 62, 63],
])

zigzag_vector = np.ravel(zigzag_matrix)

#OPTIMIZATION DONE BY INTEGRATION TEAM
@jit(nopython=True, parallel=True)#TODO: CUDA?!
def zigzag_Inverse (matrix):
    matrix = np.ravel(matrix)
    new_matrix = np.zeros(matrix.shape, dtype=np.float32)
    for i in range(64): 
        new_matrix[zigzag_vector[i]] = matrix[i]
    return new_matrix.reshape((8,8))

#OPTIMIZATION DONE BY INTEGRATION TEAM
@jit(nopython=True, parallel=True)#TODO: CUDA?!
def zigzag (matrix):
    new_matrix = np.zeros(matrix.shape, dtype=np.float32)
    matrix = np.ravel(matrix)
    for i in range(8): 
        for j in range(8): 
            new_matrix[i,j] = matrix[zigzag_matrix[i,j]]
    return new_matrix

def DCTQ_Encode (image):
    DCT_Quantized = DCT_Q_Encode(image-128.0)
    return DCT_Quantized

def DCTQ_Decode (Quantized_data):
    DCT_IQuantized = DCT_Q_Decode(Quantized_data)+128.0
    return DCT_IQuantized

#OPTIMIZATION DONE BY INTEGRATION TEAM
@jit(nopython=True, parallel=True)#TODO: CUDA?!
def DCT_Q_Decode (Quantized_data):
    # change read image
    IDCT = np.zeros(Quantized_data.shape,dtype=np.float32)
    out = np.zeros((8,8),dtype=np.float32)
    # Do 8x8 DCT on image (in-place)
    for i in range(0,IDCT.shape[0],8):
        for j in range(0,IDCT.shape[1],8):
            #TO OPIMIZE LOOPS
            IDCT[i:(i+8),j:(j+8),0] = block_idct2(rnd1(zigzag_Inverse( Quantized_data[i:(i+8),j:(j+8),0] )* Q, decimals=0, out=out)) #Y

            IDCT[i:(i+8),j:(j+8),1] = block_idct2(rnd1(zigzag_Inverse( Quantized_data[i:(i+8),j:(j+8),1] )* Q, decimals=0, out=out)) #U
            
            IDCT[i:(i+8),j:(j+8),2] = block_idct2(rnd1(zigzag_Inverse( Quantized_data[i:(i+8),j:(j+8),2] )* Q, decimals=0, out=out)) #V
            
    return IDCT

@jit(nopython=True)
def rnd1(x, decimals, out):
    return np.round_(x, decimals, out)

#OPTIMIZATION DONE BY INTEGRATION TEAM
@jit(nopython=True, parallel=True)#TODO: CUDA?!
def DCT_Q_Encode (image):
    # change read image
    dct = np.zeros(image.shape,dtype=np.float32)
    out = np.zeros((8,8),dtype=np.float32)
    # Do 8x8 DCT on image (in-place)
    for i in range(0,image.shape[0],8):
        for j in range(0,image.shape[1],8):
            #TO OPIMIZE LOOPS
            dct[i:(i+8),j:(j+8),0] = zigzag ( rnd1(block_dct2( image[i:(i+8),j:(j+8),0] ) / Q, decimals=0, out=out ))#Y
            #TODO: Optimize with less than 4:4:4
            dct[i:(i+8),j:(j+8),1] = zigzag ( rnd1(block_dct2( image[i:(i+8),j:(j+8),1] ) / Q, decimals=0, out=out ))#U
            #TODO: Optimize with less than 4:4:4
            dct[i:(i+8),j:(j+8),2] = zigzag ( rnd1(block_dct2( image[i:(i+8),j:(j+8),2] ) / Q, decimals=0, out=out ))#V      
    return dct

#OPTIMIZATION DONE BY INTEGRATION TEAM
@jit(nopython=True, parallel=True)#TODO: CUDA?!
def block_dct2(block):
    dct_block = np.zeros(block.shape,dtype=np.float32)
    for k1 in range(0, block.shape[0]):
        for k2 in range(0, block.shape[1]):
            out = 0
            for x in range(0, block.shape[0]):
                for y in range(0, block.shape[1]):
                    out += ( block[x,y] * math.cos( ( ( (2.0*x) +1.0)/16.0)*k1 *  pi_rad) * math.cos( (( (2.0*y) +1.0)/16.0)*k2 *  pi_rad))#TODO: optimize with LUT
            if k1 == 0:
                c_k1 = 1.0/math.sqrt(2.0)
            else:
                c_k1 = 1.0
            if k2 == 0:
                c_k2 = 1.0/math.sqrt(2.0)
            else:
                c_k2 = 1.0
            out = round(float32(out * c_k1 * c_k2 * (1.0/4.0)))
            dct_block[k1,k2] = out
    return dct_block #Round the floating point to the nearest integer

#OPTIMIZATION DONE BY INTEGRATION TEAM
@jit(nopython=True, parallel=True)#TODO: CUDA?!
def block_idct2(block):
    dct_block = np.zeros(block.shape,dtype=np.float32)
    for k1 in range(0, block.shape[0]):
        for k2 in range(0, block.shape[1]):
            out = 0
            for x in range(0, block.shape[0]):
                for y in range(0, block.shape[1]):
                    if x == 0:
                        c_x = 1.0/math.sqrt(2.0)
                    else:
                        c_x = 1.0
                    if y == 0:
                        c_y = 1/math.sqrt(2.0)
                    else:
                        c_y = 1.0
                    out += ( c_x * c_y *block[x,y] * math.cos( ( ( (2.0*k1) +1.0)/16.0)*x *  pi_rad) * math.cos( (( (2.0*k2) +1.0)/16.0)*y *  pi_rad))

            out = (out * (1.0/4.0))
            dct_block[k1,k2] = out
    return dct_block




"""
print (zigzag_Inverse (zigzag_matrix))
def quantize(img_channel , numberOfChannel):
    imsize = img_channel.shape
    Rquantize = np.zeros((imsize[0],imsize[1] ))
    Rlist = np.zeros ((imsize[0]*imsize[1]))

    for i in r_[:imsize[0]:8]:
        for j in r_[:imsize[1]:8]: 
            Rquantize[i:(i+8),j:(j+8)] =np.divide(img_channel[i:(i+8),j:(j+8), numberOfChannel],quantization_matrix)
            Rquantize = np.rint (Rquantize)
            Rquantize = np.clip(Rquantize,0,225).astype(int)
            
    for i in r_[:imsize[0]:8]:
        for j in r_[:imsize[1]:8]: 
            Rlist [i:(i+64)]= zigzag (Rquantize[i:(i+8),j:(j+8)] )

    return  Rlist
def zigzagInverse(input, vmax = 8, hmax = 8):
	
	#print input.shape

	# initializing the variables
	#----------------------------------
	h = 0
	v = 0

	vmin = 0
	hmin = 0

	output = np.zeros((vmax, hmax))

	i = 0
    #----------------------------------

	while ((v < vmax) and (h < hmax)): 
		#print ('v:',v,', h:',h,', i:',i)   	
		if ((h + v) % 2) == 0:                 # going up
            
			if (v == vmin):
				#print(1)
				
				output[v, h] = input[i]        # if we got to the first line

				if (h == hmax):
					v = v + 1
				else:
					h = h + 1                        

				i = i + 1

			elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
				#print(2)
				output[v, h] = input[i] 
				v = v + 1
				i = i + 1

			elif ((v > vmin) and (h < hmax -1 )):    # all other cases
				#print(3)
				output[v, h] = input[i] 
				v = v - 1
				h = h + 1
				i = i + 1

        
		else:                                    # going down

			if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
				#print(4)
				output[v, h] = input[i] 
				h = h + 1
				i = i + 1
        
			elif (h == hmin):                  # if we got to the first column
				#print(5)
				output[v, h] = input[i] 
				if (v == vmax -1):
					h = h + 1
				else:
					v = v + 1
				i = i + 1
        		        		
			elif((v < vmax -1) and (h > hmin)):     # all other cases
				output[v, h] = input[i] 
				v = v + 1
				h = h - 1
				i = i + 1




		if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
			#print(7)        	
			output[v, h] = input[i] 
			break

	return output

def inverse_quantize (img_channel ):
    RIquantize = np.zeros((imsize[0],imsize[1] ))
    Rmatrix = np.zeros((imsize[0],imsize[1] ))

    for i in r_[:imsize[0]:8]:
        for j in r_[:imsize[1]:8]: 
            Rmatrix[i:(i+8),j:(j+8)] = zigzagInverse (img_channel [i:(i+64)])            
            
    for i in r_[:imsize[0]:8]:
        for j in r_[:imsize[1]:8]:             
            RIquantize[i:(i+8),j:(j+8)] = np.multiply(Rmatrix[i:(i+8),j:(j+8)],quantization_matrix)
    return RIquantize

"""