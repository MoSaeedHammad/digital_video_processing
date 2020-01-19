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

quantization_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                [12, 12, 14, 19, 26, 58, 60, 55],
                                [14, 13, 16, 24, 40, 57, 69, 56],
                                [14, 17, 22, 29, 51, 87, 80, 62],
                                [18, 22, 37, 56, 68, 109, 103, 77],
                                [24, 35, 55, 64, 81, 104, 113, 92 ],
                                [49, 64, 78, 87, 103, 121, 120, 101],
                                [72, 92, 95,98, 112, 100, 103, 99]])


def zigzag (matrix):
    rows = 8
    columns =  8
    
    solution=[[] for i in range(rows+columns-1)] 
    final = []
    for i in range(rows): 
        for j in range(columns): 
            sum=i+j 
            if(sum%2 ==0): 

                #add at beginning 
                solution[sum].insert(0,matrix[i][j]) 
            else: 

                #add at end of the list 
                solution[sum].append(matrix[i][j]) 

    # print the solution as it as 
    for i in solution: 
        for j in i: 
            final .append (j) 
    return final

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

def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct2(a):
    return scipy.fftpack.idct( scipy.fftpack.idct( a, axis=0 , norm='ortho'), axis=1 , norm='ortho')

def DCT_Encode (image):
    # change read image
    imsize = image.shape
    dct = np.zeros(imsize)

    # Do 8x8 DCT on image (in-place)
    for i in r_[:imsize[0]:8]:
        for j in r_[:imsize[1]:8]:
            dct[i:(i+8),j:(j+8)] = dct2( image[i:(i+8),j:(j+8)] )
            
    # apply quantization 
    Rquantize = quantize(dct , 0)
    Gquantize = quantize(dct , 1)
    Bquantize = quantize(dct , 2)
    
    final = np.vstack ((Rquantize, Gquantize, Bquantize))
    
    return final

def read_decoder(final):
    rinverse = final[0]
    ginverse = final[1]
    binverse = final[2]

    #print (rinverse.shape)
    Rquantize = inverse_quantize(rinverse)
    Gquantize = inverse_quantize(ginverse)
    Bquantize = inverse_quantize(binverse)

    rgbafterinverse = np.dstack((Rquantize,Gquantize,Bquantize))
    imsize = im.shape 
    im_dct = np.zeros(imsize)

    for i in r_[:imsize[0]:8]:
        for j in r_[:imsize[1]:8]:
            im_dct[i:(i+8),j:(j+8)] = idct2( rgbafterinverse[i:(i+8),j:(j+8)] )

    im_dct  = np.rint (im_dct)+128 
    im_dct=np.clip(im_dct,0,255).astype(int)
    plt.figure()
    # plt.imshow (im)
    plt.imshow (im_dct)

    # plt.imshow( np.hstack( (im, im_dct) ) ,cmap='gray')
    plt.title("Comparison between original and DCT compressed images" )

    R, G, B = cv2.split (im_dct)
    R = R.flatten ()
    G = G.flatten ()
    B = B.flatten ()

    pixels = np.vstack( (R, G, B))
    return pixels