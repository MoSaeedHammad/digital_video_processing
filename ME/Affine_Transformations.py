# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 14:56:20 2019
@author: Ibrahim El-Shal

-------------------------------------------------------------------------------
Affine transformation is defined as q= Ap,
where:   
        A is the transformation matrix,
        q and p are corresponding image coordinates in image 1 and 2.
          
The transformation matrix is computed using the least squares estimates 
(i.e. by mininmizing the sum of the squared residuals).

Input: Corresponding Image coordinates in the two images
Outputs: A 3x3 affine transformation matrix for transforming the coordinates 
         from one frame to another.
------------------------------------------------------------------------------
"""
# In[1]: Immport Packages

from ._Init_ import *
#from Read_Seq import Read_Sequence

# In[2]: Get Neede Frames
    
#shape = [2160,3840,3]
#ins = np.float32([[2, 1], [-2, 0], [0, 2]])
#out = np.float32([[5, 0], [-2, 2], [2, 4]])
#fileName = 'F:\Education\Python Scripts\9. Video Processing\Project\Beauty_3840x2160_120fps_420_10bit_YUV.yuv'
#y_frames, u_frames, v_frames = Read_Sequence(shape,fileName)

# In[3]: 

def AffineTransformer(Coordinates_Image_1, Coordinates_Image_2):

    if (len(Coordinates_Image_1[0]) != 2 or len(Coordinates_Image_1[2]) != 2):
        raise ValueError("Incorrect input dimensions")
    elif (len(Coordinates_Image_1) != len(Coordinates_Image_2)):
        raise ValueError("Mismatch between number of points in the image pair")

    # The P and Q matrix
    P = np.float64([Coordinates_Image_1.transpose()[0,:],
                    Coordinates_Image_1.transpose()[1,:],
                    np.ones(shape=(len(Coordinates_Image_1)), dtype=int)])
    
    Q = np.float64([Coordinates_Image_2.transpose()[0,:],
                    Coordinates_Image_2.transpose()[1,:],
                    np.ones(shape=(len(Coordinates_Image_2)), dtype=int)])

    # The least squares estimate of A can be derived as <<Tr(X).X.A = Tr(X).X'>>
    # Least squares: Matrix problems Lecture "shorturl.at/ajpuT"
    LHS = np.linalg.pinv(np.matmul(P, np.transpose(P)))
    RHS = np.matmul(np.transpose(P), LHS)
    A = np.matmul(Q, RHS)[:-1, :]
    
    Mat_A = A[:, 0:2]
    Mat_B = A[:, -1].reshape(2,1)
    return Mat_A, Mat_B

def Apply_Transformation(A_Mat,B_Mat,P_Mat):
    
     return (np.matmul(A_Mat,P_Mat.T) + B_Mat).T
 
def Get_Tri(Status, block, Blk_Size= 256):
    
    new_range = 1 
    tri_points = []

    if(Status == 'Upper'):
        for row in range(block[0],block[0]+Blk_Size):
            line = [] 
            for col in range(block[1]+new_range, block[1]+Blk_Size):
                line.append((row,col))
    
            tri_points.append(line)    
            new_range+=1
    
    elif(Status == 'Lower'):
        for row in range(block[0],block[0]+Blk_Size):
            line = [] 
            variablerange = range(block[1], block[1]+new_range)
            for col in variablerange:
                line.append((row,col))
            tri_points.append(line)    
            new_range+=1
    
    else:
        raise ValueError("Incorrect Input Status")
    return (tri_points)
    
# In[4]: Test Functions
    
#A, B = AffineTransformer(ins, out)

# In[5]: Test Functions

#out2 = Apply_Transformation(A,B,ins)
#affine_matrix_lib = cv2.getAffineTransform(ins, out)