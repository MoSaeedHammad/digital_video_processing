# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 17:14:41 2019

@author: Ibrahim El-Shal
"""

# In[1]: Import Packages

from ._Init_ import *
from .Three_Step_Search import PSNR, MotionEstimation
from .Affine_Transformations import AffineTransformer, Apply_Transformation, Get_Tri
from .Split import split_block, get_triangle_idx, get_blocks_tts, getVariableTriangle,getBlocksInLayers

# In[2]:   
'''
MeshMotionEstimation Function Arguments:
    f0 = initial frame
    f1 = frame will be compressed 
    build_struct = True will build a new structure between f0 and f1, 
                    False will not build new structure and you have to give it a structure
    struct = the structure you want to get the motion vectors for
    layers = number of layers in the structure len(structure)
    max_block = block size to start with
    psnr_threshold = the threshold that the algorithm will split based on it
    returns: over all image psnr , if build_struct = True [biggest block size,structure,motion vectors] 
    if build_struct = False [-1,[],motion vectors]
'''
def MeshMotionEstimation(f0,f1,build_struct=True,struct=[],layers=1,max_block=256,psnr_threshold=31):
    block_size = max_block
    new_size = max_block//2
    shape = f0.shape
    #adding pad to the image if the image shape is not divisble by the block size
    if (shape[1]/block_size)-(int(shape[1]/block_size))!=0:
        padding_size = [int((1-((shape[0]/block_size)-(int(shape[0]/block_size))))*block_size),
                        int((1-((shape[1]/block_size)-(int(shape[1]/block_size))))*block_size)]
    else:
        padding_size = [int((1-((shape[0]/block_size)-(int(shape[0]/block_size))))*block_size),0]
        
    
    f0 = cv2.copyMakeBorder(f0, 0, padding_size[0]+1, 0, padding_size[1]+1, cv2.BORDER_CONSTANT, value=0)
    f1 = cv2.copyMakeBorder(f1, 0, padding_size[0]+1, 0, padding_size[1]+1, cv2.BORDER_CONSTANT, value=0)
    img_shape = f0.shape
    #calculating the loop range for the first layer
    loop_range = np.array(np.array(f0.shape)/block_size,dtype=np.uint16)
    psnrThreshold = psnr_threshold
    f1_dash = np.copy(f0) #copying f0 to f1 dash to act as a background
    mv_layers = [] #list of lists have all the motion vectors for every layer
    memo = {} #hash map for appending the calculated motion vectors instead of calculating it again (memoization)
    struct_list = [] #list of lists that have the 
    
    #code for building new structure
    if build_struct == True:
        #loop for layers
        for k in range(4):
            n = 0
            splitted_blocks = [] #list of lists that temporarly saves the splitted blocks indcies for the layer the loop in now
            MV_arr_layer = [] #list saves motion vectors for every layer
            struct_array = [] #list saves structure for every layer
            # blocks loop
            for i in range(loop_range[0]):
                #setting blocks indcies for layers
                if k == 0:
                    x = i*block_size
                for j in range(loop_range[1]):
                    if k == 0:
                        y = j*block_size
                    else:
                        x,y = iter_blocks[i][j]
                        
                    
                    motion_vectors = [] #list 4 saves motion vectors for every block
                    #getting blocks that will apply three step search on them
                    blocks = get_blocks_tts([(x,y),block_size],img_shape)
                    #three step search for every block
                    for block in blocks:
                        if str(block) in memo:
                            motion_vectors.append(memo[str(block)])
                        else:
                            me = MotionEstimation(f1,f0,block[0],block[1],block[2],False)
                            memo[str(block)] = me
                            motion_vectors.append(me)
                    #arranging blocks indcies as triangles
                    f0_triangles_vertecies = [[motion_vectors[0],motion_vectors[2],motion_vectors[1]],
                                         [motion_vectors[3],motion_vectors[1],motion_vectors[2]]]
                    size,f1_triangles_vertcies = get_triangle_idx([(x,y),block_size])
                    affine_matrices = [] #the affine matrices for the 2 trangles of the block
                    f0_var_triangles = [] #list saves the points inside the 2 deformed triangles
                    # getting the affine matrices and the points inside the deformed triangles
                    for tri1,tri2 in zip(f0_triangles_vertecies,f1_triangles_vertcies):
                        affine_matrices.append(AffineTransformer(np.float32(tri1),np.float32(tri2)))
                        f0_var_triangles.append(getVariableTriangle(tri1))
                        
                    f1_dash_block = []
                    #applying transformation to every point at the deformed triangle to get the new points in f1 dash 
                    for affine,tri_pts in zip(affine_matrices,f0_var_triangles):
                        f1_dash_block.append(Apply_Transformation(affine[0],affine[1],tri_pts))
                    #coloring the pixels after applying the transformation it has to be parallelized about 95% of the algorithm time
                    for f0_tri,f1_tri in zip(f0_var_triangles,f1_dash_block):
                        for f0_pixel,f1_pixel in zip(f0_tri,f1_tri):
                            if f0_pixel[0]>=img_shape[0]:
                                f0_pixel[0] = img_shape[0]-1
                            if f0_pixel[1]>=img_shape[1]:
                                f0_pixel[1] = img_shape[1]-1
                            if f1_pixel[0]>=img_shape[0]:
                                f1_pixel[0] = img_shape[0]-1
                            if f1_pixel[1]>=img_shape[1]:
                                f1_pixel[1] = img_shape[1]-1
                            if f0_pixel[0]<0:
                                f0_pixel[0] = 0
                            if f0_pixel[1]<0:
                                f0_pixel[1] = 0
                            if f1_pixel[0]<0:
                                f1_pixel[0] = 0
                            if f1_pixel[1]<0:
                                f1_pixel[1] = 0
                            
                            f1_dash[int(f1_pixel[0]),int(f1_pixel[1])] = f0[int(f0_pixel[0]),int(f0_pixel[1])]
                    #comparing the real block and the predicted block and splitting the blocks with low psnr and appending the block motion vectors
                    if PSNR(f1_dash[x:x+block_size,y:y+block_size],f1[x:x+block_size,y:y+block_size])<psnrThreshold:
                         MV_arr_layer.append(motion_vectors)
                         struct_array.append(1)
                         new_size,new_blocks = split_block([(x,y),block_size])
                         splitted_blocks.append(new_blocks)
                         
                    else:
                        MV_arr_layer.append(motion_vectors)
                        struct_array.append(0)
            mv_layers.append(MV_arr_layer)
            struct_list.append(np.array(struct_array))
            block_size = new_size
            #calculating the psnr after finishinh every layer  
            img_psnr = PSNR(f1_dash,f1)
            if img_psnr>psnrThreshold:
                #return_iframe = False
                break
            #else:
                #return_iframe = True
            if len(splitted_blocks) == 0:
                break
            #updating the loop range to the splitted blocks
            loop_range[0] = len(splitted_blocks)
            loop_range[1] = len(splitted_blocks[0])
            iter_blocks = []
            iter_blocks = splitted_blocks
        #if return_iframe:
        #    return -1
        return img_psnr,[max_block,struct_list,mv_layers]
    
    elif build_struct==False:
        struct_l = struct.copy()
        for k in range(layers):
            splitted_blocks = []
            MV_arr_layer = []
            if k == 0:
                struct_l[k] = struct_l[k].reshape(loop_range)
            else:
                struct_l[k] = struct_l[k].reshape(-1,4)
            for i in range(loop_range[0]):
                if k == 0:
                    x = i*block_size
                for j in range(loop_range[1]):
                    if k == 0:
                        y = j*block_size
                    else:
                        x,y = iter_blocks[i][j]
                    motion_vectors = []
                    blocks = get_blocks_tts([(x,y),block_size],img_shape)
                    for block in blocks:
                        if str(block) in memo:
                            motion_vectors.append(memo[str(block)])
                        else:
                            me = MotionEstimation(f1,f0,block[0],block[1],block[2],False)
                            memo[str(block)] = me
                            motion_vectors.append(me)
                    f0_triangles_vertecies = [[motion_vectors[0],motion_vectors[2],motion_vectors[1]],
                                         [motion_vectors[3],motion_vectors[1],motion_vectors[2]]]
                    size,f1_triangles_vertcies = get_triangle_idx([(x,y),block_size])
                    affine_matrices = []
                    f0_var_triangles = []
                    for tri1,tri2 in zip(f0_triangles_vertecies,f1_triangles_vertcies):
                        affine_matrices.append(AffineTransformer(np.float32(tri1),np.float32(tri2)))
                        f0_var_triangles.append(getVariableTriangle(tri1))
                    f1_dash_block = []
                    for affine,tri_pts in zip(affine_matrices,f0_var_triangles):
                        f1_dash_block.append(Apply_Transformation(affine[0],affine[1],tri_pts))
                    for f0_tri,f1_tri in zip(f0_var_triangles,f1_dash_block):
                        for f0_pixel,f1_pixel in zip(f0_tri,f1_tri):
                            if f0_pixel[0]>=img_shape[0]:
                                f0_pixel[0] = img_shape[0]-1
                            if f0_pixel[1]>=img_shape[1]:
                                f0_pixel[1] = img_shape[1]-1
                            if f1_pixel[0]>=img_shape[0]:
                                f1_pixel[0] = img_shape[0]-1
                            if f1_pixel[1]>=img_shape[1]:
                                f1_pixel[1] = img_shape[1]-1
                            if f0_pixel[0]<0:
                                f0_pixel[0] = 0
                            if f0_pixel[1]<0:
                                f0_pixel[1] = 0
                            if f1_pixel[0]<0:
                                f1_pixel[0] = 0
                            if f1_pixel[1]<0:
                                f1_pixel[1] = 0
                            
                            f1_dash[int(f1_pixel[0]),int(f1_pixel[1])] = f0[int(f0_pixel[0]),int(f0_pixel[1])]
                    MV_arr_layer.append(motion_vectors)
                    if struct_l[k][i][j] == 1:
                        new_size,new_blocks = split_block([(x,y),block_size])
                        splitted_blocks.append(new_blocks)
            mv_layers.append(MV_arr_layer)
            block_size = new_size
            img_psnr = PSNR(f1_dash,f1)
            if img_psnr>psnrThreshold:
                #return_iframe = False
                break
            #else:
                #return_iframe = True
            if len(splitted_blocks) == 0:
                break
            loop_range[0] = len(splitted_blocks)
            loop_range[1] = len(splitted_blocks[0])
            iter_blocks = []
            iter_blocks = splitted_blocks
       # if return_iframe:
       #     return -1
        return img_psnr,[-1,[],mv_layers]
    
def MeshMotionCompensation(f0,structure,motion_vectors,block_size):
    shape = f0.shape
    #print(shape)
    padding_size = [0,0]
    #adding pad to the image if the image shape is not divisble by the block size
    if (shape[1]/block_size)-(int(shape[1]/block_size))!=0:
        padding_size = [int((1-((shape[0]/block_size)-(int(shape[0]/block_size))))*block_size),
                        int((1-((shape[1]/block_size)-(int(shape[1]/block_size))))*block_size)]
    else:
        padding_size = [int((1-((shape[0]/block_size)-(int(shape[0]/block_size))))*block_size),0]


    f0 = cv2.copyMakeBorder(f0, 0, padding_size[0]+1, 0, padding_size[1]+1, cv2.BORDER_CONSTANT, value=0)
    img_shape = f0.shape
    #calculating the loop range for the first layer
    f1_dash = np.copy(f0) #copying f0 to f1 dash to act as a background
    blocks_layers = getBlocksInLayers(img_shape,block_size,structure)
    motion_v = motion_vectors.copy()
    for f0_blocks,f1_blocks in zip(reversed(motion_v),reversed(blocks_layers)):
        for f0_block,f1_block in zip(f0_blocks,f1_blocks):
            if not f1_block:
                continue
            else:
                f0_triangles_vertecies = [[f0_block[0],f0_block[2],f0_block[1]],
                                         [f0_block[3],f0_block[1],f0_block[2]]]
                size,f1_triangles_vertcies = get_triangle_idx([(f1_block[0],f1_block[1]),f1_block[2]])
                affine_matrices = []
                f0_var_triangles = []
                for tri1,tri2 in zip(f0_triangles_vertecies,f1_triangles_vertcies):
                    affine_matrices.append(AffineTransformer(np.float32(tri1),np.float32(tri2)))
                    f0_var_triangles.append(getVariableTriangle(tri1))
                f1_dash_block = []
                for affine,tri_pts in zip(affine_matrices,f0_var_triangles):
                    f1_dash_block.append(Apply_Transformation(affine[0],affine[1],tri_pts))
                for f0_tri,f1_tri in zip(f0_var_triangles,f1_dash_block):
                    for f0_pixel,f1_pixel in zip(f0_tri,f1_tri):
                        if f0_pixel[0]>=img_shape[0]:
                            f0_pixel[0] = img_shape[0]-1
                        if f0_pixel[1]>=img_shape[1]:
                            f0_pixel[1] = img_shape[1]-1
                        if f1_pixel[0]>=img_shape[0]:
                            f1_pixel[0] = img_shape[0]-1
                        if f1_pixel[1]>=img_shape[1]:
                            f1_pixel[1] = img_shape[1]-1
                        if f0_pixel[0]<0:
                            f0_pixel[0] = 0
                        if f0_pixel[1]<0:
                            f0_pixel[1] = 0
                        if f1_pixel[0]<0:
                            f1_pixel[0] = 0
                        if f1_pixel[1]<0:
                            f1_pixel[1] = 0

                        f1_dash[int(f1_pixel[0]),int(f1_pixel[1])] = f0[int(f0_pixel[0]),int(f0_pixel[1])]
    if (shape[1]/block_size)-(int(shape[1]/block_size))==0:
        f1_dash = f1_dash[:-(padding_size[0]+1),:-1]
    else:
        f1_dash = f1_dash[:-(padding_size[0]+1),:-(padding_size[1]+1)]
    return f1_dash

# FOR TESTING PLEASE USE 8BIT YUV IMAGE FILE TO USE Read_Sequence 
#fileName = 'D:\Downloads\ShakeNDry_3840x2160_120fps_420_8bit_YUV_RAW\ShakeNDry_3840x2160.yuv'
#shape=[2160,3840,3]
#y_frames, u_frames, v_frames = Read_Sequence(shape,fileName)
#f0,f1 = y_frames

#psnr,motion_vector = MeshMotionEstimation(f0,f1)
#layers = len(motion_vector[1])
#f1_dash = MeshMotionCompensation(f0,motion_vector[1],motion_vector[2],256)
#plt.imshow(f1_dash,cmap='gray')
#plt.show()

#print(psnr)
#new_motion_vectors = MeshMotionEstimation(f0,f1,build_struct=False,layers=layers,struct=motion_vectors[1])
        

# In[3]:   

