# coding: utf-8

# In[1]: Immport Packages

from ._Init_ import *

# In[2]:

def split_block(block):
    new_block_size = block[1]//2
    return new_block_size,[[block[0][0],block[0][1]],[block[0][0],block[0][1]+new_block_size],
                           [block[0][0]+new_block_size,block[0][1]],[block[0][0]+new_block_size,block[0][1]+new_block_size]]


def get_triangle_idx(block):
    triangle_idx = [[[block[0][0],block[0][1]],[block[0][0]+block[1],block[0][1]],[block[0][0],block[0][1]+block[1]]]
                ,[[block[0][0]+block[1],block[0][1]+block[1]],[block[0][0],block[0][1]+block[1]],[block[0][0]+block[1],block[0][1]]]]
    return block[1], triangle_idx


def get_blocks_tts(block,img_size,vertix_block_size=8):
    if(block[0][0]+block[1] < img_size[0]-1) and (block[0][1]+block[1] < img_size[1]-1) :
        return [[block[0][0],block[0][1],vertix_block_size],[block[0][0],block[0][1]+block[1],vertix_block_size]
                    ,[block[0][0]+block[1],block[0][1],vertix_block_size],
                    [block[0][0]+block[1],block[0][1]+block[1],vertix_block_size]]
    elif (block[0][0]+block[1] >= img_size[0]-1):
        if (block[0][1]+block[1] >= img_size[1]-1):
            return [[block[0][0],block[0][1],vertix_block_size],[block[0][0],block[0][1]+block[1]-8,vertix_block_size]
                    ,[block[0][0]+block[1]-8,block[0][1],vertix_block_size],
                    [block[0][0]+block[1]-8,block[0][1]+block[1]-8,vertix_block_size]]
        else:
            return [[block[0][0],block[0][1],vertix_block_size],[block[0][0],block[0][1]+block[1],vertix_block_size]
                    ,[block[0][0]+block[1]-8,block[0][1],vertix_block_size],
                    [block[0][0]+block[1]-8,block[0][1]+block[1],vertix_block_size]]
    elif (block[0][1]+block[1] >= img_size[1]-1):
        if (block[0][0]+block[1] >= img_size[0]-1):
            return [[block[0][0],block[0][1],vertix_block_size],[block[0][0],block[0][1]+block[1]-8,vertix_block_size]
                    ,[block[0][0]+block[1]-8,block[0][1],vertix_block_size],
                    [block[0][0]+block[1]-8,block[0][1]+block[1]-8,vertix_block_size]]
        else:
            return [[block[0][0],block[0][1],vertix_block_size],[block[0][0],block[0][1]+block[1]-8,vertix_block_size]
                    ,[block[0][0]+block[1],block[0][1],vertix_block_size],
                    [block[0][0]+block[1],block[0][1]+block[1]-8,vertix_block_size]]

def getLine(p1,p2):
    if (p2[0]-p1[0])!=0:  
        m = (p2[1]-p1[1])/(p2[0]-p1[0])
        b = p1[1]-m*p1[0]
        return m,b
    elif (p2[0]-p1[0])==0:
        return 1,0
    

def getVariableTriangle(triangle_pts):
   triangle_pts = sorted(triangle_pts,key = lambda x: x[0])
   triangle_vertices = []
   m1,b1 = getLine(triangle_pts[0],triangle_pts[2])
   m2,b2 = getLine(triangle_pts[0],triangle_pts[1])
   m3,b3 = getLine(triangle_pts[1],triangle_pts[2])
   #print(triangle_pts[0][0],triangle_pts[1][0]+1)
   for i in range(triangle_pts[0][0],triangle_pts[1][0]):
       triangle_vertices.append(getHorizontalLine((i,int(m1*i+b1)),(i,int(m2*i+b2))))
   for j in range(triangle_pts[1][0],triangle_pts[2][0]+1):
       triangle_vertices.append(getHorizontalLine((j,int(m1*j+b1)),(j,int(m3*j+b3))))
   triangle_vertices = np.array([item for sublist in triangle_vertices for item in sublist],dtype=np.float32)
   return triangle_vertices
def getHorizontalLine(p1,p2):
    pointsOnLine = []
    x = p1[0]
    if p1[1]<p2[1]:
        for y in range(p1[1],p2[1]):
            pointsOnLine.append((x,y))
    elif p1[1]>p2[1]:
        for y in range(p2[1],p1[1]):
            pointsOnLine.append((x,y))
    elif p1[1]==p2[1]:
        pointsOnLine.append((x,p1[1]))
    return pointsOnLine
def getBlocksInLayers(img_shape,block_size,structure):
    image_vertcies_layers = []
    loop_range = np.array(np.array(img_shape)/block_size,dtype=np.uint16)
    struct = structure.copy()
    iter_blocks = []
    for k in range(len(struct)):
        splitted_blocks = []
        image_vertcies_arr = []
        if k == 0:
            struct[k] = struct[k].reshape(loop_range)
        else:
            struct[k] = struct[k].reshape(-1,4)
        for i in range(loop_range[0]):
            if k ==0:
                x = i*block_size
            for j in range(loop_range[1]):
                if k == 0:
                    y = j*block_size
                else:
                    x,y = iter_blocks[i][j]
                if k != (len(struct)-1):
                    if struct[k][i][j]==0:
                        image_vertcies_arr.append([x,y,block_size])
                    elif struct[k][i][j]==1:
                        image_vertcies_arr.append([])
                        new_size,new_blocks = split_block([(x,y),block_size])
                        splitted_blocks.append(new_blocks)
                elif k == (len(struct)-1):
                    if struct[k][i][j]==0:
                        image_vertcies_arr.append([x,y,block_size])
                    elif struct[k][i][j]==1:
                        image_vertcies_arr.append([x,y,block_size])
        image_vertcies_layers.append(image_vertcies_arr)
        block_size = new_size
        if k !=3:
            loop_range[0] = len(splitted_blocks)
            loop_range[1] = len(splitted_blocks[0])
            iter_blocks = []
            iter_blocks = splitted_blocks
    return image_vertcies_layers
        
# In[2]:
'''def getBlockFromStructure(structure,max_size,block_index,img_shape):
    number_of_blocks = img_shape//max_size
    for i in range(len(structure)):
        if i == 0:
            structure[i] = structure[i].reshape(number_of_blocks)
        else:
            structure[i] = structure[i].reshape(-1,4)
    if block_index[]
    return block,size'''
'''def getVariableContour(contour_pts):
    m1,b1 = getLine(contour_pts[0],contour_pts[2])
    m2,b2 = getLine(contour_pts[0],contour_pts[1])
    m3,b3 = getLine(contour_pts[1],contour_pts[3])
    m4,b4 = getLine(contour_pts[2],contour_pts[3])
    sorted_contour_pts = sorted(contour_pts,key = lambda x: x[0])
    if sorted_contour_pts[0] == contour_pts[0]:
        for''' 
#pts = getVariableTriangle([[0,0],[0,10],[12,8]])
#new_blocks = split_block([[0,255],256])
'''if(getLine(triangle_pts[0],triangle_pts[2])=='inf' and getLine(triangle_pts[0],triangle_pts[1])[2]==True and getLine(triangle_pts[1],triangle_pts[2])[2]==True):
        m2,b2,_ = getLine(triangle_pts[0],triangle_pts[1])
        m3,b3,_ = getLine(triangle_pts[1],triangle_pts[2])
        
        for i,j in zip(range(triangle_pts[0][0],triangle_pts[1][0]),range(triangle_pts[0][1],triangle_pts[1][1])):
            triangle_vertices.append(getHorizontalLine((i,j)(i,int(m2*i+b2))))
        for i,j in zip(range(triangle_pts[1][0],triangle_pts[2][0]+1),range(triangle_pts[1][1],triangle_pts[2][1]+1)):
            triangle_vertices.append(getHorizontalLine((i,j),(i,int(m3*i+b3))))
            

    elif getLine(triangle_pts[0],triangle_pts[2])[2]==True and getLine(triangle_pts[0],triangle_pts[1])=='inf' and getLine(triangle_pts[1],triangle_pts[2])[2]==True:
        m1,b1,_ = getLine(triangle_pts[0],triangle_pts[2])
        m3,b3,_ = getLine(triangle_pts[1],triangle_pts[2])
        for i,j in zip(range(triangle_pts[0][0],triangle_pts[1][0]),range(triangle_pts[0][1],triangle_pts[1][1])):
            triangle_vertices.append(getHorizontalLine((i,int(m1*i+b1))(i,j)))
        for i,j in zip(range(triangle_pts[1][0],triangle_pts[2][0]+1),range(triangle_pts[1][1],triangle_pts[2][1]+1)):
            triangle_vertices.append(getHorizontalLine((i,int(m1*i+b1)),(i,int(m3*i+b3))))
        
    elif getLine(triangle_pts[0],triangle_pts[2])[2]==True and getLine(triangle_pts[0],triangle_pts[1])[2]==True and getLine(triangle_pts[1],triangle_pts[2])=='inf':
        m1,b1,_ = getLine(triangle_pts[0],triangle_pts[2])
        m2,b2,_ = getLine(triangle_pts[0],triangle_pts[1])
        for i,j in zip(range(triangle_pts[0][0],triangle_pts[1][0]),range(triangle_pts[0][1],triangle_pts[1][1])):
            triangle_vertices.append(getHorizontalLine((i,int(m1*i+b1))(i,int(m2*i+b2))))
        for i,j in zip(range(triangle_pts[1][0],triangle_pts[2][0]+1),range(triangle_pts[1][1],triangle_pts[2][1]+1)):
            triangle_vertices.append(getHorizontalLine((i,int(m1*i+b1)),(i,j)))
       '''
# In[3]: Test Functions
    
'''
f0 = y_frames[0]
f1 = y_frames[1]

size = f0.shape
block_size = 256
vertex_block_size = 8
vertics_array = []
blocks_tts = []
triangles_pts = []
vertics_matrix = []

for x in  range(0,size[0],block_size):
    vertics = []
    for y in range(0,size[1]+1,block_size):
        vertics_array.append((x,y))
        vertics.append((y,x))
    vertics_matrix.append(vertics)
    
for i in range(len(vertics_matrix)-1):
    for j in range(len(vertics_matrix[0])-1):
        triangles_pts.append([[vertics_matrix[i][j],vertics_matrix[i][j+1],vertics_matrix[i+1][j]],
                            [vertics_matrix[i][j+1],vertics_matrix[i+1][j],vertics_matrix[i+1][j+1]]])


print(vertics_array[30])        
print(split_block([vertics_array[30],block_size]))
print(get_triangle_idx([vertics_array[30],block_size]))
print(get_blocks_tts([vertics_array[30],block_size],size))
#print(triangles_pts)
for vertex in vertics_array:
    if vertex[0] == 0 and vertex[1] == 0:
        blocks_tts.append([vertex[0],vertex[1],vertex_block_size])
    elif vertex[0] == 0 and vertex[1] == size[1]:
        blocks_tts.append([vertex[0],vertex[1]-1,-vertex_block_size])
    elif vertex[0] == 0 and vertex[1]>0 and vertex[1]<size[1]:
        blocks_tts.append([vertex[0],vertex[1]-4,vertex_block_size])
    elif (vertex[1] == 0 and vertex[0]>0):
        blocks_tts.append([vertex[0]-4,vertex[1],vertex_block_size])
    elif (vertex[1] == size[1] and vertex[0]>0):
        blocks_tts.append([vertex[0]-4,vertex[1]-1,-vertex_block_size])
    else:
        blocks_tts.append([vertex[0]-4,vertex[1]-4,vertex_block_size])
        
print(blocks_tts[0])
#print(vertics_array)
#print(blocks_tts)
img = np.copy(f0)
plt.imshow(img)
plt.show()
#block1 = blocks_tts[0]
#block2 = blocks_tts[1]
#block3 = blocks_tts[2]
#block144 = blocks_tts[143]
#print(block144)
#print((block144[0]+(block144[2]*32),block144[1]+(block144[2]*32)))
#print((block144[0],block144[1]))
#img = cv2.rectangle(img,((3839-64),(252+64)),(3839,252),(0,0,0),-1)
#img = cv2.rectangle(img,((3580+64),(252+64)),(3580,252),(0,0,0),-1)
#print(blocks_tts)
for block in blocks_tts:
    if block[1] == 3839:
        img = cv2.rectangle(img,(block[1],block[0]),(block[1]+(block[2]*8),block[0]+(-block[2]*8)),(0,0,0),-1)
    else:
        img = cv2.rectangle(img,(block[1],block[0]),(block[1]+(block[2]*8),block[0]+(block[2]*8)),(0,0,0),-1)
        
        
img2 = np.copy(f0)
for triangles_pt in triangles_pts:
    triangle_cnt = np.array(triangles_pt[0])
    img2 = cv2.drawContours(img2, [triangle_cnt] , 0, (0,255,0), 20)
    triangle_cnt = np.array(triangles_pt[1])
    img2 = cv2.drawContours(img2, [triangle_cnt] , 0, (0,255,0), 20)

#triangle_cnt = np.array(get_triangle_idx([vertics_array[30],block_size])[0])
#img2 = cv2.drawContours(img2, [triangle_cnt] , 0, (0,255,0), 20)
plt.imshow(img)
plt.show()
plt.imshow(img2)
plt.show()

'''