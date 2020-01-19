import cv2
import numpy as np

#Pre-Processing..
shape = [2160,3840,3]
y_frames = []
u_frames = []
v_frames = []
fileName = 'D:\\Self-Learning\\Datasets\\Beauty_3840x2160_120fps_420_10bit_YUV.yuv'
with open(fileName, "rb") as f:
    for i in range(120):
        frame = np.empty(shape)
        for j in range(3):
            if j == 0:
                raw = f.read(2160*3840*2)
                channel = np.frombuffer(raw, dtype=np.uint16)
                channel = channel.reshape(2160,3840)
                y_frames.append(channel)
                #plt.imshow(channel)
                #plt.show()
            elif j == 1:
                raw = f.read(1080*1920*2)
                channel = np.frombuffer(raw, dtype=np.uint16)
                channel = channel.reshape(1080,1920)
                u_frames.append(channel)
                #plt.imshow(channel)
                #plt.show()
            elif j == 2:
                raw = f.read(1080*1920*2)
                channel = np.frombuffer(raw, dtype=np.uint16)
                channel = channel.reshape(1080,1920)
                v_frames.append(channel)
                #plt.imshow(channel)
                #plt.show()
Y1, Y2 = y_frames[0], y_frames[5]

#%%

def PSNR(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

#%% TSS (Image1, Image2, X-axis, Y-axis, Size or length)
#Hints:
#   1. it adds padding to each input to avoid boundaries
#   2. those boundaries must be handled through Motion Compensation
    
def MotionEstimation(Ch1,Ch2,x,y,s):
    Mv = [] # Motion Vector
    Ch1 = cv2.copyMakeBorder(Ch1, s*7, s*7, s*7, s*7, cv2.BORDER_CONSTANT, value=0)
    Ch2 = cv2.copyMakeBorder(Ch2, s*7, s*7, s*7, s*7, cv2.BORDER_CONSTANT, value=0)
    #print(Ch1.shape)
    SAD = 10000000
    xi , xj = x, y
    block1 = Ch1[x:x+s,y:y+s]
    Step1 = [Ch2[x:x+s,y:y+s], Ch2[x:x+s,y+(4*s):y+(4*s)+s], Ch2[x-(4*s):x-(4*s)+s,y:y+s], Ch2[x:x+s,y-(4*s):y-(4*s)+s], Ch2[x+(4*s):x+(4*s)+s,y:y+s], Ch2[x+(2*s):x+(2*s)+s,y+(2*s):y+(2*s)+s], Ch2[x+(2*s):x+(2*s)+s,y-(2*s):y-(2*s)+s], Ch2[x-(2*s):x-(2*s)+s,y-(2*s):y-(2*s)+s], Ch2[x-(2*s):x-(2*s)+s,y+(2*s):y+(2*s)+s]]
    for i in range(0,9): # Using a grid of (9) indices & 4 steps block search
        Sad = np.sum(np.abs(block1-Step1[i]))
        #print(Sad)
        if (Sad<SAD):
            SAD = Sad
            if i == 0: xi, xj = x, y
            if i == 1: xi, xj = x, y+(4*s)
            if i == 2: xi, xj = x-(4*s), y
            if i == 3: xi, xj = x, y-(4*s)
            if i == 4: xi, xj = x+(4*s), y
            if i == 5: xi, xj = x+(2*s), y+(2*s)
            if i == 6: xi, xj = x+(2*s), y-(2*s)
            if i == 7: xi, xj = x-(2*s), y-(2*s)
            if i == 8: xi, xj = x-(2*s), y+(2*s)
            Step2 = [Ch2[xi:xi+s,xj+(2*s):xj+(2*s)+s], Ch2[xi-(2*s):xi-(2*s)+s,xj:xj+s], Ch2[xi:xi+s,xj-(2*s):xj-(2*s)+s], Ch2[xi+(2*s):xi+(2*s)+s,xj:xj+s], Ch2[xi+(s):xi+(s)+s,xj+(s):xj+(s)+s], Ch2[xi+(s):xi+(s)+s,xj-(s):xj-(s)+s], Ch2[xi-(s):xi-(s)+s,xj-(s):xj-(s)+s], Ch2[xi-(s):xi-(s)+s,xj+(s):xj+(s)+s]]
            for j in range(0,8): # Using a grid of (8) indices & 2 steps block search
                Sad = np.sum(np.abs(block1-Step2[j]))
                #print(Sad)
                if (Sad<SAD):
                    SAD = Sad
                    if j == 0: xi, xj = xi, xj+(2*s)
                    if j == 1: xi, xj = xi-(2*s), xj
                    if j == 2: xi, xj = xi, xj-(2*s)
                    if j == 3: xi, xj = xi+(2*s), xj
                    if j == 4: xi, xj = xi+(s), xj+(s)
                    if j == 5: xi, xj = xi+(s), xj-(s)
                    if j == 6: xi, xj = xi-(s), xj-(s)
                    if j == 7: xi, xj = xi-(s), xj+(s)
                    Step3 = [Ch2[xi:xi+s,xj+(s):xj+(s)+s], Ch2[xi-(s):xi-(s)+s,xj:xj+s], Ch2[xi:xi+s,xj-(s):xj-(s)+s], Ch2[xi+(s):xi+(s)+s,xj:xj+s], Ch2[xi+(s/2):xi+(s/2)+s,xj+(s/2):xj+(s/2)+s], Ch2[xi+(s/2):xi+(s/2)+s,xj-(s/2):xj-(s/2)+s], Ch2[xi-(s/2):xi-(s/2)+s,xj-(s/2):xj-(s/2)+s], Ch2[xi-(s/2):xi-(s/2)+s,xj+(s/2):xj+(s/2)+s]]
                    for k in range(0,8): # Using a grid of (4) indices & 1 step block search
                        Sad = np.sum(np.abs(block1-Step3[k]))
                        #print(Sad)
                        if (Sad<SAD):
                            SAD = Sad
                            if k == 0: xi, xj = xi, xj+(s)
                            if k == 1: xi, xj = xi-(s), xj
                            if k == 2: xi, xj = xi, xj-(s)
                            if k == 3: xi, xj = xi+(s), xj
                            if k == 4: xi, xj = xi+(s/2), xj+(s/2)
                            if k == 5: xi, xj = xi+(s/2), xj-(s/2)
                            if k == 6: xi, xj = xi-(s/2), xj-(s/2)
                            if k == 7: xi, xj = xi-(s/2), xj+(s/2)
    Mv = [xi,xj]
    return Mv
#%%
Mv = MotionEstimation(Y1,Y2,1803,1812,256)
