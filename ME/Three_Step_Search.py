# In[1]: Immport Packages

from ._Init_ import *

# In[2]: 

def PSNR(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

# In[3]: TSS (Image1, Image2, X-axis, Y-axis, Size or length)
#Hints:
#   1. it adds padding to each input to avoid boundaries
#   2. those boundaries must be handled through Motion Compensation
def MotionEstimation(Ch1,Ch2,x,y,s,appr):
    Mv = [] # Motion Vector
    
    if (appr==0) :
        
        Dic1 = {0: [x, y], 1: [x, y+(4+s)], 2: [x-(4+s), y], 3: [x, y-(4+s)], 4: [x+(4+s), y]}
        #Dic2 = {0: [x, y+(2+s)], 1: [x-(2+s), y], 2: [x, y-(2+s)], 3: [x+(2+s), y]}
        #Dic3 = {0: [x, y+(s)], 1: [x-(s), y], 2: [x, y-(s)], 3: [x+(s), y]}
        
        Ch1 = cv2.copyMakeBorder(Ch1, s, s, s, s, cv2.BORDER_CONSTANT, value=0)
        Ch2 = cv2.copyMakeBorder(Ch2, s, s, s, s, cv2.BORDER_CONSTANT, value=0)
        
        B1    = [Ch1[x:x+s,y:y+s], Ch1[x:x+s,y:y+s], Ch1[x:x+s,y:y+s], Ch1[x:x+s,y:y+s], Ch1[x:x+s,y:y+s]]
        #B1F   = [Ch1[x:x+s,y:y+s], Ch1[x:x+s,y:y+s], Ch1[x:x+s,y:y+s], Ch1[x:x+s,y:y+s]]
        Step1 = [Ch2[x:x+s,y:y+s], Ch2[x:x+s,y+(4+s):y+(4+s)+s], Ch2[x-(4+s):x-(4+s)+s,y:y+s], Ch2[x:x+s,y-(4+s):y-(4+s)+s], Ch2[x+(4+s):x+(4+s)+s,y:y+s]]
        #Step2 = [Ch2[x:x+s,y+(2+s):y+(2+s)+s], Ch2[x-(2+s):x-(2+s)+s,y:y+s], Ch2[x:x+s,y-(2+s):y-(2+s)+s], Ch2[x+(2+s):x+(2+s)+s,y:y+s]]
        #Step3 = [Ch2[x:x+s,y+(s):y+(s)+s], Ch2[x-(s):x-(s)+s,y:y+s], Ch2[x:x+s,y-(s):y-(s)+s], Ch2[x+(s):x+(s)+s,y:y+s]]
        #for i in range(0,4):
        #    if len(Step3[i]) == [0,8] or [8,0]:
        #        Step3[i] = np.zeros((s,s))
        B1, B2 =  np.reshape(B1,(5,-1)), np.reshape(Step1,(5,-1))#, np.reshape(Step2,(4,-1)), np.reshape(Step3,(4,-1))
        
        sad_list = np.sum(np.abs(B1-B2),axis=1)
        Sad1 = np.argmin(sad_list)
        
        #loc1 = Dic1[np.argmin(np.sum(np.abs(B1-B2),axis=1))]
        #loc2 = Dic2[np.argmin(np.sum(np.abs(B1F-B3F),axis=1))]
        #loc3 = Dic3[np.argmin(np.sum(np.abs(B1F-B4F),axis=1))]
        
        #rlist = np.argmin([10000000, Sad1, Sad2, Sad3])
        #rDic  = {0: [x, y], 1:Dic1[np.argmin(sad_list[0])],
        #         2: Dic2[np.argmin(sad_list[1])]
        #         ,3:Dic3[np.argmin(sad_list[2])]}
        xi, xj  = Dic1[Sad1]
        Mv = [xi,xj]
        
    elif (appr==1) :
        
        Dic1 = {0: [x, y], 1: [x, y+(4*s)], 2: [x-(4*s), y], 3: [x, y-(4*s)], 4: [x+(4*s), y]}
        Dic2 = {0: [x, y+(2*s)], 1: [x-(2*s), y], 2: [x, y-(2*s)], 3: [x+(2*s), y]}
        Dic3 = {0: [x, y+(s)], 1: [x-(s), y], 2: [x, y-(s)], 3: [x+(s), y]}
        
        Ch1 = cv2.copyMakeBorder(Ch1, s*7, s*7, s*7, s*7, cv2.BORDER_CONSTANT, value=0)
        Ch2 = cv2.copyMakeBorder(Ch2, s*7, s*7, s*7, s*7, cv2.BORDER_CONSTANT, value=0)
        
        B1    = [Ch1[x:x+s,y:y+s], Ch1[x:x+s,y:y+s], Ch1[x:x+s,y:y+s], Ch1[x:x+s,y:y+s], Ch1[x:x+s,y:y+s]]
        B1F   = [Ch1[x:x+s,y:y+s], Ch1[x:x+s,y:y+s], Ch1[x:x+s,y:y+s], Ch1[x:x+s,y:y+s]]
        Step1 = [Ch2[x:x+s,y:y+s], Ch2[x:x+s,y+(4*s):y+(4*s)+s], Ch2[x-(4*s):x-(4*s)+s,y:y+s], Ch2[x:x+s,y-(4*s):y-(4*s)+s], Ch2[x+(4*s):x+(4*s)+s,y:y+s]]
        Step2 = [Ch2[x:x+s,y+(2*s):y+(2*s)+s], Ch2[x-(2*s):x-(2*s)+s,y:y+s], Ch2[x:x+s,y-(2*s):y-(2*s)+s], Ch2[x+(2*s):x+(2*s)+s,y:y+s]]
        Step3 = [Ch2[x:x+s,y+(s):y+(s)+s], Ch2[x-(s):x-(s)+s,y:y+s], Ch2[x:x+s,y-(s):y-(s)+s], Ch2[x+(s):x+(s)+s,y:y+s]]
        for i in range(0,4):
            if len(Step3[i]) == [0,s] or [s,0]:
                Step3[i] = np.zeros((s,s))
        B1, B2, B1F, B3F, B4F =  np.reshape(B1,(5,-1)), np.reshape(Step1,(5,-1)), np.reshape(B1F,(4,-1)), np.reshape(Step2,(4,-1)), np.reshape(Step3,(4,-1))
        
        Sad1 = np.min(np.sum(np.abs(B1-B2),axis=1))
        Sad2 = np.min(np.sum(np.abs(B1F-B3F),axis=1))
        Sad3 = np.min(np.sum(np.abs(B1F-B4F),axis=1))
        
        loc1 = Dic1[np.argmin(np.sum(np.abs(B1-B2),axis=1))]
        loc2 = Dic2[np.argmin(np.sum(np.abs(B1F-B3F),axis=1))]
        loc3 = Dic3[np.argmin(np.sum(np.abs(B1F-B4F),axis=1))]
        
        rlist = np.argmin([10000000, Sad1, Sad2, Sad3])
        rDic  = {0: [x, y], 1: loc1, 2: loc2, 3: loc3}
        xi, xj  = rDic[rlist][0], rDic[rlist][1]
        Mv = [xi,xj]
        
    return Mv
'''def MotionEstimation(Ch1,Ch2,x,y,s,appr):
    Mv = [] # Motion Vector
    
    if (appr==0) :
        
        Dic1 = {0: [x, y], 1: [x, y+(4+s)], 2: [x-(4+s), y], 3: [x, y-(4+s)], 4: [x+(4+s), y]}
        Dic2 = {0: [x, y+(2+s)], 1: [x-(2+s), y], 2: [x, y-(2+s)], 3: [x+(2+s), y]}
        Dic3 = {0: [x, y+(s)], 1: [x-(s), y], 2: [x, y-(s)], 3: [x+(s), y]}
        
        Ch1 = cv2.copyMakeBorder(Ch1, s+8, s+8, s+8, s+8, cv2.BORDER_CONSTANT, value=0)
        Ch2 = cv2.copyMakeBorder(Ch2, s+8, s+8, s+8, s+8, cv2.BORDER_CONSTANT, value=0)
        
        B1    = [Ch1[x:x+s,y:y+s], Ch1[x:x+s,y:y+s], Ch1[x:x+s,y:y+s], Ch1[x:x+s,y:y+s], Ch1[x:x+s,y:y+s]]
        B1F   = [Ch1[x:x+s,y:y+s], Ch1[x:x+s,y:y+s], Ch1[x:x+s,y:y+s], Ch1[x:x+s,y:y+s]]
        Step1 = [Ch2[x:x+s,y:y+s], Ch2[x:x+s,y+(4+s):y+(4+s)+s], Ch2[x-(4+s):x-(4+s)+s,y:y+s], Ch2[x:x+s,y-(4+s):y-(4+s)+s], Ch2[x+(4+s):x+(4+s)+s,y:y+s]]
        Step2 = [Ch2[x:x+s,y+(2+s):y+(2+s)+s], Ch2[x-(2+s):x-(2+s)+s,y:y+s], Ch2[x:x+s,y-(2+s):y-(2+s)+s], Ch2[x+(2+s):x+(2+s)+s,y:y+s]]
        Step3 = [Ch2[x:x+s,y+(s):y+(s)+s], Ch2[x-(s):x-(s)+s,y:y+s], Ch2[x:x+s,y-(s):y-(s)+s], Ch2[x+(s):x+(s)+s,y:y+s]]

        B1, B2, B1F, B3F, B4F =  np.reshape(B1,(5,-1)), np.reshape(Step1,(5,-1)), np.reshape(B1F,(4,-1)), np.reshape(Step2,(4,-1)), np.reshape(Step3,(4,-1))
        
        print("B4F {}".format(B4F))
        print("B1F {}".format(B1F))
        Sad1 = np.min(np.sum(np.abs(B1-B2),axis=1))
        Sad2 = np.min(np.sum(np.abs(B1F-B3F),axis=1))
        Sad3 = np.min(np.sum(np.abs(B1F-B4F),axis=1))
        
        loc1 = Dic1[np.argmin(np.sum(np.abs(B1-B2),axis=1))]
        loc2 = Dic2[np.argmin(np.sum(np.abs(B1F-B3F),axis=1))]
        loc3 = Dic3[np.argmin(np.sum(np.abs(B1F-B4F),axis=1))]
        
        rlist = np.argmin([10000000, Sad1, Sad2, Sad3])
        rDic  = {0: [x, y], 1: loc1, 2: loc2, 3: loc3}
        xi, xj  = rDic[rlist][0], rDic[rlist][1]
        Mv = [xi,xj]
        
    elif (appr==1) :
        
        Dic1 = {0: [x, y], 1: [x, y+(4*s)], 2: [x-(4*s), y], 3: [x, y-(4*s)], 4: [x+(4*s), y]}
        Dic2 = {0: [x, y+(2*s)], 1: [x-(2*s), y], 2: [x, y-(2*s)], 3: [x+(2*s), y]}
        Dic3 = {0: [x, y+(s)], 1: [x-(s), y], 2: [x, y-(s)], 3: [x+(s), y]}
        
        Ch1 = cv2.copyMakeBorder(Ch1, s*7, s*7, s*7, s*7, cv2.BORDER_CONSTANT, value=0)
        Ch2 = cv2.copyMakeBorder(Ch2, s*7, s*7, s*7, s*7, cv2.BORDER_CONSTANT, value=0)
        
        B1    = [Ch1[x:x+s,y:y+s], Ch1[x:x+s,y:y+s], Ch1[x:x+s,y:y+s], Ch1[x:x+s,y:y+s], Ch1[x:x+s,y:y+s]]
        B1F   = [Ch1[x:x+s,y:y+s], Ch1[x:x+s,y:y+s], Ch1[x:x+s,y:y+s], Ch1[x:x+s,y:y+s]]
        Step1 = [Ch2[x:x+s,y:y+s], Ch2[x:x+s,y+(4*s):y+(4*s)+s], Ch2[x-(4*s):x-(4*s)+s,y:y+s], Ch2[x:x+s,y-(4*s):y-(4*s)+s], Ch2[x+(4*s):x+(4*s)+s,y:y+s]]
        Step2 = [Ch2[x:x+s,x+(2*s):x+(2*s)+s], Ch2[x-(2*s):x-(2*s)+s,x:x+s], Ch2[x:x+s,x-(2*s):x-(2*s)+s], Ch2[x+(2*s):x+(2*s)+s,x:x+s]]
        Step3 = [Ch2[x:x+s,x+(s):x+(s)+s], Ch2[x-(s):x-(s)+s,x:x+s], Ch2[x:x+s,x-(s):x-(s)+s], Ch2[x+(s):x+(s)+s,x:x+s]]

        B1, B2, B1F, B3F, B4F =  np.reshape(B1,(5,-1)), np.reshape(Step1,(5,-1)), np.reshape(B1F,(4,-1)), np.reshape(Step2,(4,-1)), np.reshape(Step3,(4,-1))
        
        Sad1 = np.min(np.sum(np.abs(B1-B2),axis=1))
        Sad2 = np.min(np.sum(np.abs(B1F-B3F),axis=1))
        Sad3 = np.min(np.sum(np.abs(B1F-B4F),axis=1))
        
        loc1 = Dic1[np.argmin(np.sum(np.abs(B1-B2),axis=1))]
        loc2 = Dic2[np.argmin(np.sum(np.abs(B1F-B3F),axis=1))]
        loc3 = Dic3[np.argmin(np.sum(np.abs(B1F-B4F),axis=1))]
        
        rlist = np.argmin([10000000, Sad1, Sad2, Sad3])
        rDic  = {0: [x, y], 1: loc1, 2: loc2, 3: loc3}
        xi, xj  = rDic[rlist][0], rDic[rlist][1]
        Mv = [xi,xj]
        
    return Mv'''

'''def MotionEstimation(Ch1,Ch2,x,y,s,appr):
    Mv = [] # Motion Vector
    if (appr==0) :
        SAD = 10000000
        xi , xj = x, y
        Ch1 = cv2.copyMakeBorder(Ch1, s+7, s+7, s+7, s+7, cv2.BORDER_CONSTANT, value=0)
        Ch2 = cv2.copyMakeBorder(Ch2, s*7, s+7, s+7, s+7, cv2.BORDER_CONSTANT, value=0)
        block1 = Ch1[x:x+s,y:y+s]
        Step1 = [Ch2[x:x+s,y:y+s], Ch2[x:x+s,y+(4+s):y+(4+s)+s], Ch2[x-(4+s):x-(4+s)+s,y:y+s], Ch2[x:x+s,y-(4+s):y-(4+s)+s], Ch2[x+(4+s):x+(4+s)+s,y:y+s]]
        for i in range(0,5): # Using a grid of (5) indices & 4 steps block search
            Sad = np.sum(np.abs(block1-Step1[i]))
            if (Sad<SAD):
                SAD = Sad
                if i == 0: xi, xj = x, y
                if i == 1: xi, xj = x, y+(4+s)
                if i == 2: xi, xj = x-(4+s), y
                if i == 3: xi, xj = x, y-(4+s)
                if i == 4: xi, xj = x+(4+s), y
                Step2 = [Ch2[xi:xi+s,xj+(2+s):xj+(2+s)+s], Ch2[xi-(2+s):xi-(2+s)+s,xj:xj+s], Ch2[xi:xi+s,xj-(2+s):xj-(2+s)+s], Ch2[xi+(2+s):xi+(2+s)+s,xj:xj+s]]
                for j in range(0,4): # Using a grid of (4) indices & 2 steps block search
                    Sad = np.sum(np.abs(block1-Step2[j]))
                    if (Sad<SAD):
                        SAD = Sad
                        if j == 0: xi, xj = xi, xj+(2+s)
                        if j == 1: xi, xj = xi-(2+s), xj
                        if j == 2: xi, xj = xi, xj-(2+s)
                        if j == 3: xi, xj = xi+(2+s), xj
                        Step3 = [Ch2[xi:xi+s,xj+(s):xj+(s)+s], Ch2[xi-(s):xi-(s)+s,xj:xj+s], Ch2[xi:xi+s,xj-(s):xj-(s)+s], Ch2[xi+(s):xi+(s)+s,xj:xj+s]]
                        for k in range(0,4): # Using a grid of (4) indices & 1 step block search
                            Sad = np.sum(np.abs(block1-Step3[k]))
                            if (Sad<SAD):
                                SAD = Sad
                                if k == 0: xi, xj = xi, xj+(s)
                                if k == 1: xi, xj = xi-(s), xj
                                if k == 2: xi, xj = xi, xj-(s)
                                if k == 3: xi, xj = xi+(s), xj
        Mv = [xi,xj]
    elif (appr==1) :
        SAD = 10000000
        xi , xj = x, y
        Ch1 = cv2.copyMakeBorder(Ch1, s*7, s*7, s*7, s*7, cv2.BORDER_CONSTANT, value=0)
        Ch2 = cv2.copyMakeBorder(Ch2, s*7, s*7, s*7, s*7, cv2.BORDER_CONSTANT, value=0)
        block1 = Ch1[x:x+s,y:y+s]
        Step1 = [Ch2[x:x+s,y:y+s], Ch2[x:x+s,y+(4*s):y+(4*s)+s], Ch2[x-(4*s):x-(4*s)+s,y:y+s], Ch2[x:x+s,y-(4*s):y-(4*s)+s], Ch2[x+(4*s):x+(4*s)+s,y:y+s]]
        for i in range(0,5): # Using a grid of (5) indices & 4 steps block search
            Sad = np.sum(np.abs(block1-Step1[i]))
            if (Sad<SAD):
                SAD = Sad
                if i == 0: xi, xj = x, y
                if i == 1: xi, xj = x, y+(4*s)
                if i == 2: xi, xj = x-(4*s), y
                if i == 3: xi, xj = x, y-(4*s)
                if i == 4: xi, xj = x+(4*s), y
                Step2 = [Ch2[xi:xi+s,xj+(2*s):xj+(2*s)+s], Ch2[xi-(2*s):xi-(2*s)+s,xj:xj+s], Ch2[xi:xi+s,xj-(2*s):xj-(2*s)+s], Ch2[xi+(2*s):xi+(2*s)+s,xj:xj+s]]
                for j in range(0,4): # Using a grid of (4) indices & 2 steps block search
                    Sad = np.sum(np.abs(block1-Step2[j]))
                    if (Sad<SAD):
                        SAD = Sad
                        if j == 0: xi, xj = xi, xj+(2*s)
                        if j == 1: xi, xj = xi-(2*s), xj
                        if j == 2: xi, xj = xi, xj-(2*s)
                        if j == 3: xi, xj = xi+(2*s), xj
                        Step3 = [Ch2[xi:xi+s,xj+(s):xj+(s)+s], Ch2[xi-(s):xi-(s)+s,xj:xj+s], Ch2[xi:xi+s,xj-(s):xj-(s)+s], Ch2[xi+(s):xi+(s)+s,xj:xj+s]]
                        for k in range(0,4): # Using a grid of (4) indices & 1 step block search
                            Sad = np.sum(np.abs(block1-Step3[k]))
                            if (Sad<SAD):
                                SAD = Sad
                                if k == 0: xi, xj = xi, xj+(s)
                                if k == 1: xi, xj = xi-(s), xj
                                if k == 2: xi, xj = xi, xj-(s)
                                if k == 3: xi, xj = xi+(s), xj
        Mv = [xi,xj]
        
    return Mv'''

# In[4]: Test Function
    
#Mv = MotionEstimation(Y1,Y2,1803,1812,256)
