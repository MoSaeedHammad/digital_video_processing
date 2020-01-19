import time
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image
from 

if __name__ == "__main__":

    mode_list = ['4:4:4','4:2:2', '4:2:0', '4:1:1', '4:4:0','4:2:0_Average', '4:2:0_Median', '4:2:2_Average', '4:1:1_Average', '4:1:1_Median']
    img_list = ['f0.bmp']
    fig, (ax1, ax2) = plt.subplots(2,2, figsize=(20,20))
    for image_name in img_list:
        RGB_img = Image.open("video_input/"+image_name)
        start_time=time.time()
        YUV = Convert_RGB_YUV(RGB_img)
        print("RGB->YUV444 Conversion Execution Time : {0} seconds".format((time.time() - start_time)))
        for Mode in mode_list:
            
            Compressed_YUV = COLOR_Compression(YUV,Mode)
            
            start_time=time.time()
            Reconstructed_RGB = Convert_YUV_RGB(Compressed_YUV, True)
            #print("Reconstructed_RGB Conversion Execution Time : {} seconds".format((time.time() - start_time)))


            MSE = np.mean( (RGB_img - Reconstructed_RGB)**2 )

            if MSE == 0:
                PSNR = np.Infinity
            else:
                PSNR = 20*math.log10(255.0/math.sqrt(MSE))

            diff_mat = RGB_img-Reconstructed_RGB

            plt.ion()
            fig.suptitle('Color Compression mode: '+str(Mode)+', PSNR: '+str(PSNR)+' ,MSE: '+str(MSE), fontsize = 16)
            ax1[0].imshow(np.uint8(RGB_img))
            ax1[0].title.set_text('Original RGB image')
            ax1[1].imshow(np.uint8(Compressed_YUV))
            ax1[1].title.set_text('YUV Image')
            ax2[0].imshow(np.uint8(Reconstructed_RGB))
            ax2[0].title.set_text('Reconstructed RGB')
            ax2[1].imshow(np.uint8(diff_mat))
            ax2[1].title.set_text('Difference')
#             plt.show()
#             plt.pause(0.05)
            nimage_name = image_name.split('.')[0]
            Mode = Mode.replace(':','_')
            plt.savefig("csc_output/"+str(Mode)+'_'+nimage_name+'.png', format='png')