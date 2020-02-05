/********************** TODO ********************/
//support AVX512 __m512

/********************** INCLUDES ********************/
#include <iostream>
#include <vector>
#include <chrono>
#include "image_utils.h"
/********************** DEFINES ********************/


/********************* Name Spaces *************************/
using namespace  std;
using namespace std::chrono;

/********************** Static Variables ********************/
// 3 channel YUV images
static tImage<unsigned char> bufYUV;
static tImage<unsigned char> bufYUV422;

// 3 channel RGB images
static tImage<unsigned char> bufRGB;
static tImage<unsigned char> bufRGB1;

// SIMD buffers
static unsigned char SIMD_bufRGB[IMAGE_WIDTH*IMAGE_HEIGH*IMAGE_CH + IMAGE_OFF];
static unsigned char SIMD_bufRGB1[IMAGE_WIDTH*IMAGE_HEIGH*IMAGE_CH + IMAGE_OFF];
static unsigned char SIMD_bufYUV[IMAGE_WIDTH*IMAGE_HEIGH*IMAGE_CH + IMAGE_OFF];
static unsigned char SIMD_bufYUV422[IMAGE_WIDTH*IMAGE_HEIGH*IMAGE_CH + IMAGE_OFF];

static float fSIMD_bufYUV[IMAGE_WIDTH*IMAGE_HEIGH*IMAGE_CH + IMAGE_OFF];
static float fSIMD_bufYUV422[IMAGE_WIDTH*IMAGE_HEIGH*IMAGE_CH + IMAGE_OFF];


static double psnr;
// file name and color type, parsed from command line
static string fileName1;
static Image_Type imType;

int main(int argc, char *argv[])
{
    imType = BMP;
    bool isGray = false;
    fileName1 = "test.bmp";
    cout<<"[+]Reading Image\n";
    if(!readImage(fileName1,bufRGB,imType,isGray))
    {
        std::cerr<<"Failed to open "<<fileName1<<std::endl;
        return -1;
    }
    cout<<"[+]Converting to YUV\n";
    auto start = high_resolution_clock::now();
    converRGBtoYUV(bufRGB,bufYUV);
    auto stop = high_resolution_clock::now();
    convertYUVtoYUV422(bufYUV,bufYUV422,AVERAGE);
    auto stop1 = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    auto duration1 = duration_cast<milliseconds>(stop1 - start);
    cout<<"[+]Elapsed for YUV "<<duration.count() << " ms" <<endl;
    cout<<"[+]Elapsed for YUV422 "<<duration1.count() << " ms" <<endl;

    cout<<"[+]Converting to RGB\n";
    start = high_resolution_clock::now();
    convertYUVtoRGB(bufYUV,bufRGB1);
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    cout<<"[+]Elapsed "<<duration.count() <<" ms" <<endl;
    cout<<"[+]Calculating PSNR\n";
    psnr = calculatePSNR(bufRGB1,bufRGB);
    cout<<"psnr "<<psnr<<endl;
    if(!writeImage(std::string(fileName1+"_converted.bmp"),bufRGB1,imType))
    {
        std::cerr<<"Failed to write image"<<std::endl;
        return -1;
    }

    /***************** Testing SIMD ******************************/
    unsigned int  len = (IMAGE_WIDTH*IMAGE_HEIGH*IMAGE_CH + IMAGE_OFF);
    cout<<"[+]Testing using SIMD\n";
    //read Image from buffer directly so data are next to each other
    readImageSimd(fileName1,(char*) SIMD_bufRGB);
    cout<<"[+]Converting to YUV\n";
    start = high_resolution_clock::now();
    converRGBtoYUV(SIMD_bufRGB,SIMD_bufYUV,len);
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    cout<<"[+]Elapsed for YUV "<<duration.count() << " ms" <<endl;
    start = high_resolution_clock::now();
    converRGBtoYUV422(SIMD_bufRGB,SIMD_bufYUV422,len);
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    cout<<"[+]Elapsed for YUV422 "<<duration.count() << " ms" <<endl;
    cout<<"[+]Converting to RGB\n";
    start = high_resolution_clock::now();
    convertYUVtoRGB(SIMD_bufYUV422,SIMD_bufRGB1,len);
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    cout<<"[+]Elapsed "<<duration.count() <<" ms" <<endl;
    cout<<"[+]Calculating PSNR\n";
    psnr = calculatePSNR((unsigned char*)SIMD_bufRGB1,(unsigned char*)SIMD_bufRGB,len);
    cout<<"psnr "<<psnr<<endl;
    writeImageSimd(fileName1+"Testing_SIMD.bmp",(char*)SIMD_bufRGB1);

    /***************** floatingTesting SIMD ******************************/
    cout<<"[+]Testing using floating point SIMD"<<endl;
    cout<<"[+]Converting to YUV\n";
    start = high_resolution_clock::now();
    converRGBtoYUV(SIMD_bufRGB,fSIMD_bufYUV,len);
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    cout<<"[+]Elapsed for YUV "<<duration.count() << " ms" <<endl;
    start = high_resolution_clock::now();
    converRGBtoYUV422(SIMD_bufRGB,fSIMD_bufYUV,len);
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    cout<<"[+]Elapsed for YUV422 "<<duration.count() << " ms" <<endl;
    cout<<"[+]Converting to RGB\n";
    start = high_resolution_clock::now();
    convertYUVtoRGB(fSIMD_bufYUV,SIMD_bufRGB1,len);
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    cout<<"[+]Elapsed "<<duration.count() <<" ms" <<endl;
    cout<<"[+]Calculating PSNR\n";
    psnr = calculatePSNR((unsigned char*)SIMD_bufRGB1,(unsigned char*)SIMD_bufRGB,len);
    cout<<"psnr "<<psnr<<endl;
    writeImageSimd(fileName1+"Testing_fSIMD.bmp",(char*)SIMD_bufRGB1);



    return 0;
}
