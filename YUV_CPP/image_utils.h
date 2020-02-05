#ifndef IMAGE_UTILS_H  /*IMAGE_UTILS_H*/
#define IMAGE_UTILS_H
#include <string>
#include <vector>
#include <math.h>
#include <iostream>
#include <immintrin.h>

/**************************** CONFIGURATIOS ****************************/
#define IMAGE_WIDTH 3840
#define IMAGE_HEIGH 2160
#define IMAGE_CH 3u
#define IMAGE_OFF 54u

/************************* MACROS **************************************/
#define SATURATE(x)  ( (x<0)?(0):((x>255)?(255):x) )

#define YUV_Y 0
#define YUV_U 1
#define YUV_V 2
#define RGB_R 0
#define RGB_G 1
#define RGB_B 2

/*********************************** TYPES *****************************/

enum YUV_Conversion {AVERAGE,SUBSAMPELING};
enum Image_Type {RAW,BMP};
template <typename T>
struct tImage {
    std::vector<std::vector<T>> buf[3];
    int width;
    int height;
    unsigned char meta[54];
    bool isGray;
};

/*********************** FUNCTIONS ************************************/
bool readImage(std::string imageName,tImage<unsigned char>& buf,Image_Type imtype,bool gray );

bool writeImage(std::string imageName,tImage<unsigned char>& buf,Image_Type imtype);

bool readImageSimd(std::string imageName,char* );

bool writeImageSimd(std::string imageName,char*);

void converRGBtoYUV(unsigned char* bufRGB,unsigned char*  bufYUV,unsigned int len);
void converRGBtoYUV(unsigned char* bufRGB,float*  bufYUV,unsigned int len);

void convertYUVtoRGB(float* bufYUV,unsigned char* bufRGB,unsigned int len);
void converRGBtoYUV_ref(unsigned char* bufRGB,float*  bufYUV,unsigned int len);
void convertYUVtoRGB_ref(float* bufYUV,unsigned char* bufRGB,unsigned int len);

void convertYUVtoRGB(unsigned char* bufYUV,unsigned char* bufRGB,unsigned int len);
void convertYUVtoRGB_ref(unsigned char* bufYUV,unsigned char* bufRGB,unsigned int len);
void converRGBtoYUV422(unsigned char* bufRGB,unsigned char *bufYUV,unsigned int len);
void converRGBtoYUV422(unsigned char* bufRGB,float*  bufYUV,unsigned int len);

template <typename T>
void converRGBtoYUV(tImage<T> &bufRGB,tImage<T> &bufYUV);

template <typename T>
void convertYUVtoRGB(tImage<T> &bufYUV,tImage<T> &bufRGB);
void convertYUVtoYUV422(tImage<unsigned char> &bufYUV,tImage<unsigned char> &bufYUV422, enum YUV_Conversion ConvType);
double calculatePSNR(unsigned char* bufYUV, unsigned char* bufRGB, unsigned int len);
void convertYUVtoYUV422(unsigned char* bufYUV, unsigned char* bufYUV422 , int len);
void convertYUVtoYUV422(float* bufYUV, float* bufYUV422 , int len);


template <typename T>
double calculatePSNR(tImage<T> &bufYUV, tImage<T> &bufRGB)
{
    double MSER = 0;
    double MSEG = 0;
    double MSEB = 0;
    double MSE = 0;
    double PSNR =0;

    for(int i = 0 ; i < bufYUV.height;i++)
    {
        for(int j = 0 ; j < bufYUV.width ; j++)
        {
            MSER = pow((bufYUV.buf[YUV_Y][i][j] -bufRGB.buf[RGB_R][i][j]),2);
            MSEG = pow((bufYUV.buf[YUV_U][i][j] -bufRGB.buf[RGB_G][i][j]),2);
            MSEB = pow((bufYUV.buf[YUV_V][i][j] -bufRGB.buf[RGB_B][i][j]),2);
            MSE+=MSER+MSEG+MSEB;
        }
    }
    MSE = (MSE)/(3.0*bufYUV.width*bufYUV.height);
    PSNR = 10*log10((255.0*255.0)/MSE);
    std::cout<<"MSE "<<MSE<<std::endl;
    return PSNR;
}

template <typename T>
void converRGBtoYUV(tImage<T> &bufRGB,tImage<T> &bufYUV)
{
    bufYUV.buf[YUV_Y].resize(bufRGB.height);
    bufYUV.buf[YUV_U].resize(bufRGB.height);
    bufYUV.buf[YUV_V].resize(bufRGB.height);
    bufYUV.width = bufRGB.width;
    bufYUV.height = bufRGB.height;
    bufYUV.isGray = bufRGB.isGray;
    for(int i = 0 ; i < 54;i++)
        bufYUV.meta[i] = bufRGB.meta[i];

    for(int i = 0 ; i < bufYUV.height ; i++)
    {
        bufYUV.buf[YUV_Y][i].resize(bufYUV.width);
        bufYUV.buf[YUV_U][i].resize(bufYUV.width);
        bufYUV.buf[YUV_V][i].resize(bufYUV.width);
    }

    for(int i = 0 ; i < bufYUV.height; i++)
    {
        for(int j = 0 ; j < bufYUV.width ; j++)
        {
            int r = bufRGB.buf[RGB_R][i][j];
            int g = bufRGB.buf[RGB_G][i][j];
            int b = bufRGB.buf[RGB_B][i][j];

            bufYUV.buf[YUV_Y][i][j] = ((66*r + 129*g +25*b + 128) >> 8 ) + 16 ;
            bufYUV.buf[YUV_U][i][j] = ((-38*r - 74*g +112*b + 128) >> 8 ) + 128 ;
            bufYUV.buf[YUV_V][i][j] = ((112*r - 94*g  -18*b + 128) >> 8 ) + 128 ;
        }
    }
}
template <typename T>
void convertYUVtoRGB(tImage<T> &bufYUV,tImage<T> &bufRGB)
{
    bufRGB.buf[RGB_R].resize(bufYUV.height);
    bufRGB.buf[RGB_G].resize(bufYUV.height);
    bufRGB.buf[RGB_B].resize(bufYUV.height);
    bufRGB.width = bufYUV.width;
    bufRGB.height = bufYUV.height;
    bufRGB.isGray = bufYUV.isGray;
    for(int i = 0 ; i < 54;i++)
        bufRGB.meta[i] = bufYUV.meta[i];

    for(int i = 0 ; i < bufRGB.height ; i++)
    {
        bufRGB.buf[RGB_R][i].resize(bufRGB.width);
        bufRGB.buf[RGB_G][i].resize(bufRGB.width);
        bufRGB.buf[RGB_B][i].resize(bufRGB.width);
    }

    for(int i = 0 ; i < bufRGB.height; i++)
    {
        for(int j = 0 ; j < bufRGB.width;j++)
        {
            int c = bufYUV.buf[YUV_Y][i][j] - 16;
            int d = bufYUV.buf[YUV_U][i][j] - 128;
            int e = bufYUV.buf[YUV_V][i][j] - 128;
            bufRGB.buf[RGB_R][i][j]=( 298 * c + 409 * e + 128) >> 8;
            bufRGB.buf[RGB_G][i][j]= ( 298 * c - 100 * d - 208*e + 128) >> 8;
            bufRGB.buf[RGB_B][i][j] = ( 298 * c + 516 * d + 128) >> 8;
        }
    }
}

#endif /*IMAGE_UTILS_H*/
