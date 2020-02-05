#include <stdio.h>
#include <iostream>     
#include <fstream>
#include <chrono>

#define IMAGE_WIDTH 3840
#define IMAGE_HEIGH 2160
#define IMAGE_CH 3u
#define IMAGE_OFF 54u


__global__
void kernelYUV2RGB(unsigned char *a, unsigned char *b) {
    int i = 3*blockIdx.x;
	int c = b[i+0] - 16;
    int d = b[i+1] - 128;
    int e = b[i+2] - 128;
    a[i+0]=( 298 * c + 409 * e + 128) >> 8;
    a[i+1]= ( 298 * c - 100 * d - 208*e + 128) >> 8;
    a[i+2] = ( 298 * c + 516 * d + 128) >> 8;
}

__global__
void kernelRGB2YUV(unsigned char *a, unsigned char *db) {
	int i = 3*blockIdx.x;
	int r = db[i+0];
    int g = db[i+1];
    int b = db[i+2];
	a[i+0] = ((66*r + 129*g +25*b + 128) >> 8 ) + 16 ;
	a[i+1] = ((-38*r - 74*g +112*b + 128) >> 8 ) + 128 ;
	a[i+2] = ((112*r - 94*g  -18*b + 128) >> 8 ) + 128 ;
}

__global__
void kernelRGB2YUV422(unsigned char *a, unsigned char *db) {
	int i = 3*blockIdx.x;
	int r = db[i+0];
    int g = db[i+1];
    int b = db[i+2];
	int r1 = db[i+0+3];
    int g1 = db[i+1+3];
    int b1 = db[i+2+3];
	int u1,u2;
	a[i+0] = ((66*r + 129*g +25*b + 128) >> 8 ) + 16 ;
	a[i+0+3] = ((66* r1 + 129*g1 +25* b1 + 128) >> 8 ) + 16 ;
	u1 = ((-38*r - 74*g +112*b + 128) >> 8 ) + 128 ;
	u2 = ((-38*r1 - 74* g1 +112*b1 + 128) >> 8 ) + 128 ;
	a[i+1] = a[i+1+3] = (u1+u2)/2;
	u1 = ((112*r - 94*g  -18*b + 128) >> 8 ) + 128 ;
	u2 = ((112*r1 - 94* g1  -18*b1 + 128) >> 8 ) + 128 ;
	a[i+2] = a[i+2+3] = (u1+u2)/2;	
}
using namespace std;
using namespace std::chrono;

double calculatePSNR(unsigned char* bufYUV, unsigned char* bufRGB, unsigned int len)
{
    double MSER = 0;
    double MSEG = 0;
    double MSEB = 0;
    double MSE = 0;
    double PSNR =0;

    for(unsigned int i = 0 ; i < len;i++)
    {
        MSE += pow((bufYUV[i] -bufRGB[i]),2);
    }
    MSE = (MSE)/(len);
    PSNR = 10*log10((255.0*255.0)/MSE);
    std::cout<<"MSE "<<MSE<<std::endl;
    return PSNR;
}

bool readImageSimd(std::string imageName ,  char* buf )
{
    bool retVal = true ;
    int off = 0;
    int wdth = 0 ;
    int height = 0;
    std::ifstream ImageFile(imageName, std::ios::binary | std::ios::ate);
    if(ImageFile.fail())
        return false;
    //temproray buffer to hold the image as char buffer
    auto sz= ImageFile.tellg();
    //temproray buffer to hold the image as char buffer
    std::cout<<"reading "<<sz<<std::endl;
    ImageFile.seekg(0, std::ios::beg);
    ImageFile.read(buf, sz);
    ImageFile.close();
    return  true;
}
void converRGBtoYUV(unsigned char* bufRGB,unsigned char *bufYUV,unsigned int len)
{
    int off = IMAGE_OFF;
    for(int i = 0 ; i < off ;i++)
        bufYUV[i] = bufRGB[i];
    //for each pixel
    for(int i = off ; i < len ; i+=3)
    {
        int r = bufRGB[i+0];
        int g = bufRGB[i+1];
        int b = bufRGB[i+2];

        bufYUV[i+0] = ((66*r + 129*g +25*b + 128) >> 8 ) + 16 ;
        bufYUV[i+1] = ((-38*r - 74*g +112*b + 128) >> 8 ) + 128 ;
        bufYUV[i+2] = ((112*r - 94*g  -18*b + 128) >> 8 ) + 128 ;
    }
}
static unsigned char SIMD_bufRGB[IMAGE_WIDTH*IMAGE_HEIGH*IMAGE_CH + IMAGE_OFF];
static unsigned char SIMD_bufRGB1[IMAGE_WIDTH*IMAGE_HEIGH*IMAGE_CH + IMAGE_OFF];
static unsigned char SIMD_bufYUV[IMAGE_WIDTH*IMAGE_HEIGH*IMAGE_CH + IMAGE_OFF];

int main() {
	string fileName1 = "testo.bmp";

	unsigned int  len = (IMAGE_WIDTH*IMAGE_HEIGH*IMAGE_CH + IMAGE_OFF);
	unsigned char *da, *db;
	cudaMalloc((void **)&da, len*sizeof(char));
    cudaMalloc((void **)&db, len*sizeof(char));

	
    cout<<"[+]Testing using SIMD\n";
    //read Image from buffer directly so data are next to each other
    readImageSimd(fileName1,(char*) SIMD_bufRGB);
    cout<<"[+]Converting to YUV\n";
	auto start = high_resolution_clock::now();
	cudaMemcpy(db, SIMD_bufRGB, len*sizeof(char), cudaMemcpyHostToDevice);   
    kernelRGB2YUV<<<IMAGE_WIDTH*IMAGE_HEIGH, 1>>>(da, db);
	cudaMemcpy(SIMD_bufYUV, da, len*sizeof(char), cudaMemcpyDeviceToHost);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout<<"[+]Elapsed "<<duration.count() << " ms" <<endl;
    //converRGBtoYUV(SIMD_bufRGB,SIMD_bufYUV,len);
	
	cout<<"[+]Converting to YUV422\n";
	start = high_resolution_clock::now();
	cudaMemcpy(db, SIMD_bufRGB, len*sizeof(char), cudaMemcpyHostToDevice);   
    kernelRGB2YUV422<<<(IMAGE_WIDTH*IMAGE_HEIGH)/2, 1>>>(da, db);
	cudaMemcpy(SIMD_bufYUV, da, len*sizeof(char), cudaMemcpyDeviceToHost);
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    
	cout<<"[+]Converting to RGB\n";
	start = high_resolution_clock::now();
	cudaMemcpy(db, SIMD_bufYUV, len*sizeof(char), cudaMemcpyHostToDevice);   
    kernelYUV2RGB<<<IMAGE_WIDTH*IMAGE_HEIGH, 1>>>(da, db);
	cudaMemcpy(SIMD_bufRGB1, da, len*sizeof(char), cudaMemcpyDeviceToHost);
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    cout<<"[+]Elapsed "<<duration.count() << " ms" <<endl;
    cudaFree(da);
    cudaFree(db);
	for(int i = 0 ; i < IMAGE_OFF ; i++)
		SIMD_bufRGB1[i] = SIMD_bufYUV[i];

	cout<<"[+]Calculating PSNR\n";
    double psnr = calculatePSNR((unsigned char*)SIMD_bufRGB1,(unsigned char*)SIMD_bufRGB,len);
    cout<<"psnr "<<psnr<<endl;

    return 0;
}