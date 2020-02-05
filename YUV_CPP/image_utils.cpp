#include <string>
#include <iostream>
#include <fstream>
#include "image_utils.h"

bool readImage(std::string imageName,tImage<unsigned char>& bufImage,Image_Type imtype,bool gray )
{
    bool retVal = true ;
    int off = 0;
    int wdth = 0 ;
    int height = 0;
    std::ifstream ImageFile(imageName, std::ios::binary | std::ios::ate);
    if(ImageFile.fail())
        return false;
    std::streamsize ImageSize = ImageFile.tellg();
    //temproray buffer to hold the image as char buffer
    std::vector<char> temp_buffer(ImageSize);
    std::cout<<"reading "<<temp_buffer.size()<<std::endl;

    ImageFile.seekg(0, std::ios::beg);
    if (!ImageFile.read(temp_buffer.data(), ImageSize))
    {
        return  false;

    }
    ImageFile.close();
    bufImage.isGray =gray;
    //copy to temprory buffer
    if(imtype == BMP)
    {
        off=54; //skip BMP header
        for(int j = 0 ; j < off ; j++)
        {
            bufImage.meta[j] = temp_buffer[j];
        }
        wdth  =*reinterpret_cast<uint32_t *>(&bufImage.meta[18]);
        height = *reinterpret_cast<uint32_t *>(&bufImage.meta[22]);
        bufImage.width = wdth;
        bufImage.height = height;
        std::cout<<"Image width "<<wdth<<std::endl;
        std::cout<<"Image Height "<<height<<std::endl;

    }

    bufImage.buf[RGB_R].resize(height);
    bufImage.buf[RGB_G].resize(height);
    bufImage.buf[RGB_B].resize(height);

    for(int i = 0 ; i < height ; i++)
    {
        bufImage.buf[RGB_R][i].resize(wdth);
        bufImage.buf[RGB_G][i].resize(wdth);
        bufImage.buf[RGB_B][i].resize(wdth);
    }


    if(gray)
    {
        for(int i = 0 ; i < height ; i++ )
        {
            for(int j = 0 ; j < wdth;j++)
            {
                bufImage.buf[RGB_R][i][j] =  (unsigned char)temp_buffer[i*height+j+off];
                bufImage.buf[RGB_G][i][j] =  (unsigned char)temp_buffer[i*height+j+off];
                bufImage.buf[RGB_B][i][j] =  (unsigned char)temp_buffer[i*height+j+off];

            }
        }
    }
    else
    {
        for(int i = 0 ; i < height ; i++ )
        {
            for(int j = 0 ; j < wdth;j++)
            {
                bufImage.buf[RGB_R][i][j] =  (unsigned char)temp_buffer[3*(i*wdth+j)+off];
                bufImage.buf[RGB_G][i][j]=  (unsigned char)temp_buffer[3*(i*wdth+j)+off+1];
                bufImage.buf[RGB_B][i][j] =  (unsigned char)temp_buffer[3*(i*wdth+j)+off+2];

            }
        }
    }
    return retVal;
}
bool writeImage(std::string imageName,tImage<unsigned char>& bufImage,Image_Type imtype)
{
    std::ofstream ImageFile(imageName,std::ofstream::binary);
    int off = 0 ;
    if(imtype == BMP)
    {
        off = 54;
    }
    std::vector<char> temproray_buffer;
    if(bufImage.isGray)
        temproray_buffer.resize(bufImage.width*bufImage.height+off);
    else
        temproray_buffer.resize(3*bufImage.width*bufImage.height+off);

    std::cout<<"writing "<<temproray_buffer.size()<<std::endl;
    for(int i = 0 ; i < off;i++)
    {
        temproray_buffer[i] =  bufImage.meta[i];
    }
    if(bufImage.isGray)
    {
        for(int i = 0 ; i < bufImage.height ; i++ )
        {
            for(int j = 0 ; j < bufImage.width;j++)
            {
                temproray_buffer[i*bufImage.height + j +off] = bufImage.buf[RGB_R][i][j] ;
            }
        }
    }
    else
    {
        for(int i = 0 ; i < bufImage.height ; i++ )
        {
            for(int j = 0 ; j < bufImage.width;j++)
            {
                temproray_buffer[3*(i*bufImage.width+j) +off] = bufImage.buf[RGB_R][i][j];
                temproray_buffer[3*(i*bufImage.width+j) +off + 1] = bufImage.buf[RGB_G][i][j];
                temproray_buffer[3*(i*bufImage.width+j) +off + 2 ] = bufImage.buf[RGB_B][i][j];
            }
        }
    }

    ImageFile.write (temproray_buffer.data(),temproray_buffer.size());
    ImageFile.close();
    return true;
}

//simplified function to read images as it is for SIMD processing
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
bool writeImageSimd(std::string imageName, char*buf)
{
    std::ofstream ImageFile(imageName,std::ofstream::binary);
    ImageFile.write (buf,IMAGE_WIDTH*IMAGE_HEIGH*IMAGE_CH  + IMAGE_OFF);
    ImageFile.close();
    return true;
}

__m256i mul_epi8_to_16(__m128i v0, __m256i v1)
{
    __m256i tmp0 = _mm256_cvtepi8_epi16 (v0); //printf("\ntmp0 = ");_mm256_print_epi16(tmp0);
    __m256i tmp1 = _mm256_mullo_epi16(tmp0, v1);
    return tmp1;
}
void converRGBtoYUV422(unsigned char* bufRGB,unsigned char *bufYUV,unsigned int len)
{
    int off = IMAGE_OFF;
    for(int i = 0 ; i < off ;i++)
        bufYUV[i] = bufRGB[i];
    //for each pixel

    __m128i ssse3_red_indeces_0 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 15, 12, 9, 6, 3, 0);
    __m128i ssse3_red_indeces_1 = _mm_set_epi8(-1, -1, -1, -1, -1, 14, 11, 8, 5, 2, -1, -1, -1, -1, -1, -1);
    __m128i ssse3_red_indeces_2 = _mm_set_epi8(13, 10, 7, 4, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    __m128i ssse3_green_indeces_0 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 13, 10, 7, 4, 1);
    __m128i ssse3_green_indeces_1 = _mm_set_epi8(-1, -1, -1, -1, -1, 15, 12, 9, 6, 3, 0, -1, -1, -1, -1, -1);
    __m128i ssse3_green_indeces_2 = _mm_set_epi8(14, 11, 8, 5, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    __m128i ssse3_blue_indeces_0 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 14, 11, 8, 5, 2);
    __m128i ssse3_blue_indeces_1 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, 13, 10, 7, 4, 1, -1, -1, -1, -1, -1);
    __m128i ssse3_blue_indeces_2 = _mm_set_epi8(15, 12, 9, 6, 3, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    __m128i  consts_16 = _mm_set1_epi8(16);
    __m128i  consts_128 =_mm_set1_epi8(128);
    __m256i  consts_66 =_mm256_set1_epi16(66);
    __m256i  consts_129 =_mm256_set1_epi16(129);
    __m256i  consts_25 =_mm256_set1_epi16(25);
    __m256i  consts_38 =_mm256_set1_epi16(38);
    __m256i  consts_74 =_mm256_set1_epi16(74);
    __m256i  consts_112 =_mm256_set1_epi16(112);
    __m256i  consts_94 =_mm256_set1_epi16(94);
    __m256i  consts_18 =_mm256_set1_epi16(18);
    __m256i  consts_128_2 =_mm256_set1_epi16(128);
    __m256i  consts_8_2 =_mm256_set1_epi16(8);
    __m256i  consts_16_2 =_mm256_set1_epi16(16);

    __m128i  consts_8 =_mm_set_epi64x(0,8);
    for(unsigned int i = off ; i < len ; i+=48)
    {
        const __m128i red_chunk0 = _mm_loadu_si128((const __m128i*)(bufRGB+ i));
        const __m128i red_chunk1 = _mm_loadu_si128((const __m128i*)(bufRGB+ i + 16));
        const __m128i red_chunk2 = _mm_loadu_si128((const __m128i*)(bufRGB+ i + 32));
        const __m128i red = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(red_chunk0, ssse3_red_indeces_0),
                                                      _mm_shuffle_epi8(red_chunk1, ssse3_red_indeces_1)), _mm_shuffle_epi8(red_chunk2, ssse3_red_indeces_2));
        const __m128i green_chunk0 = _mm_loadu_si128((const __m128i*)(bufRGB+ i));
        const __m128i green_chunk1 = _mm_loadu_si128((const __m128i*)(bufRGB+ i + 16));
        const __m128i green_chunk2 = _mm_loadu_si128((const __m128i*)(bufRGB+ i + 32));
        const __m128i green = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(green_chunk0, ssse3_green_indeces_0),
                                                        _mm_shuffle_epi8(green_chunk1, ssse3_green_indeces_1)), _mm_shuffle_epi8(green_chunk2, ssse3_green_indeces_2));
        const __m128i blue_chunk0 = _mm_loadu_si128((const __m128i*)(bufRGB+ i));
        const __m128i blue_chunk1 = _mm_loadu_si128((const __m128i*)(bufRGB+ i + 16));
        const __m128i blue_chunk2 = _mm_loadu_si128((const __m128i*)(bufRGB+ i + 32));
        const __m128i blue = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(blue_chunk0, ssse3_blue_indeces_0),
                                                       _mm_shuffle_epi8(blue_chunk1, ssse3_blue_indeces_1)), _mm_shuffle_epi8(blue_chunk2, ssse3_blue_indeces_2));


        //r
        __m256i x1 = mul_epi8_to_16(red,consts_66);
        __m256i x2 = mul_epi8_to_16(green,consts_129);
        __m256i x3 = mul_epi8_to_16(blue,consts_25);

        __m256i x4 = _mm256_add_epi16(x1,x2);
        x4 = _mm256_add_epi16(x4,x3);
        x4 = _mm256_add_epi16(x4,consts_128_2);
        __m256i x5 = _mm256_srl_epi16(x4,consts_8);
        x5 = _mm256_add_epi16(x5,consts_16_2);
        unsigned short toto[16];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(toto), x5);
        for(int j = 0 ; j < 16 ; j++ )
        {
            bufYUV[i+(3*j)] = toto[j];
        }
        //g
        x1= mul_epi8_to_16(red,consts_38);
        x2= mul_epi8_to_16(green,consts_74);
        x4= mul_epi8_to_16(blue,consts_112);
        x3 = _mm256_sub_epi16(x4,x1);
        x3 = _mm256_sub_epi16(x3,x2);
        x4 = _mm256_add_epi16(x3,consts_128_2);
        x5 = _mm256_srl_epi16(x4,consts_8);
        x5 = _mm256_add_epi16(x5,consts_128_2);
        __m256i x6 = _mm256_srl_epi16(x5,_mm_set1_epi8(1));
        x6 = _mm256_hadd_epi16(x6,x6);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(toto), x6);
        for(int j = 0 ; j < 16 ; j++ )
        {
            bufYUV[i+(3*j)+1] = toto[j];
        }

        //b

        x1= mul_epi8_to_16(red,consts_112);
        x2= mul_epi8_to_16(green,consts_94);
        x3= mul_epi8_to_16(blue,consts_18);
        x4 = _mm256_sub_epi16(x1,x2);
        x4 = _mm256_sub_epi16(x4,x3);
        x4 = _mm256_add_epi16(x4,consts_128_2);
        x5 = _mm256_srl_epi16(x4,consts_8);
        x5 = _mm256_add_epi16(x5,consts_128_2);

        x6 = _mm256_srl_epi16(x5,_mm_set1_epi8(1));
        x6 = _mm256_hadd_epi16(x6,x6);

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(toto), x6);

        for(int j = 0 ; j < 16 ; j++ )
        {
            bufYUV[i+(3*j)+2] = toto[j];
        }
    }
}

void converRGBtoYUV(unsigned char* bufRGB,unsigned char *bufYUV,unsigned int len)
{
    int off = IMAGE_OFF;
    for(int i = 0 ; i < off ;i++)
        bufYUV[i] = bufRGB[i];
    //for each pixel

    __m128i ssse3_red_indeces_0 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 15, 12, 9, 6, 3, 0);
    __m128i ssse3_red_indeces_1 = _mm_set_epi8(-1, -1, -1, -1, -1, 14, 11, 8, 5, 2, -1, -1, -1, -1, -1, -1);
    __m128i ssse3_red_indeces_2 = _mm_set_epi8(13, 10, 7, 4, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    __m128i ssse3_green_indeces_0 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 13, 10, 7, 4, 1);
    __m128i ssse3_green_indeces_1 = _mm_set_epi8(-1, -1, -1, -1, -1, 15, 12, 9, 6, 3, 0, -1, -1, -1, -1, -1);
    __m128i ssse3_green_indeces_2 = _mm_set_epi8(14, 11, 8, 5, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    __m128i ssse3_blue_indeces_0 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 14, 11, 8, 5, 2);
    __m128i ssse3_blue_indeces_1 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, 13, 10, 7, 4, 1, -1, -1, -1, -1, -1);
    __m128i ssse3_blue_indeces_2 = _mm_set_epi8(15, 12, 9, 6, 3, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    __m128i  consts_16 = _mm_set1_epi8(16);
    __m128i  consts_128 =_mm_set1_epi8(128);
    __m256i  consts_66 =_mm256_set1_epi16(66);
    __m256i  consts_129 =_mm256_set1_epi16(129);
    __m256i  consts_25 =_mm256_set1_epi16(25);
    __m256i  consts_38 =_mm256_set1_epi16(38);
    __m256i  consts_74 =_mm256_set1_epi16(74);
    __m256i  consts_112 =_mm256_set1_epi16(112);
    __m256i  consts_94 =_mm256_set1_epi16(94);
    __m256i  consts_18 =_mm256_set1_epi16(18);
    __m256i  consts_128_2 =_mm256_set1_epi16(128);
    __m256i  consts_8_2 =_mm256_set1_epi16(8);
    __m256i  consts_16_2 =_mm256_set1_epi16(16);

    __m128i  consts_8 =_mm_set_epi64x(0,8);
    for(unsigned int i = off ; i < len ; i+=48)
    {
        const __m128i red_chunk0 = _mm_loadu_si128((const __m128i*)(bufRGB+ i));
        const __m128i red_chunk1 = _mm_loadu_si128((const __m128i*)(bufRGB+ i + 16));
        const __m128i red_chunk2 = _mm_loadu_si128((const __m128i*)(bufRGB+ i + 32));
        const __m128i red = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(red_chunk0, ssse3_red_indeces_0),
                                                      _mm_shuffle_epi8(red_chunk1, ssse3_red_indeces_1)), _mm_shuffle_epi8(red_chunk2, ssse3_red_indeces_2));
        const __m128i green_chunk0 = _mm_loadu_si128((const __m128i*)(bufRGB+ i));
        const __m128i green_chunk1 = _mm_loadu_si128((const __m128i*)(bufRGB+ i + 16));
        const __m128i green_chunk2 = _mm_loadu_si128((const __m128i*)(bufRGB+ i + 32));
        const __m128i green = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(green_chunk0, ssse3_green_indeces_0),
                                                        _mm_shuffle_epi8(green_chunk1, ssse3_green_indeces_1)), _mm_shuffle_epi8(green_chunk2, ssse3_green_indeces_2));
        const __m128i blue_chunk0 = _mm_loadu_si128((const __m128i*)(bufRGB+ i));
        const __m128i blue_chunk1 = _mm_loadu_si128((const __m128i*)(bufRGB+ i + 16));
        const __m128i blue_chunk2 = _mm_loadu_si128((const __m128i*)(bufRGB+ i + 32));
        const __m128i blue = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(blue_chunk0, ssse3_blue_indeces_0),
                                                       _mm_shuffle_epi8(blue_chunk1, ssse3_blue_indeces_1)), _mm_shuffle_epi8(blue_chunk2, ssse3_blue_indeces_2));


        //r
        __m256i x1 = mul_epi8_to_16(red,consts_66);
        __m256i x2 = mul_epi8_to_16(green,consts_129);
        __m256i x3 = mul_epi8_to_16(blue,consts_25);

        __m256i x4 = _mm256_add_epi16(x1,x2);
        x4 = _mm256_add_epi16(x4,x3);
        x4 = _mm256_add_epi16(x4,consts_128_2);
        __m256i x5 = _mm256_srl_epi16(x4,consts_8);
        x5 = _mm256_add_epi16(x5,consts_16_2);
        unsigned short toto[16];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(toto), x5);
        for(int j = 0 ; j < 16 ; j++ )
        {
            bufYUV[i+(3*j)] = toto[j];
        }
        //g
        x1= mul_epi8_to_16(red,consts_38);
        x2= mul_epi8_to_16(green,consts_74);
        x4= mul_epi8_to_16(blue,consts_112);
        x3 = _mm256_sub_epi16(x4,x1);
        x3 = _mm256_sub_epi16(x3,x2);
        x4 = _mm256_add_epi16(x3,consts_128_2);
        x5 = _mm256_srl_epi16(x4,consts_8);
        x5 = _mm256_add_epi16(x5,consts_128_2);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(toto), x5);
        for(int j = 0 ; j < 16 ; j++ )
        {
            bufYUV[i+(3*j)+1] = toto[j];
        }

        //b

        x1= mul_epi8_to_16(red,consts_112);
        x2= mul_epi8_to_16(green,consts_94);
        x3= mul_epi8_to_16(blue,consts_18);
        x4 = _mm256_sub_epi16(x1,x2);
        x4 = _mm256_sub_epi16(x4,x3);
        x4 = _mm256_add_epi16(x4,consts_128_2);
        x5 = _mm256_srl_epi16(x4,consts_8);
        x5 = _mm256_add_epi16(x5,consts_128_2);

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(toto), x5);
        for(int j = 0 ; j < 16 ; j++ )
        {
            bufYUV[i+(3*j)+2] = toto[j];
        }
    }

}
void convertYUVtoRGB_ref(unsigned char* bufYUV,unsigned char *bufRGB,unsigned int len)
{
    int off = IMAGE_OFF;
    for(int i = 0 ; i < off ;i++)
        bufRGB[i] = bufYUV[i];
    //for each pixel
    for(int i = off ; i < len ; i+=3)
    {
        int c = bufYUV[i+0] - 16;
        int d = bufYUV[i+1] - 128;
        int e = bufYUV[i+2] - 128;
        bufRGB[i+0]=( 298 * c + 409 * e + 128) >> 8;
        bufRGB[i+1]= ( 298 * c - 100 * d - 208*e + 128) >> 8;
        bufRGB[i+2] = ( 298 * c + 516 * d + 128) >> 8;
    }
}


void convertYUVtoRGB(unsigned char* bufYUV,unsigned char *bufRGB,unsigned int len)
{
    int off = IMAGE_OFF;
    for(int i = 0 ; i < off ;i++)
        bufRGB[i] = bufYUV[i];
    //for each pixel

    for(unsigned int i = off ; i < len ; i+=48)
    {
        __m128i ssse3_red_indeces_0 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 15, 12, 9, 6, 3, 0);
        __m128i ssse3_red_indeces_1 = _mm_set_epi8(-1, -1, -1, -1, -1, 14, 11, 8, 5, 2, -1, -1, -1, -1, -1, -1);
        __m128i ssse3_red_indeces_2 = _mm_set_epi8(13, 10, 7, 4, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        __m128i ssse3_green_indeces_0 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 13, 10, 7, 4, 1);
        __m128i ssse3_green_indeces_1 = _mm_set_epi8(-1, -1, -1, -1, -1, 15, 12, 9, 6, 3, 0, -1, -1, -1, -1, -1);
        __m128i ssse3_green_indeces_2 = _mm_set_epi8(14, 11, 8, 5, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        __m128i ssse3_blue_indeces_0 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 14, 11, 8, 5, 2);
        __m128i ssse3_blue_indeces_1 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, 13, 10, 7, 4, 1, -1, -1, -1, -1, -1);
        __m128i ssse3_blue_indeces_2 = _mm_set_epi8(15, 12, 9, 6, 3, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        __m128i  consts_16 = _mm_set1_epi8(16);
        __m128i  consts_128 =_mm_set1_epi8(128);
        __m256i  consts_298 =_mm256_set1_epi16(298);
        __m256i  consts_409 =_mm256_set1_epi16(409);
        __m256i  consts_516 =_mm256_set1_epi16(516);
        __m256i  consts_208 =_mm256_set1_epi16(208);
        __m256i  consts_100 =_mm256_set1_epi16(100);
        __m256i  consts_128_2 =_mm256_set1_epi16(128);
        __m128i  consts_8 =_mm_set_epi64x(0,8);

        const __m128i red_chunk0 = _mm_loadu_si128((const __m128i*)(bufYUV+ i));
        const __m128i red_chunk1 = _mm_loadu_si128((const __m128i*)(bufYUV+ i + 16));
        const __m128i red_chunk2 = _mm_loadu_si128((const __m128i*)(bufYUV+ i + 32));
        const __m128i red = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(red_chunk0, ssse3_red_indeces_0),
                                                      _mm_shuffle_epi8(red_chunk1, ssse3_red_indeces_1)), _mm_shuffle_epi8(red_chunk2, ssse3_red_indeces_2));
        const __m128i green_chunk0 = _mm_loadu_si128((const __m128i*)(bufYUV+ i));
        const __m128i green_chunk1 = _mm_loadu_si128((const __m128i*)(bufYUV+ i + 16));
        const __m128i green_chunk2 = _mm_loadu_si128((const __m128i*)(bufYUV+ i + 32));
        const __m128i green = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(green_chunk0, ssse3_green_indeces_0),
                                                        _mm_shuffle_epi8(green_chunk1, ssse3_green_indeces_1)), _mm_shuffle_epi8(green_chunk2, ssse3_green_indeces_2));
        const __m128i blue_chunk0 = _mm_loadu_si128((const __m128i*)(bufYUV+ i));
        const __m128i blue_chunk1 = _mm_loadu_si128((const __m128i*)(bufYUV+ i + 16));
        const __m128i blue_chunk2 = _mm_loadu_si128((const __m128i*)(bufYUV+ i + 32));
        const __m128i blue = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(blue_chunk0, ssse3_blue_indeces_0),
                                                       _mm_shuffle_epi8(blue_chunk1, ssse3_blue_indeces_1)), _mm_shuffle_epi8(blue_chunk2, ssse3_blue_indeces_2));

        //data are orderd now
        __m128i c = _mm_sub_epi8(red ,consts_16);
        __m128i d = _mm_sub_epi8(green , consts_128);
        __m128i e = _mm_sub_epi8(blue ,consts_128) ;


        //r
        __m256i x1 = mul_epi8_to_16(c,consts_298);
        __m256i x2 = mul_epi8_to_16(e,consts_409);
        __m256i x3 = _mm256_add_epi16(x1,x2);
        __m256i x4 = _mm256_add_epi16(x3,consts_128_2);
        __m256i x5 = _mm256_srl_epi16(x4,consts_8);
        unsigned short toto[16];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(toto), x1);
        for(int j = 0 ; j < 16 ; j++ )
        {
            bufRGB[i+(3*j)] = toto[j];
        }
        //g
        x1= mul_epi8_to_16(c,consts_298);
        x2= mul_epi8_to_16(e,consts_208);
        x4= mul_epi8_to_16(d,consts_100);
        x3 = _mm256_sub_epi16(x1,x4);
        x3 = _mm256_sub_epi16(x3,x2);
        x4 = _mm256_add_epi16(x3,consts_128_2);
        x5 = _mm256_srl_epi16(x4,consts_8);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(toto), x5);
        for(int j = 0 ; j < 16 ; j++ )
        {
            bufRGB[i+(3*j)+1] = toto[j];
        }

        //b
        x1= mul_epi8_to_16(c,consts_298);
        x2= mul_epi8_to_16(d,consts_516);
        x3 = _mm256_add_epi16(x1,x2);
        x4 = _mm256_add_epi16(x3,consts_128_2);
        x5 = _mm256_srl_epi16(x4,consts_8);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(toto), x5);
        for(int j = 0 ; j < 16 ; j++ )
        {
            bufRGB[i+(3*j)+2] = toto[j];
        }
    }
}

void converRGBtoYUV(unsigned char* bufRGB,float*  bufYUV,unsigned int len)
{
    int off = IMAGE_OFF;
    for(int i = 0 ; i < off ;i++)
        bufYUV[i] = bufRGB[i];

    int x = 0xFFFFFFFF;
    float  *F_MASK = reinterpret_cast<float*>(&x);

    __m256i ssse3_red_indeces_0 = _mm256_set_epi32(-1, -1, -1, -1,-1, 6, 3, 0);
    __m256i ssse3_red_indeces_1 = _mm256_set_epi32(-1, -1, 7, 4, 1, -1, -1, -1);
    __m256i ssse3_red_indeces_2 = _mm256_set_epi32(5, 2, -1, -1, 1, -1, -1, -1);
    __m256i ssse3_green_indeces_0 = _mm256_set_epi32(-1, -1, -1, -1, -1, 7,4,1);
    __m256i ssse3_green_indeces_1 = _mm256_set_epi32(-1, -1, -1, 5,2, -1, -1, -1);
    __m256i ssse3_green_indeces_2 = _mm256_set_epi32(6, 3, 0, -1, -1, -1, -1, -1);
    __m256i ssse3_blue_indeces_0 = _mm256_set_epi32(-1, -1, -1, -1, -1, -1,5, 2);
    __m256i ssse3_blue_indeces_1 = _mm256_set_epi32(-1, -1, -1, 6, 3,0, -1, -1);
    __m256i ssse3_blue_indeces_2 = _mm256_set_epi32(7, 4, 1, -1,-1, -1, -1, -1);
    __m256  consts_0_29 = _mm256_set1_ps(0.299f);
    __m256  consts_0_58 =_mm256_set1_ps(0.587f);
    __m256  consts_0_11 =_mm256_set1_ps(0.114f);
    __m256  consts_0_14 =_mm256_set1_ps(0.147f);
    __m256  consts_0_28=_mm256_set1_ps(0.289f);
    __m256  consts_0_43 =_mm256_set1_ps(0.436f);
    __m256  consts_0_61 =_mm256_set1_ps(0.615f);
    __m256  consts_0_51 =_mm256_set1_ps(0.515f);
    __m256  consts_0_10 =_mm256_set1_ps(0.1f);


    //for each pixel
    for(unsigned int i = off ; i < len ; i+=24)
    {
        const __m128i chunk_0 = _mm_loadu_si128((__m128i*)(bufRGB+ i));
        const __m128i chunk_1 = _mm_loadu_si128((__m128i*)(bufRGB+ i + 8));
        const __m128i chunk_2 = _mm_loadu_si128((__m128i*)(bufRGB+ i + 16));
        const __m256 chunk0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(chunk_0));
        const __m256 chunk1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(chunk_1));
        const __m256 chunk2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(chunk_2));
        const __m256 red1 =  _mm256_and_ps(_mm256_permutevar8x32_ps (chunk0, ssse3_red_indeces_0),_mm256_set_ps(0,0,0,0,0, *F_MASK,*F_MASK ,*F_MASK));
        const __m256 red2 =  _mm256_and_ps(_mm256_permutevar8x32_ps(chunk1, ssse3_red_indeces_1),_mm256_set_ps(0,0,*F_MASK,*F_MASK,*F_MASK, 0,0 ,0));
        const __m256 red3 =  _mm256_and_ps(_mm256_permutevar8x32_ps(chunk2, ssse3_red_indeces_2),_mm256_set_ps(*F_MASK,*F_MASK,0,0,0, 0,0 ,0));
        const __m256 red = _mm256_or_ps(_mm256_or_ps(red1,red2) ,red3);
        const __m256 green1 = _mm256_and_ps(_mm256_permutevar8x32_ps(chunk0, ssse3_green_indeces_0),_mm256_set_ps(0,0,0,0,0, *F_MASK,*F_MASK ,*F_MASK));
        const __m256 green2 =  _mm256_and_ps(_mm256_permutevar8x32_ps(chunk1, ssse3_green_indeces_1),_mm256_set_ps(0,0,0,*F_MASK,*F_MASK, 0,0 ,0));
        const __m256 green3 =  _mm256_and_ps(_mm256_permutevar8x32_ps(chunk2, ssse3_green_indeces_2),_mm256_set_ps(*F_MASK,*F_MASK,*F_MASK,0,0, 0,0 ,0));
        const __m256 green = _mm256_or_ps(_mm256_or_ps(green1,green2) ,green3);
        const __m256 blue1 = _mm256_and_ps(_mm256_permutevar8x32_ps(chunk0,  ssse3_blue_indeces_0),_mm256_set_ps(0,0,0,0,0, 0,*F_MASK ,*F_MASK));
        const __m256 blue2 =  _mm256_and_ps(_mm256_permutevar8x32_ps(chunk1, ssse3_blue_indeces_1),_mm256_set_ps(0,0,0,*F_MASK,*F_MASK, *F_MASK,0 ,0));
        const __m256 blue3 =  _mm256_and_ps(_mm256_permutevar8x32_ps(chunk2, ssse3_blue_indeces_2),_mm256_set_ps(*F_MASK,*F_MASK,*F_MASK,0,0, 0,0 ,0));
        const __m256 blue = _mm256_or_ps(_mm256_or_ps(blue1,blue2) ,blue3);

        //r
        __m256 x1 = _mm256_mul_ps(red,consts_0_29);
        __m256 x2 = _mm256_mul_ps(green,consts_0_58);
        __m256 x3 = _mm256_mul_ps(blue,consts_0_11);

        __m256 x4 = _mm256_add_ps(x2,x1);
        x4 = _mm256_add_ps(x4,x3);
        float  toto[8];
        _mm256_storeu_ps((toto), x4);
        for(int j = 0 ; j < 8 ; j++ )
        {
            bufYUV[i+(3*j)] = toto[j];
        }
        //g
        x1 = _mm256_mul_ps(red,consts_0_14);
        x2 = _mm256_mul_ps(green,consts_0_28);
        x3 = _mm256_mul_ps(blue,consts_0_43);
        x4 = _mm256_sub_ps(x3,x1);
        x4 = _mm256_sub_ps(x4,x2);
        _mm256_storeu_ps((toto), x4);
        for(int j = 0 ; j < 8 ; j++ )
        {
            bufYUV[i+(3*j)+1] = toto[j];
        }
        //b
        x1 = _mm256_mul_ps(red,consts_0_61);
        x2 = _mm256_mul_ps(green,consts_0_51);
        x3 = _mm256_mul_ps(blue,consts_0_10);

        x4 = _mm256_sub_ps(x1,x2);
        x4 = _mm256_sub_ps(x4,x3);

        _mm256_storeu_ps((toto), x4);
        for(int j = 0 ; j < 8 ; j++ )
        {
            bufYUV[i+(3*j)+2] = toto[j];
        }
    }
}

void converRGBtoYUV422(unsigned char* bufRGB,float*  bufYUV,unsigned int len)
{
    int off = IMAGE_OFF;
    for(int i = 0 ; i < off ;i++)
        bufYUV[i] = bufRGB[i];

    int x = 0xFFFFFFFF;
    float  *F_MASK = reinterpret_cast<float*>(&x);

    __m256i ssse3_red_indeces_0 = _mm256_set_epi32(-1, -1, -1, -1,-1, 6, 3, 0);
    __m256i ssse3_red_indeces_1 = _mm256_set_epi32(-1, -1, 7, 4, 1, -1, -1, -1);
    __m256i ssse3_red_indeces_2 = _mm256_set_epi32(5, 2, -1, -1, 1, -1, -1, -1);
    __m256i ssse3_green_indeces_0 = _mm256_set_epi32(-1, -1, -1, -1, -1, 7,4,1);
    __m256i ssse3_green_indeces_1 = _mm256_set_epi32(-1, -1, -1, 5,2, -1, -1, -1);
    __m256i ssse3_green_indeces_2 = _mm256_set_epi32(6, 3, 0, -1, -1, -1, -1, -1);
    __m256i ssse3_blue_indeces_0 = _mm256_set_epi32(-1, -1, -1, -1, -1, -1,5, 2);
    __m256i ssse3_blue_indeces_1 = _mm256_set_epi32(-1, -1, -1, 6, 3,0, -1, -1);
    __m256i ssse3_blue_indeces_2 = _mm256_set_epi32(7, 4, 1, -1,-1, -1, -1, -1);
    __m256  consts_0_29 = _mm256_set1_ps(0.299f);
    __m256  consts_0_58 =_mm256_set1_ps(0.587f);
    __m256  consts_0_11 =_mm256_set1_ps(0.114f);
    __m256  consts_0_14 =_mm256_set1_ps(0.147f);
    __m256  consts_0_28=_mm256_set1_ps(0.289f);
    __m256  consts_0_43 =_mm256_set1_ps(0.436f);
    __m256  consts_0_61 =_mm256_set1_ps(0.615f);
    __m256  consts_0_51 =_mm256_set1_ps(0.515f);
    __m256  consts_0_10 =_mm256_set1_ps(0.1f);


    //for each pixel
    for(unsigned int i = off ; i < len ; i+=24)
    {
        const __m128i chunk_0 = _mm_loadu_si128((__m128i*)(bufRGB+ i));
        const __m128i chunk_1 = _mm_loadu_si128((__m128i*)(bufRGB+ i + 8));
        const __m128i chunk_2 = _mm_loadu_si128((__m128i*)(bufRGB+ i + 16));
        const __m256 chunk0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(chunk_0));
        const __m256 chunk1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(chunk_1));
        const __m256 chunk2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(chunk_2));
        const __m256 red1 =  _mm256_and_ps(_mm256_permutevar8x32_ps (chunk0, ssse3_red_indeces_0),_mm256_set_ps(0,0,0,0,0, *F_MASK,*F_MASK ,*F_MASK));
        const __m256 red2 =  _mm256_and_ps(_mm256_permutevar8x32_ps(chunk1, ssse3_red_indeces_1),_mm256_set_ps(0,0,*F_MASK,*F_MASK,*F_MASK, 0,0 ,0));
        const __m256 red3 =  _mm256_and_ps(_mm256_permutevar8x32_ps(chunk2, ssse3_red_indeces_2),_mm256_set_ps(*F_MASK,*F_MASK,0,0,0, 0,0 ,0));
        const __m256 red = _mm256_or_ps(_mm256_or_ps(red1,red2) ,red3);
        const __m256 green1 = _mm256_and_ps(_mm256_permutevar8x32_ps(chunk0, ssse3_green_indeces_0),_mm256_set_ps(0,0,0,0,0, *F_MASK,*F_MASK ,*F_MASK));
        const __m256 green2 =  _mm256_and_ps(_mm256_permutevar8x32_ps(chunk1, ssse3_green_indeces_1),_mm256_set_ps(0,0,0,*F_MASK,*F_MASK, 0,0 ,0));
        const __m256 green3 =  _mm256_and_ps(_mm256_permutevar8x32_ps(chunk2, ssse3_green_indeces_2),_mm256_set_ps(*F_MASK,*F_MASK,*F_MASK,0,0, 0,0 ,0));
        const __m256 green = _mm256_or_ps(_mm256_or_ps(green1,green2) ,green3);
        const __m256 blue1 = _mm256_and_ps(_mm256_permutevar8x32_ps(chunk0,  ssse3_blue_indeces_0),_mm256_set_ps(0,0,0,0,0, 0,*F_MASK ,*F_MASK));
        const __m256 blue2 =  _mm256_and_ps(_mm256_permutevar8x32_ps(chunk1, ssse3_blue_indeces_1),_mm256_set_ps(0,0,0,*F_MASK,*F_MASK, *F_MASK,0 ,0));
        const __m256 blue3 =  _mm256_and_ps(_mm256_permutevar8x32_ps(chunk2, ssse3_blue_indeces_2),_mm256_set_ps(*F_MASK,*F_MASK,*F_MASK,0,0, 0,0 ,0));
        const __m256 blue = _mm256_or_ps(_mm256_or_ps(blue1,blue2) ,blue3);

        //r
        __m256 x1 = _mm256_mul_ps(red,consts_0_29);
        __m256 x2 = _mm256_mul_ps(green,consts_0_58);
        __m256 x3 = _mm256_mul_ps(blue,consts_0_11);

        __m256 x4 = _mm256_add_ps(x2,x1);
        x4 = _mm256_add_ps(x4,x3);
        float  toto[8];
        _mm256_storeu_ps((toto), x4);
        for(int j = 0 ; j < 8 ; j++ )
        {
            bufYUV[i+(3*j)] = toto[j];
        }
        //g
        x1 = _mm256_mul_ps(red,consts_0_14);
        x2 = _mm256_mul_ps(green,consts_0_28);
        x3 = _mm256_mul_ps(blue,consts_0_43);
        x4 = _mm256_sub_ps(x3,x1);
        x4 = _mm256_sub_ps(x4,x2);
        __m256 x6 = _mm256_hadd_ps(x4,x4);
        x6 = _mm256_div_ps(x6,_mm256_set1_ps(2));

        _mm256_storeu_ps((toto), x4);
        for(int j = 0 ; j < 8 ; j++ )
        {
            bufYUV[i+(3*j)+1] = toto[j];
        }
        //b
        x1 = _mm256_mul_ps(red,consts_0_61);
        x2 = _mm256_mul_ps(green,consts_0_51);
        x3 = _mm256_mul_ps(blue,consts_0_10);

        x4 = _mm256_sub_ps(x1,x2);
        x4 = _mm256_sub_ps(x4,x3);

        x6 = _mm256_hadd_ps(x4,x4);
        x6 = _mm256_div_ps(x6,_mm256_set1_ps(2));

        _mm256_storeu_ps((toto), x6);
        for(int j = 0 ; j < 8 ; j++ )
        {
            bufYUV[i+(3*j)+2] = toto[j];
        }
    }
}
void convertYUVtoRGB(float* bufYUV,unsigned char* bufRGB,unsigned int len)
{
    int off = IMAGE_OFF;
    int x = 0xFFFFFFFF;
    float  *F_MASK = reinterpret_cast<float*>(&x);
    for(int i = 0 ; i < off ;i++)
        bufRGB[i] = bufYUV[i];

    __m256i ssse3_red_indeces_0 = _mm256_set_epi32(-1, -1, -1, -1,-1, 6, 3, 0);
    __m256i ssse3_red_indeces_1 = _mm256_set_epi32(-1, -1, 7, 4, 1, -1, -1, -1);
    __m256i ssse3_red_indeces_2 = _mm256_set_epi32(5, 2, -1, -1, 1, -1, -1, -1);
    __m256i ssse3_green_indeces_0 = _mm256_set_epi32(-1, -1, -1, -1, -1, 7,4,1);
    __m256i ssse3_green_indeces_1 = _mm256_set_epi32(-1, -1, -1, 5,2, -1, -1, -1);
    __m256i ssse3_green_indeces_2 = _mm256_set_epi32(6, 3, 0, -1, -1, -1, -1, -1);
    __m256i ssse3_blue_indeces_0 = _mm256_set_epi32(-1, -1, -1, -1, -1, -1,5, 2);
    __m256i ssse3_blue_indeces_1 = _mm256_set_epi32(-1, -1, -1, 6, 3,0, -1, -1);
    __m256i ssse3_blue_indeces_2 = _mm256_set_epi32(7, 4, 1, -1,-1, -1, -1, -1);
    __m256  consts_1_14 = _mm256_set1_ps(1.14f);
    __m256  consts_0_39 =_mm256_set1_ps(0.3947f);
    __m256  consts_0_58 =_mm256_set1_ps(0.5808f);
    __m256  consts_2_03 =_mm256_set1_ps(2.033f);

    //for each pixel
    for(unsigned int i = off ; i < len ; i+=24)
    {
        const __m256 chunk0 = _mm256_loadu_ps((bufYUV+i));
        const __m256 chunk1 = _mm256_loadu_ps((bufYUV+ i + 8));
        const __m256 chunk2 = _mm256_loadu_ps((bufYUV+ i + 16));
        const __m256 red1 =  _mm256_and_ps(_mm256_permutevar8x32_ps (chunk0, ssse3_red_indeces_0),_mm256_set_ps(0,0,0,0,0, *F_MASK,*F_MASK ,*F_MASK));
        const __m256 red2 =  _mm256_and_ps(_mm256_permutevar8x32_ps(chunk1, ssse3_red_indeces_1),_mm256_set_ps(0,0,*F_MASK,*F_MASK,*F_MASK, 0,0 ,0));
        const __m256 red3 =  _mm256_and_ps(_mm256_permutevar8x32_ps(chunk2, ssse3_red_indeces_2),_mm256_set_ps(*F_MASK,*F_MASK,0,0,0, 0,0 ,0));
        const __m256 red = _mm256_or_ps(_mm256_or_ps(red1,red2) ,red3);
        const __m256 green1 = _mm256_and_ps(_mm256_permutevar8x32_ps(chunk0, ssse3_green_indeces_0),_mm256_set_ps(0,0,0,0,0, *F_MASK,*F_MASK ,*F_MASK));
        const __m256 green2 =  _mm256_and_ps(_mm256_permutevar8x32_ps(chunk1, ssse3_green_indeces_1),_mm256_set_ps(0,0,0,*F_MASK,*F_MASK, 0,0 ,0));
        const __m256 green3 =  _mm256_and_ps(_mm256_permutevar8x32_ps(chunk2, ssse3_green_indeces_2),_mm256_set_ps(*F_MASK,*F_MASK,*F_MASK,0,0, 0,0 ,0));
        const __m256 green = _mm256_or_ps(_mm256_or_ps(green1,green2) ,green3);
        const __m256 blue1 = _mm256_and_ps(_mm256_permutevar8x32_ps(chunk0,  ssse3_blue_indeces_0),_mm256_set_ps(0,0,0,0,0, 0,*F_MASK ,*F_MASK));
        const __m256 blue2 =  _mm256_and_ps(_mm256_permutevar8x32_ps(chunk1, ssse3_blue_indeces_1),_mm256_set_ps(0,0,0,*F_MASK,*F_MASK, *F_MASK,0 ,0));
        const __m256 blue3 =  _mm256_and_ps(_mm256_permutevar8x32_ps(chunk2, ssse3_blue_indeces_2),_mm256_set_ps(*F_MASK,*F_MASK,*F_MASK,0,0, 0,0 ,0));
        const __m256 blue = _mm256_or_ps(_mm256_or_ps(blue1,blue2) ,blue3);

        //r
        __m256 x1 = _mm256_mul_ps(blue,consts_1_14);
        __m256 x2 = _mm256_add_ps(red,x1);
        float  toto[8];
        _mm256_storeu_ps((toto), x2);
        for(int j = 0 ; j < 8 ; j++ )
        {
            bufRGB[i+(3*j)] = toto[j];
        }
        //g
        x1 = _mm256_mul_ps(green,consts_0_39);
        x2 = _mm256_mul_ps(blue,consts_0_58);
        __m256 x3 = _mm256_sub_ps(red,x1);
        __m256 x4 = _mm256_sub_ps(x3,x2);
        _mm256_storeu_ps((toto), x4);
        for(int j = 0 ; j < 8 ; j++ )
        {
            bufRGB[i+(3*j)+1] = toto[j];
        }
        //b
        x1 = _mm256_mul_ps(green,consts_2_03);
        x2 = _mm256_add_ps(red,x1);

        _mm256_storeu_ps((toto), x2);
        for(int j = 0 ; j < 8 ; j++ )
        {
            bufRGB[i+(3*j)+2] = toto[j];
        }
    }
}
void converRGBtoYUV_ref(unsigned char* bufRGB,float*  bufYUV,unsigned int len)
{
    int off = IMAGE_OFF;
    for(int i = 0 ; i < off ;i++)
        bufYUV[i] = bufRGB[i];
    //for each pixel
    for(int i = off ; i < len ; i+=3)
    {
        //Y                R        G       B
        bufYUV[i+0] =  0.299*double(bufRGB[i+0]) + 0.587*double(bufRGB[i+1]) + 0.114*double(bufRGB[i+2]);
        bufYUV[i+1]=  -0.147*double(bufRGB[i+0]) - 0.289*double(bufRGB[i+1]) + 0.436*double(bufRGB[i+2]);
        bufYUV[i+2] =  0.615*double(bufRGB[i+0]) - 0.515*double(bufRGB[i+1]) - 0.100*double(bufRGB[i+2]);
    }
}
void convertYUVtoRGB_ref(float* bufYUV,unsigned char* bufRGB,unsigned int len)
{
    int off = IMAGE_OFF;
    for(int i = 0 ; i < off ;i++)
        bufRGB[i] = bufYUV[i];
    //for each pixel
    for(int i = off ; i < len ; i+=3)
    {
        bufRGB[i+0]= bufYUV[i+0]+1.140*bufYUV[i+2];
        bufRGB[i+1]= bufYUV[i+0]- 0.3947*bufYUV[i+1]- 0.5808*bufYUV[i+2];
        bufRGB[i+2]= bufYUV[i+0]+2.033*bufYUV[i+1];
    }
}
void convertYUVtoYUV422(tImage<unsigned char> &bufYUV,tImage<unsigned char> &bufYUV422, enum YUV_Conversion ConvType)
{

    bufYUV422.buf[RGB_R].resize(bufYUV.height);
    bufYUV422.buf[RGB_G].resize(bufYUV.height);
    bufYUV422.buf[RGB_B].resize(bufYUV.height);
    bufYUV422.width = bufYUV.width;
    bufYUV422.height = bufYUV.height;
    bufYUV422.isGray = bufYUV.isGray;
    for(int i = 0 ; i < 54;i++)
        bufYUV422.meta[i] = bufYUV.meta[i];

    for(int i = 0 ; i < bufYUV422.height ; i++)
    {
        bufYUV422.buf[RGB_R][i].resize(bufYUV422.width);
        bufYUV422.buf[RGB_G][i].resize(bufYUV422.width);
        bufYUV422.buf[RGB_B][i].resize(bufYUV422.width);
    }


    switch(ConvType)
    {
    case AVERAGE:
        for(int i = 0 ; i < bufYUV422.height; i++)
        {
            for(int j = 0 ; j < bufYUV422.width;j+=2)
            {
                bufYUV422.buf[YUV_Y][i][j]= bufYUV.buf[YUV_Y][i][j];
                bufYUV422.buf[YUV_U][i][j]= (bufYUV.buf[YUV_U][i][j]+bufYUV.buf[YUV_U][i][j+1])/2;
                bufYUV422.buf[YUV_V][i][j]= (bufYUV.buf[YUV_V][i][j]+bufYUV.buf[YUV_V][i][j+1])/2;
                bufYUV422.buf[YUV_Y][i][j+1]= bufYUV.buf[YUV_Y][i][j+1];
                bufYUV422.buf[YUV_U][i][j+1]= (bufYUV.buf[YUV_U][i][j]+bufYUV.buf[YUV_U][i][j+1])/2;
                bufYUV422.buf[YUV_V][i][j+1]=(bufYUV.buf[YUV_V][i][j]+bufYUV.buf[YUV_V][i][j+1])/2;
            }
        }
        break;
    case SUBSAMPELING:
        for(int i = 0 ; i < bufYUV422.height; i++)
        {
            for(int j = 0 ; j < bufYUV422.width;j+=2)
            {
                bufYUV422.buf[YUV_Y][i][j]= bufYUV.buf[YUV_Y][i][j];
                bufYUV422.buf[YUV_U][i][j]= bufYUV.buf[YUV_U][i][j];
                bufYUV422.buf[YUV_V][i][j]= bufYUV.buf[YUV_V][i][j];
                bufYUV422.buf[YUV_Y][i][j+1]= bufYUV.buf[YUV_Y][i][j+1];
                bufYUV422.buf[YUV_U][i][j+1]= bufYUV.buf[YUV_U][i][j];
                bufYUV422.buf[YUV_V][i][j+1]= bufYUV.buf[YUV_V][i][j];
            }
        }
        break;
    }
}
void convertYUVtoYUV422(unsigned char* bufYUV, unsigned char* bufYUV422,int len)
{
    for(int i = 0 ; i < IMAGE_OFF ;i++)
        bufYUV422[i] = bufYUV[i];
    for(int i = IMAGE_OFF ; i < len; i+=6)
    {
            bufYUV422[i+1] = bufYUV422[i+4]  =  (bufYUV[i+1]+bufYUV[i+4])/2;
            bufYUV422[i+1] = bufYUV422[i+4]  =  (bufYUV[i+2]+bufYUV[i+5])/2;
    }
}
void convertYUVtoYUV422(float* bufYUV, float* bufYUV422,int len)
{
    for(int i = 0 ; i < IMAGE_OFF ;i++)
        bufYUV422[i] = bufYUV[i];
    for(int i = IMAGE_OFF ; i < len; i+=6)
    {
        bufYUV422[i+1] = bufYUV422[i+4]  =  (bufYUV[i+1]+bufYUV[i+4])/2;
        bufYUV422[i+1] = bufYUV422[i+4]  =  (bufYUV[i+2]+bufYUV[i+5])/2;
    }
}
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
