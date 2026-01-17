#include <bits/stdc++.h>
#include "CImg.h"
#include "kernel.cuh"
int main()
{
    cimg_library::CImg<unsigned char> img("/home/muhammad/projects/cuda_tutorials/sharpen_rgb_image/butterfly.bmp");
    const int h=img.height(),w=img.width();
    uchar4 *img_ptr=(uchar4*) malloc(h*w*sizeof(uchar4));
    for (int i=0;i<h;i++)
    {
        for (int j=0;j<w;j++)
        {
            img_ptr[i*w+j].x=img(j,i,0);
            img_ptr[i*w+j].y=img(j,i,1);
            img_ptr[i*w+j].z=img(j,i,2);
        }
    }
    sharpen_image(img_ptr,w,h,false);

    for (int i=0;i<h;i++)
    {
        for (int j=0;j<w;j++)
        {
            img(j,i,0)=img_ptr[i*w+j].x;
            img(j,i,1)=img_ptr[i*w+j].y;
            img(j,i,2)=img_ptr[i*w+j].z;
        }
    }
    img.save_bmp("/home/muhammad/projects/cuda_tutorials/sharpen_rgb_image/res.bmp");
    free(img_ptr);

    return 0;
}