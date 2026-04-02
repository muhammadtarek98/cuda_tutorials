#include <bits/stdc++.h>
#include "interactions.h"
#include "visualization_utils.h"
#include "kernel.cuh"


int main(int argc,char **argv)
{
    auto bytes=NX*NY*NZ*sizeof(float);
    cudaMalloc(&d_vol,bytes);
    volume_kernel_launch(d_vol,vol_size,id,params);
    print_instructions();
    initGLUT(&argc,argv);
    create_menu();
    gluOrtho2D(0,W,H,0);
    glutKeyboardFunc(keyboard);
    glutSpecialFunc(handle_keys);
    glutDisplayFunc(display);
    initPixelBuffer();
    glutMainLoop();
    atexit(exitfunc);
    cudaDeviceReset();
    return 0;
}