#pragma once
#define W 640
#define H 640
#define DT 1.f
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <GL/glew.h>
#include <GL/glut.h>
#define ITERS_PEER_RENDER 50
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

struct uchar4;
typedef struct
{
    int X,Y;
    float Chamfer,rad,t_s,t_a,t_g;
}
BC;
extern  BC bc;
extern   int iterationCount;
extern   struct cudaGraphicsResource *resource;
extern   int w;
extern   int h;
extern   GLuint pbo;
extern   GLuint tex;
extern  float* d_temp;



