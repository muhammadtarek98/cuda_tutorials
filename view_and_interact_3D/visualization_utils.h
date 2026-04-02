#pragma once
#include <bits/stdc++.h>
#include<GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include<cuda_gl_interop.h>
#include "kernels_launch.h"
#include "interactions.h"
extern GLuint pbo,tex;
extern struct cudaGraphicsResource *cuda_pbo_resource;
void render();
void draw_texture();
void display();
void initGLUT(int *argc,char **argv);
void initPixelBuffer();
void exitfunc();