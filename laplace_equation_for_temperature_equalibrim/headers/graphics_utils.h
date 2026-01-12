#pragma once
#define GLEW_STATIC
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <bits/stdc++.h>
#include "common_utils.h"
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#endif
#include "kernel.cuh"

void render();
void draw_texture();
void display();
void init_GLUT(int *argc,char **argv);
void init_pixel_buffer();
void exit_func();