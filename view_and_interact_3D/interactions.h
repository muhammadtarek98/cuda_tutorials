#pragma once
#include <vector_functions.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <vector_types.h>
#define H 600
#define W 600
#define DELTA 5
#define NX 128
#define NY 128
#define NZ 128
extern float *d_vol;
extern int id;
extern int method;
extern const int3 vol_size;
extern const float4 params;
extern float zs, dist, theta, threshold;
void menu(int val);
void create_menu();
void keyboard(unsigned char key,int x,int y);
void handle_keys(int key,int x,int y);
void print_instructions();
