#pragma once
#include "kernel.cuh"
#include <bits/stdc++.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/freeglut.h>
#include "common_utils.h"
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

void keyboard(unsigned char key,int x,int y);
void mouse(int button,int state,int x,int y);
void idle();
void print_instruction();