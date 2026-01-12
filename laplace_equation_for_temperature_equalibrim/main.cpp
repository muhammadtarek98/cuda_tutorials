#include "graphics_utils.h"
#include "intersection.h"
#include "common_utils.h"
#define NOMINMAX
#include "kernel.cuh"
BC bc = {W / 2, H / 2, W / 10.f, 150, 212.f, 70.f, 0.f};
int iterationCount = 0;
float *d_temp = nullptr;
struct cudaGraphicsResource *resource ;
uchar4 *dout = nullptr;
GLuint pbo = 0, tex = 0;
int h=H,w=W;

int main(int argc,char **argv)
{
    cudaMalloc(reinterpret_cast<void**>(&d_temp),w*h*sizeof(float));
    reset_temp(d_temp,W,H,bc);
    init_GLUT(&argc,argv);
    gluOrtho2D(0,W,H,0);
    glutIdleFunc(idle);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutDisplayFunc(display);
    init_pixel_buffer();
    glutMainLoop();
    atexit(exit_func);
    return 0;
}