#include "interactions.h"
#include "kernel.cuh"
float *d_vol = nullptr;
int id = 1;
int method = 2;
const int3 vol_size = make_int3(NX, NY, NZ);
const float4 params = {make_float4(NX/4.0f, NY/6.0f, NZ/16.0f, 1.0f)};
float zs = NZ, dist = 0.0f, theta = 0.0f, threshold = 0.0f;
void menu(int val)
{
    if (val==0)
    {
        return;
    }
    else if (val==1)
    {
        id=0;
    }
    else if (val==2)
    {
        id=1;
    }
    else if (val==3)
    {
        id=2;
    }
    volume_kernel_launch(d_vol,vol_size,id,params);
    glutPostRedisplay();
}

void create_menu()
{
    glutCreateMenu(menu);
    glutAddMenuEntry("Object Selector",0);
    glutAddMenuEntry("Sphere",1);
    glutAddMenuEntry("Torus",2);
    glutAddMenuEntry("Block",3);
    glutAttachMenu(GLUT_RIGHT_BUTTON);
}

void keyboard(unsigned char key, int x, int y)
{
    if (key=='+')
    {
        zs-=DELTA;
    }
    if (key=='-')
    {
        zs+=DELTA;
    }
    if (key=='d')
    {
        --dist;
    }
    if (key=='D')
    {
        ++dist;
    }
    if (key=='z')
    {
        zs=NZ;
        theta=0.0f;
        dist=0.0f;
    }
    if (key=='v')
    {
        method=0;
    }
    if (key=='f')
    {
        method=1;
    }
    if (key=='r')
    {
        method=2;
    }
    if (key==27)
    {
        exit(0);
    }
    glutPostRedisplay();
}

void handle_keys(int key, int x, int y)
{
    if (key==GLUT_KEY_LEFT)
    {
        theta-=0.1f;
    }
    if (key==GLUT_KEY_RIGHT)
    {
        theta+=0.1f;
    }
    if (key==GLUT_KEY_DOWN)
    {
        theta-=0.1f;
    }
    if (key==GLUT_KEY_UP)
    {
        theta+=0.1f;
    }
    glutPostRedisplay();
}

void print_instructions()
{
    printf("3D Volume Visualizer\n"
         "Controls:\n"
         "Volume render mode                          : v\n"
         "Slice render mode                           : f\n"
         "Raycast mode                                : r\n"
         "Zoom out/in                                 : -/+\n"
         "Rotate view                                 : left/right\n"
         "Decr./Incr. Offset (intensity in slice mode): down/up\n"
         "Decr./Incr. distance (only in slice mode)   : d/D\n"
         "Reset parameters                            : z\n"
         "Right-click for object selection menu\n");
}
