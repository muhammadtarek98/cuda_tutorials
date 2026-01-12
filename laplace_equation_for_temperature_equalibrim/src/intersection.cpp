#include "intersection.h"
#include "common_utils.h"

void idle()
{
    ++iterationCount;
    glutPostRedisplay();
}

void mouse(int button, int state, int x, int y)
{
    bc.X=x,bc.Y=y;
    glutPostRedisplay();
}

void keyboard( unsigned char key,int x,int y)
{  if (key == 'S') bc.t_s += DT;
    if (key == 's') bc.t_s -= DT;
    if (key == 'A') bc.t_a += DT;
    if (key == 'a') bc.t_a -= DT;
    if (key == 'G') bc.t_g += DT;
    if (key == 'g') bc.t_g -= DT;
    if (key == 'R') bc.rad += DT;
    if (key == 'r') bc.rad = MAX(0.f, bc.rad - DT);
    if (key == 'C') ++bc.Chamfer;
    if (key == 'c') --bc.Chamfer;
    if (key == 'z') reset_temp(d_temp, W, H, bc);
    if (key == 27) exit(0);
    glutPostRedisplay();
}

void print_instruction()
{

}
