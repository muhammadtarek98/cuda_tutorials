#include "graphics_utils.h"
void render()
{
    uchar4 *d_out = 0;
    cudaGraphicsMapResources(1, &resource, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&d_out, NULL,
      resource);
    for (int i = 0; i < ITERS_PEER_RENDER; ++i) {
        kernel_launcher(d_out, d_temp, W, H, bc);
    }
    cudaGraphicsUnmapResources(1, &resource, 0);
    char title[128];
    sprintf(title, "Temperature Visualizer - Iterations=%4d, "
                    "T_s=%3.0f, T_a=%3.0f, T_g=%3.0f",
                    iterationCount, bc.t_s, bc.t_a, bc.t_g);
    glutSetWindowTitle(title);

}
void draw_texture()
{
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA,
        GL_UNSIGNED_BYTE, NULL);
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(0, 0);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(0, H);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(W, H);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(W, 0);
    glEnd();
    glDisable(GL_TEXTURE_2D);
}

void display()
{
    render();
    draw_texture();
    glutSwapBuffers();
}


void init_GLUT(int *argc,char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(W, H);
    glutCreateWindow("Temp. Vis.");
#ifndef __APPLE__
    glewInit();
#endif
}


void init_pixel_buffer()
{
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, W*H*sizeof(GLubyte)* 4, 0,
      GL_STREAM_DRAW);
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    cudaGraphicsGLRegisterBuffer(&resource, pbo,
      cudaGraphicsMapFlagsWriteDiscard);
}

void exit_func()
{
    if (pbo) {
        cudaGraphicsUnregisterResource(resource);
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &tex);
    }
    cudaFree(d_temp);
}
