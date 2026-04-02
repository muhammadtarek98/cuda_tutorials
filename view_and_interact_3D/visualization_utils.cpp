#include "visualization_utils.h"
#include "kernel.cuh"
#include <iostream>

GLuint pbo=0,tex=0;
struct cudaGraphicsResource *cuda_pbo_resource=nullptr;

void render()
{
    uchar4 *d_out;
    cudaGraphicsMapResources(1,&cuda_pbo_resource,0);
    cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&d_out),nullptr,cuda_pbo_resource);
    kernel_launch(d_out,d_vol,W,H,vol_size,method,zs,theta,threshold,dist);
    cudaGraphicsUnmapResources(1,&cuda_pbo_resource,0);
    char title[128];
    sprintf(title, "Volume Visualizer : objId =%d, method = %d,"
          " dist = %.1f, theta = %.1f", id, method, dist,
          theta);
    glutSetWindowTitle(title);
}

void draw_texture()
{
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,W,H,0,GL_RGBA,GL_UNSIGNED_BYTE,NULL);
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f,0.0f); glVertex2f(0,0);
    glTexCoord2f(0.0f,1.0f); glVertex2f(0,H);
    glTexCoord2f(1.0f,1.0f); glVertex2f(W,H);
    glTexCoord2f(1.0f,0.0f); glVertex2f(W,0);
    glEnd();
    glDisable(GL_TEXTURE_2D);
}

void display()
{
    render();
    draw_texture();
    glutSwapBuffers();
}

void initGLUT(int* argc, char** argv)
{
    glutInit(argc,argv);
    glutInitDisplayMode(GLUT_RGBA|GLUT_DOUBLE);
    glutInitWindowSize(W,H);
    glutCreateWindow("volume visualizer");
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (GLEW_OK != err) {
        std::cerr << "GLEW Error: " << glewGetErrorString(err) << std::endl;
    }
}

void initPixelBuffer()
{
    glGenBuffers(1,&pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER,pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER,W*H*sizeof(GLubyte)*4,0,GL_STREAM_DRAW);
    glGenTextures(1,&tex);
    glBindTexture(GL_TEXTURE_2D,tex);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
    cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource,pbo,cudaGraphicsMapFlagsWriteDiscard);
}

void exitfunc()
{
    if (pbo)
    {
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        glDeleteBuffers(1,&pbo);
        glDeleteTextures(1,&tex);
    }
    cudaFree(d_vol);
}
