#include "volume_render.cuh"
__device__ uchar4 volume_render_shader(float *d_vol,int3 vol_size,Ray ray,float threshold,int num_steps)
{
    uchar4 shade{make_uchar4(96,0,192,0)};
    const float dt{1.f/num_steps};
    const float len{length(ray.dir)/num_steps};
    float accum=0.0;
    auto current_pos=ray.orign;
    float density_val=density(d_vol,vol_size,current_pos);
    for (float i=0.0f;i<1.0f;i+=dt)
    {
        if (density_val-threshold<0.0f)
        {
            accum+=(fabsf(density_val-threshold))+len;
        }
        current_pos=param_ray(ray,i);
        density_val=density(d_vol,vol_size,current_pos);
    }
    if (clip(accum,0.0,255.0)>0.f)
    {
        shade.y=clip(accum,0.0,255.0);
    }
    return shade;
}
