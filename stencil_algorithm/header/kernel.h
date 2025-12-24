#pragma once
#define TPB 64
#define RAD 1
#include <bits/stdc++.h>
__global__ void smem(float *in,float *out,int sz,float h);
void run(float *in,float *out,const int &n,const float &h);
std::tuple<std::vector<float>,std::vector<float>> init(const float &h,const float &pi,const int &n);
