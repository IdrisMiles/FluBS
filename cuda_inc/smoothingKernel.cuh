#ifndef _SMOOTHINGKERNELS__H_
#define _SMOOTHINGKERNELS__H_

#include <cuda_runtime.h>
#include <math_constants.h>

#include "vec_ops.cuh"




__device__ float SpikyKernel_Kernel(const float &_r, const float &_h)
{
    if(_r > _h || _r < 0.0f)
    {
        return 0.0f;
    }
    else
    {
        return (15.0f/(CUDART_PI_F * pow(_h, 6.0f))) * pow((_h-_r), 3.0f);
    }
}

__device__ float SpikyKernelGradientF_Kernel(const float &_r, const float &_h)
{
    return ((_r > _h || _r <= FLT_EPSILON) ? 0.0f : - (45.0f/(CUDART_PI_F*pow(_h,6.0f))) * pow((_h-_r), 2.0f));
}

__device__ float3 SpikyKernelGradientV_Kernel(const float3 _a, const float3 _b, const float _h)
{
    float3 dir = _a - _b;
    float distance = length(dir);
    return ((distance<=FLT_EPSILON) ? make_float3(0.f, 0.f, 0.f) : SpikyKernelGradientF_Kernel(distance, _h) * (dir/distance));
}


__device__ float ViscosityKernel(const float &_r, const float &_h)
{
    if (_r > _h || _r < 0.0f)
    {
        return 0.0f;
    }
    else
    {
        return ( 15.0f/(2.0f*CUDART_PI_F*pow(_h,3.0f)) ) * ( -(pow(_r,3.0f)/(2.0f*pow(_h,2.0f))) + (pow(_r,2.0f)/pow(_h,2.0f)) + (_h/(2.0f*_r)) - 1.0f );
    }

}

__device__ float Poly6Kernel_Kernel(const float &_r, const float &_h)
{
    if (_r > _h)
    {
        return 0.0f;
    }

    return (315.0f / (64.0f*CUDART_PI_F*pow(_h,9.0f))) * pow((_h*_h)-(_r*_r), 3.0f);
}

__device__ float Poly6Laplacian_Kernel(const float &_r, const float &_h)
{
    return ((_r > _h && _r < 0.0f) ? 0.0f : (-945.0 / (32.0*CUDART_PI_F*pow(_h, 9.0f))) * ((_h*_h) - (_r*_r)) * ((3.0f * (_h*_h)) - (7.0f * (_r*_r))));
}

__device__ float SplineGaussianKernel_Kernel(const float &_r, const float &_h)
{
    if(_r > /*2.0f**/_h || _r < 0.0f)
    {
        return 0.0f;
    }
    else if(_r > _h)
    {
        return (1/(CUDART_PI_F * _h)) * 0.25f * pow(2-(_r/_h), 3);
    }
    else
    {
        return (1/(CUDART_PI_F * _h)) * (1.0f - (1.5f*(pow((_r/_h), 2))) + (0.75f*(pow((_r/_h), 3))));
    }
}



#endif //_SMOOTHINGKERNELS__H_
