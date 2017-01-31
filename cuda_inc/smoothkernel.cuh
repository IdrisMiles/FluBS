#ifndef SMOOTHKERNEL_H
#define SMOOTHKERNEL_H

#include <cuda_runtime.h>
#include <math_constants.h>

#include <math.h>
#include <float.h>

#include "../cuda_inc/vec_ops.cuh"

class SmoothKernel
{
public:
    SmoothKernel(){}
    virtual ~SmoothKernel(){}

    __host__ __device__ virtual float Eval(float _x) = 0;
    __host__ __device__ virtual float Grad(float _x) = 0;
    __host__ __device__ virtual float3 Grad(float3 _x) = 0;
    __host__ __device__ virtual float Laplace(float _x) = 0;

};

#endif // SMOOTHKERNEL_H
