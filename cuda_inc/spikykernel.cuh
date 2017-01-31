#ifndef SPIKYKERNEL_H
#define SPIKYKERNEL_H

#include "../cuda_inc/smoothkernel.cuh"


class SpikyKernel : public SmoothKernel
{
public:
    SpikyKernel(float _h) :
        SmoothKernel(),
        m_h(_h)
    {

    }

    virtual ~SpikyKernel(){}



    __host__ __device__ float Eval(float _x)
    {
        if(fabs(_x) > m_h || _x < 0.0f)
        {
            return 0.0f;
        }
        else
        {
            return (15.0f/(CUDART_PI_F * pow(m_h, 6.0f))) * pow((m_h-_x), 3.0f);
        }
    }

    __host__ __device__ float Grad(float _x)
    {
        if(fabs(_x) > m_h || fabs(_x) <= FLT_EPSILON)
        {
            return 0.0f;
        }
        else
        {
            float coeff = - (45.0f/(CUDART_PI_F*pow(m_h,6.0f)));
            return coeff * pow((m_h-_x), 2.0f);
        }
    }

    __host__ __device__ float3 Grad(float3 _x)
    {
        float distance = length(_x);
        if(fabs(distance) <= FLT_EPSILON)
        {
            return make_float3(0.f, 0.f, 0.f);
        }
        else
        {
            float c = Grad(distance);

            return (c * _x/distance);
        }
    }

    __host__ __device__ float Laplace(float _x)
    {
        return 0.0f;
    }

private:

    float m_h;
};

#endif // SPIKYKERNEL_H
