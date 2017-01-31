#ifndef POLY6KERNEL_H
#define POLY6KERNEL_H

#include "../cuda_inc/smoothkernel.cuh"


class Poly6Kernel : public SmoothKernel
{
public:
    Poly6Kernel(float _h) :
        SmoothKernel(),
        m_h(_h)
    {

    }

    virtual ~Poly6Kernel(){}



    __host__ __device__ float Eval(float _x)
    {
        if (fabs(_x) > m_h)
        {
            return 0.0f;
        }

        return (315.0f / (64.0f*CUDART_PI_F*pow(m_h,9.0f))) * pow((m_h*m_h)-(_x*_x), 3.0f);
    }

    __host__ __device__ float Grad(float _x)
    {
        return 0.0f;
    }

    __host__ __device__ float3 Grad(float3 _x)
    {
        return make_float3(0.0f,0.0f,0.0f);

    }

    __host__ __device__ float Laplace(float _x)
    {
        if(_x > m_h && _x < 0.0f)
        {
            return 0.0f;
        }
        else
        {
            float a = -945.0 / (32.0*CUDART_PI_F*pow(m_h, 9.0f));
            float b = (m_h*m_h) - (_x*_x);
            float c = 3.0f * (m_h*m_h) - 7.0f * (_x*_x);
            return a * b * c;
        }
    }


private:

    float m_h;
};

#endif // POLY6KERNEL_H
