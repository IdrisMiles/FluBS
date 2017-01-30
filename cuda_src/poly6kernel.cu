#include "../cuda_inc/poly6kernel.cuh"

//Poly6Kernel::Poly6Kernel(float _h) : SmoothKernel()
//{
//    m_h = _h;
//}

//Poly6Kernel::~Poly6Kernel()
//{
//}

//__host__ __device__ float Poly6Kernel::Eval(float _x)
//{
//    if (fabs(_x) > m_h)
//    {
//        return 0.0f;
//    }

//    return (315.0f / (64.0f*CUDART_PI_F*pow(m_h,9.0f))) * pow((m_h*m_h)-(_x*_x), 3.0f);
//}

//__host__ __device__ float Poly6Kernel::Grad(float _x)
//{
//    return 0.0f;
//}

//__host__ __device__ float3 Poly6Kernel::Grad(float3 _x)
//{
//    return make_float3(0.0f,0.0f,0.0f);

//}

//__host__ __device__ float Poly6Kernel::Laplace(float _x)
//{
//    if(_x > m_h && _x < 0.0f)
//    {
//        return 0.0f;
//    }
//    else
//    {
//        float a = -945.0 / (32.0*CUDART_PI_F*pow(m_h, 9.0f));
//        float b = (m_h*m_h) - (_x*_x);
//        float c = 3.0f * (m_h*m_h) - 7.0f * (_x*_x);
//        return a * b * c;
//    }
//}

