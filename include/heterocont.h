#ifndef HETEROCONT_H
#define HETEROCONT_H


template<typename T>
class HeteroCont
{
public:
    HeteroCont();

    HeteroCont(const T &_copy)
    {
        m_cpu = _copy;
    }

    ~HeteroCont();

    void Set(const T &_value)
    {
        m_cpu = _value;
    }

    T Get(const bool _getGPU = false)
    {
        if(_getGPU)
        {
            return m_gpu;
        }
        else
        {
            return m_cpu;
        }
    }


private:
    T m_cpu;
    T m_gpu;
};

#endif // HETEROCONT_H
