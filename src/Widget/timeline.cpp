#include "include/Widget/timeline.h"

TimeLine::TimeLine(int duration, QObject *parent) : QTimeLine(duration, parent)
{
    m_frameCacheStates.resize(endFrame() - startFrame(), CacheState::NotCached);


    connect(this, &TimeLine::frameChanged, [this](int frame){
        m_frameCacheStates[frame] = Cached;
    });

}


TimeLine::~TimeLine()
{

}

void TimeLine::SetFrameRange(int start, int end)
{
    m_frameCacheStates.resize(end - start, CacheState::NotCached);
    setFrameRange(start, end);
}



void TimeLine::OnFrameCached(int frame)
{
    if(frame = -1)
    {
        for(auto &c : m_frameCacheStates)
        {
            c = CacheState::Cached;
        }
        return;
    }

    if(frame < m_frameCacheStates.size())
    {
        m_frameCacheStates[frame] = CacheState::Cached;
    }
    else
    {
        while(frame >= m_frameCacheStates.size())
        {
            m_frameCacheStates.push_back(CacheState::NotCached);
        }
    }
}

void TimeLine::OnFrameCacheStale(int frame)
{
    if(frame = -1)
    {
        for(auto &c : m_frameCacheStates)
        {
            c = CacheState::StaleCache;
        }
        return;
    }

    if(frame < m_frameCacheStates.size())
    {
        m_frameCacheStates[frame] = CacheState::StaleCache;
    }
    else
    {
        while(frame >= m_frameCacheStates.size())
        {
            m_frameCacheStates.push_back(CacheState::NotCached);
        }
    }
}
