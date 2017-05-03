#include "include/Widget/timeline.h"

//------------------------------------------------------------------------------------

TimeLine::TimeLine(int duration, QObject *parent) : QTimeLine(duration, parent)
{
    m_frameCacheStates.resize(endFrame() - startFrame(), CacheState::NotCached);


    connect(this, &TimeLine::frameChanged, [this](int frame){
        m_frameCacheStates[frame] = Cached;
    });


    // connect on frame changed
    // if play every frame set
    // - save current state
    // - pause

    // connect receiving a frame finished signal
    // - resume saved state

}

//------------------------------------------------------------------------------------

TimeLine::~TimeLine()
{

}

//------------------------------------------------------------------------------------

void TimeLine::SetFrameRange(int start, int end)
{
    m_frameCacheStates.resize(end - start, CacheState::NotCached);
    setFrameRange(start, end);
}

//------------------------------------------------------------------------------------

void TimeLine::OnFrameCached(int frame)
{
}

//------------------------------------------------------------------------------------

void TimeLine::OnFrameCacheStale(int frame)
{
}

//------------------------------------------------------------------------------------

void TimeLine::OnFrameFinished(int frame)
{
}
