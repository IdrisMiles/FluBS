#include "include/Widget/timeline.h"

TimeLine::TimeLine(int duration, QObject *parent) : QTimeLine(duration, parent)
{
    frameCacheStates.resize(endFrame() - startFrame(), CacheState::NotCached);


    connect(this, &TimeLine::frameChanged, [this](int frame){
        frameCacheStates[frame] = Cached;
    });

}


TimeLine::~TimeLine()
{

}

void TimeLine::SetFrameRange(int start, int end)
{
    frameCacheStates.resize(end - start, CacheState::NotCached);
    setFrameRange(start, end);
}
