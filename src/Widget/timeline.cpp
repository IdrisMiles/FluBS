#include "include/Widget/timeline.h"
#include <iostream>

//------------------------------------------------------------------------------------

TimeLine::TimeLine(int duration, QObject *parent) : QTimeLine(duration, parent)
{
    m_frameCacheStates.resize(endFrame() - startFrame(), CacheState::NotCached);


    connect(this, &TimeLine::frameChanged, [this](int frame){
//        setPaused(true);
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

void TimeLine::SetSavedState(TimeLine::State _savedState)
{
    m_savedState = _savedState;
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
//    if(currentFrame() == frame)
//    {
//        std::cout<<"\t|matching frames\n";
//    }
//    else
//    {
//        std::cout<<"\t|NOT matching frames"<<frame<<", "<<currentFrame()<<"\n";
//    }

//    std::cout<<"saved state:\t";
//    if((m_savedState == TimeLine::State::NotRunning || m_savedState == TimeLine::State::Paused) &&
//            (State() != TimeLine::State::NotRunning && State() != TimeLine::State::Paused))
//    {
//        setPaused(true);
//        std::cout<<"pause\n";
//    }
//    else if(m_savedState == TimeLine::State::Running && state() != TimeLine::State::Running)
//    {
//        resume();
//        std::cout<<"resume\n";
//    }
}
