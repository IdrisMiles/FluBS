#ifndef TIMELINE_H
#define TIMELINE_H

#include <QTimeLine>

class TimeLine : public QTimeLine
{
    Q_OBJECT

public:

    enum CacheState
    {
        NotCached,
        Cached,
        StaleCache
    };

    TimeLine(int duration = 1000, QObject *parent = nullptr);
    ~TimeLine();

    void SetFrameRange(int start, int end);

public slots:
    void OnFrameCached(int frame);
    void OnFrameCacheStale(int frame);


protected:

private:
    std::vector<CacheState> m_frameCacheStates;

};

#endif // TIMELINE_H
