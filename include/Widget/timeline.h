#ifndef TIMELINE_H
#define TIMELINE_H

//--------------------------------------------------------------------------------------------------------------

#include <QTimeLine>

//--------------------------------------------------------------------------------------------------------------
/// @author Idris Miles
/// @version 1.0
/// @date 01/06/2017
//--------------------------------------------------------------------------------------------------------------

/// @class TimeLine
/// @brief Inherits from QTimeLine, a custom timeline class
class TimeLine : public QTimeLine
{
    Q_OBJECT

public:

    /// @brief constructor
    TimeLine(int duration = 1000, QObject *parent = nullptr);

    /// @brief destructor
    ~TimeLine();

    /// @brief Method to set frame range
    void SetFrameRange(int start, int end);

    /// @brief Method to save state of timeline
    void SetSavedState(TimeLine::State _savedState);

public slots:
    /// @brief Slot to handle frame being cached
    void OnFrameCached(int frame);

    /// @brief Slot to handle frames cache becoming stale
    void OnFrameCacheStale(int frame);

    /// @brief Slot to handle frame being finished
    void OnFrameFinished(int frame);


protected:

private:
    TimeLine::State m_currentState;
    TimeLine::State m_savedState;

};

//--------------------------------------------------------------------------------------------------------------

#endif // TIMELINE_H
