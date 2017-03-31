#ifndef TESTTIMELINE_H
#define TESTTIMELINE_H

#include <QFrame>

namespace Ui {
class TestTimeLine;
}

class TestTimeLine : public QFrame
{
    Q_OBJECT

public:
    explicit TestTimeLine(QWidget *parent = 0);
    ~TestTimeLine();

private:
    Ui::TestTimeLine *ui;
};

#endif // TESTTIMELINE_H
