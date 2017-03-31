#ifndef ALGAEPROPERTYWIDGET_H
#define ALGAEPROPERTYWIDGET_H

#include <QWidget>
#include "Widget/sphparticlepropertywidget.h"

namespace Ui {
class AlgaePropertyWidget;
}

class AlgaePropertyWidget : public SphParticlePropertyWidget
{
    Q_OBJECT

public:
    explicit AlgaePropertyWidget(QWidget *parent = 0);
    ~AlgaePropertyWidget();

private:
    Ui::AlgaePropertyWidget *ui;
};

#endif // ALGAEPROPERTYWIDGET_H
