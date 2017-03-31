#ifndef RIGIDPROPERTYWIDGET_H
#define RIGIDPROPERTYWIDGET_H

#include <QWidget>
#include "Widget/sphparticlepropertywidget.h"

namespace Ui {
class RigidPropertyWidget;
}

class RigidPropertyWidget : public SphParticlePropertyWidget
{
    Q_OBJECT

public:
    explicit RigidPropertyWidget(QWidget *parent = 0);
    ~RigidPropertyWidget();

private:
    Ui::RigidPropertyWidget *ui;
};

#endif // RIGIDPROPERTYWIDGET_H
