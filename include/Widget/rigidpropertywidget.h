#ifndef RIGIDPROPERTYWIDGET_H
#define RIGIDPROPERTYWIDGET_H

#include <QWidget>
#include "Widget/sphparticlepropertywidget.h"

#include "SPH/rigidproperty.h"

namespace Ui {
class RigidPropertyWidget;
}

class RigidPropertyWidget : public SphParticlePropertyWidget
{
    Q_OBJECT

public:
    explicit RigidPropertyWidget(QWidget *parent = 0);
    ~RigidPropertyWidget();

    virtual void SetProperty(std::shared_ptr<RigidProperty> _rigidProperty);
    virtual RigidProperty *GetProperty();

private:
    Ui::RigidPropertyWidget *ui;

    std::shared_ptr<RigidProperty> m_property;
};

#endif // RIGIDPROPERTYWIDGET_H
