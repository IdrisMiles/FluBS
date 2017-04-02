#ifndef FLUIDPROPERTYWIDGET_H
#define FLUIDPROPERTYWIDGET_H

#include <QWidget>
#include "Widget/sphparticlepropertywidget.h"

#include "SPH/fluidproperty.h"

namespace Ui {
class FluidPropertyWidget;
}

class FluidPropertyWidget : public SphParticlePropertyWidget
{
    Q_OBJECT

public:
    explicit FluidPropertyWidget(QWidget *parent = 0, std::shared_ptr<FluidProperty> _property = nullptr);
    ~FluidPropertyWidget();

    virtual void SetProperty(std::shared_ptr<FluidProperty> _fluidProperty);
    virtual FluidProperty *GetProperty();

private:
    Ui::FluidPropertyWidget *ui;

    std::shared_ptr<FluidProperty> m_property;
};

#endif // FLUIDPROPERTYWIDGET_H
