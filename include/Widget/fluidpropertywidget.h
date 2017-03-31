#ifndef FLUIDPROPERTYWIDGET_H
#define FLUIDPROPERTYWIDGET_H

#include <QWidget>
#include "Widget/sphparticlepropertywidget.h"

namespace Ui {
class FluidPropertyWidget;
}

class FluidPropertyWidget : public SphParticlePropertyWidget
{
    Q_OBJECT

public:
    explicit FluidPropertyWidget(QWidget *parent = 0);
    ~FluidPropertyWidget();

private:
    Ui::FluidPropertyWidget *ui;
};

#endif // FLUIDPROPERTYWIDGET_H
