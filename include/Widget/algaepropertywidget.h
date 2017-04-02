#ifndef ALGAEPROPERTYWIDGET_H
#define ALGAEPROPERTYWIDGET_H

#include <QWidget>
#include "Widget/sphparticlepropertywidget.h"

#include "SPH/algaeproperty.h"

namespace Ui {
class AlgaePropertyWidget;
}

class AlgaePropertyWidget : public SphParticlePropertyWidget
{
    Q_OBJECT

public:
    explicit AlgaePropertyWidget(QWidget *parent = 0);
    ~AlgaePropertyWidget();

    virtual void SetProperty(std::shared_ptr<AlgaeProperty> _algaeProperty);
    virtual AlgaeProperty *GetProperty();

private:
    Ui::AlgaePropertyWidget *ui;

    std::shared_ptr<AlgaeProperty> m_property;
};

#endif // ALGAEPROPERTYWIDGET_H
