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
    explicit RigidPropertyWidget(QWidget *parent = 0, RigidProperty *_property = nullptr);
    ~RigidPropertyWidget();

    virtual void SetProperty(RigidProperty *_property);
    virtual RigidProperty *GetProperty();

signals:
    /// @brief Qt Signal to communicate that the FluidProperty has changed to other classes
    void PropertyChanged(RigidProperty *_property);

public slots:
    /// @brief Qt Slot to be connected to any changes on this widget, emits PropertyChanged(m_property)
    virtual void OnPropertyChanged();

private:
    Ui::RigidPropertyWidget *ui;

    RigidProperty *m_property;
};

#endif // RIGIDPROPERTYWIDGET_H
