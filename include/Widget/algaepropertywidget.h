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
    explicit AlgaePropertyWidget(QWidget *parent = 0, AlgaeProperty *_algaeProperty = nullptr);
    ~AlgaePropertyWidget();

    virtual void SetProperty(AlgaeProperty *_property);
    virtual AlgaeProperty *GetProperty();


signals:
    /// @brief Qt Signal to communicate that the FluidProperty has changed to other classes
    void PropertyChanged(AlgaeProperty *_property);


public slots:
    /// @brief Qt Slot to be connected to any changes on this widget, emits PropertyChanged(m_property)
    virtual void OnPropertyChanged();

private:
    Ui::AlgaePropertyWidget *ui;

    AlgaeProperty *m_property;
};

#endif // ALGAEPROPERTYWIDGET_H
