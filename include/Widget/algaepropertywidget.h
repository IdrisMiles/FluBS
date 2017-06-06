#ifndef ALGAEPROPERTYWIDGET_H
#define ALGAEPROPERTYWIDGET_H

//--------------------------------------------------------------------------------------------------------------

#include <QWidget>
#include "Widget/sphparticlepropertywidget.h"

#include "SPH/algaeproperty.h"

//--------------------------------------------------------------------------------------------------------------
/// @author Idris Miles
/// @version 1.0
/// @date 01/06/2017
//--------------------------------------------------------------------------------------------------------------


namespace Ui {
class AlgaePropertyWidget;
}

/// @class AlgaePropertyWidget
/// @brief GUI widget for an algae property object. This class inherits from SphParticlePropertyWidget.
class AlgaePropertyWidget : public SphParticlePropertyWidget
{
    Q_OBJECT

public:
    /// @brief constructor
    explicit AlgaePropertyWidget(QWidget *parent = 0, AlgaeProperty _algaeProperty = AlgaeProperty());

    /// @brief destructor
    ~AlgaePropertyWidget();

    /// @brief Method to set property associated with this widget
    virtual void SetProperty(AlgaeProperty _property);

    /// @brief Method to get the properties associated with this widget
    AlgaeProperty GetProperty();


signals:
    /// @brief Qt Signal to communicate that the FluidProperty has changed to other classes
    void PropertyChanged(AlgaeProperty _property);


public slots:
    /// @brief Qt Slot to be connected to any changes on this widget, emits PropertyChanged(m_property)
    virtual void OnPropertyChanged();

private:
    Ui::AlgaePropertyWidget *ui;

    AlgaeProperty m_property;
};

//--------------------------------------------------------------------------------------------------------------

#endif // ALGAEPROPERTYWIDGET_H
