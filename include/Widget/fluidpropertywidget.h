#ifndef FLUIDPROPERTYWIDGET_H
#define FLUIDPROPERTYWIDGET_H

//-----------------------------------------------------------------------------------------------------------

#include "Widget/sphparticlepropertywidget.h"

#include "SPH/fluidproperty.h"

//-----------------------------------------------------------------------------------------------------------

/// @author Idris Miles
/// @version 0.1.0
/// @date 20/04/2017

//-----------------------------------------------------------------------------------------------------------


namespace Ui {
class FluidPropertyWidget;
}


/// @class FluidPropertyWidget
/// @brief GUI widget for a fluid property object. This class inherits from SphParticlePropertyWidget.
class FluidPropertyWidget : public SphParticlePropertyWidget
{
    Q_OBJECT

public:

    /// @brief constructor
    explicit FluidPropertyWidget(QWidget *parent = 0, std::shared_ptr<FluidProperty> _property = nullptr);

    /// @brief destructor
    virtual ~FluidPropertyWidget();

    /// @brief Setter for the m_property attribute
    virtual void SetProperty(std::shared_ptr<FluidProperty> _property);

    /// @brief Geter for the m_property attribute
    virtual FluidProperty *GetProperty();


signals:
    /// @brief Qt Signal to communicate that the FluidProperty has changed to other classes
    void PropertyChanged(std::shared_ptr<FluidProperty> _property);

public slots:
    /// @brief Qt Slot to be connected to any changes on this widget, emits PropertyChanged(m_property)
    virtual void OnPropertyChanged();


signals:


private:
    /// @brief
    Ui::FluidPropertyWidget *ui;

    /// @brief The underlying fluid property that is affected by this widget
    std::shared_ptr<FluidProperty> m_property;
};

//-----------------------------------------------------------------------------------------------------------

#endif // FLUIDPROPERTYWIDGET_H
