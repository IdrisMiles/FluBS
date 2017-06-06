#ifndef RIGIDPROPERTYWIDGET_H
#define RIGIDPROPERTYWIDGET_H

//--------------------------------------------------------------------------------------------------------------

#include <QWidget>
#include "Widget/sphparticlepropertywidget.h"

#include "SPH/rigidproperty.h"

//--------------------------------------------------------------------------------------------------------------
/// @author Idris Miles
/// @version 1.0
/// @date 01/06/2017
//--------------------------------------------------------------------------------------------------------------


namespace Ui {
class RigidPropertyWidget;
}

/// @class RigidPropertyWidget
/// @brief GUI widget for a rigid property object. This class inherits from SphParticlePropertyWidget.
class RigidPropertyWidget : public SphParticlePropertyWidget
{
    Q_OBJECT

public:
    /// @brief constructor
    explicit RigidPropertyWidget(QWidget *parent = 0, RigidProperty _property = RigidProperty(),
                                 float posX = 0.0f, float posY = 0.0f, float posZ = 0.0f, float rotX = 0.0f, float rotY = 0.0f, float rotZ = 0.0f);

    /// @brief destructor
    ~RigidPropertyWidget();

    /// @brief Setter for the m_property attribute
    virtual void SetProperty(RigidProperty _property);

    /// @brief Geter for the m_property attribute
    virtual RigidProperty GetProperty();

    /// @brief Method to set transform values on widget
    void SetTransform(float posX, float posY, float posZ, float rotX, float rotY, float rotZ);

signals:
    /// @brief Qt Signal to communicate, that the FluidProperty has changed, to other classes
    void PropertyChanged(RigidProperty _property);

    /// @brief Qt Signal to communicate, that the transform has changed, to other classes
    void TransformChanged(float posX, float posY, float posZ, float rotX, float rotY, float rotZ, float scaleX, float scaleY, float scaleZ);

public slots:
    /// @brief Qt Slot to be connected to any changes on this widget, emits PropertyChanged(m_property)
    virtual void OnPropertyChanged();
    void OnTransformChanged();

private:
    Ui::RigidPropertyWidget *ui;

    RigidProperty m_property;
};

//--------------------------------------------------------------------------------------------------------------

#endif // RIGIDPROPERTYWIDGET_H
