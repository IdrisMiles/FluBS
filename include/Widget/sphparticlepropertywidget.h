#ifndef SPHPARTICLEPROPERTYWIDGET_H
#define SPHPARTICLEPROPERTYWIDGET_H

//-----------------------------------------------------------------------------------------------------------

#include <QWidget>
#include <memory>
#include "SPH/sphparticlepropeprty.h"


//--------------------------------------------------------------------------------------------------------------
/// @author Idris Miles
/// @version 1.0
/// @date 01/06/2017
//--------------------------------------------------------------------------------------------------------------



namespace Ui {
class SphParticlePropertyWidget;
}


/// @class SphParticlePropertyWidget
/// @brief GUI widget for a sph property object. This class inherits from QWidget.
class SphParticlePropertyWidget : public QWidget
{
    Q_OBJECT

public:
    /// @brief constructor
    explicit SphParticlePropertyWidget(QWidget *parent = 0, SphParticleProperty _property = SphParticleProperty());

    /// @brief destructor
    virtual ~SphParticlePropertyWidget();

    /// @brief Method to add widgets to the layout dynamically
    void AddWidgetToGridLayout(QWidget *w, int col = 0, int rowSpan = 1, int colSpan = 1);

    /// @brief Setter for the m_property attribute
    virtual void SetProperty(SphParticleProperty _property);

    /// @brief Geter for the m_property attribute
    SphParticleProperty GetProperty();

public slots:
    /// @brief Qt Slot to be connected to any changes on this widget, emits PropertyChanged(m_property)
    virtual void OnPropertyChanged();

signals:
    /// @brief Qt Signal to communicate that the FluidProperty has changed to other classes
    void PropertyChanged(SphParticleProperty _property);

protected:
    void SetNumParticles(const int _numParticles);
    void SetParticleMass(const float _particlesMass);
    void SetParticleRadius(const float _particlesRadius);
    void SetRestDensity(const float _restDensity);

    int GetNumParticles();
    float GetParticleMass();
    float GetParticleRadius();
    float GetRestDensity();

private:
    Ui::SphParticlePropertyWidget *ui;
    int m_numRow;

    SphParticleProperty m_property;
};

//-----------------------------------------------------------------------------------------------------------

#endif // SPHPARTICLEPROPERTYWIDGET_H
