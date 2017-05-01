#ifndef SOLVERPROPERTYWIDGET_H
#define SOLVERPROPERTYWIDGET_H

#include <QWidget>
#include <QLabel>
#include <QDoubleSpinBox>
#include <QGridLayout>
#include <QPushButton>

#include <memory>

#include "FluidSystem/fluidsystem.h"
#include "FluidSystem/fluidsolverproperty.h"


namespace Ui {
class SolverPropertyWidget;
}

class SolverPropertyWidget : public QWidget
{
    Q_OBJECT

public:
    explicit SolverPropertyWidget(QWidget *parent = 0, std::shared_ptr<FluidSolverProperty> _property = nullptr);
    virtual ~SolverPropertyWidget();



    /// @brief Setter for the m_property attribute
    virtual void SetProperty(std::shared_ptr<FluidSolverProperty> _property);

    /// @brief Geter for the m_property attribute
    virtual FluidSolverProperty *GetProperty();


signals:
    /// @brief Qt Signal to communicate that the FluidProperty has changed to other classes
    void PropertyChanged(std::shared_ptr<FluidSolverProperty> _property);

public slots:
    /// @brief Qt Slot to be connected to any changes on this widget, emits PropertyChanged(m_property)
    virtual void OnPropertyChanged();

private:
    Ui::SolverPropertyWidget *ui;

    std::shared_ptr<FluidSolverProperty> m_property;


};

#endif // SOLVERPROPERTYWIDGET_H
