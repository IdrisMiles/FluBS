#ifndef SPHPARTICLEPROPERTYWIDGET_H
#define SPHPARTICLEPROPERTYWIDGET_H

#include <QWidget>
#include <memory>
#include "SPH/sphparticlepropeprty.h"


namespace Ui {
class SphParticlePropertyWidget;
}

class SphParticlePropertyWidget : public QWidget
{
    Q_OBJECT

public:
    explicit SphParticlePropertyWidget(QWidget *parent = 0, std::shared_ptr<SphParticleProperty> _property = nullptr);
    ~SphParticlePropertyWidget();

    void AddWidgetToGridLayout(QWidget *w, int col = 0, int rowSpan = 1, int colSpan = 1);

    virtual void SetProperty(std::shared_ptr<SphParticleProperty> _property);
    virtual SphParticleProperty *GetProperty();

public slots:

signals:
    void propertiesChanged();

private:
    Ui::SphParticlePropertyWidget *ui;
    int m_numRow;

    std::shared_ptr<SphParticleProperty> m_property;
};

#endif // SPHPARTICLEPROPERTYWIDGET_H
