#ifndef SPHPARTICLEPROPERTYWIDGET_H
#define SPHPARTICLEPROPERTYWIDGET_H

#include <QWidget>

namespace Ui {
class SphParticlePropertyWidget;
}

class SphParticlePropertyWidget : public QWidget
{
    Q_OBJECT

public:
    explicit SphParticlePropertyWidget(QWidget *parent = 0);
    ~SphParticlePropertyWidget();

    void AddWidgetToGridLayout(QWidget *w, int col = 0, int rowSpan = 1, int colSpan = 1);

private:
    Ui::SphParticlePropertyWidget *ui;
    int m_numRow;
};

#endif // SPHPARTICLEPROPERTYWIDGET_H
