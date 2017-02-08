#ifndef FLUIDSOLVERPROPERTY_H
#define FLUIDSOLVERPROPERTY_H


class FluidSolverProperty
{
public:
    FluidSolverProperty(float _smoothingLength = 1.2f,
                        float _deltaTime = 0.005f,
                        unsigned int _solveIterations = 1,
                        unsigned int _gridResolution = 10,
                        float _gridCellWidth = 1.2f,
                        bool _play = false);

    ~FluidSolverProperty();



    float smoothingLength;

    float deltaTime;
    unsigned int solveIterations;
    unsigned int gridResolution;
    float gridCellWidth;
    float gridVolume;

    bool play;
};

#endif // FLUIDSOLVERPROPERTY_H
