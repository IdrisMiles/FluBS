#ifndef CACHESYSTEM_H
#define CACHESYSTEM_H

#include <vector>

#include <glm/glm.hpp>

#include "json/src/json.hpp"

#include "FluidSystem/fluidsystem.h"


using json = nlohmann::json;

namespace glm {

void to_json(json& j, const glm::vec3& v);

void from_json(const json& j, glm::vec3& v);

}



class CacheSystem
{
public:
    CacheSystem(const int _numFrames = 100);
    ~CacheSystem();

    void Cache(const int _frame,
               std::shared_ptr<FluidSystem> _fluidSystem);

    void Load(const int _frame,
              std::shared_ptr<FluidSystem> _fluidSystem);

    void WriteCache(const int _frame = -1);

    bool IsFrameCached(const int _frame);

    void ClearCache(const int frame = -1);


private:
    const struct DataId{
        std::string pos = "p";
        std::string vel = "v";
        std::string particlId = "id";
        std::string bioluminescentIntensoty = "bio";
        std::string mass = "mass";
        std::string radius = "radius";

        std::string deltaTime = "dt";
        std::string solveIterations = "solve iterations";
        std::string gridRes = "grid res";
        std::string cellWidth = "cell width";
    } m_dataId;

    void Cache(const int _frame,
               const std::string &_object,
               std::shared_ptr<FluidSystem> _fluidSystem);

    void Cache(const int _frame,
               const std::string &_object,
               const std::shared_ptr<Fluid> _fluid);

    void Cache(const int _frame,
               const std::string &_object,
               const std::shared_ptr<Algae> _algae);

    void Cache(const int _frame,
               const std::string &_object,
               const std::shared_ptr<Rigid> _rigid);

    void Load(const int _frame,
               const std::string &_object,
               std::shared_ptr<FluidSystem> _fluidSystem);

    void Load(const int _frame,
               const std::string &_object,
               const std::shared_ptr<Fluid> _fluid);

    void Load(const int _frame,
               const std::string &_object,
               const std::shared_ptr<Algae> _algae);

    void Load(const int _frame,
               const std::string &_object,
               const std::shared_ptr<Rigid> _rigid);


    std::vector<json> m_cachedFrames;

};



#endif // CACHESYSTEM_H
