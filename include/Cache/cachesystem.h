#ifndef CACHESYSTEM_H
#define CACHESYSTEM_H

#include <vector>
#include <thread>
#include <utility>
#include <functional>

#include <glm/glm.hpp>

#include "json/src/json.hpp"

#include "FluidSystem/fluidsystem.h"


#define IS_CACHED(status) (status & CacheStatus::CACHED)
#define IS_CACHED_TO_MEMORY(status) (status & CacheStatus::MEMORY)
#define IS_CACHED_TO_DISK(status) (status & CacheStatus::DISK)
#define IS_NOT_CACHED(status) (status & CacheStatus::NOTCACHED)
#define IS_DIRTY(status) (status & CacheStatus::DIRTY)

using json = nlohmann::json;

namespace glm {

void to_json(json& j, const glm::vec3& v);

void from_json(const json& j, glm::vec3& v);

}

typedef uint8_t foo;
enum CacheStatus
{
    NOTCACHED = (1<<0), // 00000001
    DIRTY = (1<<1),     // 00000010
    CACHED = (1<<2),    // 00000100
    MEMORY = (1<<3),    // 00001000
    DISK = (1<<4),      // 00010000
};


class CacheSystem
{
public:
    CacheSystem(const int _numFrames = 10);
    ~CacheSystem();

    void Cache(const int _frame, std::shared_ptr<FluidSystem> _fluidSystem);

    void Load(const int _frame, std::shared_ptr<FluidSystem> _fluidSystem);

    void WriteCache(const int _frame = -1);

    void CacheOutToDisk(std::string _fileName);

    bool IsFrameCached(const int _frame);

    void ClearCache(const int frame = -1);

    void SetFrameRange(int start, int end);


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

    void Cache(json &_frame, const std::string &_object, std::shared_ptr<FluidSystem> _fluidSystem);

    void Cache(json &_frame, const std::string &_object, const std::shared_ptr<Fluid> _fluid);

    void Cache(json &_frame, const std::string &_object, const std::shared_ptr<Algae> _algae);

    void Cache(json &_frame, const std::string &_object, const std::shared_ptr<Rigid> _rigid);

    void Load(const json &_frame, const std::string &_object, std::shared_ptr<FluidSystem> _fluidSystem);

    void Load(const json &_frame, const std::string &_object, const std::shared_ptr<Fluid> _fluid);

    void Load(const json &_frame, const std::string &_object, const std::shared_ptr<Algae> _algae);

    void Load(const json &_frame, const std::string &_object, const std::shared_ptr<Rigid> _rigid);


    void WriteToDisk(const std::string _file, const json &_object);

    void LoadFromMemory(json &_frame, std::shared_ptr<FluidSystem> _fluidSystem);

    void LoadFromDisk(const std::string _file, json &_object);


    std::vector<json> m_cachedFrames;
    std::vector<foo> m_isFrameCached;
    std::vector<std::thread> m_threads;
    int m_threadHead;

};



#endif // CACHESYSTEM_H
