#ifndef CACHESYSTEM_H
#define CACHESYSTEM_H

#include <vector>
#include "json/src/json.hpp"
#include <glm/glm.hpp>

using json = nlohmann::json;

class CacheSystem
{
public:
    CacheSystem();
    ~CacheSystem();

    void CachePoints(const int _np,
                     const std::vector<glm::vec3> &_pos,
                     const std::vector<glm::vec3> &_vel,
                     const std::vector<float> &_illum,
                     const std::vector<int> &_id,
                     const int _frame,
                     const bool _append);



private:
    void CacheGlmVec3(const int _frame, const bool _append, const std::string _id, const std::vector<glm::vec3> &_vec);

    template<typename T>
    void CachePod(const int _frame, const bool _append, const std::string _id, const std::vector<T> &_vec);


    std::vector<json> m_cachedFrames;

};


//--------------------------------------------------------------------------------------------------------------------

template<typename T>
void CacheSystem::CachePod(const int _frame, const bool _append, const std::string _id, const std::vector<T> &_vec)
{
    if(_append)
    {
        for(auto &v : _vec)
        {
            m_cachedFrames[_frame][_id].push_back(json(v));
        }
    }
    else
    {
        m_cachedFrames[_frame][_id] = json::array();
        for(auto &v : _vec)
        {
            m_cachedFrames[_frame][_id].push_back(json(v));
        }
    }
}


#endif // CACHESYSTEM_H
