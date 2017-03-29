#include "Cache/cachesystem.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>


//--------------------------------------------------------------------------------------------------------------------

CacheSystem::CacheSystem()
{
}

//--------------------------------------------------------------------------------------------------------------------

CacheSystem::~CacheSystem()
{
    // write to file

    // destroy
    m_cachedFrames.clear();
}

//--------------------------------------------------------------------------------------------------------------------

void CacheSystem::CachePoints(const int _frame,
                              const bool _append,
                              const std::vector<glm::vec3> &_pos,
                              const std::vector<glm::vec3> &_vel,
                              const std::vector<float> &_illum,
                              const std::vector<int> &_id)
{
    if(m_cachedFrames.size() <= _frame)
    {
        if(m_cachedFrames.size() == _frame)
        {
            m_cachedFrames.push_back(json());
        }
        else
        {
            while(m_cachedFrames.size() <= _frame)
            {
                m_cachedFrames.push_back(json());
            }
        }
    }

    CacheGlmVec3(_frame, _append, "p", _pos);
    CacheGlmVec3(_frame, _append, "v", _vel);
    CachePod(_frame, _append, "i", _illum);
    CachePod(_frame, _append, "id", _id);
    m_cachedFrames[_frame]["np"] = _pos.size();
}

//--------------------------------------------------------------------------------------------------------------------

void CacheSystem::WriteCache(const int _frame)
{
    if(_frame == -1)
    {
        // cache all frames
        return;
    }

    if(_frame >= m_cachedFrames.size())
    {
        return;
    }


    std::stringstream ss;
    ss << std::setw(4) << std::setfill('0') << _frame;
    std::ofstream ofs("data_"+ss.str());
    if(ofs.is_open())
    {
        ofs << std::setw(4)<< m_cachedFrames[_frame] << std::endl;

        ofs.close();
    }

}

//--------------------------------------------------------------------------------------------------------------------

void CacheSystem::ReadPoints(const int _frame,
                             std::vector<glm::vec3> &_pos,
                             std::vector<glm::vec3> &_vel,
                             std::vector<float> &_illum,
                             std::vector<int> &_id)
{

}

//--------------------------------------------------------------------------------------------------------------------

void CacheSystem::CacheGlmVec3(const int _frame, const bool _append, const std::string _id, const std::vector<glm::vec3> &_vec)
{
    if(_append)
    {
        for(auto &v : _vec)
        {
            m_cachedFrames[_frame][_id].push_back(json({v.x, v.y, v.z}));
        }
    }
    else
    {
        m_cachedFrames[_frame][_id] = json::array({});
        for(auto &v : _vec)
        {
            m_cachedFrames[_frame][_id].push_back(json({v.x, v.y, v.z}));
        }
    }
}


//--------------------------------------------------------------------------------------------------------------------
