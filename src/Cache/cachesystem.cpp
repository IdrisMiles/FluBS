//#include "Cache/cachesystem.h"

//#include <iostream>


////--------------------------------------------------------------------------------------------------------------------

//CacheSystem::CacheSystem()
//{
//}

////--------------------------------------------------------------------------------------------------------------------

//CacheSystem::~CacheSystem()
//{
//    // write to file

//    // destroy
//    m_cachedFrames.clear();
//}

////--------------------------------------------------------------------------------------------------------------------

//void CacheSystem::CachePoints(const int _np,
//                              const std::vector<glm::vec3> &_pos,
//                              const std::vector<glm::vec3> &_vel,
//                              const std::vector<float> &_illum,
//                              const std::vector<int> &_id,
//                              const int _frame,
//                              const bool _append)
//{
//    CacheGlmVec3(_frame, _append, "p", _pos);
//    CacheGlmVec3(_frame, _append, "v", _vel);
//    CachePod(_frame, _append, "i", _illum);
//    CachePod(_frame, _append, "id", _id);
//    m_cachedFrames[_frame]["np"] = _np;
//}

////--------------------------------------------------------------------------------------------------------------------

//void CacheSystem::CacheGlmVec3(const int _frame, const bool _append, const std::string _id, const std::vector<glm::vec3> &_vec)
//{
//    if(_append)
//    {
//        for(auto &v : _vec)
//        {
//            m_cachedFrames[_frame][_id].push_back(json({v.x, v.y, v.z}));
//        }
//    }
//    else
//    {
//        for(auto &v : _vec)
//        {
//            m_cachedFrames[_frame][_id] = json::array({});
//            m_cachedFrames[_frame][_id].push_back(json({v.x, v.y, v.z}));
//        }
//    }
//}


////--------------------------------------------------------------------------------------------------------------------
