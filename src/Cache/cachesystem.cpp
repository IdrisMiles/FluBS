#include "Cache/cachesystem.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>


void glm::to_json(json& j, const glm::vec3& v)
{
    j = json({v.x, v.y, v.z});
}

void glm::from_json(const json& j, glm::vec3& v)
{
    v.x = j[0];
    v.y = j[1];
    v.z = j[2];
}



//--------------------------------------------------------------------------------------------------------------------

CacheSystem::CacheSystem(const int _numFrames)
{
    m_cachedFrames.resize(_numFrames);
}

//--------------------------------------------------------------------------------------------------------------------

CacheSystem::~CacheSystem()
{
    // write to file

    // destroy
    m_cachedFrames.clear();
}

//--------------------------------------------------------------------------------------------------------------------

void CacheSystem::Cache(const int _frame,
                        std::shared_ptr<FluidSystem> _fluidSystem)
{
    // Make sure our container is large enough to hold this frame
    while(m_cachedFrames.size() <= _frame)
    {
        m_cachedFrames.push_back(json());
    }


    Cache(_frame, "Fluid System", _fluidSystem);


    auto fluid = _fluidSystem->GetFluid();
    Cache(_frame, "Fluid", fluid);


    auto algae = _fluidSystem->GetAlgae();
    Cache(_frame, "Algae", algae);


    auto activeRigids = _fluidSystem->GetActiveRigids();
    for (int i=0; i<activeRigids.size(); ++i)
    {
        std::stringstream ss;
        ss << std::setw(4) << std::setfill('0') << i;

        Cache(_frame, "Active Rigid"+ss.str(), activeRigids[i]);
    }


    // Static rigids won't change throughout sim
    if(_frame == 0)
    {
        auto staticRigids = _fluidSystem->GetStaticRigids();
        for (int i=0; i<staticRigids.size(); ++i)
        {
            std::stringstream ss;
            ss << std::setw(4) << std::setfill('0') << i;

            Cache(_frame, "Static Rigid"+ss.str(), staticRigids[i]);
        }
    }
}

//--------------------------------------------------------------------------------------------------------------------

void CacheSystem::Load(const int _frame,
                       std::shared_ptr<FluidSystem> _fluidSystem)
{
    // Make sure our container is large enough to hold this frame
    if(m_cachedFrames.size() <= _frame)
    {
        return;
    }


    Load(_frame, "Fluid System", _fluidSystem);


    auto fluid = _fluidSystem->GetFluid();
    Load(_frame, "Fluid", fluid);


    auto algae = _fluidSystem->GetAlgae();
    Load(_frame, "Algae", algae);


    auto activeRigids = _fluidSystem->GetActiveRigids();
    for (int i=0; i<activeRigids.size(); ++i)
    {
        std::stringstream ss;
        ss << std::setw(4) << std::setfill('0') << i;

        Load(_frame, "Active Rigid"+ss.str(), activeRigids[i]);
    }


    // Static rigids won't change throughout sim
    if(_frame == 0)
    {
        auto staticRigids = _fluidSystem->GetStaticRigids();
        for (int i=0; i<staticRigids.size(); ++i)
        {
            std::stringstream ss;
            ss << std::setw(4) << std::setfill('0') << i;

            Load(_frame, "Static Rigid"+ss.str(), staticRigids[i]);
        }
    }
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

bool CacheSystem::IsFrameCached(const int _frame)
{
    if(m_cachedFrames.size() <= _frame)
    {
        return false;
    }

    return (!m_cachedFrames[_frame].empty());
}

//--------------------------------------------------------------------------------------------------------------------

void CacheSystem::ClearCache(const int frame)
{
    if(frame == -1)
    {
        for(auto &cf : m_cachedFrames)
        {
            cf.clear();
        }
    }
    else if(m_cachedFrames.size() > frame)
    {
        m_cachedFrames[frame].clear();
    }
}

//--------------------------------------------------------------------------------------------------------------------

void CacheSystem::Cache(const int _frame,
                        const std::string &_object,
                        std::shared_ptr<FluidSystem> _fluidSystem)
{
    auto props = _fluidSystem->GetProperty();
    m_cachedFrames[_frame][_object][m_dataId.deltaTime] = props->deltaTime;
    m_cachedFrames[_frame][_object][m_dataId.solveIterations] = props->solveIterations;
    m_cachedFrames[_frame][_object][m_dataId.gridRes] = props->gridResolution;
    m_cachedFrames[_frame][_object][m_dataId.cellWidth] = props->gridCellWidth;
}

//--------------------------------------------------------------------------------------------------------------------

void CacheSystem::Cache(const int _frame,
           const std::string &_object,
           const std::shared_ptr<Fluid> _fluid)
{
    std::vector<glm::vec3> pos;
    std::vector<glm::vec3> vel;
    std::vector<int> id;

    _fluid->GetPositions(pos);
    _fluid->GetVelocities(vel);
    _fluid->GetParticleIds(id);
    auto props = _fluid->GetProperty();

    m_cachedFrames[_frame][_object][m_dataId.pos] = pos;
    m_cachedFrames[_frame][_object][m_dataId.vel] = vel;
    m_cachedFrames[_frame][_object][m_dataId.particlId] = id;

    m_cachedFrames[_frame][_object][m_dataId.mass] = props->particleMass;
    m_cachedFrames[_frame][_object][m_dataId.radius] = props->particleRadius;
}

//--------------------------------------------------------------------------------------------------------------------

void CacheSystem::Cache(const int _frame,
           const std::string &_object,
           const std::shared_ptr<Algae> _algae)
{
    std::vector<glm::vec3> pos;
    std::vector<glm::vec3> vel;
    std::vector<int> id;
    std::vector<float> bio;

    _algae->GetPositions(pos);
    _algae->GetVelocities(vel);
    _algae->GetParticleIds(id);
    _algae->GetBioluminescentIntensities(bio);

    m_cachedFrames[_frame][_object][m_dataId.pos] = pos;
    m_cachedFrames[_frame][_object][m_dataId.vel] = vel;
    m_cachedFrames[_frame][_object][m_dataId.particlId] = id;
    m_cachedFrames[_frame][_object][m_dataId.bioluminescentIntensoty] = bio;
}

//--------------------------------------------------------------------------------------------------------------------

void CacheSystem::Cache(const int _frame,
                        const std::string &_object,
                        const std::shared_ptr<Rigid> _rigid)
{

}

//--------------------------------------------------------------------------------------------------------------------

void CacheSystem::Load(const int _frame,
                       const std::string &_object,
                       std::shared_ptr<FluidSystem> _fluidSystem)
{

}

//--------------------------------------------------------------------------------------------------------------------

void CacheSystem::Load(const int _frame,
                       const std::string &_object,
                       const std::shared_ptr<Fluid> _fluid)
{
    std::vector<glm::vec3> pos;
    std::vector<glm::vec3> vel;
    std::vector<int> id;
    auto props = _fluid->GetProperty();

    pos = m_cachedFrames[_frame][_object].at(m_dataId.pos).get<std::vector<glm::vec3>>();
    vel = m_cachedFrames[_frame][_object].at(m_dataId.vel).get<std::vector<glm::vec3>>();
    id = m_cachedFrames[_frame][_object].at(m_dataId.particlId).get<std::vector<int>>();

    props->particleMass = m_cachedFrames[_frame][_object][m_dataId.mass];
    props->particleRadius = m_cachedFrames[_frame][_object][m_dataId.radius];

    _fluid->SetPositions(pos);
    _fluid->SetVelocities(vel);
    _fluid->SetParticleIds(id);


}

//--------------------------------------------------------------------------------------------------------------------

void CacheSystem::Load(const int _frame,
                       const std::string &_object,
                       const std::shared_ptr<Algae> _algae)
{

}

//--------------------------------------------------------------------------------------------------------------------

void CacheSystem::Load(const int _frame,
                       const std::string &_object,
                       const std::shared_ptr<Rigid> _rigid)
{

}

//--------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------
