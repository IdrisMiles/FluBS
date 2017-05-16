#include "Cache/cachesystem.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <mutex>


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
    m_cacheDir = ".cache_";
    m_cacheFileName = "data_";

    m_cachedFrames.resize(_numFrames);
    m_isFrameCached.resize(_numFrames, CacheStatus::NOTCACHED);

    m_threads.resize(std::thread::hardware_concurrency()-1);
    m_threadHead = 0;
}

//--------------------------------------------------------------------------------------------------------------------

CacheSystem::~CacheSystem()
{
    // write to file

    // destroy
    m_cachedFrames.clear();
    m_isFrameCached.clear();

    if(m_threads[m_threadHead].joinable())
    {
        m_threads[m_threadHead].join();
    }
}

//--------------------------------------------------------------------------------------------------------------------

void CacheSystem::Cache(const int _frame,
                        std::shared_ptr<FluidSystem> _fluidSystem)
{
    // Make sure our container is large enough to hold this frame
    if(m_cachedFrames.size() <= _frame)
    {
        m_cachedFrames.resize(_frame+50, json());
        m_isFrameCached.resize(_frame+50, CacheStatus::NOTCACHED);
    }


    m_isFrameCached[_frame] &= ~(CacheStatus::NOTCACHED | CacheStatus::DIRTY);
    m_isFrameCached[_frame] |= (CacheStatus::CACHED | CacheStatus::MEMORY);

    Cache(m_cachedFrames[_frame], "Fluid System", _fluidSystem);


    auto fluid = _fluidSystem->GetFluid();
    Cache(m_cachedFrames[_frame], "Fluid", fluid);


    auto algae = _fluidSystem->GetAlgae();
    Cache(m_cachedFrames[_frame], "Algae", algae);


    auto activeRigids = _fluidSystem->GetActiveRigids();
    for (int i=0; i<activeRigids.size(); ++i)
    {
        std::stringstream ss;
        ss << std::setw(4) << std::setfill('0') << i;

        Cache(m_cachedFrames[_frame], "Active Rigid"+ss.str(), activeRigids[i]);
    }


    // Static rigids won't change throughout sim
    if(_frame == 0)
    {
        auto staticRigids = _fluidSystem->GetStaticRigids();
        for (int i=0; i<staticRigids.size(); ++i)
        {
            std::stringstream ss;
            ss << std::setw(4) << std::setfill('0') << i;

            Cache(m_cachedFrames[_frame], "Static Rigid"+ss.str(), staticRigids[i]);
        }
    }


    // write to disk
//    WriteCacheToDisk(_frame);
}

//--------------------------------------------------------------------------------------------------------------------

void CacheSystem::Load(const int _frame, std::shared_ptr<FluidSystem> _fluidSystem)
{
    // Make sure our container holds this frame
    if(m_cachedFrames.size() <= _frame && m_isFrameCached.size() <= _frame)
    {
        return;
    }

    // Make sure this frame has been cached
    if(IS_NOT_CACHED(m_isFrameCached[_frame]) || (m_isFrameCached[_frame] & CacheStatus::DIRTY) || m_cachedFrames[_frame].empty())
    {
        return;
    }


    // Load cached from disk to memory if necessary
    if(IS_CACHED(m_isFrameCached[_frame]) && IS_CACHED_TO_DISK(m_isFrameCached[_frame]) && !IS_CACHED_TO_MEMORY(m_isFrameCached[_frame]))
    {
        std::stringstream ss;
        ss << std::setw(4) << std::setfill('0') << _frame;
        std::string fileName = m_cacheDir+m_cacheFileName+ss.str()+".json";

        std::cout<<"cached to disk not memory\n";
        LoadFromDisk(fileName, m_cachedFrames[_frame]);
    }


    LoadFromMemory(m_cachedFrames[_frame], _fluidSystem);

}


//--------------------------------------------------------------------------------------------------------------------

void CacheSystem::WriteCacheToDisk(const int _frame)
{
    if(_frame == -1)
    {
        // cache all frames
        return;
    }

    if(_frame >= m_cachedFrames.size() && _frame > m_isFrameCached.size())
    {
        return;
    }

    std::stringstream ss;
    ss << std::setw(4) << std::setfill('0') << _frame;
    std::string fileName = m_cacheDir+m_cacheFileName+ss.str()+".json";


    // make thread avaiable
    if(m_threads[m_threadHead].joinable())
    {
        m_threads[m_threadHead].join();
    }

    // write cache to disk asynchronously
    m_threads[m_threadHead] = std::thread(&CacheSystem::CacheToDisk, this, fileName, std::ref(m_cachedFrames[_frame]));
//    m_threads[m_threadHead].detach();
    m_threadHead = (m_threadHead+1)%m_threads.size();


    // mark frame as cached to disk
    m_isFrameCached[_frame] |= (CacheStatus::CACHED | CacheStatus::DISK);
    m_isFrameCached[_frame] &= ~(CacheStatus::NOTCACHED | CacheStatus::DIRTY);
}

//--------------------------------------------------------------------------------------------------------------------


void CacheSystem::CacheOutToDisk(std::string _fileName, QProgressBar *_progress)
{
    int counter = 0;
    std::mutex progressMutex;
    std::thread::id mainThreadId = std::this_thread::get_id();
    auto threadFunc = [this, &counter, &progressMutex, _progress, mainThreadId, _fileName](int startId, int endId){
        for(int i=startId; i<endId; i++)
        {
            if(!IS_CACHED(m_isFrameCached[i]))
            {
                continue;
            }

            std::stringstream ss;
            ss << std::setw(4) << std::setfill('0') << i;

            bool notInMemory = false;
            if(!IS_CACHED_TO_MEMORY(m_isFrameCached[i]) && IS_CACHED_TO_DISK(m_isFrameCached[i]))
            {
                std::string fileName = m_cacheDir+m_cacheFileName+ss.str()+".json";
                LoadFromDisk(fileName, m_cachedFrames[i]);
                m_isFrameCached[i] |= CacheStatus::MEMORY;
                notInMemory = true;
            }

            if(!IS_CACHED_TO_MEMORY(m_isFrameCached[i]))
            {
//                if(IS_CACHED_TO_DISK(m_isFrameCached[i]))
//                {
//                    // TODO
//                    // Just copy file
//                }

                continue;
            }

            // write to disk
            std::string fileName = _fileName+ss.str()+".json";
            CacheToDisk(fileName, m_cachedFrames[i]);

            if(notInMemory)
            {
                m_cachedFrames[i].clear();
            }

            progressMutex.lock();
            counter++;
            if(std::this_thread::get_id() == mainThreadId)
            {
                _progress->setValue(counter);
            }
            progressMutex.unlock();
        }
    };

    int numThreads = m_threads.size() + 1;
    int dataSize = m_cachedFrames.size();
    int chunkSize = dataSize / numThreads;
    int numBigChunks = dataSize % numThreads;
    int bigChunkSize = chunkSize + (numBigChunks>0 ? 1 : 0);
    int startChunk = 0;
    int threadId=0;

    // join threads in case
    for(int i=0; i<m_threads.size(); i++)
    {
        if(m_threads[i].joinable())
        {
            m_threads[i].join();
        }
    }

    _progress->setMaximum(dataSize);


    // multi-threaded cache to disk
    for(threadId=0; threadId<numBigChunks; threadId++)
    {
       m_threads[threadId] = std::thread(threadFunc, startChunk, startChunk+bigChunkSize);
       startChunk+=bigChunkSize;
    }
    for(; threadId<numThreads-1; threadId++)
    {
       m_threads[threadId] = std::thread(threadFunc, startChunk, startChunk+chunkSize);
       startChunk+=chunkSize;
    }
    threadFunc(startChunk, m_cachedFrames.size());


    // clean up and join threads
    for(int i=0; i<m_threads.size(); i++)
    {
       if(m_threads[i].joinable())
       {
           m_threads[i].join();
       }
    }

    _progress->setValue(dataSize);
}

//--------------------------------------------------------------------------------------------------------------------

void CacheSystem::LoadCacheFromDisk(std::vector<std::string> _fileNames, QProgressBar *_progress)
{
    m_cachedFrames.clear();
    m_isFrameCached.clear();

    m_cachedFrames.resize(_fileNames.size(), json());
    m_isFrameCached.resize(_fileNames.size(), CacheStatus::NOTCACHED);

    int counter = 0;
    std::mutex progressMutex;
    std::thread::id mainThreadId = std::this_thread::get_id();
    auto threadFunc = [this, &counter, &progressMutex, _progress, mainThreadId, _fileNames](int startId, int endId){
        for(int i=startId; i<endId; i++)
        {
            LoadFromDisk(_fileNames[i], m_cachedFrames[i]);
            m_isFrameCached[i] = (CacheStatus::CACHED | CacheStatus::MEMORY);

            progressMutex.lock();
            counter++;
            if(std::this_thread::get_id() == mainThreadId)
            {
                _progress->setValue(counter);
            }
            progressMutex.unlock();
        }
    };


    int numThreads = m_threads.size() + 1;
    int dataSize = m_cachedFrames.size();
    int chunkSize = dataSize / numThreads;
    int numBigChunks = dataSize % numThreads;
    int bigChunkSize = chunkSize + (numBigChunks>0 ? 1 : 0);
    int startChunk = 0;
    int threadId=0;

    // join threads in case
    for(int i=0; i<m_threads.size(); i++)
    {
        if(m_threads[i].joinable())
        {
            m_threads[i].join();
        }
    }

    _progress->setMaximum(dataSize);

    // multi-threaded cache to disk
    for(threadId=0; threadId<numBigChunks; threadId++)
    {
       m_threads[threadId] = std::thread(threadFunc, startChunk, startChunk+bigChunkSize);
       startChunk+=bigChunkSize;
    }
    for(; threadId<numThreads-1; threadId++)
    {
       m_threads[threadId] = std::thread(threadFunc, startChunk, startChunk+chunkSize);
       startChunk+=chunkSize;
    }
    threadFunc(startChunk, m_cachedFrames.size());


    // clean up and join threads
    for(int i=0; i<m_threads.size(); i++)
    {
       if(m_threads[i].joinable())
       {
           m_threads[i].join();
       }
    }

    _progress->setValue(dataSize);
}

//--------------------------------------------------------------------------------------------------------------------

void CacheSystem::CacheToDisk(const std::string _file, const json &_object)
{
    std::ofstream ofs(_file);
    if(ofs.is_open())
    {
        ofs << std::setw(4)<< _object << std::endl;

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

    return (IS_CACHED(m_isFrameCached[_frame]));
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

        for(auto &fr : m_isFrameCached)
        {
            fr = CacheStatus::NOTCACHED;
        }
    }
    else if(m_cachedFrames.size() > frame)
    {
        m_cachedFrames[frame].clear();
        m_isFrameCached[frame] = CacheStatus::NOTCACHED;
    }
}

//--------------------------------------------------------------------------------------------------------------------

void CacheSystem::SetFrameRange(int start, int end)
{
    assert(start <= end);

    int range = end - start;
    range = range < 1 ? 1 : range;

    m_cachedFrames.resize(range);
    m_isFrameCached.resize(range, CacheStatus::NOTCACHED);
}

//--------------------------------------------------------------------------------------------------------------------

void CacheSystem::Cache(json &_frame, const std::string &_object, std::shared_ptr<FluidSystem> _fluidSystem)
{
    auto props = _fluidSystem->GetProperty();
    _frame[_object][m_dataId.deltaTime] = props.deltaTime;
    _frame[_object][m_dataId.solveIterations] = props.solveIterations;
    _frame[_object][m_dataId.gridRes] = props.gridResolution;
    _frame[_object][m_dataId.cellWidth] = props.gridCellWidth;
}

//--------------------------------------------------------------------------------------------------------------------

void CacheSystem::Cache(json &_frame, const std::string &_object, const std::shared_ptr<Fluid> _fluid)
{
    std::vector<glm::vec3> pos;
    std::vector<glm::vec3> vel;
    std::vector<int> id;

    _fluid->GetPositions(pos);
    _fluid->GetVelocities(vel);
    _fluid->GetParticleIds(id);
    auto props = _fluid->GetProperty();

    _frame[_object][m_dataId.pos] = pos;
    _frame[_object][m_dataId.vel] = vel;
    _frame[_object][m_dataId.particlId] = id;

    _frame[_object][m_dataId.mass] = props->particleMass;
    _frame[_object][m_dataId.radius] = props->particleRadius;
}

//--------------------------------------------------------------------------------------------------------------------

void CacheSystem::Cache(json &_frame, const std::string &_object, const std::shared_ptr<Algae> _algae)
{
    std::vector<glm::vec3> pos;
    std::vector<glm::vec3> vel;
    std::vector<int> id;
    std::vector<float> bio;
    auto props = _algae->GetProperty();

    _algae->GetPositions(pos);
    _algae->GetVelocities(vel);
    _algae->GetParticleIds(id);
    _algae->GetBioluminescentIntensities(bio);

    _frame[_object][m_dataId.pos] = pos;
    _frame[_object][m_dataId.vel] = vel;
    _frame[_object][m_dataId.particlId] = id;
    _frame[_object][m_dataId.bioluminescentIntensoty] = bio;

    _frame[_object][m_dataId.mass] = props->particleMass;
    _frame[_object][m_dataId.radius] = props->particleRadius;
}

//--------------------------------------------------------------------------------------------------------------------

void CacheSystem::Cache(json &_frame, const std::string &_object, const std::shared_ptr<Rigid> _rigid)
{

}

//--------------------------------------------------------------------------------------------------------------------

void CacheSystem::Load(const json &_frame, const std::string &_object, std::shared_ptr<FluidSystem> _fluidSystem)
{
    if(_frame[_object].empty())
    {
        return;
    }

    try
    {
        auto props = _fluidSystem->GetProperty();
        props.deltaTime        = _frame[_object][m_dataId.deltaTime];
        props.solveIterations  = _frame[_object][m_dataId.solveIterations];
        props.gridResolution   = _frame[_object][m_dataId.gridRes];
        props.gridCellWidth    = _frame[_object][m_dataId.cellWidth];
    }
    catch(std::exception e)
    {
        std::cout<<e.what()<<"\t|FluidSystem\n";
    }
}

//--------------------------------------------------------------------------------------------------------------------

void CacheSystem::Load(const json &_frame, const std::string &_object, const std::shared_ptr<Fluid> _fluid)
{
    if(_frame[_object].empty())
    {
        return;
    }

    std::vector<glm::vec3> pos;
    std::vector<glm::vec3> vel;
    std::vector<int> id;
    auto props = _fluid->GetProperty();

    try
    {
        pos = _frame[_object].at(m_dataId.pos).get<std::vector<glm::vec3>>();
        vel = _frame[_object].at(m_dataId.vel).get<std::vector<glm::vec3>>();
        id = _frame[_object].at(m_dataId.particlId).get<std::vector<int>>();

        props->particleMass = _frame[_object][m_dataId.mass];
        props->particleRadius = _frame[_object][m_dataId.radius];

        _fluid->SetPositions(pos);
        _fluid->SetVelocities(vel);
        _fluid->SetParticleIds(id);
    }
    catch(std::exception e)
    {
        std::cout<<e.what()<<"\t|Fluid\n";
    }


}

//--------------------------------------------------------------------------------------------------------------------

void CacheSystem::Load(const json &_frame, const std::string &_object, const std::shared_ptr<Algae> _algae)
{
    if(_frame[_object].empty())
    {
        return;
    }
    std::vector<glm::vec3> pos;
    std::vector<glm::vec3> vel;
    std::vector<int> id;
    std::vector<float> bio;
    auto props = _algae->GetProperty();

    try
    {
        pos = _frame[_object].at(m_dataId.pos).get<std::vector<glm::vec3>>();
        vel = _frame[_object].at(m_dataId.vel).get<std::vector<glm::vec3>>();
        id = _frame[_object].at(m_dataId.particlId).get<std::vector<int>>();
        bio = _frame[_object].at(m_dataId.bioluminescentIntensoty).get<std::vector<float>>();

        props->particleMass = _frame[_object][m_dataId.mass];
        props->particleRadius = _frame[_object][m_dataId.radius];

        _algae->SetPositions(pos);
        _algae->SetVelocities(vel);
        _algae->SetParticleIds(id);
        _algae->SetBioluminescentIntensities(bio);
    }
    catch(std::exception e)
    {
        std::cout<<e.what()<<"\t|Algae\n";
    }

}

//--------------------------------------------------------------------------------------------------------------------

void CacheSystem::Load(const json &_frame,
                       const std::string &_object,
                       const std::shared_ptr<Rigid> _rigid)
{

}

//--------------------------------------------------------------------------------------------------------------------

void CacheSystem::LoadFromMemory(json &_frame, std::shared_ptr<FluidSystem> _fluidSystem)
{
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
    auto staticRigids = _fluidSystem->GetStaticRigids();
    for (int i=0; i<staticRigids.size(); ++i)
    {
        std::stringstream ss;
        ss << std::setw(4) << std::setfill('0') << i;

        Load(m_cachedFrames[0], "Static Rigid"+ss.str(), staticRigids[i]);
    }
}

//--------------------------------------------------------------------------------------------------------------------

void CacheSystem::LoadFromDisk(const std::string _file, json &_object)
{
    std::ifstream i(_file);
    i >> _object;
}

//--------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------
