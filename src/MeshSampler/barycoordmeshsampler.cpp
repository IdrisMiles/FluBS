#include "MeshSampler/barycoordmeshsampler.h"
#include <vector>
#include <algorithm>
#include <random>
#include <iostream>

struct AcceptSample
{
    AcceptSample(const glm::vec3 &_samplePoint, const float _sampleDist) :
        accept(false),
        samplePoint(_samplePoint),
        sampleDist(_sampleDist)
    {}

    void operator()(const glm::vec3 &activePoint)
    {
        accept |= (glm::distance(activePoint, samplePoint) >= sampleDist);
    }

    glm::vec3 samplePoint;
    float sampleDist;
    bool accept;
};

Mesh MeshSampler::BaryCoord::SampleMesh(const Mesh &_mesh, const int _numSamples)
{

    // iterate through triangles and get their area
    std::vector<float> triAreas;
    for(auto &tri : _mesh.tris)
    {
        glm::vec3 v1 = _mesh.verts[tri[0]];
        glm::vec3 v2 = _mesh.verts[tri[1]];
        glm::vec3 v3 = _mesh.verts[tri[2]];

        glm::vec3 e1 = v2 - v1;
        glm::vec3 e2 = v3 - v1;

        float area = 0.5f * glm::length(glm::cross(e1, e2));

        triAreas.push_back(area);
    }


    // Get smallest triangle area
    float smallestArea = *std::min_element(triAreas.begin(), triAreas.end());


    // Get the proprtion of all triangle areas relative to smallest area
    std::vector<unsigned int> triAreaProportions;
    for(auto &triArea : triAreas)
    {
        unsigned int propirtion = ceil(triArea / smallestArea);
        triAreaProportions.push_back(propirtion);
    }


    // Generate Sample Probability Vector
    std::vector<unsigned int> sampleTriProbability;
    unsigned int triIndex=0;
    for(auto &triAreaPro : triAreaProportions)
    {
        for(unsigned int j=0; j<triAreaPro; j++)
        {
            sampleTriProbability.push_back(triIndex);
        }

        triIndex++;
    }


    // figure out our sampling criteria
    float totalArea = std::accumulate(triAreas.begin(), triAreas.end(), 0);
    float sampleArea = totalArea / _numSamples;
    float sampleDist = (sqrt(sampleArea));


    // initiliase random number generator
    std::default_random_engine prg;
    std::uniform_int_distribution<unsigned int> randDistTriIndex(0, sampleTriProbability.size()-1);
    std::uniform_real_distribution<float> randDistBarycentric(0.0f, 1.0f);


    // Get sample points
    int currIteration;
    int maxIterations = 1000;
    Mesh samples;
    while(samples.verts.size() < _numSamples && currIteration < maxIterations)
    {
        unsigned int sampleTriProbabilityIndex = randDistTriIndex(prg);
//        if(sampleTriProbabilityIndex >= sampleTriProbability.size())
//        {
//            sampleTriProbabilityIndex = sampleTriProbability.size()-1;
//        }
        triIndex = sampleTriProbability[sampleTriProbabilityIndex];
//        std::cout<<"sampleTriProb size "<<sampleTriProbability.size()<<"\n";
//        std::cout<<"sampleTriProbIndex "<<sampleTriProbabilityIndex<<"\n";

        glm::vec3 v1 = _mesh.verts[_mesh.tris[triIndex][0]];
        glm::vec3 v2 = _mesh.verts[_mesh.tris[triIndex][1]];
        glm::vec3 v3 = _mesh.verts[_mesh.tris[triIndex][2]];

        float u = randDistBarycentric(prg);
        float v = randDistBarycentric(prg);
        if(u+v > 1.0)
        {
            v = 1.0f - u;
        }
        float w = 1.0f - (u+v);

        glm::vec3 samplePoint = (u*v1) + (v*v2) + (w*v3);


        if(currIteration == 0)
        {
            samples.verts.push_back(samplePoint);
            samples.norms.push_back(_mesh.norms[_mesh.tris[triIndex][0]]);
        }
        else
        {
            // Check samplePoint meets sampling criteria before adding to sample list
            AcceptSample acceptSample(samplePoint, sampleDist);
            acceptSample = std::for_each(samples.verts.begin(), samples.verts.end(), acceptSample);

            if(acceptSample.accept)
            {
                samples.verts.push_back(samplePoint);
                samples.norms.push_back(_mesh.norms[_mesh.tris[triIndex][0]]);
            }
        }


        currIteration++;
    }


    return samples;
}
