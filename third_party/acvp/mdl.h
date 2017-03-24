#ifndef MDL_H
#define MDL_H

//#include "stdafx.h"

#include <vector> // STL
#include <cstddef>

#define LIBALIGN_IMEXPORT

namespace align {

enum MdlStatus {MDL_INVALID, MDL_VALID, MDL_ACCEPT};
    class MdlGraph;

class LIBALIGN_IMEXPORT  MdlSet {
    friend class MdlGraph;
public:
    typedef double (*ComputeNFA)(const MdlGraph& g, const MdlSet& set);

    MdlSet(void* p = NULL)
    : edges(), id(p), _cacheNFA(0), _cache(false), _status(MDL_VALID) {}
    ~MdlSet() {}

    enum MdlStatus status() const { return _status; }
    inline double nfa(const MdlGraph& g, ComputeNFA cmpNFA) const;

    std::vector<unsigned int> edges; // Forward edges
    void* id; // Identification
private:
    double _cacheNFA; // Cached value of NFA
    bool _cache; // Above value up to date?
    enum MdlStatus _status;
};

class LIBALIGN_IMEXPORT  MdlPoint {
public:
    MdlPoint() : edges() {}
    ~MdlPoint() {}
    std::vector<unsigned int> edges; // Backward edges
};

class LIBALIGN_IMEXPORT  MdlEdge {
    friend class MdlGraph;
public:
    MdlEdge(unsigned int sn, unsigned int pn)
    : setNode(sn), pointNode(pn), _status(MDL_VALID) {}
    ~MdlEdge() {}

    inline MdlEdge& operator=(const MdlEdge& edge);
    enum MdlStatus status() const { return _status; }
    const unsigned int setNode, pointNode;
private:
    enum MdlStatus _status;
};

class LIBALIGN_IMEXPORT  MdlGraph {
public:
    explicit MdlGraph(unsigned int nPoints);
    ~MdlGraph() {}

    unsigned int new_set(void* p = NULL);
    void link(unsigned int iSet, unsigned int iPoint);
    void mdl(MdlSet::ComputeNFA nfa, double maxNFA);

    std::vector<MdlEdge> edges;
    std::vector<MdlSet> setNodes;
    std::vector<MdlPoint> pointNodes;
private:
    unsigned int best_set(MdlSet::ComputeNFA nfa, double maxNFA);
};

// Defined explicitely because const-cast necessary
MdlEdge& MdlEdge::operator=(const MdlEdge& edge)
{
    if(this != &edge) {
	const_cast<unsigned int&>(setNode) = edge.setNode;
	const_cast<unsigned int&>(pointNode) = edge.pointNode;
	_status = edge._status;
    }
    return *this;
}

// Return NFA of set
double MdlSet::nfa(const MdlGraph& g, ComputeNFA cmpNFA) const
{
    if(! _cache) {
	const_cast<MdlSet*>(this)->_cache = true;
	const_cast<MdlSet*>(this)->_cacheNFA = cmpNFA(g, *this);
    }
    return _cacheNFA;
}

}

#undef LIBALIGN_IMEXPORT 
#endif
