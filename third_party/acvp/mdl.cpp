#include "mdl.h"

using namespace align;

// Constructor
MdlGraph::MdlGraph(unsigned int nPoints)
: edges(), setNodes(), pointNodes(nPoints, MdlPoint())
{}

// Build a new set with data `p' and return position in vector
unsigned int MdlGraph::new_set(void* p)
{
    unsigned int i = (unsigned int) setNodes.size();
    setNodes.push_back(MdlSet(p));
    return i;
}

// Link set and point in the graph
void MdlGraph::link(unsigned int iSet, unsigned int iPoint)
{
    setNodes[iSet].edges.push_back((unsigned int)edges.size());
    pointNodes[iPoint].edges.push_back((unsigned int) edges.size());
    edges.push_back(MdlEdge(iSet, iPoint));
}

// Return index of set of minimal NFA
unsigned int MdlGraph::best_set(MdlSet::ComputeNFA nfa, double maxNFA)
{
    unsigned int iBest = (unsigned int) setNodes.capacity();
    double bestNFA = maxNFA;
    for(unsigned int i = 0; i < setNodes.size(); i++)
	if(setNodes[i]._status == MDL_VALID) {
	    double v = setNodes[i].nfa(*this, nfa);
	    if(v > maxNFA)
		setNodes[i]._status = MDL_INVALID;
	    else if(v <= bestNFA) {
		bestNFA = v;
		iBest = i;
	    }
	}
    return iBest;
}

// Apply Minimum Description Length principle
void MdlGraph::mdl(MdlSet::ComputeNFA nfa, double maxNFA)
{
    while(true) {
	unsigned int i = best_set(nfa, maxNFA);
	if(i >= setNodes.size())
	    break;
	setNodes[i]._status = MDL_ACCEPT;
	const std::vector<unsigned int>& forwEdges = setNodes[i].edges;
	for(unsigned int j = 0; j < forwEdges.size(); j++)
	    if(edges[forwEdges[j]]._status == MDL_VALID) {
		unsigned int pt = edges[forwEdges[j]].pointNode;
		std::vector<unsigned int>& backEdges = pointNodes[pt].edges;
		for(unsigned int k = 0; k < backEdges.size(); k++) {
		    edges[backEdges[k]]._status = MDL_INVALID;
		    setNodes[edges[backEdges[k]].setNode]._cache = false;
		}
		edges[forwEdges[j]]._status = MDL_ACCEPT;
		setNodes[i]._cache = true;
	    }
    }
}
