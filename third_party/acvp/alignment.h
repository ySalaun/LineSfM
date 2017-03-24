#ifndef ALIGNMENT_H
#define ALIGNMENT_H

//#include "stdafx.h"

#include "LWImage.h"

#define LIBALIGN_IMEXPORT

#include <vector> // STL
class Binomial;

namespace align {

class Segment;
class SegmentIterator;
class MdlGraph;
class Bloc { public: int start, end; }; // Internal usage

class LIBALIGN_IMEXPORT Alignment
{
public:
    Alignment(float minNorm = 2.0, int qGradient = 16,
	      int nLevels = 1, int nDirections = 96);
    ~Alignment();

    void fixPoint(float x, float y, float dist);
    void angles(float frad1, float frad2);
    void detect(const LWImage<unsigned char>& im,
		std::vector<Segment>& seg, double eps = 0,
                std::vector<Segment>* candidates = 0);
private:
    float m_minNorm, p;
    int nLevels, nTheta;
    float xFixed, yFixed, qDist; // Look only for lines through fixed point
    float minRad, maxRad; // Limit angles

    bool check_angle(float theta) const;
    bool check_line(float x, float y, float dx, float dy) const;
    static float error_direction(float obs, float ref);
    static int find_blocs(SegmentIterator& it,
			  const LWImage<float>& im, float v, float prec,
                          Bloc* bloc);
    static void
        mark_pixels(MdlGraph& g, const LWImage<float>& im, float val,float prec,
		const Segment& seg, const SegmentIterator& line,
		const Binomial* test);
    static void
    extract_segments(SegmentIterator& it,
		     const LWImage<float>& im, float v, float prec,
		     Bloc* bloc, const Binomial& test, double max_nfa,
		     std::vector<Segment>& seg,
		     std::vector<bool>& maximal, MdlGraph& g);
    static void mdl(MdlGraph& g, float maxNFA, float mNFA,
                    std::vector<Segment>& seg);
};

} // namespace align

#endif
