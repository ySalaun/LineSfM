#ifndef VPOINT_H
#define VPOINT_H

#include "alignment.h"
#include "segment.h"

#include "LWImage.h"

class Tiling; // Local usage

namespace align {
    
class LIBALIGN_IMEXPORT  Vpoint
{
public:
    Vpoint() {}
    ~Vpoint() {}

    static void detect(std::vector<Vpoint>& vp,
                       const std::vector<Segment>& segIni, int w, int h,
                       double eps = 0);
    static void detect(std::vector<Vpoint>& vp,
                       const LWImage<unsigned char>& image,
                       double eps = 0,
                       float minNorm = 2.f, int qGradient = 16,
                       int nLevels = 1, int nDirections = 96);
    void pos(float& x, float& y, float& z) const;

    float x1, y1, x2, y2, x3, y3, x4, y4;
    double weight; // -log10(number of false alarms)
    std::vector<Segment> seg;
    std::vector<int> segIndex;
private:
    static void find_vp(std::vector<Vpoint>& vp, Tiling** Tilings, int nLevels,
                        const std::vector<Segment>& seg);

};

}

#endif
