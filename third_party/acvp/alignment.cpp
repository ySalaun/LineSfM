#include "alignment.h"
//#include "LLAngle.h"
#include "segment.h"
#include "mdl.h"
#include "binomial.h"

#include <algorithm> // STL
#include <functional> // STL

#include <stdio.h>
#include <float.h>
#include <math.h>
#include <assert.h>
#ifndef M_PI
#define M_PI 3.14159265358
#endif
#ifndef M_PI_2
#define M_PI_2 M_PI/2.
#endif

static const float doublePi = 2 * (float)M_PI;
static const float DIR_INVALID = FLT_MAX;

using namespace align;

// To iterate on lines meeting a side of the image
struct iter_side_t {
    int ox, oy; // Origin coordinates, to be multiplied by w and h
    int mx, my; // Step to next point
    float otheta; // First angle of lines
};

static iter_side_t iter_sides[] =
{
    {0, 0, 1, 0, 0.0f},   // Upper side
    {1, 0, 0, 1, (float) M_PI_2}, // Right side
    {0, 1, 1, 0, (float)M_PI},   // Lower side
    {0, 0, 0, 1, -(float)M_PI_2} // Left side
};

static bool vertical(int side)
{ return (::iter_sides[side].my != 0); }


// Constructor
Alignment::Alignment(float minNorm, int qGradient, int levels, int nDirections)
: m_minNorm(minNorm), p(1.0f / (float)qGradient), nLevels(levels),
  nTheta(nDirections / 2), xFixed(0), yFixed(0), qDist(-1.0),
  minRad(0), maxRad(2.0f*(float)M_PI)
{}

// Destructor
Alignment::~Alignment()
{}

// Restrict next detection to lines through point at `dist' from (x,y).
// dist<0 means no restriction.
void Alignment::fixPoint(float x, float y, float dist)
{
    xFixed = x;
    yFixed = y;
    qDist = ((dist >= 0) ? dist*dist : -1.0f);
}

// Is line thru (x,y) of direction (dx,dy) within max distance of fixed point?
bool Alignment::check_line(float x, float y, float dx, float dy) const
{
    return (Line(x, y, dx, dy).qdistance(xFixed, yFixed) <= qDist);
}

// Restrict next detection to angular sector (clockwise orientation)
void Alignment::angles(float frad1, float frad2)
{
    while(frad1 >= doublePi)
        frad1 -= doublePi;
    while(frad1 < 0)
        frad1 += doublePi;
    while(frad2 >= doublePi)
        frad2 -= doublePi;
    while(frad2 <= frad1)
        frad2 += doublePi;
    minRad = frad1;
    maxRad = frad2;
}

// Is angle within angular sector?
bool Alignment::check_angle(float theta) const
{
    while(theta < minRad)
        theta += (float)M_PI;
    while(theta > maxRad) {
        theta -= (float)M_PI;
        if(theta < minRad)
            return false;
    }
    return true;
}

// Return error between observed and reference angles in [0,2pi]
float Alignment::error_direction(float obs, float ref)
{
    static const float TWO_PI = static_cast<float>(2*M_PI);
    if(obs != DIR_INVALID) {
        obs -= ref;
        while(obs <= -M_PI) obs += TWO_PI;
        while(obs >   M_PI) obs -= TWO_PI;
        if(obs < 0) obs = -obs;
    }
    return obs;
}

// Find pixels on segment where direction is compatible. Return result
// in blocs of adjacent indices.
int Alignment::find_blocs(SegmentIterator& it,
                          const LWImage<float>& im, float val, float prec,
                          Bloc* bloc) // Output
{
    int n = 0;
    bool inBloc = false;
    for(; it.valid(); ++it) {
        if(error_direction(im.data[it.xy()], val) <= prec) {
            if(! inBloc) {
                bloc[n].start = it.pos();
                inBloc = true;
            }
        } else
            if(inBloc) {
                bloc[n++].end = it.pos()-1;
                inBloc = false;
            }
    }
    if(inBloc)
        bloc[n++].end = it.pos()-1;
    return n;
}

// Comparison of weights, used for sorting
static bool less_weight(const Segment& s1, const Segment& s2)
{
    return (s1.weight < s2.weight);
}

// Test if segments with same orientation intersect
static bool intersect(const Segment& s1, const Segment& s2)
{
    if(s1.x1 <= s1.x2) {
        if(s2.x1 > s1.x2 || s2.x2 < s1.x1)
            return false;
    } else
        if(s2.x1 < s1.x2 || s2.x2 > s1.x1)
            return false;
    if(s1.y1 <= s1.y2) {
        if(s2.y1 > s1.y2 || s2.y2 < s1.y1)
            return false;
    } else
        if(s2.y1 < s1.y2 || s2.y2 > s1.y1)
            return false;
    return true;
}

// Keep only maximal meaningful segments among aligned segments
static void keep_maximal(std::vector<Segment>& seg, const unsigned int nBefore,
                         std::vector<bool>& maximal)
{
    if(nBefore == seg.size())
        return;
    maximal.assign(seg.size()-nBefore, true);
    std::sort(seg.begin()+nBefore, seg.end(), less_weight);
    unsigned int n = nBefore;
    for(unsigned int i = nBefore; i < seg.size(); i++)
        if(maximal[i-nBefore]) {
            for(unsigned int j = i+1; j < seg.size(); j++)
                if(intersect(seg[i], seg[j]))
                    maximal[j-nBefore] = false;
                if(n < i)
                    seg[n] = seg[i];
                ++ n;
            }
    if(n < seg.size())
        seg.erase(seg.begin()+n, seg.end());
}

struct DataSet {
    DataSet(const SegmentIterator& si, const Binomial* t)
    : it(si), test(t) {}
    SegmentIterator it;
    const Binomial* test;
};

// Append to graph segment and pixels
void Alignment::mark_pixels(MdlGraph& g, const LWImage<float>& im,
                            float val, float prec,
                            const Segment& seg, const SegmentIterator& line,
                            const Binomial* test)
{
    SegmentIterator it(line);
    it.init(seg, &line);
    unsigned int s = g.new_set(new struct DataSet(it, test));
    for(int delta = -1; delta <= 1; delta++)
        for(SegmentIterator it2 = it.shift((float) delta); it2.valid(); ++it2)
            if(delta != 0 || error_direction(im.data[it2.xy()], val) <= prec)
                g.link(s, it2.xy());
}

// Extract maximal meaningful segments on a given line
void Alignment::extract_segments(SegmentIterator& it,
                                 const LWImage<float>& im, float v, float prec,
                                 Bloc* bloc,const Binomial& test,double max_nfa,
                                 std::vector<Segment>& seg,
                                 std::vector<bool>& maximal, MdlGraph& g)
{
    int nBlocs = find_blocs(it, im, v, prec, bloc);
    unsigned int nBefore = (unsigned int) seg.size();
    for(int i = 0; i < nBlocs; i++) {
        int k = 0; // #pts with good direction
        for( int j = i; j < nBlocs; j++) {
            k += (bloc[j].end - bloc[j].start+1);
            int l = bloc[j].end - bloc[i].start+1;
            double nfa = test(l/2, k/2);
            if(nfa < max_nfa)
                seg.push_back(Segment(it.at(bloc[i].start).fx(),
                                      it.at(bloc[i].start).fy(),
                                      it.at(bloc[j].end).fx(),
                                      it.at(bloc[j].end).fy(), nfa));
        }
    }
    keep_maximal(seg, nBefore, maximal);
    for(unsigned int i = nBefore; i < seg.size(); i++)
        mark_pixels(g, im, v, prec, seg[i], it, &test);
}

// Comparison of edges by arrival node
struct LessEdge : public std::binary_function<unsigned int, unsigned int, bool> {
    LessEdge(const std::vector<MdlEdge>& e) : edges(e) {}
    bool operator()(unsigned int i, unsigned int j) const
    { return edges[i].pointNode < edges[j].pointNode; }
private:
    const std::vector<MdlEdge>& edges;
};

// Return nb of valid edges for a set in the graph (i.e., segment).
static unsigned int valid_edges(const MdlGraph& g, const MdlSet& set,
                                int& iFirst, int& iLast)
{
    struct LessEdge le(g.edges);
    MdlEdge& lastEdge = const_cast<std::vector<MdlEdge>&>(g.edges).back();
    unsigned int& pn = const_cast<unsigned int&>(lastEdge.pointNode);

    struct DataSet* data = static_cast<struct DataSet*>(set.id);
    SegmentIterator& it = data->it.at(0);

    iFirst = -1, iLast = -2;
    unsigned int k = 0;
    for(; it.valid(); ++it) {
        pn = static_cast<unsigned int>(it.xy());
        std::vector<unsigned int>::iterator t =
			std::lower_bound(const_cast<MdlSet&>(set).edges.begin(),
                        const_cast<MdlSet&>(set).edges.end(),
                        (unsigned int) g.edges.size()-1, le);
        if(t != set.edges.end() && ! le((unsigned int) g.edges.size()-1, *t) &&
           g.edges[*t].status() != MDL_INVALID) {
            ++ k;
            if(iFirst < 0)
                iFirst = it.pos();
            iLast = it.pos();
        }
    }
    return k;
}

// Function to compute NFA
static double nfa(const MdlGraph& g, const MdlSet& set)
{
    int iFirst, iLast;
    unsigned int k = valid_edges(g, set, iFirst, iLast);
    return (*static_cast<struct DataSet*>(set.id)->test)
        ((iLast-iFirst+1)/2, k/2);
}

// Apply Minimum Description Length principle to extract segments
void Alignment::mdl(MdlGraph& g, float maxNFA, float mNFA,
                    std::vector<Segment>& seg)
{
    struct LessEdge le(g.edges);
    for(unsigned int i = 0; i < g.setNodes.size(); i++)
        std::sort(g.setNodes[i].edges.begin(), g.setNodes[i].edges.end(), le);

    g.edges.push_back(MdlEdge(0,0)); // Used for comparison
    g.mdl(nfa, maxNFA);

    for(unsigned int i = 0; i < g.setNodes.size(); i++) {
        if(g.setNodes[i].status() == MDL_ACCEPT) {
            double w = -log10(mNFA * g.setNodes[i].nfa(g, nfa));
            SegmentIterator& it =
                static_cast<struct DataSet*>(g.setNodes[i].id)->it;
            int j1, j2;
            valid_edges(g, g.setNodes[i], j1, j2);
            seg.push_back(Segment(it.at(j1).fx(), it.at(j1).fy(),
                                  it.at(j2).fx(), it.at(j2).fy(), w));
        }
        delete static_cast<struct DataSet*>(g.setNodes[i].id);
    }
}

// Find intersection of half line supporting segment with image frame
static int inter_frame(const Segment& s, int w, int h,
                       float& x1, float& y1, float& x2, float& y2)
{
    float dx = s.x2 - s.x1, dy = s.y2 - s.y1;
    float l[3];
    l[0] = 1.0f; l[1] = 0.f; l[2] = -(float) (w-1);
    Segment s2(s);
    bool bHorizontal;
    int side = -1;

    if(dx > 0 && InterLines(l, s).compute(x1, y1) && y1 >= 0 && y1 <= h-1) {
        bHorizontal = (dx > dy && dx > -dy);
        side = 1;
    }
    if(side < 0) {
        l[2] = 0;
        if(dx < 0 && InterLines(l, s).compute(x1, y1) && y1 >= 0 && y1 <=h-1) {
            bHorizontal = (dx < dy && dx < -dy);
            side = 3;
        }
    }

    if(side >= 0) {
        if(bHorizontal) {
            ++s2.y1; --s2.y2; InterLines(l,Line(s2)).compute(x1, y1);
            if(y1 < 0) y1 = 0;
            s2 = s;
            --s2.y1; ++s2.y2; InterLines(l,Line(s2)).compute(x2, y2);
            if(y2 > h-1.f) y2 = h-1.f;
        } else { // Vertical
            ++s2.x1; --s2.x2; InterLines(l,Line(s2)).compute(x1, y1);
            if(y1 < 0.f) y1 = 0.f;
            s2 = s;
            --s2.x1; ++s2.x2; InterLines(l,Line(s2)).compute(x2, y2);
            if(y2 > h-1.f) y2 = h-1.f;
            if(y1 > y2)
                std::swap(y1, y2);
        }
        return side;
    }

    if(side < 0) {
        l[0] = 0; l[1] = 1.0f; l[2] = -(h-1.f);
        if(dy > 0 && InterLines(l, s).compute(x1, y1) && x1 >= 0 && x1 <=w-1) {
            bHorizontal = (dx > dy || dx < -dy);
            side = 2;
        }
    }
    if(side < 0) {
        l[2] = 0;
        if(dy < 0 && InterLines(l, s).compute(x1, y1) && x1 >= 0 && x1 <=w-1) {
            bHorizontal = (dx > -dy || dx < dy);
            side = 0;
        }
    }
    assert(side >= 0);

    if(bHorizontal) {
        ++s2.y1; --s2.y2; InterLines(l,Line(s2)).compute(x1, y1);
        if(x1 < 0) x1 = 0;
        s2 = s;
        --s2.y1; ++s2.y2; InterLines(l,Line(s2)).compute(x2, y2);
        if(x2 > w-1.f) x2 = w-1.f;
        if(x1 > x2)
            std::swap(x1, x2);
    } else {
        ++s2.x1; --s2.x2; InterLines(l,Line(s2)).compute(x1, y1);
        if(x1 < 0) x1 = 0;
        s2 = s;
        --s2.x1; ++s2.x2; InterLines(l,Line(s2)).compute(x2, y2);
        if(x2 > w-1.f) x2 = w-1.f;
    }
    return side;
}

// Find angles of tolerance for the direction of the segment
static void find_angles(const Segment& s, int side,
                        float& angle1, float& angle2)
{
    float dx = s.x2 - s.x1, dy = s.y2 - s.y1;
    if(dx<0) { dx = -dx; dy = -dy; }
    if(dx >= dy && dx >= -dy) { // Rather horizontal line
        angle1 = atan2(dy+2, dx);
        angle2 = atan2(dy-2, dx);
    } else { // Rather vertical line
        angle1 = atan2(dy, dx+2);
        angle2 = atan2(dy, dx-2);
    }
    if(angle1 > angle2)
        std::swap(angle1, angle2);
    float minAngle = ::iter_sides[side].otheta;
    float maxAngle = minAngle + (float) M_PI;
    while(angle1 < minAngle) {
        angle1 += (float) M_PI;
        angle2 += (float) M_PI;
    }
    while(angle1 >= maxAngle) {
        angle1 -= (float) M_PI;
        angle2 -= (float) M_PI;
    }
}

// Detect meaningful segments in image
void Alignment::detect(const LWImage<unsigned char>& im,
                       std::vector<Segment>& seg,
                       double eps, std::vector<Segment>* candidates)
{
    const double n4 = im.w*(double)im.w*(double)im.h*(double)im.h;
    double maxNFA = pow(10.0, -eps) / n4 / nLevels;
    const float dTheta = float(M_PI / nTheta);

    int nBlocs = (int)ceil(hypot(im.w, im.h)) / 2 + 2;
    Bloc* bloc = new Bloc[nBlocs];
    std::vector<bool> maximal;
    maximal.reserve(nBlocs*nBlocs);
    Binomial** test = new Binomial*[nLevels];
    MdlGraph g(im.w*im.h);
    std::vector<Segment> temp, filter;
    std::vector<Segment>& iniSegments = (candidates != 0) ? *candidates : temp;
    if(! seg.empty())
        std::swap(seg, filter);

    LWImage<float> img(new float[im.w*im.h], im.w, im.h);
    SegmentIterator it(im.w, im.h);
    float p = this->p;
    for(int level = 0; level < nLevels; level++) { // Loop 0: precision level
        test[level] = new Binomial(p);
        const float prec = float(M_PI * p);
        //ll_angle(im, &img, NULL, m_minNorm/prec);
        if(! filter.empty())
            for(std::vector<Segment>::const_iterator s = filter.begin();
                s != filter.end(); ++s) {
                float x1, y1, x2, y2;
                int side = inter_frame(*s, im.w, im.h, x1, y1, x2, y2);
                fixPoint(.5f*((*s).x1+(*s).x2), .5f*((*s).y1+(*s).y2), 1.0f);
                if(vertical(side))
                    y1 = (int)y1 + 0.5f;
                else
                    x1 = (int)x1 + 0.5f;
                float theta, angle1, angle2;
                find_angles(*s, side, angle1, angle2);
                int iTheta = int((angle1 - iter_sides[side].otheta) / dTheta);
                while((theta=iter_sides[side].otheta+iTheta++*dTheta)<angle2) {
                    if(! check_angle(theta))
                        continue;
                    float x=x1, y=y1, dx=cosf(theta), dy=sinf(theta);
                    while(x <= x2 && y <= y2) {
                        if(qDist <= 0 || check_line(x, y, dx, dy)) {
                            it.init(x, y, dx, dy);
                            extract_segments(it, img, theta, prec, bloc,
                                             *test[level], maxNFA, iniSegments,
                                             maximal, g);
                        }
                        if(iTheta == 0)
                            break;
                        x += iter_sides[side].mx;
                        y += iter_sides[side].my;
                    }
                }
            }
        else for(int side = 0; side < 4; side++) { // Loop 1: the four sides
            int posmax = im.w*iter_sides[side].mx+ im.h*iter_sides[side].my - 1;

            for(int iTheta = 0; iTheta < nTheta; iTheta++) { // Loop 2: angles
                printf("level: %d, side: %d, theta:%2d\n", level,side,iTheta);
                float theta = iter_sides[side].otheta + iTheta*dTheta;
                if(! check_angle(theta))
                    continue;
                float x, y, dx = cosf(theta), dy = sinf(theta);
                
                for(int pos = 0; pos < posmax; pos++) { // Loop 3: position
                    x = iter_sides[side].ox*(im.w-1) +
                        pos*iter_sides[side].mx + .5f;
                    y = iter_sides[side].oy*(im.h-1) +
                        pos*iter_sides[side].my + .5f;
                    if(qDist > 0 && ! check_line(x, y, dx, dy))
                        continue;
                    it.init(x, y, dx, dy);
                    extract_segments(it, img, theta, prec, bloc,
                                     *test[level], maxNFA, iniSegments,
                                     maximal, g);
                    if(iTheta == 0)
                        break;
                }
            }
        }

        p *= .5f;
    } // Loop on levels

    delete [] img.data;
    delete [] bloc;
    printf("output:%ld segments\n", iniSegments.size());
    mdl(g, (float)maxNFA, (float)(n4*nLevels), seg);

    for(int i = nLevels-1; i >= 0; i--)
        delete test[i];
    delete [] test;
}
