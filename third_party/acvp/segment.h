#ifndef SEGMENT_H
#define SEGMENT_H

#include <stdlib.h>

namespace align {

class Point {
public:
    Point() : x(0), y(0) {}
    Point(float fx, float fy) : x(fx), y(fy) {}
    
    float qdistance(float fx, float fy) const;
    
    float x, y;
};

class Segment {
public:
    Segment() : x1(0), y1(0), x2(0), y2(0), weight(0) {}
    Segment(float ix1, float iy1, float ix2, float iy2, double w = 0)
    : x1(ix1), y1(iy1), x2(ix2), y2(iy2), weight(w) {}

    bool isPoint() const { return (x1 == x2 && y1 == y2); }
    float qlength() const;
    float manhattanLength() const;
    void project(float& x, float& y) const;
    static bool inter(const Segment &seg1, const Segment &seg2, float &x, float &y);
    float qdistance(float x, float y) const;

    float x1, y1, x2, y2;
    double weight;
};

class Line {
public:
    Line() : weight(0) { l[0] = l[1] = l[2] = 0; }
    Line(const float* L, double w = 0);
    Line(float l0, float l1, float l2);
    Line(float x, float y, float dx, float dy, double w = 0);
    Line(const Segment& s);

    bool isValid() const { return (l[0] != 0 || l[1] != 0); }
    float x(float y0) const;
    float y(float x0) const;
    void project(float& x, float& y) const;
    float qdistance(float x, float y) const;
    bool interRect(Segment &seg, float x1, float y1, float x2, float y2) const;
    void normalize() const;
    float angle() const;
    bool horizontal() const {return (l[0]*l[0] < l[1]*l[1]);}

    float l[3];
    double weight;
};

class InterLines {
public:
    explicit InterLines(bool useWeight = false);
    InterLines(const Line& l1, const Line& l2, bool useWeight = false);

    void add(const Line& line);
    bool compute(float& x, float& y) const;
    bool compute(float& x, float& y, float& z) const;
private:
    bool useW;
    double l00, l01, l11, l02, l12, l22;
};

// Iterate over points included in segment
class SegmentIterator {
public:
    SegmentIterator(int w, int h);
    ~SegmentIterator() {}

    void init(float ox, float oy, float dx, float dy, int len = -1);
    void init(const Segment& seg, const SegmentIterator* line = NULL);
    int w() const { return _w; }
    int h() const { return _h; }
    int x() const { return static_cast<int>(_x); }
    int y() const { return static_cast<int>(_y); }
    int xy() const { return _w*y() + x(); }
    float fx() const { return _x; }
    float fy() const { return _y; }
    int pos() const { return _l; }
    inline void operator++();
    inline SegmentIterator& at(int l);
    bool valid() const
    { return (_l != _len && _x >= 0 && _x < _w && _y >= 0 && _y < _h); }

    SegmentIterator shift(float delta) const;
private:
    const int _w, _h;
    float _ox, _oy, _dx, _dy;
    int _len, _l;
    float _x, _y;
    static void extract_vect(const Segment& seg,
			     float& dx, float& dy, float& l);
};

// Next point
void SegmentIterator::operator++()
{
    ++ _l;
    _x = _ox + _l*_dx;
    _y = _oy + _l*_dy;
}

// Set position
SegmentIterator& SegmentIterator::at(int l)
{
    if(_l != l) {
	_l = l;
	_x = _ox + l*_dx;
	_y = _oy + l*_dy;
    }
    return *this;
}

}

#endif
