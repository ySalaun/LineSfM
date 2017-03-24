#include "segment.h"
//#include "libNumerics/numerics.h"

#include <math.h>
#include <float.h>

using namespace align;

static const float EPS = 1E-4f;

/// Quadratic distance to point (fx,fy).
float Point::qdistance(float fx, float fy) const
{
    return (x-fx)*(x-fx) + (y-fy)*(y-fy);
}

// Length of segment
float Segment::qlength() const
{
    return (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2);
}

float Segment::manhattanLength() const
{
	float dx = x2-x1;
	if (dx<0.f) dx = -dx;
	float dy = y2-y1;
	if (dy<0.f) dy = -dy;
	return(dx+dy);
}

// Project point into segment
void Segment::project(float& x, float& y) const
{
    if(isPoint()) {
        x = x1;
        y = y1;
        return;
    }
    float dx = x1 - x2;
    float dy = y1 - y2;
    float lambda = dx * (x - x2) + dy * (y - y2);
    lambda /= (dx*dx + dy*dy);
    if(lambda <= 0.0f)
        lambda = 0.0f;
    else if(lambda >= 1.0f)
        lambda = 1.0f;
    x = lambda*x1 + (1-lambda)*x2;
    y = lambda*y1 + (1-lambda)*y2;
}

// Quadratic distance of point to segment
float Segment::qdistance(float x, float y) const
{
    float xp = x, yp = y;
    project(xp, yp);
    xp -= x; yp -= y;
    return xp*xp + yp*yp;
}

// compute the intersection of two segments, return false if empty
bool Segment::inter(const Segment &seg1, const Segment &seg2, float &x, float &y)
{
    Line line1(seg1), line2(seg2);
    InterLines inter(line1,line2);
    if(!inter.compute(x,y))
        return false;
    float f1 = (line1.horizontal()? (x-seg1.x1)*(x-seg1.x2):(y-seg1.y1)*(y-seg1.y2));
    float f2 = (line2.horizontal()? (x-seg2.x1)*(x-seg2.x2):(y-seg2.y1)*(y-seg2.y2));
    return(f1<=.0f && f2<=0.f);
}

// Constructor
Line::Line(const float* L, double w)
: weight(w)
{
    l[0] = L[0]; l[1] = L[1]; l[2] = L[2];
}

// Constructor. Do not take weight as optional argument in order to avoid
// confusion with other constructor (float, float, float, float, double=0)
Line::Line(float l0, float l1, float l2)
: weight(0)
{
    l[0] = l0; l[1] = l1; l[2] = l2;
}

// Constructor from point and direction vector
Line::Line(float x, float y, float dx, float dy, double w)
: weight(w)
{
    l[0] = dy;
    l[1] = -dx;
    l[2] = y*dx - x*dy;
}

// Constructor: line supporting segment
Line::Line(const Segment& s)
: weight(s.weight)
{
    l[0] = s.y2 - s.y1;
    l[1] = s.x1 - s.x2;
    if(l[0] == 0.f) l[1] = 1.f;
    if(l[1] == 0.f) l[0] = 1.f;
    l[2] = -(l[0]*s.x1 + l[1]*s.y1);
}

// x associated to ordinate
float Line::x(float y0) const
{
    return -(l[1]*y0+l[2]) / l[0];
}

// y associated to abscissa
float Line::y(float x0) const
{
    return -(l[0]*x0+l[2]) / l[1];
}

// Transform homogeneous vector into (cos(phi), sin(phi), -rho), rho >= 0
void Line::normalize() const
{
    float r = (float)hypot(l[0], l[1]);
    if(r == 0)
        return;
    r = 1.0f / r;
    if(l[2] > 0)
        r = -r;
    float* L = const_cast<float*>(l);
    L[0] *= r;
    L[1] *= r;
    L[2] *= r;
}

// Angle (in rad) with horizontal, clockwise orientation
float Line::angle() const
{
    normalize();
    return (float)atan2(l[0], -l[1]);
}

// Project point into line
void Line::project(float& x, float& y) const
{
    float s = l[0]*l[0] + l[1]*l[1];
    if(s == 0)
        return;
    s = 1.0f / s;
    float z = x;
    x = s * (l[1]*l[1]*x - l[0]*(l[1]*y+l[2]));
    y = s * (l[0]*l[0]*y - l[1]*(l[0]*z+l[2]));
}

// Quadratic distance of point to line
float Line::qdistance(float x, float y) const
{
    float xp = x; float yp = y;
    project(xp, yp);
    xp -= x; yp -= y;
    return xp*xp + yp*yp;
}

bool Line::interRect(Segment &seg, float x1, float y1, float x2, float y2) const
{
    static const float NaN = -FLT_MAX;
    align::Segment s;
    s.x1 = s.x2 = s.y1 = s.y2 = NaN;
    float x, y;

    y = y1; x = this->x(y);
    if(x1 <= x && x < x2){
        if(s.x1 == NaN) {
            s.x1 = x; s.y1 = y;
        } else {
            s.x2 = x; s.y2 = y;
        }
    }
    y = y2; x = this->x(y);
    if(x1 <= x && x < x2){
        if(s.x1 == NaN) {
            s.x1 = x; s.y1 = y;
        } else {
            s.x2 = x; s.y2 = y;
        }
    }
    x = x1; y = this->y(x);
    if(y1 <= y && y < y2){
        if(s.x1 == NaN) {
            s.x1 = x; s.y1 = y;
        } else {
            s.x2 = x; s.y2 = y;
        }
    }
    x = x2; y = this->y(x);
    if(y1 <= y && y < y2){
        if(s.x1 == NaN) {
            s.x1 = x; s.y1 = y;
        } else {
            s.x2 = x; s.y2 = y;
        }
    }
    if(s.x2 == NaN)
        return false;

    seg = s;
    return true;
}

// Constructor
InterLines::InterLines(bool useWeight)
: useW(useWeight), l00(0), l01(0), l11(0), l02(0), l12(0), l22(0)
{}

// Constructor
InterLines::InterLines(const Line& l1, const Line& l2, bool useWeight)
: useW(useWeight), l00(0), l01(0), l11(0), l02(0), l12(0), l22(0)
{
    add(l1);
    add(l2);
}

// Add line, defined by homogeneous coordinates, for intersection
void InterLines::add(const Line& line)
{
    float s = line.l[0]*line.l[0] + line.l[1]*line.l[1];
    if(s >= EPS) {
        s = (useW ? (float) line.weight : 1.0f) / s;
        l00 += line.l[0]*line.l[0] * s;
        l01 += line.l[0]*line.l[1] * s;
        l11 += line.l[1]*line.l[1] * s;
        l02 += line.l[0]*line.l[2] * s;
        l12 += line.l[1]*line.l[2] * s;
        l22 += line.l[2]*line.l[2] * s;
    }
}

// Return intersection point of lines
bool InterLines::compute(float& x, float& y) const
{ // Total Least Squares error minimization
    float det = (float)(l00 * l11 - l01 * l01);
    if(det < EPS)
        return false;
    det = 1.0f / det;
    x = static_cast<float>((l01 * l12 - l11 * l02) * det);
    y = static_cast<float>((l01 * l02 - l00 * l12) * det);
    return true;
}

// Return intersection point of lines
bool InterLines::compute(float& x, float& y, float& z) const
{
    // Normalization factor for better matrix condition
    /*double lambda = (l00+l11 >= l22)? 1.0: sqrt((l00+l11)/l22);
    libNumerics::matrix<double> u(3,3);
    u(0,0) = l00; u(0,1) = l01; u(0,2) = l02*lambda;
    u(1,1) = l11; u(1,2) = l12*lambda; u(2,2) = l22*lambda*lambda;
    u.symUpper();
    libNumerics::SVD svd(u);
    if(svd.W()(1) == svd.W()(2))
        return false;
    x = (float) svd.U()(0,2);
    y = (float) svd.U()(1,2);
    z = (float) (svd.U()(2,2)*lambda);*/
    return true;
}

// Constructor
SegmentIterator::SegmentIterator(int w, int h)
: _w(w), _h(h), _ox(0), _oy(0), _dx(1.0f), _dy(0), _len(-1), _l(-1),
  _x(-1.0f), _y(-1.0f)
{}

// Init with origin (ox, oy), direction (dx,dy) and length `len'
void SegmentIterator::init(float ox, float oy, float dx, float dy, int len)
{
    _x = _ox = ox; _y = _oy = oy; _dx = dx; _dy = dy;
    _len = len;
    if(_len >= 0)
	++ _len;
    _l = 0;
}

// Init to describe a segment point by point
void SegmentIterator::init(const Segment& seg, const SegmentIterator* line)
{
    float dx, dy, l;
    extract_vect(seg, dx, dy, l);
    if(line != NULL) {
	dx = line->_dx;
	dy = line->_dy;
    }
    init(seg.x1, seg.y1, dx, dy, int(l+.5f));
}

// Norm and direction of a `seg' considered as vector
void SegmentIterator::extract_vect(const Segment& seg,
				  float& dx, float& dy, float& l)
{
    dx = seg.x2 - seg.x1;
    dy = seg.y2 - seg.y1;
    l = (float)hypot(dx, dy);
    dx /= l;
    dy /= l;
}

// Shift segment in orthogonal direction
SegmentIterator SegmentIterator::shift(float delta) const
{
    SegmentIterator it(*this);
    it.at(0);
    if((_dx >= _dy && _dx >= -_dy) ||
       (_dx <= _dy && _dx <= -_dy)) // Horizontal line
	it._oy += delta;
    else
	it._ox += delta;
    it._x = it._ox; it._y = it._oy;
    if(it._len > 0) {
	while(it._l < it._len && ! it.valid())
	    ++ it;
	it._ox = it._x;
	it._oy = it._y;
	it._len -= it._l;
	it._l = 0;
    }
    return it;
}
