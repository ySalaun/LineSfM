/* Original author: Andres Almansa */
#include "vpoint.h"
#include "mdl.h"
#include "binomial.h"

#include <set> // STL
#include <algorithm> // STL

#include <stdio.h>
#include <math.h>

#ifndef M_PI
static const double M_PI = 3.14159265359;
#endif
#ifndef M_2PI
static const double M_2PI = 2.0*M_PI;
#endif
#ifndef M_PI_2
static const double M_PI_2 = 0.5*M_PI;
#endif

using namespace align;

static const double EPS = 1.0e-11; // Min acceptable value of positive cos
#ifndef INFINITY // Already defined in some GLIBC versions
#define INFINITY (-log(0.0))
#endif

// Probability for an exterior tile delimited by angular sector of width
// `theta' and radii 1/cos(beta1) and 1/cos(beta2)
double pext(double beta1, double beta2, double theta)
{
    double f1 = cos(beta1);
    if(f1 < EPS)
        f1 = M_PI_2;
    else
        f1 = beta1 + 1.0/f1 - tan(beta1);

    double f2 = cos(beta2);
    if(f2 < EPS)
        f2 = M_PI_2;
    else
        f2 = beta2 + 1.0/f2 - tan(beta2);

    return (2.0*theta + f2-f1) / M_PI;
}

inline double pinf(double beta1, double theta)
{
    return pext(beta1, M_PI_2, theta);
}

// Partial derivative of `pext' w.r.t. second variable
double pext_prime(double beta2)
{
    double fp2 = cos(beta2);
    if (fp2 < EPS)
        fp2 = -0.5;
    else
        fp2 = (sin(beta2)-1.0) / (fp2*fp2);

    return (1.0 + fp2) / M_PI;
}

// Finds zero of convex function `pext(beta1,x,dtheta)-p' in interval [a,b]
double fzero_convex(double a, double b, const double tolx, const double tolf,
                    double beta1, double dtheta, const double p)
{
    double fa = pext(beta1, a, dtheta) - p;
    double fb = pext(beta1, b, dtheta) - p;

    while(b-a > tolx && fb-fa > tolf) {
        double fpb = pext_prime(b);
        a -= fa * (b-a) / (fb-fa); // Intersect chord (A,B) and x-axis
        fa = pext(beta1, a, dtheta) - p;
        if(fa >= -tolf)
            return a;
        b -= fb / fpb; // Intersect tangent at B and x-axis
        fb = pext(beta1, b, dtheta) - p;
        if(fb <= tolf)
            return b;
    }
    return (a + b)*.5;
}

// Cell: part of plane storing references to intersecting lines
class Cell {
public:
    Cell() : meaning(0), vp(false), segs(new std::set<unsigned int>) {}
    Cell(const Cell& c) : meaning(c.meaning), vp(c.vp), segs(c.segs)
    { steal(c); } // "Steal" copy
    ~Cell() { delete segs; }
    
    Cell& operator=(const Cell& c)
    { if(this != &c) { meaning= c.meaning; vp= c.vp; segs= c.segs; steal(c); }
      return *this; }

    double meaning;
    bool vp;
    std::set<unsigned int>* segs;
private:
    static void steal(const Cell& c) { const_cast<Cell&>(c).segs = NULL; }
};

// Tiling: covering of projective plane with cells
class Tiling {
public:
    typedef enum { INTERIOR, EXTERIOR } INT_EXT;
    class CellID { // Identification of cell
    public:
        CellID() : T(NULL), ie(INTERIOR), ix(-1), iy(-1) {}
        CellID(Tiling* t, INT_EXT e, int x, int y)
        : T(t), ie(e), ix(x), iy(y) {}
        
        Tiling* T;
        INT_EXT ie;
        int ix, iy;
    };

public:
    Tiling(int nTheta, int w, int h);
    ~Tiling() {}

    int nb_cells() const { return nCells; }
    int nb_segments() const { return nSegments; }
    double proba() const { return p; }
    double proba_inf() const { return pInf; }

    Cell& cell(INT_EXT ie, unsigned int ix, unsigned int iy);
    Cell& cell(const CellID& id) { return cell(id.ie, id.ix, id.iy); }
    const Cell& cell(INT_EXT ie, unsigned int ix, unsigned int iy) const
    { return const_cast<Tiling*>(this)->cell(ie, ix, iy); }
    const Cell& cell(const CellID& id) const
    { return const_cast<Tiling*>(this)->cell(id); }

    void add(const Segment& seg, unsigned int iSeg);
    int compute_vp(double threshold);
    void vp(std::vector<CellID>& list) const;

    static double nfa(const MdlGraph& g, const MdlSet& set);
    static int mdl(Tiling** Tilings, int nTilings, unsigned int n,
                   double threshold);

    void convert(Vpoint& vp,
                 const CellID& id, const std::vector<Segment>& seg) const;


    std::vector<double> x, y, t, d;
    std::vector<Cell> intCells, extCells;

private: // Coordinates conversions
    double X0, Y0; // Image center
    static void cross_prod(const double* a, const double* b, double* c);
    void proj_coords(const Segment& seg, double* L) const;
    static void proj2polar(double* l, double& phi);
    void polar2cart(double theta, double q, float& x, float& y) const;

private:
    static double qhypot(double x, double y) { return x*x + y*y; }
    static int ifloor(double x);
    void compute_radii(double R, double dTheta);

    void add_to_cell(unsigned int iSeg, INT_EXT ie, int ix, int iy);
    void inter_horizontal_lines(unsigned int iSeg, double* l);
    void inter_vertical_lines(unsigned int iSeg, double* l);
    void inter_int(unsigned int iSeg, double* l);
    void inter_circles(unsigned int iSeg, const double* l, double phi);
    void inter_rays(unsigned int iSeg, const double* l, double phi);
    void inter_ext(unsigned int iSeg, double* l);

    void compute_meaning();
    int local_maxima(INT_EXT ie, double threshold);
    int opposite_angle(int ix) const;
    double meaning(INT_EXT ie, int ix, int iy) const;

    double p, pInf;
    unsigned int nCells, nSegments;
};

// Create a new Tiling of the plane into vanishing regions
// for a given angular precision level (ntheta orientations)
Tiling::Tiling(int ntheta, int w, int h)
: x(), y(), t(), d(), intCells(), extCells(),
  X0(w*.5), Y0(h*.5),
  p(0), pInf(0), nCells(0), nSegments(0)
{
    double dtheta  = M_PI / ntheta; // Angular precision
    p = 4.0 * sin(dtheta) / M_PI; // Probability of internal tiles

    // Boundaries of internal cells in normalized coordinates (x,y)
    double R = hypot(X0, Y0);
    double dxy = 2.0 * R * sin(dtheta);
    int nx = (int)ceil(2.0 * R / dxy);
    for(int i = 0; i <= nx; i++)
        x.push_back(X0 - R + dxy*i);
    int ny = nx;
    for(int i = 0; i <= ny; i++)
        y.push_back(Y0 - R + dxy*i);
    for(int i = (int)((x.size()-1) * (y.size()-1))-1; i >= 0; i--)
        intCells.push_back(Cell());

    // Boundaries of external cells in normalized polar coordinates (theta,d)
    for(int i = 0; i <= ntheta ; i++)
        t.push_back(2.0 * dtheta * i);
    compute_radii(R, dtheta); // Fill fields `d' and `pInf'
    for(int i = (int)((t.size()-1) * (d.size()-1))-1; i >= 0; i--)
        extCells.push_back(Cell());

    // Remove interior cells outside unit circle
    R *= R;
    nCells = (unsigned int)(intCells.size() + extCells.size());
    for(unsigned int iy = 0; iy+1 < y.size(); iy++)
        for(unsigned int ix = 0; ix+1 < x.size(); ix++)
            if(qhypot(x[ix  ]-X0, y[iy  ]-Y0) > R &&
               qhypot(x[ix+1]-X0, y[iy  ]-Y0) > R &&
               qhypot(x[ix  ]-X0, y[iy+1]-Y0) > R &&
               qhypot(x[ix+1]-X0, y[iy+1]-Y0) > R) {
                -- nCells;
                delete cell(INTERIOR, ix, iy).segs;
                cell(INTERIOR, ix, iy).segs = NULL;
            }
}

// Compute radii for exterior tiles such that they have same probability as
// interior tiles. `pInf' gets probability of nonbounded tiles
void Tiling::compute_radii(double R, double dtheta)
{
    const double tol = 1.0e-8;

    d.push_back(R);
    double b = 0.0;
    while((pInf = pinf(b, dtheta)) > p) {
        b = fzero_convex(b, M_PI_2, tol, tol, b, dtheta, p);
        d.push_back(R / cos(b));
    }
    d.push_back(INFINITY);
}

// Return cell structure
Cell& Tiling::cell(INT_EXT ie, unsigned int ix, unsigned int iy)
{
    return (ie == INTERIOR) ?
        intCells[iy*(x.size()-1) + ix] :
        extCells[iy*(t.size()-1) + ix];
}

// Identification of opposite tiles of last ring
inline int Tiling::opposite_angle(int x) const
{
    int mid = ((int) t.size()-1) / 2;
    return (x >= mid) ? x - mid : x + mid;
}

// Meaningfulness of cell, wrapping off-by-one indices for exterior tiling
double Tiling::meaning(INT_EXT ie, int ix, int iy) const
{
    int nx = (ie == INTERIOR) ? (int)x.size() - 1 : (int)t.size() - 1;
    int ny = (ie == INTERIOR) ? (int)y.size() - 1 : (int)d.size() - 1;

    if(ie == EXTERIOR) { // Wrap around
        if(ix < 0)
            ix += nx;
        else if(ix == nx)
            ix = 0;
        if(iy < 0 || iy > ny)
            return -INFINITY;
        if(iy == ny) {
            ix = opposite_angle(ix);
            iy = ny-1;
        }
    }
    return (ix < 0 || iy < 0 || ix >= nx || iy >= ny) ?
        -INFINITY : cell(ie, ix, iy).meaning;
}

// Cross-product in R³ or join or meet in P²
void Tiling::cross_prod(const double* a, const double* b, double* c)
{
    c[2] = a[0]*b[1] - a[1]*b[0];
    c[0] = a[1]*b[2] - a[2]*b[1];
    c[1] = a[2]*b[0] - a[0]*b[2];
}

// Normalized projective coordinates of the line supporting a segment
void Tiling::proj_coords(const Segment& seg, double* L) const
{
    double p[3], q[3];
    p[0] = seg.x1; p[1] = seg.y1; p[2] = 1.0;
    q[0] = seg.x2; q[1] = seg.y2; q[2] = 1.0;
    cross_prod(p, q, L); // Line through p,q
}

// Transform homogeneous vector `l' into (cos(phi), sin(phi), -rho), rho >= 0
void Tiling::proj2polar(double* l, double& phi)
{
    double r = 1.0 / hypot(l[0], l[1]);
    if(l[2] <= 0) {
        l[0] *= r;
        l[1] *= r;
        l[2] *= r;
    } else {
        l[0] = - l[0] * r;
        l[1] = - l[1] * r;
        l[2] = - l[2] * r;
    }
    phi = atan2(l[1], l[0]);
    if(phi < 0)
        phi += M_2PI;
}

// Convert polar (relative to center) to cartesian coordinates
void Tiling::polar2cart(double theta, double q, float& x, float& y) const
{
    x = static_cast<float>(q * cos(theta) + X0);
    y = static_cast<float>(q * sin(theta) + Y0);
}

// Simplified (and faster) version of `floor' for positive values
inline int Tiling::ifloor(double x)
{
    if(x < 0)
        return -1;
    return static_cast<int>(x);
}

// Add segment number `j' to cell
void Tiling::add_to_cell(unsigned int j, INT_EXT ie, int ix, int iy)
{
    if(ix < 0 || iy < 0 ||
       (ie == INTERIOR && (ix+1 >= (int)x.size() || iy+1 >= (int)y.size())) ||
       (ie == EXTERIOR && (ix+1 >= (int)t.size() || iy+1 >= (int)d.size())) )
        return;
    std::set<unsigned int>* indices = cell(ie,ix,iy).segs;
    if(indices == NULL)
        return;
    if(binary_search(indices->begin(), indices->end(), j)) {
        if(ie == INTERIOR)
            indices->erase(j);
        return;
    }
    indices->insert(j);
}

// Find intersection of line (given by homogeneous coordinates) with
// horizontal lines delimiting interior tiling
void Tiling::inter_horizontal_lines(unsigned int iSeg, double* l)
{
    l[1] /= -l[0]; l[2] /= -l[0]; l[0] = -1;
    double min = x[0];
    double step = 1.0 / (x[1] - min);
    for(unsigned int iy = 0; iy < y.size(); iy++) {
        double x = l[1] * y[iy] + l[2];
        int ix = ifloor((x - min) * step);
        add_to_cell(iSeg, INTERIOR, ix, (int)iy);
        if(iy > 0)
            add_to_cell(iSeg, INTERIOR, ix, (int)iy-1);
    }
}

// Find intersection of line (given by homogeneous coordinates) with
// vertical lines delimiting interior tiling
void Tiling::inter_vertical_lines(unsigned int iSeg, double* l)
{
    l[0] /= -l[1]; l[2] /= -l[1]; l[1] = -1;
    double min = y[0];
    double step = 1.0 / (y[1] - min);
    for(unsigned int ix = 0; ix < x.size(); ix++) {
        double y = l[0] * x[ix] + l[2];
        int iy = ifloor((y - min) * step);
        add_to_cell(iSeg, INTERIOR, (int)ix, iy);
        if(ix > 0)
            add_to_cell(iSeg, INTERIOR, (int)ix-1, iy);
    }
}

// Find intersections with interior cells of line given in homogeneous
// coordinates
void Tiling::inter_int(unsigned int iSeg, double* l)
{
    if((l[0] >= l[1] && l[0] >= -l[1]) ||
       (l[0] <= l[1] && l[0] <= -l[1])) // "vertical" line
        inter_horizontal_lines(iSeg, l);
    else // "horizontal" line
        inter_vertical_lines(iSeg, l);
}

// Intersection of line (given in polar coordinates) with circles limiting
// exterior cells 
void Tiling::inter_circles(unsigned int iSeg, const double* l, double phi)
{
    double dxInv = 1.0 / (t[1] - t[0]);
    double rho = -(l[0]*X0 + l[1]*Y0 + l[2]);
    for(unsigned int iy = 0; iy+1 < d.size(); iy++) {
        double psi = acos(rho / d[iy]);
        double x = phi - psi; // -M_PI <= x < M_2PI
        if(x < t[0])
            x += M_2PI;
        int ix = ifloor((x - t[0]) * dxInv);
        add_to_cell(iSeg, EXTERIOR, ix, (int)iy);
        if(iy > 0)
            add_to_cell(iSeg, EXTERIOR, ix, (int)iy-1);

        x = phi + psi; // 0 <= x < 3 M_PI
        if(x >= t.back())
            x -= M_2PI;
        if(x < t[0])
            x += M_2PI;
        ix = ifloor((x - t[0]) * dxInv);
        add_to_cell(iSeg, EXTERIOR, ix, (int)iy);
        if(iy > 0)
            add_to_cell(iSeg, EXTERIOR, ix, (int)iy-1);
    }
}

// Intersection of line (given in polar coordinates) with radii limiting
// exterior cells
void Tiling::inter_rays(unsigned int iSeg, const double* l, double phi)
{
    double rho = -(l[0]*X0 + l[1]*Y0 + l[2]);
    for(unsigned int ix = 0; ix+1 < t.size(); ix++) {
        double y = cos(phi-t[ix]);
        y = (y < EPS && y > -EPS) ? d[d.size()-2] : rho / y;
        if(y >= d[0]) {
            int iy = 0;
            while(d[iy+1] <= y)
                ++ iy;
            add_to_cell(iSeg, EXTERIOR, (int)ix, iy);
            add_to_cell(iSeg, EXTERIOR, (ix>0)? (int)ix-1: (int)t.size()-2, iy);
        }
    }
}

// Find intersections with exterior cells of line given in homogeneous
// coordinates
void Tiling::inter_ext(unsigned int iSeg, double* l)
{
    double phi;
    proj2polar(l, phi); // Polar coordinates of supporting line
    inter_circles(iSeg, l, phi);
    inter_rays(iSeg, l, phi);
}

// Update cells intersected by segment
void Tiling::add(const Segment& seg, unsigned int iSeg)
{
    ++ nSegments;
    double l[3];
    proj_coords(seg, l); // Homogeneous coordinates of supporting line
    inter_int(iSeg, l); // Intersections with interior cells
    inter_ext(iSeg, l); // Intersections with exterior cells
}

// Meaningfulness of cells
void Tiling::compute_meaning()
{
    Binomial binom(p), binomInf(pInf);
    const std::vector<double>& B = binom(nSegments);
    const std::vector<double>& Binf = binomInf(nSegments);

    std::vector<Cell>::iterator it = intCells.begin();
    for(; it != intCells.end(); ++it)
        if((*it).segs != NULL)
            (*it).meaning = -log10(nCells*B[(*it).segs->size()]);
        else
            (*it).meaning = -INFINITY;

    for(it = extCells.begin(); it != extCells.end(); ++it)
        if((*it).segs != NULL)
            (*it).meaning = -log10(nCells*B[(*it).segs->size()]);
        else
            (*it).meaning = -INFINITY;

    it = extCells.begin() + (d.size()-2)*(t.size()-1);
    for(; it != extCells.end(); ++it)
        if((*it).segs != NULL)
            (*it).meaning = -log10(nCells*Binf[(*it).segs->size()]);
        else
            (*it).meaning = -INFINITY;
}

// Mark local maxima of meaningfulness above `threshold'
int Tiling::local_maxima(INT_EXT ie, double threshold)
{
    int nx, ny;
    if(ie == INTERIOR) {
        nx = (int)x.size() - 1; ny = (int)y.size() - 1;
    } else {
        nx = (int)t.size() - 1; ny = (int)d.size() - 1;
    }
    int n = 0;
    for(int iy = 0; iy < ny; iy++)
        for(int ix = 0; ix < nx; ix++) {
            double m = cell(ie, ix, iy).meaning;
            if(m >= threshold &&
               m >= meaning(ie, ix-1, iy-1) &&
               m >= meaning(ie, ix-1, iy  ) &&
               m >= meaning(ie, ix-1, iy+1) &&
               m >  meaning(ie, ix+1, iy-1) &&
               m >  meaning(ie, ix+1, iy  ) &&
               m >  meaning(ie, ix+1, iy+1) &&
               m >  meaning(ie, ix  , iy-1) &&
               (m > meaning(ie, ix  , iy+1) ||
                (m == meaning(ie, ix, iy+1) &&
                 (iy+1 < ny || ix < nx / 2)))) {
                cell(ie, ix, iy).vp = true;
                ++ n;
            }
        }
    return n;
}

// Mark as VP local maxima of meaningfulness
int Tiling::compute_vp(double threshold)
{
    compute_meaning();
    int n = local_maxima(INTERIOR, threshold);
    n    += local_maxima(EXTERIOR, threshold);
    return n;
}

// Add meaningful cells identifiers to vector
void Tiling::vp(std::vector<CellID>& vect) const
{
    for(unsigned int iy = 0; iy+1 < y.size(); iy++)
        for(unsigned int ix = 0; ix+1 < x.size(); ix++)
            if(cell(INTERIOR, ix, iy).vp)
                vect.push_back(CellID(const_cast<Tiling*>(this),
                                      INTERIOR, ix, iy));
    
    for(unsigned int id = 0; id+1 < d.size(); id++)
        for(unsigned int it = 0; it+1 < t.size(); it++)
            if(cell(EXTERIOR, it, id).vp)
                vect.push_back(CellID(const_cast<Tiling*>(this),
                                      EXTERIOR, it, id));
}

// Function to compute NFA
double Tiling::nfa(const MdlGraph& g, const MdlSet& set)
{
    int nHits = 0;
    for(unsigned int i = 0; i < set.edges.size(); i++)
        if(g.edges[set.edges[i]].status() != MDL_INVALID)
            ++ nHits;

    unsigned int n = (unsigned int) g.pointNodes.size(); // #segments not yet assigned
    for(int i = (int)n-1; i >= 0; i--) {
        const MdlPoint& pt = g.pointNodes[i];
        for(unsigned int j = 0; j < pt.edges.size(); j++)
            if(g.edges[pt.edges[j]].status() == MDL_ACCEPT) {
                -- n;
                break;
            }
    }

    //    const CellID& id = *static_cast<const CellID*const>(set.id);
    const CellID& id = *(const CellID*const)(set.id);
    bool infCell = (id.ie == EXTERIOR && id.iy+2 == (int)id.T->d.size());
    Binomial binom(infCell ? id.T->proba_inf() : id.T->proba());
    return id.T->nb_cells() * binom(n, nHits);
}

// Minimum Description Length
int Tiling::mdl(Tiling** Tilings, int nLevels, unsigned int nIni, double threshold)
{
    MdlGraph g(nIni);
    // Build graph
    for(int i = 0; i < nLevels; i++) {
        Tiling* T = Tilings[i];
        std::vector<CellID> cells;
        T->vp(cells);
        std::vector<CellID>::const_iterator it = cells.begin();
        for(; it != cells.end(); ++it) {
            T->cell(*it).vp = false;
            unsigned int s = g.new_set(new CellID(*it));
            const std::set<unsigned int>& indices = *T->cell(*it).segs;
            std::set<unsigned int>::const_iterator ind = indices.begin();
            for(; ind != indices.end(); ++ind)
                g.link(s, *ind);
        }
    }
    g.mdl(nfa, pow(10.0,-threshold));
    // Extract results
    int n = 0;
    for(unsigned int i = 0; i < g.setNodes.size(); i++) {
        if(g.setNodes[i].status() == MDL_ACCEPT) {
            ++ n;
            const CellID& id = *static_cast<CellID*>(g.setNodes[i].id);
            Cell& cell = id.T->cell(id);
            cell.vp = true;
            cell.meaning = -log10(g.setNodes[i].nfa(g, nfa));
            cell.segs->clear();
            const std::vector<unsigned int>& forwEdges = g.setNodes[i].edges;
            for(unsigned int j = 0; j < forwEdges.size(); j++)
                if(g.edges[forwEdges[j]].status() == MDL_ACCEPT)
                    cell.segs->insert(g.edges[forwEdges[j]].pointNode);
        }
        delete static_cast<CellID*>(g.setNodes[i].id);
    }
    return n;
}

// Conversion of cell into `Vpoint'
void Tiling::convert(Vpoint& vp,
                     const CellID& id, const std::vector<Segment>& seg) const
{
    if(id.ie == INTERIOR) {
        vp.x1 = vp.x4 = static_cast<float>(x[id.ix  ]);
        vp.y1 = vp.y2 = static_cast<float>(y[id.iy  ]);
        vp.x2 = vp.x3 = static_cast<float>(x[id.ix+1]);
        vp.y3 = vp.y4 = static_cast<float>(y[id.iy+1]);
    } else {
        float t1 = static_cast<float>(t[id.ix  ]);
        float t2 = static_cast<float>(t[id.ix+1]);
        float d1 = static_cast<float>(d[id.iy  ]);
        float d2;
        if(id.iy+2 == (int)d.size())
            d2 = 2.0f * d1;
        else
            d2 = static_cast<float>(d[id.iy+1]);
        polar2cart(t1, d1, vp.x1, vp.y1);
        polar2cart(t2, d1, vp.x2, vp.y2);
        polar2cart(t2, d2, vp.x3, vp.y3);
        polar2cart(t1, d2, vp.x4, vp.y4);
    }
    const Cell& c = cell(id);
    vp.weight = c.meaning;
    std::set<unsigned int>::const_iterator it(c.segs->begin());
    for(; it != c.segs->end(); ++it){
        vp.seg.push_back(seg[*it]);
	vp.segIndex.push_back(*it);
    }
}

void Vpoint::find_vp(std::vector<Vpoint>& vp, Tiling** Tilings, int nLevels,
                     const std::vector<Segment>& seg)
{
    double correction = log10((double) nLevels);
    for(int i = 0; i < nLevels; i++) {
        Tiling* T = Tilings[i];
        std::vector<Tiling::CellID> vect;
        T->vp(vect);
        std::vector<Tiling::CellID>::const_iterator it = vect.begin();
        for(; it != vect.end(); ++it) {
            Vpoint newVP;
            T->convert(newVP, *it, seg);
            newVP.weight -= correction;
            vp.push_back(newVP);
        }
    }
}

// Compute vanishing points based on segments
void Vpoint::detect(std::vector<Vpoint>& vp, const std::vector<Segment>& seg,
                    int w, int h, double eps)
{
    const int minLevel = 6; // Precision levels
    const int nLevels = 4;
    const double threshold = eps + log10((double)nLevels);

    Tiling** Tilings = new Tiling*[nLevels];
    int ntheta = (1 << minLevel); // #orientations
    for(int i = 0; i < nLevels; i++) { // Multi resolution
        //printf("\nVanishing regions for angular precision = pi/%d\n", ntheta);

        // Tiling for this precision level
        Tilings[i] = new Tiling(ntheta, w, h);

        // For each segment, update all tiles it meets
        for(unsigned int j = 0; j < seg.size(); j++)
            Tilings[i]->add(seg[j], j);

        // VPs are local maxima of meaningfulness
        int nvp = Tilings[i]->compute_vp(threshold);
        //printf("  %d maximal meaningful regions at level\n", nvp);
        ntheta <<= 1;
    }

    // Minimum Description Length
    //printf("\nMinimum Description Length...\n");
    Tiling::mdl(Tilings, nLevels, (unsigned int) seg.size(), threshold);
    find_vp(vp, Tilings, nLevels, seg);

    // Free memory
    for(int i = nLevels-1; i >= 0; i--)
        delete Tilings[i];
    delete [] Tilings;
}

// Compute segments and then vanishing points in `image'
void Vpoint::detect(std::vector<Vpoint>& vp,
                    const LWImage<unsigned char>& image, double eps,
                    float minNorm, int qGradient, int nLevels, int nDirections)
{
    Alignment align(minNorm, qGradient, nLevels, nDirections);
    std::vector<Segment> seg;
    align.detect(image, seg, eps);
    detect(vp, seg, image.w, image.h, eps);
}

// Precise position of the vanishing point
void Vpoint::pos(float& x, float& y, float& z) const
{
    InterLines inter(true);
    std::vector<Segment>::const_iterator it;
    // Normalization
    float zx=0, zy=0, dx=0, dy=0;
    for(it = seg.begin(); it != seg.end(); ++it) {
        dx += (*it).x1 + (*it).x2;
        dy += (*it).y1 + (*it).y2;
        zx += (*it).x1 * (*it).x1 + (*it).x2 * (*it).x2;
        zy += (*it).y1 * (*it).y1 + (*it).y2 * (*it).y2;
    }
    dx /= 2*seg.size();
    dy /= 2*seg.size();
    zx = (float)sqrt(zx - 2*seg.size()*dx*dx);
    zy = (float)sqrt(zy - 2*seg.size()*dy*dy);
    for(it = seg.begin(); it != seg.end(); ++it) {
        align::Line line(*it);
        line.l[2] += dx*line.l[0] + dy*line.l[1];
        line.l[0] *= zx;
        line.l[1] *= zy;
        line.weight = sqrt( (*it).qlength() );
        inter.add(line);
    }
    if(inter.compute(x, y, z)) {
        x = zx*x + z*dx; // Back to original coordinates
        y = zy*y + z*dy;
    } else {
        x = .25f * (x1 + x2 + x3 + x4);
        y = .25f * (y1 + y2 + y3 + y4);
        z = 1.0f;
    }
}
