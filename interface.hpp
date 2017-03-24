/*----------------------------------------------------------------------------    
  Copyright (c) 2016-2017 Yohann Salaun <yohann.salaun@imagine.enpc.fr>

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU Affero General Public License as
  published by the Free Software Foundation, either version 3 of the
  License, or (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU Affero General Public License for more details.

  You should have received a copy of the GNU Affero General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.

  ----------------------------------------------------------------------------*/

#ifndef INTERFACE_HPP
#define INTERFACE_HPP

// EIGEN
#include "openMVG/numeric/numeric.h"
#include <eigen/Eigen/src/Core/products/GeneralBlockPanelKernel.h>

// OPENCV
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

// STD LIB
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <map>

// descriptors parameters
const int widthOfBand_ = 7;
const int numOfBand_ = 9;
const short descriptorSize = numOfBand_ * 8;

enum FEATURE_TYPE {POINT, LINE, PARALLEL_PAIR, COPLANAR_PAIR};
enum GT_TYPE {ONLY_K, ONLY_RELATIVE, GLOBAL};

inline
std::string toString(const FEATURE_TYPE fType){
  switch(fType){
    case POINT:
      return "point";
    case LINE:
      return "line";
    case PARALLEL_PAIR:
      return "parallel pair";
  }
}

// return true if the pictures i and j are consecutive (also count the loop end)
inline
bool isConsecutive(const bool consecutive, const bool close_loop, const int i, const int j, const int nPictures){
  if(!consecutive){return i != j;}
  if(j-i == 1){return true;}
  return close_loop && i+nPictures-j == 1;
}

// strucuture to keep track of segment detected
struct Segment{
  // segment coordinates
  double x1, y1, x2, y2;

  // segment geometric attributes
  double width, length, angle;

  // NFA related arguments
  double log_nfa, prec;

  // scale of last detection from 0 (rawest) to n (finest)
  int scale;
  
  // descriptor
  std::vector<float> descriptor;
  
  // used for matching
  openMVG::Vec2 m;
  openMVG::Vec3 line, p1, p2;
  int vpIdx;
  
  // used for coplanar constraints
  std::vector<int> planes, lines3D, coplanar_cts;
  openMVG::Vec3 homogenous_line;

  Segment(){};
  Segment(const double X1, const double Y1, const double X2, const double Y2,
    const double w, const double p, const double nfa, const double s);
  
  // CLUSTERING METHOD
  bool isSame(int &ptr_l3D, const double angle_thresh, const double dist_thresh, const void* l3D) const;
  
  // FOR CALIBRATION/RECONSTRUCTION
  void normalize(const openMVG::Mat3 &K, const openMVG::Mat3 &Kinv);
  
  // FOR MULTISCALE LSD
  void upscale(const double k);
  
  // DISTANCE METHODS
  double distTo(const openMVG::Vec2 &p) const;
  double distTo(const Segment &s) const;
  
  // I/O METHODS for segments
  void readSegment(std::ifstream &file);
  void saveSegment(std::ofstream &file) const;
  
  // I/O METHODS for descriptors
  void readDescriptor(std::ifstream &file);
  void saveDescriptor(std::ofstream &file) const;
  
  double qlength();
  openMVG::Vec2 center();
  openMVG::Vec3 equation();
};

struct Vpoint{
  openMVG::Vec3 coords;
  std::vector<int> cluster;
  
  Vpoint(){}
  
  void refineCoords(const std::vector<Segment> &lines){ 
    float a2 = 0.f, b2 = 0.f, ab = 0.f, ac = 0.f, bc = 0.f;
    for(int i = 0; i < cluster.size(); i++){
      float ai = lines[cluster[i]].line[0];
      float bi = lines[cluster[i]].line[1];
      float ci = lines[cluster[i]].line[2];
      float di = ai*ai + bi*bi;
      
      a2 += (ai*ai)/di;
      b2 += (bi*bi)/di;
      ab += (ai*bi)/di;
      ac += (ai*ci)/di;
      bc += (bi*ci)/di;
    }
    
    coords[0] = ( ab*bc - b2*ac ) / ( a2*b2 - ab*ab );
    coords[1] = -(ab*coords[0] + bc)/b2;
    coords[2] = 1.0;
  }
  
  void normalize(const openMVG::Mat3 &Kinv){
    coords = Kinv*coords;
    coords.normalize();
  }
};

struct ParallelPair{
  int li, lj;
  int vp_idx;
  openMVG::Vec3 vp;
  bool correct;
  const double thresh_angle = sin(1.f*M_PI/180.f);
  
  ParallelPair(const int i, const int j, const int idx, const std::vector<Segment> &segments){
    li = i;
    lj = j;
    vp_idx = idx;
    vp = openMVG::CrossProductMatrix(segments[i].line)*segments[j].line;
    correct = vp.norm() > thresh_angle;
    vp.normalize();
  }
};

struct Sift{
  openMVG::Vec3 pt;
  double angle, scale;
};

typedef openMVG::Vec3 Point;
typedef std::vector<Point> PicturePoints;
typedef std::vector<PicturePoints> PicturesPoints;
typedef std::vector<Sift> PictureSifts;
typedef std::vector<PictureSifts> PicturesSifts;
typedef std::vector<Segment> PictureSegments;
typedef std::vector<PictureSegments> PicturesSegments;
typedef std::vector<Vpoint> PictureVPs;
typedef std::vector<PictureVPs> PicturesVPs;
typedef std::pair<Point, Point> PointConstraint;
typedef std::vector<PointConstraint> PointConstraints;
typedef std::pair<ParallelPair, ParallelPair> ParallelConstraint;
typedef std::vector<ParallelConstraint> ParallelConstraints;
typedef std::pair<Segment, Segment> LinePair;
typedef std::vector<LinePair> LinePairs;
typedef std::pair<int,int> PicturePair;
typedef std::pair<PicturePair, std::vector<int>> PictureMatches;
typedef std::map<PicturePair, std::vector<int>> PicturesMatches;
typedef std::pair<openMVG::Mat3, openMVG::Vec3> Pose;
typedef std::pair<PicturePair, Pose> PictureRelativePoses;
typedef std::map<PicturePair, Pose> PicturesRelativePoses;
typedef std::pair<FEATURE_TYPE, int> FTypeIndex;
typedef std::pair<double, FTypeIndex> ErrorFTypeIndex;

inline
double rotationError(const openMVG::Mat3 &R1, const openMVG::Mat3 &R2){
  openMVG::Mat3 R = R1.transpose()*R2;
  double cos_theta = (R(0,0) + R(1,1) + R(2,2) - 1)/2;
  cos_theta = std::min(std::max(cos_theta, -1.0), 1.0);
  return 180*acos(cos_theta)/M_PI;
}

inline
double translationError(const openMVG::Vec3 &t1, const openMVG::Vec3 &t2){
  double cos_theta = fabs(t2.dot(t1));
  cos_theta = std::min(std::max(cos_theta, -1.0), 1.0);
  return 180*acos(cos_theta)/M_PI;
}

struct GroundTruth{
  std::vector<openMVG::Mat3> rotations;
  std::vector<openMVG::Vec3> centers;
  PicturesRelativePoses relPoses;
  std::vector<openMVG::Mat3> K, Kinv;
  GT_TYPE gt_type;
  
  // for vgg dataset
  GroundTruth(const std::string &path, const std::vector<std::string> &picName, PicturesPoints &points, PicturesSegments &segments, 
	      PicturesMatches &matches_points, PicturesMatches &matches_lines, std::vector<PointConstraints> &ptCts);
  // for personal dataset + Strecha
  GroundTruth(const std::string &path, const std::vector<std::string> &picName, const bool consecutive, const bool close_loop, const std::string ext, const GT_TYPE g);
  void compareRelativePose(const PicturesRelativePoses &foundRelPoses) const;
  void compareGlobalPose(const std::vector<Pose> &foundGlobalPoses, const std::string &dirPath) const;
  void saveComputedPose(const std::string &path, const std::vector<std::string> &picName) const;
  void saveComputedPoseHofer(const std::string &path, const std::vector<std::string> &picName) const;
  
};

void compareGlobalPose(const std::vector<Pose> &foundGlobalPoses, const std::vector<Pose> &gtGlobalPoses, const std::string &dirPath);

struct Point3D{
  std::vector<int> proj_ids, cam_ids;
  openMVG::Vec3 p;
  
  Point3D(){}
};

struct Line3D{
  // 3D extremities
  openMVG::Vec3 p1, p2;
  
  // for plane clustering
  std::vector<int> proj_ids, cam_ids;
  openMVG::Vec3 direction;
  std::set<int> planes, cop_cts;
  
  // for translation norm computation only
  openMVG::Vec3 mid;
  double lambda;
  
  Line3D(const openMVG::Vec3 &P1, openMVG::Vec3 &P2){
    p1 = P1;
    p2 = P2;
    direction = (p1-p2).normalized();
    mid = 0.5*(p1 + p2);
  }
  
  // compute 3D line up to the translation scale
  Line3D(const Segment &l, const Segment &m, const openMVG::Vec3 &d, 
	 const openMVG::Mat3 &R1, const openMVG::Mat3 &R2, const openMVG::Vec3 &t12,
	 const int li, const int mi, const int i1, const int i2, const bool invert){
    proj_ids.push_back(li);
    proj_ids.push_back(mi);
    
    cam_ids.push_back(i1);
    cam_ids.push_back(i2);
    
    direction = d.normalized();
    
    mid = (invert)?  0.5*(m.p1 + m.p2) : 0.5*(l.p1 + l.p2);
    openMVG::Vec3 t = (invert)? -R1*R2.transpose()*t12 : t12;
    lambda = (invert)? t.dot(l.line)/((R2.transpose()*mid).dot(R1.transpose()*l.line))
		     : t.dot(m.line)/((R1.transpose()*mid).dot(R2.transpose()*m.line));
    
    p1 = -t12.dot(m.line)/((R1.transpose()*l.p1).dot(R2.transpose()*m.line))*R1.transpose()*l.p1;
    p2 = -t12.dot(m.line)/((R1.transpose()*l.p2).dot(R2.transpose()*m.line))*R1.transpose()*l.p2;
  }
  
  // only for t_norm method
  double distTo(const Line3D &l, const PicturesSegments &lines) const{
    return lines[cam_ids[1]][proj_ids[1]].distTo(lines[l.cam_ids[0]][l.proj_ids[0]]);
  }
  
  // general distance function
  double distTo(const Line3D &l) const{
    return std::min(std::min((p1-l.p1).norm(), (p2-l.p1).norm()),
		    std::min((p1-l.p2).norm(), (p2-l.p2).norm()));
  }
  
  void addProjection(const int i_cam, const int i_proj){
    proj_ids.push_back(i_proj);
    cam_ids.push_back(i_cam);
  }
  
  void addCopCts(Segment &s){
    for(int i = 0; i < s.coplanar_cts.size(); i++){
      cop_cts.insert(s.coplanar_cts[i]);
    }
  }
  
  bool isEqualUpTo(const Line3D &l, const double angle_thresh, const double dist_thresh) const{
    if(fabs(direction.dot(l.direction)) > angle_thresh){return false;}
    if(distTo(l) > dist_thresh){return false;}
    return true;
  }
};


struct ClusterPlane{
  std::vector<Line3D> lines;
  std::vector<int> proj_ids;

  openMVG::Vec3 normal, centroid, proj_intersec;
  double t_norm;
  
  // precomputation for RANSAC method
  double l_den_var, l_den_fixed, l_num_var2, l_num_var, l_num_fixed;
  openMVG::Vec3 P0, d_var, d_fixed, C0C1;
  
  void precompute(const openMVG::Vec3 &p0, const openMVG::Vec3 &p1, const openMVG::Vec3 &C0, const openMVG::Vec3 &C1){
    P0 = p0 + C0;
    C0C1 = C0-C1;
    
    d_fixed = openMVG::CrossProductMatrix(P0)*C1;
    d_var = openMVG::CrossProductMatrix(P0)*p1;
    
    openMVG::Vec3 u_fixed = openMVG::CrossProductMatrix(d_fixed)*P0;
    openMVG::Vec3 u_var = openMVG::CrossProductMatrix(d_var)*P0;
    
    l_den_fixed = C0C1.dot(u_fixed);
    l_den_var = C0C1.dot(u_var);
    l_num_fixed = C1.dot(u_fixed);
    l_num_var = C1.dot(u_var) + p1.dot(u_fixed);
    l_num_var2 = p1.dot(u_var);
  }
  
  void intersection(const openMVG::Vec3 &p1, const openMVG::Vec3 &C1, const double t_norm, openMVG::Vec3 &P0, openMVG::Vec3 &P1){
    double var = 1/t_norm;
    double lambda = (l_den_fixed + l_den_var*var)/(l_num_fixed + l_num_var*var + l_num_var2*var*var);
    
    openMVG::Vec3 d = (d_fixed + d_var*var).normalized();
    
    P0 = lambda * (p1*var + C1) + C1;
    P1 = P0 + (d.dot(C0C1))*d;
  }
  
};

struct Plane{
  // for display functions
  std::vector<openMVG::Vec3> basis;
  openMVG::Vec3 normal, centroid, rangePlus, rangeMinus;
  double median_distance, width, nfa;
  
  // when cluster form
  std::vector<int> proj_ids;
  int i_picture;
  
  // plane information
  std::set<int> lines3D;
  
  Plane(){}
  Plane(const ClusterPlane &c, const int i);
  Plane(const std::vector<Plane> &planes, const std::vector<int> &planes_idx, const std::vector<Line3D> &l3D);
  
  void computeCentroid(const std::vector<Line3D> &lines3D);
  void computeNormal(const std::vector<Line3D> &lines3D);
  void computeBasis(const std::vector<Line3D> &lines3D);
  void computeRange(const std::vector<Line3D> &lines3D);
  void computePlane(const std::vector<Line3D> &lines3D);
};

struct Triplet{
  // associated translation norm
  double t_norm;
  
  // indexes of the three features
  int idx[3];
  
  // precomputation for inliers computation
  std::vector<openMVG::Vec3> precomputation;
  
  // for bundle adjustment
  int iPicture;
  FEATURE_TYPE type;
};

struct CopCts{
  int i_cam[4], i_proj[4];
};


typedef std::vector<Line3D> Lines;
typedef std::vector<Point3D> Points;
typedef std::vector<Triplet> Triplets;

/*=================== VANISHING POINT ===================*/
// interface for vanishing point detection
PictureVPs computeVanishingPoints(const cv::Mat &im, PictureSegments &lines);

/*=================== CONSTRAINTS ===================*/
// select only point matches between two pictures (simplification for calibration methods)
PointConstraints selectPointMatches(const PicturePoints &p1, const PicturePoints &p2, const std::vector<int> &matches_points, std::vector<std::pair<int, int>> &features);
// select only line matches between two pictures (simplification for multi-view calibration methods)
LinePairs selectLineMatches(const PictureSegments &l1, const PictureSegments &l2, const std::vector<int> &matches_lines);
// add line intersections to point matches
void addLineIntersections(const PictureSegments &l1, const PictureSegments &l2, const std::vector<int> &matches_lines, 
			  PointConstraints &point_pairs, const openMVG::Mat3 &K);
void addLineIntersections(const PictureSegments &l1, const PictureSegments &l2, const std::vector<int> &matches_lines, 
			  PictureSifts &pt1, PictureSifts &pt2, std::vector<int> &matches_points, const int w, const int h);
// compute parallel pairs of lines constraints wrt vpoints and matches
ParallelConstraints computeParallelPairs(const PictureSegments &l1, const PictureSegments &l2, const std::vector<int> &matches_lines, std::vector<std::pair<int, int>> &features);

/*=================== NORMALIZATION ===================*/
// normalize the points, segments and vpoints wrt K and Kinv
void normalize(PicturesPoints &points, PicturesSegments &segments, PicturesVPs &vpoints, const std::vector<openMVG::Mat3> &K, const std::vector<openMVG::Mat3> &Kinv);

/*=================== INPUT/OUTPUT ===================*/
cv::Point2d fromHomog(const openMVG::Vec3 &p);

// read picture file 
// syntax is:
// number_of_pictures
// name whole_path
void readPictureFile(const std::string path, std::vector<std::string> &picName, std::vector<std::string> &picPath);

// load/save points detected in:
// - txt file @path/name_points.txt
// - picture @path/pictures/name_points.png
// @withNumber enables/disables the display of sifts corresponding number into the saved picture
PicturePoints readPoints(const std::string path, const std::string name);
PicturePoints readPointsOpenMVG(const std::string path, const std::string name);
PictureSifts readSifts(const std::string path, const std::string name);
void savePoints(const PictureSifts &points, const std::string path, const std::string name);
void savePointsPicture(const PictureSifts &points, const cv::Mat &image, 
		       const std::string path, const std::string name, const bool withNumber);


// load/save segments detected in:
// - txt file @path/name_lines.txt
// - picture @path/pictures/name_lines.png
// @withNumber enables/disables the display of segments corresponding number into the saved picture
std::vector<Segment> readLines(const std::string path, const std::string name);
void saveLines(const std::vector<Segment> &lines, const std::string path, const std::string name);
void saveLinesPicture(const std::vector<Segment> &lines, const cv::Mat &image, 
		      const std::string path, const std::string name, const bool withNumber);

// load/save segment descriptors in:
// - txt file @path/name_descriptors.txt
void readDescriptors(std::vector<Segment> &lines, const std::string path, const std::string name);
void saveDescriptors(const std::vector<Segment> &lines, const std::string path, const std::string name);

// load/save segments detected in:
// - txt file @path/name1_name2_matches_points/lines.txt
// - picture @path/pictures/name1_name2_matches_points/lines.png
std::vector<int> readMatches(const std::string path, const std::string picName1, const std::string picName2, const FEATURE_TYPE fType);
std::vector<int> readMatchesOpenMVG(const std::string path, const int i, const int j, const int n);
void saveMatches(const std::vector<int> &matches, const std::string path, 
		 const std::string picName1, const std::string picName2, const FEATURE_TYPE fType);
void saveMatchesPicture(const PictureSifts &l1, const PictureSifts &l2, const std::vector<int> &matches, 
			const cv::Mat &im1, const cv::Mat &im2, const std::string path, const std::string picName1, const std::string picName2);
void saveTripletsPicture(const PicturePoints &p1, const PicturePoints &p2, const PicturePoints &p3, 
			 const std::vector<int> &triplets, const cv::Mat &im1, const cv::Mat &im2, const cv::Mat &im3, 
			 const std::string path, const std::string picName1, const std::string picName2, const std::string picName3, const openMVG::Mat3 &K);
void saveMatchesPicture(const std::vector<Segment> &l1, const std::vector<Segment> &l2, const std::vector<int> &matches, 
			const cv::Mat &im1, const cv::Mat &im2, const std::string path, const std::string picName1, const std::string picName2, const bool vpWise);

std::vector<Vpoint> readVanishingPoints(std::vector<Segment> &lines, const std::string path, const std::string name);
std::vector<openMVG::Vec3> readVanishingPointDirections(const openMVG::Mat3 &Kinv, const std::string path, const std::string name);
void saveVanishingPoints(const std::vector<Vpoint> &vps, const std::string path, const std::string name);
void saveVanishingPointsPicture(const std::vector<Vpoint> &vps, const std::vector<Segment> &lines, const cv::Mat &im, const std::string path, const std::string name);

// save bifocal calibration inliers
void saveInliers(const std::vector<FTypeIndex> &inliers, const PictureSegments &seg1, const PictureSegments &seg2, const PicturePoints &points,
		 const std::vector<std::pair<int, int>> &point_pairs, const std::vector<std::pair<int, int>> &line_pairs, const std::vector<int> &matches,
		 const std::string path, const std::string picName1, const std::string picName2);

// load/save segments detected in:
// - txt file @path/pictures/name1_name2_pose.txt
Pose readPose(const std::string path, const std::string picName1, const std::string picName2);
void savePose(const Pose &pose, const openMVG::Mat3 &Kinv, const std::string path, const std::string picName1, const std::string picName2);

// save plane clusters in:
// - picture @path/pictures/name_plane_clusters.png
void saveClustersPicture(const std::vector<ClusterPlane> &planes, const std::vector<Segment> &lines, const cv::Mat &im, const std::string path, const std::string name);

// save reconstructed mesh in:
// - mesh @path/name.ply
void saveMesh(const Points &points, const Lines &lines, const std::vector<Plane> &planes, const std::vector<Pose> &globalPoses, const std::string &dirPath, const std::string &name, const bool insideMode = true);

#endif