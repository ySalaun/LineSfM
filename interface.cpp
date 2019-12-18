/*----------------------------------------------------------------------------    
  Copyright (c) 2016-2017 Yohann Salaun <yohann.salaun@imagine.enpc.fr>

  This program is free software: you can redistribute it and/or modify
  it under the terms of the Mozilla Public License as
  published by the Mozilla Foundation, either version 2.0 of the
  License, or (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  Mozilla Public License for more details.
  
  You should have received a copy of the Mozilla Public License
  along with this program. If not, see <https://www.mozilla.org/en-US/MPL/2.0/>.

  ----------------------------------------------------------------------------*/

#include "interface.hpp"

using namespace std;
using namespace cv;

// for color randomization
vector<Scalar> rgb = {Scalar(255,0,0), Scalar(0,255,0), Scalar(0,0,255), Scalar(255,255,0), Scalar(0,255,255), Scalar(255,0,255), 
		      Scalar(0,150,0), Scalar(150,0,0), Scalar(0,0,150), Scalar(150,150,0), Scalar(0,150,150), Scalar(150,0,150)};
int countRandomRGB = 0;

/*=================== SEGMENT ===================*/
Segment::Segment(const double X1, const double Y1, const double X2, const double Y2,
  const double w, const double p, const double nfa, const double s){
  x1 = X1; y1 = Y1;
  x2 = X2; y2 = Y2;
  
  width = w;
  prec = p;
  log_nfa = nfa;
  scale = s;
  
  p1 = openMVG::Vec3(x1, y1, 1);
  p2 = openMVG::Vec3(x2, y2, 1);
  
  length = sqrt(qlength());
  angle = atan2(y2 - y1, x2 - x1);
  
  m = center();
  
  line = equation();
  homogenous_line = line / sqrt(line[0]*line[0] + line[1]*line[1]);
  vpIdx = -1;
}

// CLUSTERING METHOD
bool Segment::isSame(int &ptr_l3D, const double angle_thresh, const double dist_thresh, const void* l3D) const{
  const vector<Line3D> *ptr_lines3D = (const vector<Line3D> *) l3D;
  for(int i = 0; i < lines3D.size(); i++){
    if((*ptr_lines3D)[ptr_l3D].isEqualUpTo((*ptr_lines3D)[lines3D[i]], angle_thresh, dist_thresh)){
      ptr_l3D = lines3D[i];
      return true;
    }
  }
  return false;
}

// FOR CALIBRATION/RECONSTRUCTION
void Segment::normalize(const openMVG::Mat3 &K, const openMVG::Mat3 &Kinv){
  line = K.transpose()*line;
  line.normalize();
  p1 = Kinv*p1;
  p1.normalize();
  p2 = Kinv*p2;
  p2.normalize();
}

// FOR MULTISCALE LSD
void Segment::upscale(const double k){
  x1 *= k; y1 *= k;
  x2 *= k; y2 *= k;
  width *= k;
  length *= k;
}

// DISTANCE METHODS
double Segment::distTo(const openMVG::Vec2 &p) const{
  return fabs(line[0]*p[0] + line[1]*p[1] + line[2])/sqrt(line[0]*line[0] + line[1]*line[1]);
}

double Segment::distTo(const Segment &s) const{
  return std::min(std::min(openMVG::Vec2(s.x1-x1, s.y1-y1).norm(), openMVG::Vec2(s.x1-x2, s.y1-y2).norm()), 
		  std::min(openMVG::Vec2(s.x2-x1, s.y2-y1).norm(), openMVG::Vec2(s.x2-x2, s.y2-y2).norm()));
}

// I/O METHODS for segments
void Segment::readSegment(std::ifstream &file){
  file >> x1 >> y1 >> x2 >> y2 >> width >> prec >> log_nfa >> scale;
  scale = 0;
  p1 = openMVG::Vec3(x1, y1, 1);
  p2 = openMVG::Vec3(x2, y2, 1);
  length = sqrt(qlength());
  angle = atan2(y2 - y1, x2 - x1);
  m = center();
  line = equation();
  homogenous_line = line / sqrt(line[0]*line[0] + line[1]*line[1]);
  vpIdx = -1;
}

void Segment::saveSegment(std::ofstream &file) const{
  file << x1 << "  " << y1 << "  " << x2 << "  " << y2 << " " << width << " " << prec << " " << log_nfa << " " << scale << std::endl;
}
// I/O METHODS for descriptors
void Segment::readDescriptor(std::ifstream &file){
  descriptor.resize(descriptorSize);
  for(int k = 0; k < descriptorSize; k++){
    file >> descriptor[k];
  }
}

void Segment::saveDescriptor(std::ofstream &file) const{
  for(int k = 0; k < descriptorSize; k++){
    file << descriptor[k] << " ";
  }
  file << std::endl;
}

// PRIVATE METHODS
double Segment::qlength(){
  double dX = x1 - x2;
  double dY = y1 - y2;
  length = dX*dX + dY*dY;
  return length;
}

openMVG::Vec2 Segment::center(){
  return openMVG::Vec2(0.5*(x1+x2), 0.5*(y1+y2));
}

openMVG::Vec3 Segment::equation(){
  openMVG::Vec3 line(y2-y1, x1-x2, 0);
  line[2] -= line[0]*x1 + line[1]*y1;
  return line;
}

/*=================== PLANE ===================*/
Plane::Plane(const ClusterPlane &c, const int i){
  normal = c.normal;
  centroid = c.centroid;
  proj_ids = c.proj_ids;
  i_picture = i;
}

Plane::Plane(const vector<Plane> &planes, const vector<int> &planes_idx, const vector<Line3D> &l3D){
  for(int i = 0; i < planes_idx.size(); i++){
    const Plane* p = &(planes[planes_idx[i]]);
    for(set<int>::const_iterator it = p->lines3D.begin(); it != p->lines3D.end(); it++){
      lines3D.insert(*it);
    }
  }
  computeNormal(l3D);
}

void Plane::computeCentroid(const std::vector<Line3D> &l3D){
  const int size = 2*lines3D.size();
    
  centroid = openMVG::Vec3(0,0,0);
  for(set<int>::const_iterator it = lines3D.begin(); it != lines3D.end(); it++){
    centroid += l3D[*it].p1;
    centroid += l3D[*it].p2;
  }
  centroid /= size;
}

void Plane::computeNormal(const vector<Line3D> &l3D){
  const int size = 2*lines3D.size();
  
  normal = openMVG::Vec3(0,0,0);
  Eigen::MatrixXd A(size, 3);
  int i = 0;
  for(set<int>::const_iterator it = lines3D.begin(); it != lines3D.end(); it++, i++){
    for(int j = 0; j < 3; j++){
      A(2*i, j) = l3D[*it].p1[j] - centroid[j];
      A(2*i+1, j) = l3D[*it].p2[j] - centroid[j];
    }
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
  for(int k = 0; k < 3; k++){
    normal[k] = svd.matrixV()(k,2);
  }
  normal.normalize();
}

void Plane::computeBasis(const vector<Line3D> &l3D){
  const int size = 2*lines3D.size();
   
  Eigen::MatrixXd A(size, 3);
  int i = 0;
  for(set<int>::const_iterator it = lines3D.begin(); it != lines3D.end(); it++, i++){
    double f1 = rand()%10 + 1;
    double f2 = rand()%10 + 1;
    for(int j = 0; j < 3; j++){
      A(2*i, j) = f1*l3D[*it].direction[j];
      A(2*i+1, j) = -f2*l3D[*it].direction[j];
    }
  }
  
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
  basis.resize(3);
  for(int k = 0; k < 3; k++){
    basis[0][k] = svd.matrixV()(k,0);
  }
  
  basis[0].normalize();
  basis[0] -= normal.dot(basis[0])*normal;
  basis[0].normalize();
  basis[1] = openMVG::CrossProductMatrix(normal)*basis[0];
  basis[2] = normal;
}

void Plane::computeRange(const vector<Line3D> &l3D){
  const int size = 2*lines3D.size();
  
  vector<double> dist_normal(size, 0);
  int i = 0;
  for(set<int>::const_iterator it = lines3D.begin(); it != lines3D.end(); it++, i++){
    for(int j = 0; j < 3; j++){
      double dist = (l3D[*it].p1 - centroid).dot(basis[j]);
      dist_normal[2*i] = fabs(dist);
      dist = (l3D[*it].p2 - centroid).dot(basis[j]);
      dist_normal[2*i+1] = fabs(dist);
    }
  }
  sort(dist_normal.begin(), dist_normal.end());
  median_distance = 2*dist_normal[dist_normal.size()/2];
  
  rangePlus = openMVG::Vec3(0,0,0);
  rangeMinus = openMVG::Vec3(0,0,0);
  i = 0;
  for(set<int>::const_iterator it = lines3D.begin(); it != lines3D.end(); it++, i++){
    for(int j = 0; j < 3; j++){
      double dist = (l3D[*it].p1 - centroid).dot(basis[j]);
      rangePlus[j] = max(rangePlus[j], dist);
      rangeMinus[j] = max(rangeMinus[j], -dist);
      dist = (l3D[*it].p2 - centroid).dot(basis[j]);
      rangePlus[j] = max(rangePlus[j], dist);
      rangeMinus[j] = max(rangeMinus[j], -dist);
    }
  }
  for(int j = 0; j < 3; j++){
    width = (j==0)? rangePlus[j] + rangeMinus[j] : min(width, rangePlus[j] + rangeMinus[j]);
  }
  
  // impose width = 0
  rangePlus[2] = rangeMinus[2] = 0;
}

void Plane::computePlane(const vector<Line3D> &l3D){    
  // compute centroid
  computeCentroid(l3D);

  // compute normal
  computeNormal(l3D);
  
  // compute basis
  computeBasis(l3D);
  
  // compute range
  computeRange(l3D);
}

/*=================== GROUND TRUTH ===================*/
#include "third_party/openMVG/tools_precisionEvaluationToGt.hpp"
#include "third_party/arpack++/include/blas1c.h"
#include <boost/concept_check.hpp>
#include "openMVG/multiview/projection.hpp"

GroundTruth::GroundTruth(const string &path, const vector<string> &picName, const bool consecutive, const bool close_loop, const string ext, const GT_TYPE g){
  gt_type = g;
  const int nPictures = picName.size();
  
  cout << "reading intrinsic parameters" << endl;
  ifstream Kfile((path + "K.txt").c_str(), ifstream::in);
  K.resize(nPictures);
  Kinv.resize(nPictures);
  Kfile >> K[0](0,0) >> K[0](0,1) >> K[0](0,2)
	>> K[0](1,0) >> K[0](1,1) >> K[0](1,2)
	>> K[0](2,0) >> K[0](2,1) >> K[0](2,2);
  for(int i = 0; i < nPictures; i++){
    K[i] = K[0];
    Kinv[i] = K[i].inverse();
  }
  
  switch(gt_type){
    case ONLY_K:
      break;
    case ONLY_RELATIVE:
      rotations.resize(0);
      centers.resize(0);
      cout << "computing relative poses" << endl;
      for(int i = 0; i < nPictures; i++){
	for(int j = i; j < nPictures; j++){
	  if(!isConsecutive(consecutive, close_loop, i, j, nPictures)){ continue;}
	  ifstream Rfile((path + picName[i] + "_" + picName[j] + "_R" + ext + ".txt").c_str(), ifstream::in);
	  openMVG::Mat3 R;
	  Rfile >> R(0,0) >> R(0,1) >> R(0,2)
		>> R(1,0) >> R(1,1) >> R(1,2)
		>> R(2,0) >> R(2,1) >> R(2,2);
	  openMVG::Vec3 t;
	  ifstream Tfile((path + picName[i] + "_" + picName[j] + "_t" + ext + ".txt").c_str(), ifstream::in);
	  Tfile >> t[0] >> t[1] >> t[2];
	  t.normalize();
	  relPoses.insert(PictureRelativePoses(PicturePair(i,j), Pose(R,t)));
	}
      }
      break;
    case GLOBAL:
      rotations.resize(nPictures);
      centers.resize(nPictures);
      cout << "reading global poses" << endl;
      for(int i = 0; i < nPictures; i++){
	ifstream camFile((path + picName[i] + ext).c_str(), ifstream::in);
	// reread K (useless)
	float temp;
	camFile >> temp >> temp >> temp
		>> temp >> temp >> temp
		>> temp >> temp >> temp
		>> temp >> temp >> temp;
	
	camFile >> rotations[i](0,0) >> rotations[i](0,1) >> rotations[i](0,2)
		>> rotations[i](1,0) >> rotations[i](1,1) >> rotations[i](1,2)
		>> rotations[i](2,0) >> rotations[i](2,1) >> rotations[i](2,2);
	
	camFile >> centers[i][0] >> centers[i][1] >> centers[i][2];
	
	rotations[i].transposeInPlace();
      }
	  
      cout << "computing relative poses" << endl;
      for(int i = 0; i < nPictures; i++){
	for(int j = i; j < nPictures; j++){
	  if(!isConsecutive(consecutive, close_loop, i, j, nPictures)){ continue;}
	  openMVG::Mat3 R = rotations[j]*rotations[i].transpose();
	  openMVG::Vec3 t = -rotations[j]*(centers[j] - centers[i]);
	  t.normalize();
	  relPoses.insert(PictureRelativePoses(PicturePair(i,j), Pose(R,t)));
	}
      }
      break;
  }
}

void GroundTruth::saveComputedPose(const string &path, const vector<string> &picName) const{
  const int nPictures = picName.size();
  for(int i = 0; i < nPictures; i++){
    ofstream camFile((path + picName[i] + "_computedPose.txt").c_str(), ofstream::out);
    // reread K (useless)
    float temp = 0;
    camFile << temp << temp << temp
	    << temp << temp << temp
	    << temp << temp << temp
	    << temp << temp << temp;
    
    const openMVG::Mat3 R = rotations[i].transpose();
	    
    camFile << R(0,0) << R(0,1) << R(0,2)
	    << R(1,0) << R(1,1) << R(1,2)
	    << R(2,0) << R(2,1) << R(2,2);
    
    camFile << centers[i][0] << centers[i][1] << centers[i][2];
  }
}

void GroundTruth::saveComputedPoseHofer(const string &path, const vector<string> &picName) const{
  const int nPictures = picName.size();
  for(int i = 0; i < nPictures; i++){
    ofstream rotFile((path + picName[i] + "_R.txt").c_str(), ofstream::out);
    const openMVG::Mat3 R = rotations[i];
	    
    rotFile << R(0,0) << " " << R(0,1) << " " << R(0,2)
	    << R(1,0) << " " << R(1,1) << " " << R(1,2)
	    << R(2,0) << " " << R(2,1) << " " << R(2,2);
    
    ofstream centerFile((path + picName[i] + "_C.txt").c_str(), ofstream::out);
    centerFile << centers[i][0] << " " << centers[i][1] << " " << centers[i][2];
  }
}

void GroundTruth::compareRelativePose(const PicturesRelativePoses &foundRelPoses) const{
  double rotErr = 0, transErr = 0;
  int count = 0;
  ofstream gtTxt("results_relative_pose.txt", ofstream::out);
  vector<double> rError, tError;
  for(PicturesRelativePoses::const_iterator it = relPoses.begin(); it != relPoses.end(); it++){
    PicturePair pair = it->first;
    cout << "images " << pair.first << " and " << pair.second << endl;
    PicturesRelativePoses::const_iterator found = foundRelPoses.find(pair);
    
    if(found != foundRelPoses.end()){
      cout << "Error in rotation : " << rotationError(it->second.first, found->second.first) << endl;
      cout << "Error in translation : " << translationError(it->second.second, found->second.second) << endl;
      rError.push_back(rotationError(it->second.first, found->second.first));
      tError.push_back(translationError(it->second.second, found->second.second));
      rotErr += rError[count];
      transErr += tError[count];
      count++;
    }
  }
  cout << "AVERAGE RESULTS: " << endl;
  cout << "Rotation: " << rotErr/count << endl;
  cout << "Translation: " << transErr/count << endl;
  
  gtTxt << "[";
  for(int i = 0; i < count; i++){
    gtTxt << rError[i];
    if(i != count-1){gtTxt << ",";}
    else{gtTxt << "]" << endl;}
  }
  
  gtTxt << "[";
  for(int i = 0; i < count; i++){
    gtTxt << tError[i];
    if(i != count-1){gtTxt << ",";}
    else{gtTxt << "]" << endl;}
  }
  
  gtTxt << rotErr/count << endl
	<< transErr/count << endl;
}
  
void GroundTruth::compareGlobalPose(const vector<Pose> &foundGlobalPoses, const string &dirPath) const{ 
  vector<openMVG::Vec3> pos_found;
  vector<openMVG::Mat3> rot_found;
  
  for(int i = 0; i < foundGlobalPoses.size(); i++){
    pos_found.push_back(foundGlobalPoses[i].second);
    rot_found.push_back(foundGlobalPoses[i].first);
  }
  
  htmlDocument::htmlDocumentStream _htmlDocStream("openMVG Quality evaluation.");
  openMVG::EvaluteToGT(centers, pos_found, rotations, rot_found, dirPath, &_htmlDocStream);
}

void compareGlobalPose(const vector<Pose> &foundGlobalPoses, const vector<Pose> &gtGlobalPoses, const string &dirPath){ 
  vector<openMVG::Vec3> pos_found, pos_gt;
  vector<openMVG::Mat3> rot_found, rot_gt;
  
  for(int i = 0; i < foundGlobalPoses.size(); i++){
    pos_found.push_back(foundGlobalPoses[i].second);
    rot_found.push_back(foundGlobalPoses[i].first);
    
    pos_gt.push_back(gtGlobalPoses[i].second);
    rot_gt.push_back(gtGlobalPoses[i].first);
  }
  
  htmlDocument::htmlDocumentStream _htmlDocStream("openMVG Quality evaluation.");
  openMVG::EvaluteToGT(pos_gt, pos_found, rot_gt, rot_found, dirPath, &_htmlDocStream);
}

/*=================== VANISHING POINT ===================*/
#include "acvp/vpoint.h"
PictureVPs computeVanishingPoints(const Mat &im, vector<Segment> &lines){
  // vector containing vanishing points
  vector<align::Vpoint> vp;
  
  // detect line segments
  int Width = im.cols, Height = im.rows;
  vector<align::Segment> seg(lines.size());
  for(unsigned int i = 0; i < lines.size(); i++){
    seg[i] = align::Segment(lines[i].x1, lines[i].y1, lines[i].x2, lines[i].y2);
  }

  align::Vpoint::detect(vp, seg, Width, Height);

  // store vanishing points
  PictureVPs vanishingPoints;
  
  for(vector<align::Vpoint>::const_iterator it = vp.begin(); it != vp.end(); ++it) {
    Vpoint vp;
    vp.cluster.resize((*it).seg.size());
    
    for(unsigned int j = 0; j < (*it).seg.size(); j++){
      // TODO maybe a better solution ?
      // find corresponding index in lines
      int index = -1;
      double minDist = 10;
      for(unsigned int k = 0; k < lines.size(); k++){
        double dist = abs((*it).seg[j].x1 - int(lines[k].x1)) + abs((*it).seg[j].y1 - int(lines[k].y1)) + abs((*it).seg[j].x2 - int(lines[k].x2)) + abs((*it).seg[j].y2 - int(lines[k].y2));
        if( dist < minDist ){
          minDist = dist;
          index = k;
        }
      }
      vector<int>::iterator it = find(vp.cluster.begin(), vp.cluster.end(), index);
      if (it == vp.cluster.end())
      {
        lines[index].vpIdx = vanishingPoints.size();
        vp.cluster.push_back(index);
      }
    }
    vp.refineCoords(lines);
    vanishingPoints.push_back(vp);
  }
  
  return vanishingPoints;
}

/*=================== CONSTRAINTS ===================*/
PointConstraints selectPointMatches(const PicturePoints &p1, const PicturePoints &p2, const vector<int> &matches_points, vector<pair<int, int>> &features){
  PointConstraints ppairs;
  
  for(int pi = 0; pi < p1.size(); pi++){
    int qi = matches_points[pi];
    
    // if the point has no match, discard it
    if(qi == -1){ continue;}

    ppairs.push_back(PointConstraint(p1[pi], p2[qi]));
    features.push_back(pair<int, int>(pi, qi));
  }
  return ppairs;
}

LinePairs selectLineMatches(const PictureSegments &l1, const PictureSegments &l2, const vector<int> &matches_lines){
  LinePairs lpairs;
  
  for(int li = 0; li < l1.size(); li++){
    int mi = matches_lines[li];
    
    // if the point has no match, discard it
    if(mi == -1){ continue;}

    lpairs.push_back(LinePair(l1[li], l2[mi]));
  }
  return lpairs;
}

void addLineIntersections(const PictureSegments &l1, const PictureSegments &l2, const vector<int> &matches_lines, PointConstraints &point_pairs, const openMVG::Mat3 &K){
  LinePairs lpairs = selectLineMatches(l1, l2, matches_lines);
  const double w = K(0,2)*2;
  const double h = K(1,2)*2;
  const double thresh_neighbouring_line = 0.2*(w+h)/2;
  for(int i = 0; i < lpairs.size(); i++){
    openMVG::Vec3 p1, p2, q1, q2;
    p1 = K*lpairs[i].first.p1; p1 /= p1[2];
    p2 = K*lpairs[i].first.p2; p2 /= p2[2];
    q1 = K*lpairs[i].second.p1; q1 /= q1[2];
    q2 = K*lpairs[i].second.p2; q2 /= q2[2];
    for(int j = 0; j < i; j++){
      openMVG::Vec3 inter1 = openMVG::CrossProductMatrix(lpairs[i].first.line)*lpairs[j].first.line;
      openMVG::Vec3 inter2 = openMVG::CrossProductMatrix(lpairs[i].second.line)*lpairs[j].second.line;

      // check if the intersections are inside the pictures
      openMVG::Vec3 proj1 = K*inter1; proj1 /= proj1[2];
      if(proj1[0] < 0 || proj1[0] > w || proj1[1] < 0 || proj1[1] > h){ continue;}
      openMVG::Vec3 proj2 = K*inter2; proj2 /= proj2[2];
      if(proj2[0] < 0 || proj2[0] > w || proj2[1] < 0 || proj2[1] > h){ continue;}
      
      // check if the lines are close
      openMVG::Vec3 p3, p4, q3, q4;
      p3 = K*lpairs[j].first.p1; p3 /= p3[2];
      p4 = K*lpairs[j].first.p2; p4 /= p4[2];
      q3 = K*lpairs[j].second.p1; q3 /= q3[2];
      q4 = K*lpairs[j].second.p2; q4 /= q4[2];

      if(min(min((p1-p3).norm(), (p2-p3).norm()), min((p1-p4).norm(), (p2-p4).norm())) > thresh_neighbouring_line){continue;}
      if(min(min((q1-q3).norm(), (q2-q3).norm()), min((q1-q4).norm(), (q2-q4).norm())) > thresh_neighbouring_line){continue;}
      if(min(min((p1-proj1).norm(), (p2-proj1).norm()), min((p3-proj1).norm(), (p4-proj1).norm())) > thresh_neighbouring_line){continue;}
      if(min(min((q1-proj2).norm(), (q2-proj2).norm()), min((q3-proj2).norm(), (q4-proj2).norm())) > thresh_neighbouring_line){continue;}
      
      point_pairs.push_back(PointConstraint(inter1.normalized(), inter2.normalized()));
    }
  }
}

void addLineIntersections(const PictureSegments &l1, const PictureSegments &l2, const vector<int> &matches_lines, 
			  PictureSifts &pt1, PictureSifts &pt2, vector<int> &matches_points, const int w, const int h){
  LinePairs lpairs = selectLineMatches(l1, l2, matches_lines);
  int count = 0;
  const double thresh_neighbouring_line = 0.2*(w+h)/2;
  for(int i = 0; i < lpairs.size(); i++){
    openMVG::Vec3 p1, p2, q1, q2;
    p1 = lpairs[i].first.p1; p1 /= p1[2];
    p2 = lpairs[i].first.p2; p2 /= p2[2];
    q1 = lpairs[i].second.p1; q1 /= q1[2];
    q2 = lpairs[i].second.p2; q2 /= q2[2];
    for(int j = 0; j < i; j++){
      openMVG::Vec3 inter1 = openMVG::CrossProductMatrix(lpairs[i].first.line)*lpairs[j].first.line;
      openMVG::Vec3 inter2 = openMVG::CrossProductMatrix(lpairs[i].second.line)*lpairs[j].second.line;

      // check if the intersections are inside the pictures
      openMVG::Vec3 proj1 = inter1; proj1 /= proj1[2];
      if(proj1[0] < 0 || proj1[0] > w || proj1[1] < 0 || proj1[1] > h){ continue;}
      openMVG::Vec3 proj2 = inter2; proj2 /= proj2[2];
      if(proj2[0] < 0 || proj2[0] > w || proj2[1] < 0 || proj2[1] > h){ continue;}
      
      // check if the lines are close
      openMVG::Vec3 p3, p4, q3, q4;
      p3 = lpairs[j].first.p1; p3 /= p3[2];
      p4 = lpairs[j].first.p2; p4 /= p4[2];
      q3 = lpairs[j].second.p1; q3 /= q3[2];
      q4 = lpairs[j].second.p2; q4 /= q4[2];

      if(min(min((p1-p3).norm(), (p2-p3).norm()), min((p1-p4).norm(), (p2-p4).norm())) > thresh_neighbouring_line){continue;}
      if(min(min((q1-q3).norm(), (q2-q3).norm()), min((q1-q4).norm(), (q2-q4).norm())) > thresh_neighbouring_line){continue;}
      if(min(min((p1-proj1).norm(), (p2-proj1).norm()), min((p3-proj1).norm(), (p4-proj1).norm())) > 0.5*thresh_neighbouring_line){continue;}
      if(min(min((q1-proj2).norm(), (q2-proj2).norm()), min((q3-proj2).norm(), (q4-proj2).norm())) > 0.5*thresh_neighbouring_line){continue;}
      
      Sift s1; s1.pt = proj1; s1.angle = s1.scale = 0;
      Sift s2; s2.pt = proj2; s2.angle = s2.scale = 0;
      matches_points.push_back(pt2.size());
      pt1.push_back(s1);
      pt2.push_back(s2);
      count ++;
    }
  }
  cout << "added " << count << " intersections" << endl;
}


ParallelConstraints computeParallelPairs(const PictureSegments &l1, const PictureSegments &l2, const vector<int> &matches_lines, vector<pair<int,int>> &features){
  ParallelConstraints ppairs;
  
  for(int li = 0; li < l1.size(); li++){
    int mi = matches_lines[li];
    
    // if the line has no match, discard it
    if(mi == -1){ continue;}
    
    int vp_li = l1[li].vpIdx;
    int vp_mi = l2[mi].vpIdx;
    
    // test the vps
    if(vp_li == -1 || vp_mi == -1){continue;}
    
    for(int lj = 0; lj < li; lj++){
      int mj = matches_lines[lj];
      
      // if the line has no match, discard it
      if(mj == -1){ continue;}
      
      int vp_lj = l1[lj].vpIdx;
      int vp_mj = l2[mj].vpIdx;
      
      // test the vps
      if(vp_li != vp_lj || vp_mi != vp_mj){continue;}
      
      // check the matches _vps
      //if(l2[mi].vpIdx != l2[mj].vpIdx){ continue;}
      
      ParallelPair pp1(li, lj, vp_li, l1);
      ParallelPair pp2(mi, mj, vp_mi, l2);
      
      // add the pair to the list
      if(pp1.correct && pp2.correct){
	ppairs.push_back(ParallelConstraint(pp1, pp2));
	features.push_back(pair<int,int>(li, lj));
      }
    }
  }
  return ppairs;
}

/*=================== NORMALIZATION ===================*/
void normalize(PicturesPoints &points, const vector<openMVG::Mat3> &Kinv){
  for(int i = 0; i < points.size(); i++){
    for(int j = 0; j < points[i].size(); j++){
      points[i][j] = (Kinv[i]*points[i][j]).normalized();
    }
  }
}

void normalize(PicturesSegments &segments, const vector<openMVG::Mat3> &K, const vector<openMVG::Mat3> &Kinv){
  for(int i = 0; i < segments.size(); i++){
    for(int j = 0; j < segments[i].size(); j++){
      segments[i][j].normalize(K[i], Kinv[i]);
    }
  }
}

void normalize(PicturesVPs &vpoints, const vector<openMVG::Mat3> &Kinv){
  for(int i = 0; i < vpoints.size(); i++){
    for(int j = 0; j < vpoints[i].size(); j++){
      vpoints[i][j].normalize(Kinv[i]);
    }
  }
}

void normalize(PicturesPoints &points, PicturesSegments &segments, PicturesVPs &vpoints, const vector<openMVG::Mat3> &K, const vector<openMVG::Mat3> &Kinv){
  normalize(points, Kinv);
  normalize(segments, K, Kinv);
  normalize(vpoints, Kinv);
}

/*=================== INPUT/OUTPUT ===================*/
Point2d fromHomog(const openMVG::Vec3 &p){
  return Point2d(p[0]/p[2], p[1]/p[2]);
}

inline
int thicknessFromImage(const Mat &im){
  return max((im.cols + im.rows)/400, 2);
}

void readPictureFile(const string path, vector<string> &picName, vector<string> &picPath){
  ifstream picListTxt(path , ifstream::in);
  int nPictures;
  picListTxt >> nPictures;
  cout << nPictures << endl;
  picName.resize(nPictures);
  picPath.resize(nPictures);
  for(int i = 0; i < nPictures; i++){
    picListTxt >> picName[i] >> picPath[i];
    cout << picName[i] << "         " << picPath[i] << endl;
  }
}

PicturePoints readPoints(const string path, const string name){
  PicturePoints points;

  ifstream pointsTxt((path + name + "_points.txt").c_str(), ifstream::in);
  int nPoints;
  pointsTxt >> nPoints;
  points.resize(nPoints);

  for (int i = 0; i < nPoints; ++i) {
    // read point informations
    double scale, angle;
    pointsTxt >> points[i][0] >> points[i][1] >> scale >> angle;
    points[i][2] = 1;
  }
  return points;
}

PicturePoints readPointsOpenMVG(const string path, const string name){
  PicturePoints points;

  ifstream pointsTxt((path + "openMVG/" + name + ".feat").c_str(), ifstream::in);
  while(!pointsTxt.eof()){
    openMVG::Vec3 p;
    // read point informations
    double scale, angle;
    pointsTxt >> p[0] >> p[1] >> scale >> angle;
    p[2] = 1;
    points.push_back(p);
  }
  points.pop_back();
  return points;
}

PictureSifts readSifts(const string path, const string name){
  PictureSifts sifts;

  ifstream pointsTxt((path + name + "_points.txt").c_str(), ifstream::in);
  int nPoints;
  pointsTxt >> nPoints;
  sifts.resize(nPoints);

  for (int i = 0; i < nPoints; ++i) {
    // read point informations
    double scale, angle;
    pointsTxt >> sifts[i].pt[0] >> sifts[i].pt[1] >> sifts[i].scale >> sifts[i].angle;
    sifts[i].pt[2] = 1;
  }
  return sifts;
}

void savePoints(const PictureSifts &points, const string path, const string name){
  ofstream pointsTxt((path + name + "_points.txt").c_str(), ofstream::out);
  pointsTxt << points.size() << endl;

  for (int i = 0; i < points.size(); ++i) {
    // add coordinates to txt file
    pointsTxt << points[i].pt[0]/points[i].pt[2] << " " << points[i].pt[1]/points[i].pt[2] << " " << points[i].scale << " " << points[i].angle << endl;
  }
}

void savePointsPicture(const PictureSifts &points, const Mat &im, const string path, const string name, const bool withNumber){
  Mat image;
  im.copyTo(image);

  int thickness = max(3.0,double(thicknessFromImage(im)));

  for (int i = 0; i < points.size(); ++i) {
    // random colors for each segment
    countRandomRGB = (countRandomRGB+1)%rgb.size();
    
    Point2d p = fromHomog(points[i].pt);
    circle(image, p, 5.0, rgb[countRandomRGB], 2);

    // to add the segment number into the picture
    if (withNumber){
      ostringstream ss;
      ss << i;
      putText(image, ss.str(), p, FONT_HERSHEY_SCRIPT_SIMPLEX, 5, rgb[countRandomRGB], 3);
    }
  }
  imwrite(path + "pictures/" + name + "_points.jpg", image);
}

PictureSegments readLines(const string path, const string name){
  PictureSegments lines;

  ifstream linesTxt((path + name + "_lines.txt").c_str(), ifstream::in);
  int nLines;
  linesTxt >> nLines;
  lines.resize(nLines);

  for (int i = 0; i < nLines; ++i) {
    // read segment informations
    lines[i].readSegment(linesTxt);
  }
  return lines;
}

void saveLines(const PictureSegments &lines, const string path, const string name){
  ofstream linesTxt((path + name + "_lines.txt").c_str(), ofstream::out);
  linesTxt << lines.size() << endl;

  for (int i = 0; i < lines.size(); ++i) {
    // add coordinates to txt file
    lines[i].saveSegment(linesTxt);
  }
}

void saveLinesPicture(const PictureSegments &lines, const Mat &im, const string path, const string name, const bool withNumber){
  Mat image;
  im.copyTo(image);

  int thickness = thicknessFromImage(im);

  for (int i = 0; i < lines.size(); ++i) {
    // random colors for each segment
    countRandomRGB = (countRandomRGB+1)%rgb.size();

    Point2f p1(lines[i].x1, lines[i].y1);
    Point2f p2(lines[i].x2, lines[i].y2);
    line(image, p1, p2, rgb[countRandomRGB], thickness);
    circle(image, p1, thickness, rgb[countRandomRGB]);

    // to add the segment number into the picture
    if (withNumber){
      ostringstream ss;
      ss << i;
      putText(image, ss.str(), Point2f(0.5*(p1.x + p2.x), 0.5*(p1.y + p2.y)),
        FONT_HERSHEY_SCRIPT_SIMPLEX, 5, rgb[countRandomRGB], 3);
    }
  }
  imwrite(path + "pictures/" + name + "_lines.jpg", image);
}

void readDescriptors(PictureSegments &lines, const string path, const string name){
  ifstream descrTxt((path + name + "_descriptors.txt").c_str(), ifstream::in);
  int nLines;
  descrTxt >> nLines;
  
  if(lines.size() != nLines){
    cout << "ERROR, descriptor file does not correspond to line file" << endl;
    int pause; cin >> pause;
  }

  for (int i = 0; i < lines.size(); ++i) {
    // add descriptor to txt file
    lines[i].readDescriptor(descrTxt);
  }
}

void saveDescriptors(const PictureSegments &lines, const string path, const string name){
  ofstream descrTxt((path + name + "_descriptors.txt").c_str(), ofstream::out);
  descrTxt << lines.size() << endl;
  for (int i = 0; i < lines.size(); ++i) {
    // add descriptor to txt file
    lines[i].saveDescriptor(descrTxt);
  }
}

vector<int> readMatches(const string path, const string picName1, const string picName2, const FEATURE_TYPE fType){
  ifstream matchesTxt((path + picName1 + "_" + picName2 + "_matches_" + toString(fType) + ".txt").c_str(), ifstream::in);
  int nMatches;
  matchesTxt >> nMatches;

  vector<int> matches(nMatches, -1);
  while(!matchesTxt.eof()){
    int lj, mj;
    matchesTxt >> lj >> mj;
    matches[lj] = mj;
  }
  
  return matches;
}

vector<int> readMatchesOpenMVG(const string path, const int i, const int j, const int n){
  ifstream matchesTxt((path + "openMVG/matches.putative.txt").c_str(), ifstream::in);
  vector<int> matches(n, -1);

  while(!matchesTxt.eof()){
    int pi, pj;
    matchesTxt >> pi >> pj;

    bool toRead = pi == i && pj == j;
    int nMatches;
    matchesTxt >> nMatches;

    for(int i = 0; i < nMatches; i++){
      int lj, mj;
      matchesTxt >> lj >> mj;
      if(toRead){
	matches[lj] = mj;
      }
    }
    if(toRead){break;}
  }
  matchesTxt.close();

  return matches;
}

void saveMatches(const vector<int> &matches, const string path, const string picName1, const string picName2, const FEATURE_TYPE fType){  
  ofstream matchesTxt((path + picName1 + "_" + picName2 + "_matches_" + toString(fType) + ".txt").c_str(), ofstream::out);
  matchesTxt << matches.size() << endl;
  
  // save the matches inside txt file
  for(unsigned int i = 0; i < matches.size(); i++){ 
    if(matches[i] == -1){continue;}
    matchesTxt << i << " " << matches[i] << endl;
  }  
}

Mat concatPictures(const Mat &image1, const Mat &image2, bool &swapped){
  Mat im1, im2;
  if(image1.rows < image2.rows){
    image1.copyTo(im1);
    image2.copyTo(im2);
    swapped = false;
  }
  else{
    image1.copyTo(im2);
    image2.copyTo(im1); 
    swapped = true;
  }
  
  Mat enlarged;
  im2.copyTo(enlarged);

  for(int i = 0; i < enlarged.rows; i++){
    for(int j = 0; j < enlarged.cols; j++){
      if(i < im1.rows && j < im1.cols){
	enlarged.at<Vec3b>(i,j) = im1.at<Vec3b>(i,j);
      }
      else{
	enlarged.at<Vec3b>(i,j) *= 0;
      }
    }
  }
  
  Mat res;
  vconcat(enlarged, im2, res);
  return res;
}

void saveMatchesPicture(const PictureSifts &p1, const PictureSifts &p2, const vector<int> &matches, const Mat &im1, const Mat &im2, 
			const string path, const string picName1, const string picName2){  
  Mat image;
  bool swapped = false;
  if(im1.rows != im2.rows || im1.cols != im2.cols){
    cout << "pictures of different sizes, method can fail" << endl;
    image = concatPictures(im1, im2, swapped);
  }
  else{ 
    vconcat(im1, im2, image);
  }

  int thickness = thicknessFromImage(im1);
  Point2d offset(0,im1.rows);
  
  for(unsigned int i = 0; i < matches.size(); i++){    
    if(matches[i] == -1){
      continue;
    }
    countRandomRGB = (countRandomRGB+1)%rgb.size();
    
    circle(image, fromHomog(p1[i].pt), thickness, rgb[countRandomRGB]);
    circle(image, fromHomog(p2[matches[i]].pt)+offset, thickness, rgb[countRandomRGB]);
    line(image, fromHomog(p1[i].pt), fromHomog(p2[matches[i]].pt)+offset, rgb[countRandomRGB], thickness/2);   
  }  
  
  imwrite(path + "pictures/" + picName1 + "_" + picName2 + "_matches_points.jpg",image);  
}

void saveTripletsPicture(const PicturePoints &p1, const PicturePoints &p2, const PicturePoints &p3, 
			 const vector<int> &triplets, const Mat &im1, const Mat &im2, const Mat &im3, 
			 const string path, const string picName1, const string picName2, const string picName3, const openMVG::Mat3 &K){  
  if(im1.rows != im2.rows || im1.cols != im2.cols){
    cout << "pictures of different sizes, save matches is not possible" << endl;
    int pause; cin >> pause;
    return;
  }
  
  Mat image;
  vconcat(im1, im2, image);
  vconcat(image, im3, image);

  int thickness = thicknessFromImage(im1);
  Point2d offset(0,im1.rows);
  Point2d offset2(0,2*im1.rows);
  
  for(unsigned int i = 0; i < triplets.size()/3; i++){    
    countRandomRGB = (countRandomRGB+1)%rgb.size();
    
    circle(image, fromHomog(K*p1[triplets[3*i]]), thickness, rgb[countRandomRGB]);
    circle(image, fromHomog(K*p2[triplets[3*i+1]])+offset, thickness, rgb[countRandomRGB]);
    circle(image, fromHomog(K*p3[triplets[3*i+2]])+offset2, thickness, rgb[countRandomRGB]);
    line(image, fromHomog(K*p1[triplets[3*i]]), fromHomog(K*p2[triplets[3*i+1]])+offset, rgb[countRandomRGB], thickness/2);   
    line(image, fromHomog(K*p2[triplets[3*i+1]])+offset, fromHomog(K*p3[triplets[3*i+2]])+offset2, rgb[countRandomRGB], thickness/2);   
  }  
  
  imwrite(path + "pictures/" + picName1 + "_" + picName2 + "_" + picName3 + "_triplet_points.jpg",image);
}

void saveMatchesPicture(const vector<Segment> &l1, const vector<Segment> &l2, const vector<int> &matches, const Mat &im1, const Mat &im2, 
			const string path, const string picName1, const string picName2, const bool vpWise){  
  Mat image;
  bool swapped = false;
  if(im1.rows != im2.rows || im1.cols != im2.cols){
    cout << "pictures of different sizes, method can fail" << endl;
    image = concatPictures(im1, im2, swapped);
  }
  else{ 
    vconcat(im1, im2, image);
  }
  imwrite(path + "pictures/" + picName1 + "_" + picName2 + "_matches_lines" + ((vpWise)? "_vps":"") + ".jpg",image); 
  
  int thickness = thicknessFromImage(im1);
  int offset = im1.rows;
  
  for(unsigned int i = 0; i < matches.size(); i++){    
    if(matches[i] == -1 || (vpWise && l1[i].vpIdx == -1 && l2[matches[i]].vpIdx == -1)){
      continue;
    }
    countRandomRGB = (countRandomRGB+1)%rgb.size();
    
    if(swapped){
      Point2f l11(l1[i].x1, l1[i].y1 + offset);
      Point2f l12(l1[i].x2, l1[i].y2 + offset);
      Point2f l21(l2[matches[i]].x1, l2[matches[i]].y1);
      Point2f l22(l2[matches[i]].x2, l2[matches[i]].y2);
      
      line(image, l11, l12, rgb[countRandomRGB], thickness);
      line(image, l21, l22, rgb[countRandomRGB], thickness);   
      line(image, 0.5f*(l11 + l12), 0.5f*(l21+l22), rgb[countRandomRGB],thickness/2);
    }
    else{
      Point2f l11(l1[i].x1, l1[i].y1);
      Point2f l12(l1[i].x2, l1[i].y2);
      Point2f l21(l2[matches[i]].x1, l2[matches[i]].y1 + offset);
      Point2f l22(l2[matches[i]].x2, l2[matches[i]].y2 + offset);   
      
      line(image, l11, l12, rgb[countRandomRGB], thickness);
      line(image, l21, l22, rgb[countRandomRGB], thickness);   
      line(image, 0.5f*(l11 + l12), 0.5f*(l21+l22), rgb[countRandomRGB],thickness/2);
    }
  }  
  
  imwrite(path + "pictures/" + picName1 + "_" + picName2 + "_matches_lines" + ((vpWise)? "_vps":"") + ".jpg",image);  
}

vector<Vpoint> readVanishingPoints(vector<Segment> &lines, const string path, const string name){
  ifstream vpTxt((path + name + "_vps.txt").c_str(), ifstream::in);
  
  int nVPs;
  vpTxt >> nVPs;
  
  vector<Vpoint> vps(nVPs);
  for(int i = 0; i < nVPs; i++){
    vpTxt >> vps[i].coords[0] >> vps[i].coords[1] >> vps[i].coords[2];
    int sizeCluster;
    vpTxt >> sizeCluster;
    vps[i].cluster.resize(sizeCluster);
    for(int j = 0; j < sizeCluster; j++){
      vpTxt >> vps[i].cluster[j];
      lines[vps[i].cluster[j]].vpIdx = i;
    } 
  }
  
  return vps;
}

vector<openMVG::Vec3> readVanishingPointDirections(const openMVG::Mat3 &Kinv, const string path, const string name){
  ifstream vpTxt((path + name + "_vps.txt").c_str(), ifstream::in);
  
  int nVPs;
  vpTxt >> nVPs;
  
  vector<openMVG::Vec3> vps(nVPs);
  for(int i = 0; i < nVPs; i++){
    vpTxt >> vps[i][0] >> vps[i][1] >> vps[i][2];
    vps[i] = Kinv*vps[i];
    vps[i].normalize();
    int sizeCluster;
    vpTxt >> sizeCluster;
    for(int j = 0; j < sizeCluster; j++){
      double temp;
      vpTxt >> temp;
    } 
  }
  
  return vps;
}

void saveVanishingPoints(const vector<Vpoint> &vps, const string path, const string name){
  ofstream vpTxt((path + name + "_vps.txt").c_str(), ofstream::out);
  
  vpTxt << vps.size() << endl;
  
  for(int i = 0; i < vps.size(); i++){
    vpTxt << vps[i].coords.transpose() << endl;
    const int sizeCluster = vps[i].cluster.size();
    vpTxt << sizeCluster << endl;
    for(int j = 0; j < sizeCluster; j++){
      vpTxt << vps[i].cluster[j] << endl;
    }
  }
}

void saveVanishingPointsPicture(const vector<Vpoint> &vps, const vector<Segment> &lines, const Mat &im, const string path, const string name){
  Mat image;
  im.copyTo(image);

  int thickness = thicknessFromImage(im);

  for (int i = 0; i < vps.size(); ++i) {
    // random colors for each vp
    countRandomRGB = (countRandomRGB+1)%rgb.size();

    Point2f p(vps[i].coords[0]/vps[i].coords[2], vps[i].coords[1]/vps[i].coords[2]);
    circle(image, p, thickness, rgb[countRandomRGB]);
    
    for(int j = 0; j < vps[i].cluster.size(); j++){
      Point2f p1(lines[vps[i].cluster[j]].x1, lines[vps[i].cluster[j]].y1);
      Point2f p2(lines[vps[i].cluster[j]].x2, lines[vps[i].cluster[j]].y2);
      line(image, p1, p2, rgb[countRandomRGB], thickness);
    }
  }
  imwrite(path + "pictures/" + name + "_vps.jpg", image);
}

void saveInliers(const vector<FTypeIndex> &inliers, const PictureSegments &seg1, const PictureSegments &seg2, const PicturePoints &points,
		 const vector<pair<int, int>> &point_pairs, const vector<pair<int, int>> &line_pairs, const vector<int> &matches,
		 const string path, const string picName1, const string picName2){
  set<int> inliers_lines;
  {
    ofstream matchesTxt((path + picName1 + "_" + picName2 + "_inliers_matches_point.txt").c_str(), ofstream::out);
    ofstream vpsTxt((path + picName1 + "_" + picName2 + "_inliers_vps.txt").c_str(), ofstream::out);
    matchesTxt << points.size() << endl;
    for(int k = 0; k < inliers.size(); k++){
      if(inliers[k].first == POINT){
	const int idx = inliers[k].second;
	
	// save point match inside txt file
	matchesTxt << point_pairs[idx].first << " " << point_pairs[idx].second << endl;
      }
      else if(inliers[k].first == PARALLEL_PAIR){
	const int idx = inliers[k].second;
	int li = line_pairs[idx].first;
	int lj = line_pairs[idx].second;
	int mi = matches[li];
	int mj = matches[lj];
	
	// store line match inside inliers set
	inliers_lines.insert(li);
	inliers_lines.insert(lj);

	// save vanishing point match inside txt file
	openMVG::Vec3 vp1 = openMVG::CrossProductMatrix(seg1[li].line)*seg1[lj].line;
	openMVG::Vec3 vp2 = openMVG::CrossProductMatrix(seg2[mi].line)*seg2[mj].line;
	vp1.normalize();
	vp2.normalize();
	vpsTxt << vp1.transpose() << endl;
	vpsTxt << vp2.transpose() << endl;
      }
    }
  }
  {
    ofstream matchesTxt((path + picName1 + "_" + picName2 + "_inliers_matches_line.txt").c_str(), ofstream::out);
    matchesTxt << seg1.size() << endl;
    for(set<int>::iterator it = inliers_lines.begin(); it != inliers_lines.end(); it++){
      matchesTxt << *it << " " << matches[*it] << endl;
    }
  }
}


Pose readPose(const string path, const string picName1, const string picName2){
  ifstream poseTxt((path + picName1 + "_" + picName2 + "_pose.txt").c_str(), ifstream::in);
  Pose pose;
  
  poseTxt >> pose.first(0,0) >> pose.first(0,1) >> pose.first(0,2)
	  >> pose.first(1,0) >> pose.first(1,1) >> pose.first(1,2)
	  >> pose.first(2,0) >> pose.first(2,1) >> pose.first(2,2)
	  >> pose.second[0]  >> pose.second[1]  >> pose.second[2];
  return pose;
  
  std::ifstream Rfile((path + picName1 + "_" + picName2 + "_R.txt").c_str(), std::ifstream::in);
  openMVG::Mat3 R;
  Rfile >> R(0,0) >> R(0,1) >> R(0,2)
	>> R(1,0) >> R(1,1) >> R(1,2)
	>> R(2,0) >> R(2,1) >> R(2,2);
  openMVG::Vec3 t;
  std::ifstream Tfile((path + picName1 + "_" + picName2 + "_t.txt").c_str(), std::ifstream::in);
  Tfile >> t[0] >> t[1] >> t[2];
  pose.first = R;
  pose.second = t.normalized();
  return pose;
}

void savePose(const Pose &pose, const openMVG::Mat3 &Kinv, const string path, const string picName1, const string picName2){
  ofstream poseTxt((path + picName1 + "_" + picName2 + "_pose.txt").c_str(), ofstream::out);
  
  poseTxt << pose.first(0,0) << " " << pose.first(0,1) << " " << pose.first(0,2) << endl
	  << pose.first(1,0) << " " << pose.first(1,1) << " " << pose.first(1,2) << endl
	  << pose.first(2,0) << " " << pose.first(2,1) << " " << pose.first(2,2) << endl
	  << pose.second[0]  << " " << pose.second[1]  << " " << pose.second[2]  << endl;
	  
  ofstream fundamentalTxt((path + picName1 + "_" + picName2 + "_F.txt").c_str(), ofstream::out);
  openMVG::Mat3 F = Kinv.transpose()*openMVG::CrossProductMatrix(pose.second.normalized())*pose.first*Kinv;

  fundamentalTxt << F(0,0) << " " << F(0,1) << " " << F(0,2) << endl
		 << F(1,0) << " " << F(1,1) << " " << F(1,2) << endl
		 << F(2,0) << " " << F(2,1) << " " << F(2,2) << endl;
}

void saveClustersPicture(const vector<ClusterPlane> &planes, const vector<Segment> &lines, const Mat &im, const string path, const string name){
  int thickness = thicknessFromImage(im);
  
  Mat image;
  im.copyTo(image);
  for(int i = 0; i < planes.size(); i++){
    // random colors for each plane cluster
    countRandomRGB = (countRandomRGB+1)%rgb.size();
    
    // find corresponding lines 
    vector<int> l_ids;
    for(int j = 0; j < planes[i].proj_ids.size(); j++){
      l_ids.push_back(planes[i].proj_ids[j]);
    }

    // display lines
    for(int j = 0; j < l_ids.size(); j++){
      // display line
      Point2f p1(lines[l_ids[j]].x1, lines[l_ids[j]].y1);
      Point2f p2(lines[l_ids[j]].x2, lines[l_ids[j]].y2);
      line(image, p1, p2, rgb[countRandomRGB], thickness);
      for(int k = 0; k < j; k++){
	Point2f q1(lines[l_ids[k]].x1, lines[l_ids[k]].y1);
	Point2f q2(lines[l_ids[k]].x2, lines[l_ids[k]].y2);
	line(image, 0.5*(p1+p2), 0.5*(q1+q2), rgb[countRandomRGB], thickness/4);
      }
    }
  }
  imwrite(path + "pictures/" + name + "_plane_clusters.jpg", image);
}

void saveMesh(const Points &points, const Lines &lines, const vector<Plane> &planes, const vector<Pose> &globalPoses, const string &dirPath, const string &name, const bool insideMode){  
  // compute scene bounding box wrt camera positions
  openMVG::Vec3 minP, maxP;
  for(int i = 0; i < globalPoses.size(); i++){    
    if(i == 0){
      minP = globalPoses[i].second;
      maxP = globalPoses[i].second;
    }
    
    for(int j = 0; j < 3; j++){
      minP[j] = min(minP[j], globalPoses[i].second[j]);
      maxP[j] = max(maxP[j], globalPoses[i].second[j]);
    }
  }
  
  // expand a bit
  double expansion = 0;
  for(int j = 0; j < 3; j++){
    expansion = max(0.5*(maxP[j] - minP[j]), expansion);
  }
  for(int j = 0; j < 3; j++){
    minP[j] -= expansion;
    maxP[j] += expansion;
  }
  
  // filter points wrt camera scene dimension
  vector<int> selected_points;
  for(int i = 0; i < points.size(); i++){
    bool inside = true;
    for(int j = 0; j < 3 && inside; j++){
      inside = points[i].p[j] > minP[j] && points[i].p[j] < maxP[j];
    }
    if(!insideMode || inside){
      selected_points.push_back(i);
    }
  }
  const int nPoints = selected_points.size();
  
  // filter lines wrt camera scene dimension
  vector<int> selected_lines;
  for(int i = 0; i < lines.size(); i++){
    bool inside = true;
    for(int j = 0; j < 3 && inside; j++){
      inside = min(lines[i].p1[j], lines[i].p2[j]) > minP[j] && max(lines[i].p1[j], lines[i].p2[j]) < maxP[j];
    }
    
    if(!insideMode || inside){
      const Line3D* l3D = &(lines[i]);
      //if(*(l3D->planes.begin())%rgb.size() != 1){continue;}
      selected_lines.push_back(i);
    }
  }
  const int nLines = selected_lines.size();
  
  double epsilon = 0;
  for(int j = 0; j < 3; j++){
    epsilon = max(epsilon, maxP[j] - minP[j]);
  }
  double thresh_plane_width = 0.05*epsilon;
  epsilon /= 1000000;
  
  const int nCameras = globalPoses.size();
  
  ofstream lineObj(dirPath + name + ".ply", std::ios::out | std::ios::trunc);
  lineObj << "ply"                                                     << endl
          << "format ascii 1.0"                                        << endl
          << "element vertex " << 8*nLines + nPoints + nCameras        << endl
          << "property float x"                                        << endl
          << "property float y"                                        << endl
          << "property float z"                                        << endl
          << "property uchar red"                                      << endl
          << "property uchar green"                                    << endl
          << "property uchar blue"                                     << endl
          << "element face " << 6*nLines                               << endl
          << "property list uint8 int32 vertex_indices"                << endl
          << "end_header"                                              << endl;
	  
  // vertex list
  for(int i = 0; i < nLines; i++){
    const Line3D* l3D = &(lines[selected_lines[i]]);
    openMVG::Vec3 pi  = l3D->p1;
    openMVG::Vec3 qi  = l3D->p2;
    vector<Scalar> colors(1, Scalar(255, 255, 255));
    if(l3D->planes.size() > 0){
      colors.clear();
      for(set<int>::const_iterator it = l3D->planes.begin(); it != l3D->planes.end(); it++){
	colors.push_back(rgb[*it%rgb.size()]);
      }
    }
    openMVG::Vec3 dir = (qi - pi)/(qi - pi).norm();
    
    openMVG::Vec3 oX = (openMVG::CrossProductMatrix(dir+openMVG::Vec3(1,1,1))*dir).normalized();
    openMVG::Vec3 oY = (openMVG::CrossProductMatrix(oX)*dir).normalized();
    
    for(int l = 0; l < 2; l++){
      openMVG::Vec3 e = (l==0)? pi : qi;
      for(int k = 0; k < 4; k++){
	float sgn1 = (k == 0 || k == 3)? 1.f : -1.f;
	float sgn2 = (k == 0 || k == 1)? 1.f : -1.f;
	
	for(int j = 0; j < 3; j++){
	  lineObj << (e + sgn1*epsilon*oX + sgn2*epsilon*oY)[j] << " ";
	}
	int idx_color = (4*l+k)%colors.size();
	lineObj << colors[idx_color][0] << " " << colors[idx_color][1] << " " << colors[idx_color][2] << endl;
      }
    }    
  }
  
  for(int i = 0; i < nPoints; i++){
    for(int j = 0; j < 3; j++){
      lineObj << points[selected_points[i]].p[j] << " ";
    }
    lineObj << "255 50 50" << endl;
  }
  
  for(int i = 0; i < nCameras; i++){
    for(int j = 0; j < 3; j++){
      lineObj << globalPoses[i].second[j] << " ";
    }
    lineObj << "50 255 50" << endl;
  }
  
  // face list
  for(int j = 0; j < nLines; j++){
    int i = 8*j;
    lineObj << "4 " << i   << " " << i+3 << " " << i+2 << " " << i+1 << endl;
    lineObj << "4 " << i+4 << " " << i+5 << " " << i+6 << " " << i+7 << endl;
    lineObj << "4 " << i   << " " << i+1 << " " << i+5 << " " << i+4 << endl;
    lineObj << "4 " << i+3 << " " << i   << " " << i+4 << " " << i+7 << endl;
    lineObj << "4 " << i+2 << " " << i+3 << " " << i+7 << " " << i+6 << endl;
    lineObj << "4 " << i+1 << " " << i+2 << " " << i+6 << " " << i+5 << endl;
  }
  
  // planes mesh
  const int nPlanes = planes.size();
  if(nPlanes == 0){return;}
  
  ofstream planeObj(dirPath + name + "_planes.ply", std::ios::out | std::ios::trunc);
  planeObj << "ply"                                                     << endl
           << "format ascii 1.0"                                        << endl
           << "element vertex " << 8*nPlanes                            << endl
           << "property float x"                                        << endl
           << "property float y"                                        << endl
           << "property float z"                                        << endl
           << "property uchar red"                                      << endl
           << "property uchar green"                                    << endl
           << "property uchar blue"                                     << endl
           << "element face " << 6*nPlanes                              << endl
           << "property list uint8 int32 vertex_indices"                << endl
           << "end_header"                                              << endl;
    
  for(int i = 0; i < planes.size(); i++){
    Scalar color = rgb[i%rgb.size()];
    
    vector<openMVG::Vec3> local_basis = planes[i].basis;
    
    for(int k = 0; k < 8; k++){
      openMVG::Vec3 sign(1-2*(k%2), 1-2*((k/2)%2), 1-2*((k/4)%2));
      openMVG::Vec3 pt = planes[i].centroid;
      for(int j = 0; j < 3; j++){
	if(sign[j] > 0){
	  pt += planes[i].rangePlus[j]*local_basis[j];
	}
	else{
	  pt -= planes[i].rangeMinus[j]*local_basis[j];
	}
      }
      for(int j = 0; j < 3; j++){
	planeObj << pt[j] << " ";
      }
      planeObj << color[0] << " " << color[1] << " " << color[2] << endl;
    }
  }
  
  // face list
  for(int j = 0; j < nPlanes; j++){
    int i = 8*j;
    planeObj << "4 " << i   << " " << i+1 << " " << i+3 << " " << i+2 << endl;
    planeObj << "4 " << i   << " " << i+4 << " " << i+5 << " " << i+1 << endl;
    planeObj << "4 " << i   << " " << i+2 << " " << i+6 << " " << i+4 << endl;
    planeObj << "4 " << i+7 << " " << i+5 << " " << i+4 << " " << i+6 << endl;
    planeObj << "4 " << i+7 << " " << i+6 << " " << i+2 << " " << i+3 << endl;
    planeObj << "4 " << i+7 << " " << i+3 << " " << i+1 << " " << i+5 << endl;
  }
}
