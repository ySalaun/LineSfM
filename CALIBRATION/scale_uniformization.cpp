/*----------------------------------------------------------------------------  
  This code is part of the following arXiv publication :
  "Robust SfM with Little Image Overlap",
  Yohann Salaun, Renaud Marlet, and Pascal Monasse
  
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

#include "scale_uniformization.hpp"
#include "refinement.hpp"
#include <eigen/Eigen/src/Core/products/GeneralBlockPanelKernel.h>
#include <boost/concept_check.hpp>
#include "DETECTION/lsd.hpp"

using namespace openMVG;

/*=================== MISCELLANEOUS ===================*/

/// logarithm (base 10) of binomial coefficient
static double logcombi(size_t k, size_t n){
  if (k>=n || k<=0) return(0.0);
  if (n-k<k) k=n-k;
  double r = 0.0;
  for (size_t i = 1; i <= k; i++)
    r += log10((double)(n-i+1))-log10((double)i);

  return r;
}

/// tabulate logcombi(.,n)
template<typename Type>
static void makelogcombi_n(size_t n, std::vector<Type> & l){
  l.resize(n+1);
  for (size_t k = 0; k <= n; k++)
    l[k] = static_cast<Type>( logcombi(k,n) );
}

static double combi(size_t k, size_t n){
  if (k>=n || k<=0) return(1.0);
  if (n-k<k) k=n-k;
  double r = 1.0;
  for (size_t i = 1; i <= k; i++)
    r *= (double)(n-i+1)/(double)i;

  return r;
}

/// tabulate combi(.,n)
template<typename Type>
static void makecombi_n(size_t n, std::vector<Type> & l){
  l.resize(n+1);
  for (size_t k = 0; k <= n; k++)
    l[k] = static_cast<Type>( combi(k,n) );
}

inline
void addToMap(map<int, vector<int>> &lineToCopPair, const int key, const int val){
  map<int, vector<int>>::iterator it = lineToCopPair.find(key);
  if(it == lineToCopPair.end()){
    lineToCopPair.insert(pair<int, vector<int>>(key, vector<int>(1, val)));
  }
  else{
    it->second.push_back(val);
  }
}

void mapLinesToCopPairs(const vector<ClusterPlane> &clusters, map<int, vector<int>> &lineToCopPair, const int nLines){  
  for(int i = 0; i < clusters.size(); i++){
    int li = clusters[i].proj_ids[0];
    int lj = clusters[i].proj_ids[1];
   
    addToMap(lineToCopPair, li, i);
    addToMap(lineToCopPair, lj+nLines, i);
  }
}

/*=================== GEOMETRIC SUB-FUNCTIONS ===================*/

inline
Vec4 toHomog(const Vec3 &p){
  return Vec4(p[0], p[1], p[2], 1);
}

// minimize 
inline
double minimizeAngle(const Mat3 &crossProd, const Vec3 &a, const Vec3 &b){
  float aTb = a.dot(b);
  float aNorm = a.norm();
  float bNorm = b.norm();

  float auTbu = (crossProd*a).dot(crossProd*b);
  float auNorm = (crossProd*a).norm();
  float buNorm = (crossProd*b).norm();

  float alpha = aTb*buNorm*buNorm - auTbu*bNorm*bNorm;
  float gamma = auTbu*aNorm*aNorm - aTb*auNorm*auNorm;
  float beta = buNorm*buNorm*aNorm*aNorm - auNorm*auNorm*bNorm*bNorm;
  float delta = beta*beta-4*alpha*gamma;
  
  return (2*alpha)/(-beta+sqrt(delta));
}

inline
Vec3 closestPoint2Lines(const Vec3 &C0, const Vec3 &C1, const Vec3 &P0, const Vec3 &P1, const float l){
  // compute 3D point
  Vec3 d = (CrossProductMatrix(P0)*P1).normalized();

  Vec3 u = CrossProductMatrix(d)*P1;
  double lambda = (C1 - C0).dot(u)/P0.dot(u);
  return lambda*P0 + C0 + l*(d.dot(C1 - C0))*d;
}

inline
Vec3 closestPoint2Lines(const Mat3 &R0, const Mat3 &R1, const Vec3 &C0, const Vec3 &C1, const Vec3 &p0, const Vec3 &p1, const float l){
  // compute 3D point
  Vec3 P0 = R0.transpose()*p0;
  Vec3 P1 = R1.transpose()*p1;
  return closestPoint2Lines(C0, C1, P0, P1, l);
}

inline
Vec3 homogenousProjection(const Mat3 &K, const Mat3 &R, const Vec3 &C, const Vec3 &p){
  Vec3 proj = K*R*(p-C);
  return proj/proj[2];
}

/*=================== CONSTRAINTS COMPUTATION ===================*/
Lines computeCoplanarParameters(const PictureSegments &l1, const PictureSegments &l2, const vector<int> &matches, 
				const Mat3 &R1, const Mat3 &R2, const Vec3 &t12,
				const int i1, const int i2, const double imDimension, const bool invert, const bool case12){	
  Lines lines;
  // parameters
  const double length_thresh = 0.01f*imDimension;
  
  // compute 3D lines up to scale
  for(int li = 0; li < l1.size(); li++){
    int mi = matches[li];
    
    // if no match or line too short, it wont be used for coplanarity constraints
    if(mi == -1 || (imDimension > 0 && (l1[li].length < length_thresh || l2[mi].length < length_thresh))){continue;}

    Vec3 dir = CrossProductMatrix(R1.transpose()*l1[li].line)*R2.transpose()*l2[mi].line;

    // TODO refine this... when the 2D lines are too close, the cross product is instable
    if(dir.norm() < degenerate_case_thresh){continue;}
    
    // degenerate cases for coplanar cts
    if(case12){
      Vec3 t21 = (R1*R2.transpose()*t12).normalized();
      Vec3 L1 = l1[li].line.normalized();
      if(fabs(t21.dot(L1)) < coplanar_orthogonal_thresh){continue;}
      Vec3 R21p2 = (R1*R2.transpose()*l2[mi].p1).normalized();
      if(fabs(R21p2.dot(L1)) < coplanar_orthogonal_thresh){continue;}
    }
    else{
      Vec3 t23 = t12.normalized();
      Vec3 L3 = l2[mi].line.normalized();
      if(fabs(t23.dot(L3)) < coplanar_orthogonal_thresh){continue;}
      Vec3 R23p2 = (R2*R1.transpose()*l1[li].p1).normalized();
      if(fabs(R23p2.dot(L3)) < coplanar_orthogonal_thresh){continue;}
    }
   
    lines.push_back(Line3D(l1[li], l2[mi], dir, R1, R2, t12, li, mi, i1, i2, invert));
  }
  
  return lines;
}

vector<ClusterPlane> computeCoplanarCts(const PicturesSegments &lines, const PicturesMatches &matches_lines, 
					const vector<Pose> &globalPoses, const Mat3 &R01, const Vec3 &t01, const Vec3 &t12,
					const int i0, const int i1, const int i2, const double imDimension,
					Lines &l01, Lines &l12, vector<int> &cluster_left, vector<int> &cluster_right, 
					const int n_neighbours, const Mat3 &K){
  // parameters
  const float parallel_thresh = cos(15*M_PI/180);
  
  // output
  vector<ClusterPlane> candidate_planes;
  
  // parameters for pictures i0 and i1
  l01 = computeCoplanarParameters(lines[i0], lines[i1], matches_lines.find(PicturePair(i0,i1))->second, 
				  globalPoses[i0].first, globalPoses[i1].first, t01, i0, i1, imDimension, false, true);

  // parameters for pictures i1 and i2
  l12 = computeCoplanarParameters(lines[i1], lines[i2], matches_lines.find(PicturePair(i1,i2))->second, 
				  globalPoses[i1].first, globalPoses[i2].first, t12, i1, i2, imDimension, false, false);
  
  double badPair = (imDimension < 0)? 1000000 : imDimension;
  vector<pair<double, int>> distanceIdx(l12.size());
  for(int i = 0; i < l01.size(); i++){
    for(int j = 0; j < l12.size(); j++){
      distanceIdx[j] = pair<double, int>(badPair, j);

      // parallel pairs is a degenerate case
      if(fabs(l01[i].direction.dot(l12[j].direction)) > parallel_thresh){continue;}    
      
      Vec3 normal = (CrossProductMatrix(l01[i].direction)*l12[j].direction).normalized();
      if(fabs(normal.dot(globalPoses[i1].first.transpose()*l12[j].mid.normalized())) < coplanar_orthogonal_thresh 
	|| fabs(normal.dot(globalPoses[i0].first.transpose()*l01[i].mid.normalized())) < coplanar_orthogonal_thresh){
	continue;
      }
      /*if(fabs(normal[2]) > parallel_thresh){continue;}*/

      distanceIdx[j].first = l01[i].distTo(l12[j], lines);
    }
    sort(distanceIdx.begin(), distanceIdx.end());
    int addedCts = 0;
    for(int k = 0; k < distanceIdx.size() && addedCts < n_neighbours; k++){
      if(distanceIdx[k].first >= badPair){break;} 
      int j = distanceIdx[k].second;
      
      ClusterPlane c;
      c.lines.push_back(l01[i]);
      c.lines.push_back(l12[j]);
      c.proj_ids.push_back(l01[i].proj_ids[1]);
      c.proj_ids.push_back(l12[j].proj_ids[0]);

      // if the same central line, do not keep the pair
      if(c.proj_ids[0] == c.proj_ids[1]){continue;}     
      
      c.normal = (CrossProductMatrix(l01[i].direction)*l12[j].direction).normalized();
      c.t_norm = l12[j].lambda*(c.normal.dot(globalPoses[i1].first.transpose()*l12[j].mid))/(l01[i].lambda*c.normal.dot(globalPoses[i0].first.transpose()*l01[i].mid) 
		  - c.normal.dot(globalPoses[i1].first.transpose()*t01));
      c.proj_intersec = K*(CrossProductMatrix(lines[i1][c.proj_ids[0]].line)*lines[i1][c.proj_ids[1]].line);
      c.proj_intersec /= c.proj_intersec[2];
      
      candidate_planes.push_back(c);
      addedCts++;    
    }
  }

  // repartition among directions
  vector<Vec3> dir_left, dir_right;
  vector<bool> proj_left, proj_right;
  proj_left = proj_right = vector<bool>(lines[i1].size(), false);
  for(int i = 0; i < candidate_planes.size(); i++){
    ClusterPlane c = candidate_planes[i];
    
    if(!proj_left[c.proj_ids[0]]){
      bool found = false;
      for(int k = 0; k < dir_left.size() && !found; k++){
	if(fabs(c.lines[0].direction.dot(dir_left[k])) > parallel_thresh){
	  cluster_left[k]++;
	  found = true;
	}
      }
      if(!found){
	dir_left.push_back(c.lines[0].direction);
	cluster_left.push_back(1);
      }
      proj_left[c.proj_ids[0]] = true;
    }
    
    for(int p = 0; p < 2; p++){
      if(!proj_right[c.proj_ids[p]]){
	bool found = false;
	for(int k = 0; k < dir_right.size() && !found; k++){
	  if(fabs(c.lines[p].direction.dot(dir_right[k])) > parallel_thresh){
	    cluster_right[k]++;
	    found = true;
	  }
	}
	if(!found){
	  dir_right.push_back(c.lines[p].direction);
	  cluster_right.push_back(1);
	}
	proj_right[c.proj_ids[p]] = true;
      }
    }
  }

  return candidate_planes;
}

Triplets computeTripletLineCts(const PicturesSegments &lines, const PicturesMatches &matches_lines,
			       const vector<Pose> &globalPoses, const Mat3 &R01, const Mat3 &R12, 
			       const Vec3 &t01, const Vec3 &t12, const Vec3 &t10,
			       const int i0, const int i1, const int i2){
  // TODO optimize use of R10, R12 ...
  Triplets triplets;
  vector<int> match01 = matches_lines.find(PicturePair(i0,i1))->second;
  vector<int> match12 = matches_lines.find(PicturePair(i1,i2))->second;
  for(int l0 = 0; l0 < lines[i0].size(); l0++){
    int l1 = match01[l0];
    if(l1 == -1){continue;}
    int l2 = match12[l1];
    if(l2 == -1){continue;}

    double norm_t12;
    {
      // compute 3D extremities
      Vec3 p1 = lines[i0][l0].p1;
      Vec3 p2 = lines[i0][l0].p2;
      {
	Vec3 mi = lines[i1][l1].line.normalized();
      
	float zp1 = -t01.dot(mi)/(p1.dot(R01.transpose()*mi));
	float zp2 = -t01.dot(mi)/(p2.dot(R01.transpose()*mi));
    
	p1 = globalPoses[i0].first.transpose()*zp1*p1 + globalPoses[i0].second;
	p2 = globalPoses[i0].first.transpose()*zp2*p2 + globalPoses[i0].second;
      }

      // compute translation norm
      norm_t12 = minimizeAngle(CrossProductMatrix(lines[i2][l2].line)*CrossProductMatrix(globalPoses[i2].first*(p1-p2)), 
				globalPoses[i2].first*(p1 - globalPoses[i1].second), t12);
    }
    
    double norm_t01;
    {
      // compute 3D extremities
      Vec3 p1 = lines[i1][l1].p1;
      Vec3 p2 = lines[i1][l1].p2;
      {
	Vec3 mi = lines[i2][l2].line.normalized();
      
	float zp1 = -t12.dot(mi)/(p1.dot(R12.transpose()*mi));
	float zp2 = -t12.dot(mi)/(p2.dot(R12.transpose()*mi));
    
	p1 = globalPoses[i1].first.transpose()*zp1*p1 + globalPoses[i1].second;
	p2 = globalPoses[i1].first.transpose()*zp2*p2 + globalPoses[i1].second;
      }

      // compute translation norm
      norm_t01 = minimizeAngle(CrossProductMatrix(lines[i0][l0].line)*CrossProductMatrix(globalPoses[i0].first*(p1-p2)), 
				globalPoses[i0].first*(p1-globalPoses[i1].second), t10);
    }

    if(fabs(1/norm_t01 - norm_t12) > 0.1*(min(fabs(norm_t01),fabs(norm_t12)))){continue;}

    Triplet triplet;
    triplet.t_norm = 0.5*(1/norm_t01 + norm_t12);
   
    // triplet indexes
    triplet.idx[0] = l0;
    triplet.idx[1] = l1;
    triplet.idx[2] = l2;
    
    // precomputation for inliers computation
    // for norm_t12
    {
      Vec3 p1 = lines[i0][l0].p1;
      Vec3 p2 = lines[i0][l0].p2;
      
      Vec3 m1 = lines[i1][l1].line.normalized();
    
      float zp1 = -t01.dot(m1)/(p1.dot(R01.transpose()*m1));
      float zp2 = -t01.dot(m1)/(p2.dot(R01.transpose()*m1));

      triplet.precomputation.push_back(globalPoses[i0].first.transpose()*zp1*p1 + globalPoses[i0].second);
      triplet.precomputation.push_back(globalPoses[i0].first.transpose()*zp2*p2 + globalPoses[i0].second);
    }
    // for norm_t01
    {
      Vec3 p1 = lines[i1][l1].p1;
      Vec3 p2 = lines[i1][l1].p2;
      
      Vec3 m2 = lines[i2][l2].line.normalized();
    
      float zp1 = -t12.dot(m2)/(p1.dot(R12.transpose()*m2));
      float zp2 = -t12.dot(m2)/(p2.dot(R12.transpose()*m2));

      triplet.precomputation.push_back(globalPoses[i1].first.transpose()*zp1*p1);
      triplet.precomputation.push_back(globalPoses[i1].first.transpose()*zp2*p2);
    }
    
    triplet.type = LINE;
    triplet.iPicture = i1;
    
    triplets.push_back(triplet);
  } 
  return triplets;
}

Triplets computeTripletPointCts(const PicturesPoints &points, const PicturesMatches &matches_points,
				const vector<Pose> &globalPoses, const vector<Mat3> &K, const Vec3 &t12, const Vec3 &t10,
				const int i0, const int i1, const int i2){
  Triplets triplets;
  vector<int> match01 = matches_points.find(PicturePair(i0,i1))->second;
  vector<int> match12 = matches_points.find(PicturePair(i1,i2))->second;

  Vec3 C2 = globalPoses[i1].second - globalPoses[i2].first.transpose()*t12;
  
  for(int p0 = 0; p0 < points[i0].size(); p0++){
    int p1 = match01[p0];
    if(p1 == -1){continue;}
    int p2 = match12[p1];
    if(p2 == -1){continue;}

    double norm_t12;
    {
      // compute 3D point
      Vec3 P = closestPoint2Lines(globalPoses[i0].first, globalPoses[i1].first,
				  globalPoses[i0].second, globalPoses[i1].second,
				  points[i0][p0], points[i1][p1], 0.5);

      norm_t12 = minimizeAngle(CrossProductMatrix(points[i2][p2]), 
				globalPoses[i2].first*(P - globalPoses[i1].second), t12);
    }
      
    double norm_t01;
    {
      // compute 3D point
      Vec3 P = closestPoint2Lines(globalPoses[i1].first, globalPoses[i2].first,
				  globalPoses[i1].second, C2,
				  points[i1][p1], points[i2][p2], 0.5);

      // compute translation norm
      norm_t01 = minimizeAngle(CrossProductMatrix(points[i0][p0]), 
				globalPoses[i0].first*(P - globalPoses[i1].second), t10);
    }
    
    if(fabs(1/norm_t01 - norm_t12) > 0.1){continue;}
    
    Triplet triplet;
    triplet.t_norm = 0.5*(1/norm_t01 + norm_t12);
    
    // triplet indexes
    triplet.idx[0] = p0;
    triplet.idx[1] = p1;
    triplet.idx[2] = p2;
    
    // precomputation for inliers computation   
    // for norm_t12
    {	
      Vec3 P0 = closestPoint2Lines(globalPoses[i0].first, globalPoses[i1].first,
				  globalPoses[i0].second, globalPoses[i1].second,
				  points[i0][p0], points[i1][p1], 0.5);
      triplet.precomputation.push_back(P0);
      
      Vec3 p_2 = K[i2]*points[i2][p2];
      p_2 /= p_2[2];
      
      triplet.precomputation.push_back(p_2);
    }
    // for norm_t01
    {	
      Vec3 P0 = closestPoint2Lines(globalPoses[i1].first, globalPoses[i2].first,
				   globalPoses[i1].second, C2,
				   points[i1][p1], points[i2][p2], 0.5);
      triplet.precomputation.push_back(P0);
      
      Vec3 p_0 = K[i0]*points[i0][p0];
      p_0 /= p_0[2];
      
      triplet.precomputation.push_back(p_0);
    }
    
    triplet.type = POINT;
    triplet.iPicture = i1;
    
    triplets.push_back(triplet);
  }
  return triplets;
}

/*=================== NFA COMPUTATION ===================*/

const double DEFAULT = 100000;
double computeNFA(vector<ErrorFTypeIndex> &log_residuals, const vector<ErrorFTypeIndex> &log_residuals_inliers, 
 		  const vector<double> &logc_n, vector<FTypeIndex> &inliers, double &thresh_nfa, double thresh = DEFAULT){
  if(log_residuals.size() == 0){return 0;}
  
  double bestNFA = std::numeric_limits<double>::infinity();
  
  sort(log_residuals.begin(), log_residuals.end());
  int kInliers = -1;  
  
  bool coplanar = log_residuals.size() != log_residuals_inliers.size();
  
  for(int k = 2; k <= log_residuals.size(); k++){
    if(log_residuals[k-1].first >= thresh){break;}
    double nfa = (coplanar)? logc_n[k] + log_residuals[k-1].first*(double)(k-2)
			   : logc_n[k] + log10(k) + log_residuals[k-1].first*(double)(k-1);
    if(nfa < bestNFA){
      bestNFA = nfa;
      kInliers = k;
    }
  }

  thresh_nfa = log_residuals[kInliers-1].first;
#ifdef PLANE_RECO
  if(thresh != DEFAULT){
    thresh_nfa += 1;
  }
#endif
  for(int k = 0; k < log_residuals_inliers.size(); k++){
    if(log_residuals_inliers[k].first < thresh_nfa){
      inliers.push_back(log_residuals_inliers[k].second);
    }
  }

  return bestNFA;
}

vector<ErrorFTypeIndex> TranslationNormAContrario::logNFAcoplanar(const double t_norm, const PicturesSegments &lines, const vector<Pose> &globalPoses, 
								  vector<ErrorFTypeIndex> &log_residuals_coppairs, const vector<bool> &cp){  
  vector<ErrorFTypeIndex> log_residuals(nCopLines);
  
  // TODO this part is too slow !!!
  for(int j = 0; j < candidate_planes.size(); j++){
    if(!cp[j]){
      log_residuals_coppairs[j] = ErrorFTypeIndex(DEFAULT, FTypeIndex(COPLANAR_PAIR, j));
      continue;
    }
    
    // 3D points from coplanar lines
    Vec3 P1 = candidate_planes[j].lines[0].p1 + globalPoses[i0].second;
    Vec3 P2 = 1.0/t_norm * candidate_planes[j].lines[1].p1 + globalPoses[i1].second;

    // compute angular projected distance
    Vec3 P_0 = closestPoint2Lines(P1, P2, candidate_planes[j].lines[0].direction, candidate_planes[j].lines[1].direction, 0);
    Vec3 P_1 = closestPoint2Lines(P1, P2, candidate_planes[j].lines[0].direction, candidate_planes[j].lines[1].direction, 1);
    
    // point projections (homogenous coordinates)
    Vec3 p1_0 = homogenousProjection(K[i1], globalPoses[i1].first, globalPoses[i1].second, P_0);
    Vec3 p1_1 = homogenousProjection(K[i1], globalPoses[i1].first, globalPoses[i1].second, P_1);
    
    // distance
    double d = ((p1_0 - candidate_planes[j].proj_intersec).norm() + (p1_1 - candidate_planes[j].proj_intersec).norm())*0.5;
    d = 2*log10( d + epsilon)+logalpha0_point_point;
    
    log_residuals_coppairs[j] = ErrorFTypeIndex(d, FTypeIndex(COPLANAR_PAIR, j));
  }

  int idx = 0;
  for(map<int, vector<int>>::iterator it = lineToCopPair.begin(); it != lineToCopPair.end(); it++, idx++){
    log_residuals[idx] = ErrorFTypeIndex(numeric_limits<double>::infinity(), FTypeIndex(COPLANAR_PAIR, it->first));
    for(int i = 0; i < it->second.size(); i++){
      log_residuals[idx].first = min(log_residuals[idx].first, log_residuals_coppairs[it->second[i]].first);
    }
  }
  return log_residuals;
}

vector<ErrorFTypeIndex> TranslationNormAContrario::logNFAlines(const Vec3 &C2, const double t_norm, const vector<Pose> &globalPoses, 
							       const PicturesSegments &lines, const vector<bool> &lt){  
  vector<ErrorFTypeIndex> log_residuals(nLines);
  
  for(int j = 0; j < line_triplets.size(); j++){
    if(!lt[j]){
      log_residuals[j] = ErrorFTypeIndex(DEFAULT, FTypeIndex(LINE, j));
      continue;
    }
    int l0 = line_triplets[j].idx[0];
    int l2 = line_triplets[j].idx[2];

    // pixelic distance
    double d1;
    {
      Vec3 p1 = homogenousProjection(K[i2], globalPoses[i2].first, C2, line_triplets[j].precomputation[0]);
      Vec3 p2 = homogenousProjection(K[i2], globalPoses[i2].first, C2, line_triplets[j].precomputation[1]);
    
      d1 = 0.5*(fabs(p1.dot(lines[i2][l2].homogenous_line))+fabs(p2.dot(lines[i2][l2].homogenous_line)));
    }
    double d2;
    {
      Vec3 p1 = homogenousProjection(K[i0], globalPoses[i0].first, globalPoses[i0].second, line_triplets[j].precomputation[2]/t_norm + globalPoses[i1].second);
      Vec3 p2 = homogenousProjection(K[i0], globalPoses[i0].first, globalPoses[i0].second, line_triplets[j].precomputation[3]/t_norm + globalPoses[i1].second);
      d2 = 0.5*(fabs(p1.dot(lines[i0][l0].homogenous_line))+fabs(p2.dot(lines[i0][l0].homogenous_line)));
    }
    double d = 0.5*(d1+d2);
    d = log10(d + epsilon) + logalpha0_point_line;
    
    log_residuals[j] = ErrorFTypeIndex(d, FTypeIndex(LINE, j));
  }
  
  return log_residuals;
}

vector<ErrorFTypeIndex> TranslationNormAContrario::logNFApoints(const Vec3 &C2, const Vec3 &C0, const vector<Pose> &globalPoses, 
								const PicturesPoints &points, const vector<bool> &pt){  
  vector<ErrorFTypeIndex> log_residuals(nPoints);
  
  for(int j = 0; j < point_triplets.size(); j++){    
    if(!pt[j]){
      log_residuals[j] = ErrorFTypeIndex(DEFAULT, FTypeIndex(POINT, j));
      continue;
    }
    
    // triangulate 3D points
    Vec3 proj0, proj2;
    {
      Vec3 P2 = point_triplets[j].precomputation[0];
      proj2 = homogenousProjection(K[i2], globalPoses[i2].first, C2, P2);
    
      Vec3 P0 = point_triplets[j].precomputation[2];
      proj0 = homogenousProjection(K[i0], globalPoses[i0].first, C0, P0);
    }

    // homogenous projections
    Vec3 p_0 = point_triplets[j].precomputation[3];
    Vec3 p_2 = point_triplets[j].precomputation[1];
      
    // pixelic distance
    double d = 0.5*((proj0-p_0).norm() + (proj2-p_2).norm());

    d = 2*log10(d + epsilon) + logalpha0_point_point;
    
    log_residuals[j] = ErrorFTypeIndex(d, FTypeIndex(POINT, j));
  }
  
  return log_residuals;
}

/*=================== MERGE PLANES ===================*/

struct Node{
 vector<int> neighbours;
 vector<Vec3> normals;
 vector<bool> used;
 bool active;
};
bool sort_operator(ClusterPlane ci,ClusterPlane cj) { return (ci.t_norm < cj.t_norm);}

// cluster coplanar inliers into planes
vector<ClusterPlane> mergePlanes(const vector<ClusterPlane> &planes, const vector<int> &inliers,
			    const PicturesSegments &lines, const PicturesMatches &matches, const int i1){
  // TODO maybe not use Plane structure for coplanar pairs....
  // parameters
  vector<int> clusters(inliers.size(), -1);
  int nClusters = -1;
  const double thresh_similar_plane = cos(15*M_PI/180);
  for(int i = 0; i < inliers.size(); i++){
    const ClusterPlane* ci = &planes[inliers[i]];
    int li = ci->proj_ids[0];
    int mi = ci->proj_ids[1];
    for(int j = 0; j < i; j++){
      const ClusterPlane* cj = &planes[inliers[j]];
      int lj = cj->proj_ids[0];
      int mj = cj->proj_ids[1];

      if(!(li == lj || li == mj || mi == lj || mi == mj)){
	continue;
      }
      double dist = fabs(ci->normal.dot(cj->normal));
      if(dist > thresh_similar_plane){
	int c1 = clusters[i];
	int c2 = clusters[j];
	if(c1 > c2){swap(c1, c2);}
	if(c2 == -1){
	  nClusters++;
	  clusters[i] = clusters[j] = nClusters;
	}
	else if(c1 == -1){
	  clusters[i] = clusters[j] = c2;
	}
	else{
	  if(c1 == c2){continue;}
	  for(int i = 0; i < clusters.size(); i++){
	    if(clusters[i] == c2){
	      clusters[i] = c1;
	    }
	    else if(clusters[i] > c2){
	      clusters[i]--;
	    }
	  }
	  nClusters--;
	}
      }
    }
  }
  
  vector<ClusterPlane> mergedPlanes;
  for(int i = 0; i <= nClusters; i++){
    vector<int> cluster;
    for(int j = 0; j < clusters.size(); j++){
      if(clusters[j] == i){
	cluster.push_back(j);
      }
    }
    
    const int n = cluster.size();
    //if(n <= 1){continue;}
    
    ClusterPlane c;
    set<int> indexes;
    Vec3 normal(0,0,0);
    for(int j = 0; j < n; j++){
      const ClusterPlane* cp = &planes[inliers[cluster[j]]];
      indexes.insert(cp->proj_ids[0]);
      indexes.insert(cp->proj_ids[1]);
      if(normal.dot(cp->normal) < 0){
	normal -= cp->normal;
      }
      else{
	normal += cp->normal;
      }
    }
    for(set<int>::iterator it = indexes.begin(); it != indexes.end(); it++){
      c.proj_ids.push_back(*it);
    }
    c.normal = normal.normalized();
    mergedPlanes.push_back(c);
  }
  
  return mergedPlanes;
}

/*=================== MAIN FUNCTION ===================*/

TranslationNormAContrario::TranslationNormAContrario(const PicturesSegments &lines, const PicturesMatches &matches_lines, 
						     const PicturesPoints &points, const PicturesMatches &matches_points,
						     const vector<Pose> &globalPoses, const PicturesRelativePoses &relativePoses, 
						     const int iPicture, const double imDimension, const vector<Mat3> &gt_K, const bool closure){
  K = gt_K;
  Ktinv.resize(gt_K.size());
  for(int i = 0; i < gt_K.size(); i++){
    Ktinv[i] = gt_K[i].inverse().transpose(); 
  }
  
  // index for the triplet computations
  i0 = iPicture-1; i1 = iPicture; i2 = (closure)? 0:(iPicture+1);
  
  // relative rotations
  R01 = globalPoses[i1].first*globalPoses[i0].first.transpose();
  R12 = globalPoses[i2].first*globalPoses[i1].first.transpose();
  
  // relative translations
  t01 = -globalPoses[i1].first*(globalPoses[i1].second - globalPoses[i0].second);
  t12 = relativePoses.find(PicturePair(i1, i2))->second.second;
  t10 = -R01.transpose()*t01;

  // compute coplanar pairs and final associated translation norm
  vector<int> cluster_left, cluster_right;
  if(coplanar_cts){
    candidate_planes = computeCoplanarCts(lines, matches_lines, globalPoses, R01, t01, t12, i0, i1, i2, imDimension, 
					  l01, l12, cluster_left, cluster_right, n_neighbours, K[i1]);
    if(candidate_planes.size() < 2){
      candidate_planes.clear();
    }
    else{
      mapLinesToCopPairs(candidate_planes, lineToCopPair, lines[i1].size());
    }
  }
  nCopPairs = candidate_planes.size();
  nCopLines = lineToCopPair.size();

  // line triplet constraints
  if(triplets_lines_cts){
    line_triplets = computeTripletLineCts(lines, matches_lines, globalPoses, R01, R12, t01, t12, t10, i0, i1, i2);
    if(line_triplets.size() < 2){
      line_triplets.clear();
    }
  }
  nLines = line_triplets.size();

  // point triplet constraints
  if(triplets_points_cts){
    point_triplets = computeTripletPointCts(points, matches_points, globalPoses, K, t12, t10, i0, i1, i2);
    if(point_triplets.size() < 2){
      point_triplets.clear();
    }
  }
  nPoints = point_triplets.size();

  // nb of constraints & features
  nFeatures = lineToCopPair.size() + line_triplets.size() + point_triplets.size();
  nConstraints = candidate_planes.size() + line_triplets.size() + point_triplets.size();

  // repartition of different directions in cop cts
  const int nLines = lineToCopPair.size();
  if(coplanar_cts){
    int n12 = 0;
    for(int i = 0; i < cluster_left.size(); i++){
      n12 += cluster_left[i];
    }
    logc_n_cop = vector<double>(nLines+1, 0);
    vector<double> logkn;
    makelogcombi_n(nLines-2, logkn);
    for(int i = 2; i < nLines+1; i++){
      logc_n_cop[i] = log10(nLines/2) + log10(n_neighbours) + logkn[i-2];
    }
  }

  // precomputation for log nfa
  makelogcombi_n(point_triplets.size(), logc_n_points);
  makelogcombi_n(line_triplets.size(), logc_n_lines);
  
  const int w = K[0](0,2)*2, h = K[0](1,2)*2;
  const double D = sqrt(w*w + h*h); // picture diameter
  const double A = w*h;             // picture area
  
  logalpha0_point_line = log10(2.0*D/A);
  logalpha0_point_point = log10(M_PI/A);
  logalpha0_point_point_without_K = log10(M_PI);
  
  nb_points = nPoints;
  nb_lines = nLines;
  nb_cop = nCopLines;
  nb_cop_pair = nCopPairs;
}

double TranslationNormAContrario::computeTranslationRatio(const PicturesSegments &lines, const PicturesPoints &points,
							  const vector<Pose> &globalPoses, vector<FTypeIndex> &inliers,
							  FEATURE_TYPE &chosen_ratio){
  double translation_norm = 0;

  // initialization
  double bestNFA = std::numeric_limits<double>::infinity();
  
  if(true || verbose){
    cout << "nb of coplanar cts: " << candidate_planes.size() << endl;
    cout << "nb of line triplet cts: " << line_triplets.size() << endl;
    cout << "nb of point triplet cts: " << point_triplets.size() << endl;
    cout << "total number of constraints: " << nConstraints << endl;
  }
  
  const double d = 2*log10(thresh_rsc+epsilon)+logalpha0_point_point;
  const double d_line = log10(thresh_rsc + epsilon) + logalpha0_point_line;;
  double threshold_inliers_cop, threshold_inliers_lines, threshold_inliers_points;
  vector<ErrorFTypeIndex> temp_log_residuals_cop_lines;
  vector<bool> cp(candidate_planes.size(), true), lt(line_triplets.size(), true), pt(point_triplets.size(), true);

  for(int i = 0; i < nConstraints; i++){
    // select associated t_norm
    float t_norm;
    if(i < candidate_planes.size()){
      if(!cp[i]){continue;}
      t_norm = candidate_planes[i].t_norm;
    }
    else if(i < candidate_planes.size() + line_triplets.size()){
      const int idx = i-candidate_planes.size();
      if(!lt[idx]){continue;}
      t_norm = line_triplets[idx].t_norm;
    }
    else{
      const int idx = i-candidate_planes.size()-line_triplets.size();
      if(!pt[idx]){continue;}
      t_norm = point_triplets[idx].t_norm;
    }
    
    const Vec3 C2 = globalPoses[i1].second - 1.0 / t_norm * globalPoses[i2].first.transpose()*t12;
    const Vec3 C0 = globalPoses[i1].second - t_norm * globalPoses[i0].first.transpose()*t10;

    vector<FTypeIndex> current_inliers;
    current_inliers.reserve(nFeatures);
#ifndef ACRANSAC
    vector<FTypeIndex> current_inliers_rsc;
    current_inliers_rsc.reserve(nFeatures);
#endif
    // compute nfa for coplanar lines
    vector<ErrorFTypeIndex> log_residuals_coppairs(candidate_planes.size());
    vector<ErrorFTypeIndex> log_residuals_cop_lines = logNFAcoplanar(t_norm, lines, globalPoses, log_residuals_coppairs, cp);
    double nfa_cop = computeNFA(log_residuals_cop_lines, log_residuals_coppairs, logc_n_cop, current_inliers, threshold_inliers_cop, d);
#ifndef ACRANSAC
    sort(log_residuals_coppairs.begin(), log_residuals_coppairs.end());
    for(int k = 2; k <= log_residuals_coppairs.size(); k++){
      if(log_residuals_coppairs[k].first > d){
	nfa_cop = 1-k;
	break;
      }
      current_inliers_rsc.push_back(log_residuals_coppairs[k].second);
    }
#endif
    threshold_inliers_cop = pow10((threshold_inliers_cop - logalpha0_point_point)/2);

    // TODO finir de symetriser les distances de points 3D ou d'extremités 3D
    // compute nfa for line triplets
    vector<ErrorFTypeIndex> log_residuals_lines = logNFAlines(C2, t_norm, globalPoses, lines, lt);
    double nfa_lines = computeNFA(log_residuals_lines, log_residuals_lines, logc_n_lines, current_inliers, threshold_inliers_lines);
#ifndef ACRANSAC
    for(int k = 2; k <= log_residuals_lines.size(); k++){
      if(log_residuals_lines[k].first > d_line){
	nfa_lines = 1-k;
	break;
      }
      current_inliers_rsc.push_back(log_residuals_lines[k].second);
    }
#endif
    threshold_inliers_lines = pow10(threshold_inliers_lines - logalpha0_point_line);
    
    // compute nfa for point triplets
    vector<ErrorFTypeIndex> log_residuals_points = logNFApoints(C2, C0, globalPoses, points, pt);
    double nfa_points = computeNFA(log_residuals_points, log_residuals_points, logc_n_points, current_inliers, threshold_inliers_points);
#ifndef ACRANSAC
    for(int k = 2; k <= log_residuals_points.size(); k++){
      if(log_residuals_points[k].first > d){
	nfa_points = 1-k;
	break;
      }
      current_inliers_rsc.push_back(log_residuals_points[k].second);
    }
#endif
    threshold_inliers_points = pow10((threshold_inliers_points - logalpha0_point_point)/2);
    
    double nfa = nfa_cop + nfa_lines + nfa_points;
    
    // if better solution
    if(nfa < bestNFA){
      // update nfa information
      bestNFA = nfa;
      translation_norm = t_norm;
#ifdef ACRANSAC
      inliers = current_inliers;
#else
      inliers = current_inliers_rsc;
#endif
      cout << "threshold = " << threshold_inliers_cop << "/" << threshold_inliers_lines << "/" << threshold_inliers_points << endl;
      chosen_ratio = (i < candidate_planes.size())? COPLANAR_PAIR : 
		    (i < candidate_planes.size() + line_triplets.size())? LINE
		    : POINT;
      temp_log_residuals_cop_lines = log_residuals_cop_lines;
    }
  }

  if(verbose){
    cout << "###############################" << endl;
    cout << "Final nb of inliers: " << inliers.size() << "/" << nConstraints << endl;
    nb_cop_inliers = nb_points_inliers = nb_lines_inliers = 0;
    for(int i = 0; i < inliers.size(); i++){
      if(inliers[i].first == COPLANAR_PAIR){
	nb_cop_inliers++;
      }
      else if(inliers[i].first == POINT){
	nb_points_inliers++;
      }
      else if(inliers[i].first == LINE){
	nb_lines_inliers++;
      }
    }
    cout << "nb of cop inliers: " << nb_cop_inliers << "/" << nCopPairs << endl;
    cout << "nb of point inliers: " << nb_points_inliers << "/" << nPoints << endl;
    cout << "nb of line inliers: " << nb_lines_inliers << "/" << nLines << endl;
  }
  return translation_norm;
}

double TranslationNormAContrario::refineTranslationRatio(double translation_norm, const vector<FTypeIndex> &inliers, 
							 PicturesSegments &lines, const PicturesMatches &matches_lines, 
							 PicturesPoints &points, const PicturesMatches &matches_points,
							 const vector<Pose> &globalPoses, vector<Plane> &planes, vector<ClusterPlane> &clusters){ 
  // separate inliers wrt nature (line/point triplets or coplanar constraints)
  vector<int> coplanar_inliers, line_triplets_inliers, point_triplets_inliers;
  {
    for(int i = 0; i < inliers.size(); i++){
      switch(inliers[i].first){
	case COPLANAR_PAIR:
	  coplanar_inliers.push_back(inliers[i].second);
	  break;
	case LINE:
	  line_triplets_inliers.push_back(inliers[i].second);
	  break;
	case POINT:
	  point_triplets_inliers.push_back(inliers[i].second);
	  break;
      }
    }
  }

  // merge planes with inlier coplanar constraints
  if(coplanar_inliers.size() > 0){
    clusters = mergePlanes(candidate_planes, coplanar_inliers, lines, matches_lines, i1);
    
    // 0. clear planes indexes from lines
    for(int i = 0; i < lines[i1].size(); i++){
      lines[i1][i].planes.clear();
    }
    
    // 1. give plane indexes to corresponding segments
    for(int i = 0; i < clusters.size(); i++){
      ClusterPlane* c = &(clusters[i]);
      for(int j = 0; j < c->proj_ids.size(); j++){
	lines[i1][c->proj_ids[j]].planes.push_back(i);
      }
    }
    
    // 2. use segment-plane correspondencies to obtain the 3D points that belong to planes
    // TODO add only 3D extremities instead of the whole 3D line
    for(int i = 0; i < l01.size(); i++){
      Line3D L = l01[i];
      L.p1 += globalPoses[i0].second;
      L.p2 += globalPoses[i0].second;
      int li = *(++l01[i].proj_ids.begin());
      for(int j = 0; j < lines[i1][li].planes.size(); j++){
	int idx = lines[i1][li].planes[j];
	clusters[idx].lines.push_back(L);
	l01[i].planes.insert(idx);
      }
    }
    for(int i = 0; i < l12.size(); i++){
      Line3D L = l12[i];
      L.p1 = L.p1/translation_norm + globalPoses[i1].second;
      L.p2 = L.p2/translation_norm + globalPoses[i1].second;
      int li = *(L.proj_ids.begin());
      for(int j = 0; j < lines[i1][li].planes.size(); j++){
	int idx = lines[i1][li].planes[j];
	clusters[idx].lines.push_back(L);
	l12[i].planes.insert(idx);
      }
    }
    
    // 3. compute 3D planes
    // TODO really need to store 3D points ?
    for(int i = 0; i < clusters.size(); i++){
      ClusterPlane* c = &(clusters[i]);
      const int size = 2*c->lines.size();
      vector<Vec3> pointsInPlane(size);
      
      // compute centroid
      c->centroid = Vec3(0,0,0);
      for(int j = 0; j < size/2; j++){
	pointsInPlane[2*j] = c->lines[j].p1;
	c->centroid += c->lines[j].p1;
	pointsInPlane[2*j+1] = c->lines[j].p2;
	c->centroid += c->lines[j].p2;
      }
      c->centroid /= size;
      
      // compute normal
      Eigen::MatrixXd A(size, 3);
      for(int j = 0; j < size; j++){
	for(int k = 0; k < 3; k++){
	  A(j, k) = pointsInPlane[j][k] - c->centroid[k];
	}
      }
      Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
      for(int k = 0; k < 3; k++){
	c->normal[k] = svd.matrixV()(k,2);
      }
      c->normal.normalize();
    }
  }

  vector<Mat3> rot;
  vector<Vec3> trans, center;
  // TODO for now use only reconstruction on the first 2 views since the refinement is only on t12_norm
  for(int k = 0; k < 2; k++){
    rot.push_back(globalPoses[i0+k].first);
    if(k != 2){
      trans.push_back(-globalPoses[i0+k].first*globalPoses[i0+k].second);
      center.push_back(globalPoses[i0+k].second);
    }
    else{
      trans.push_back(-trans[1]+t12/translation_norm);
      center.push_back(-rot[2].transpose()*trans[2]);
    }
  }

  // compute approximate 3D lines for refinement
  vector<Vec3> lines3D;
  vector<Segment> lines_proj;
  for(int i = 0; i < line_triplets_inliers.size(); i++){
    const int idx = line_triplets_inliers[i];
    vector<Segment> l(3);
    l[0] = lines[i0][line_triplets[idx].idx[0]];
    l[1] = lines[i1][line_triplets[idx].idx[1]];
    l[2] = lines[i2][line_triplets[idx].idx[2]];
    
    Line3D line = triangulate_line(rot, center, l);

    lines3D.push_back(globalPoses[i2].first*(line.mid-globalPoses[i1].second));
    lines3D.push_back(t12);
    
    lines_proj.push_back(l[2]);
  }
  
  // compute approximate 3D points for refinement
  vector<Vec3> points3D;
  vector<Point> points_proj;
  for(int i = 0; i < point_triplets_inliers.size(); i++){
    const int idx = point_triplets_inliers[i];
    vector<Point> p(3);
    p[0] = points[i0][point_triplets[idx].idx[0]];
    p[1] = points[i1][point_triplets[idx].idx[1]];
    p[2] = points[i2][point_triplets[idx].idx[2]];
    
    Point3D point = triangulate_point(rot, trans, p);
    
    points3D.push_back(globalPoses[i2].first*(point.p - globalPoses[i1].second));
    points3D.push_back(t12);
    
    points_proj.push_back(p[2]);
  }
  
  // store info into planes
  for(int i = 0; i < clusters.size(); i++){
    planes.push_back(Plane(clusters[i], i1));
  }
  
  return translation_norm; 
}

double TranslationNormAContrario::process(PicturesSegments &lines, const PicturesMatches &matches_lines, 
					  PicturesPoints &points, const PicturesMatches &matches_points,
					  const vector<Pose> &globalPoses, 
					  vector<Plane> &planes, Triplets &final_triplets, 
					  vector<ClusterPlane> &clusters, vector<ClusterPlane> &coplanar_cts,
					  FEATURE_TYPE &chosen_ratio){
  vector<FTypeIndex> inliers;
  double t_norm = computeTranslationRatio(lines, points, globalPoses, inliers, chosen_ratio);

  // store inlier triplets
  coplanar_cts.clear();
  for(int i = 0; i < inliers.size(); i++){
    if(inliers[i].first == LINE){
      final_triplets.push_back(line_triplets[inliers[i].second]);
    }
    if(inliers[i].first == POINT){
      final_triplets.push_back(point_triplets[inliers[i].second]);
    }
    if(inliers[i].first == COPLANAR_PAIR){
      coplanar_cts.push_back(candidate_planes[inliers[i].second]);
    }
  }

  t_norm = refineTranslationRatio(t_norm, inliers, lines, matches_lines, points, matches_points, globalPoses, planes, clusters);

  return t_norm;
}

/*=================== TRIANGULATION ===================*/

// TODO replace with closestPoint2Lines ?
Point3D triangulate_point(const Mat3 &R1, const Vec3 &t1, const Mat3 &R2, const Vec3 &t2, const Point &p1, const Point &p2){
  Mat4 design;
  for (int i = 0; i < 3; ++i) {
    design(0,i) = p1[0]/p1[2] * R1(2,i) - R1(0,i);
    design(1,i) = p1[1]/p1[2] * R1(2,i) - R1(1,i);
    design(2,i) = p2[0]/p2[2] * R2(2,i) - R2(0,i);
    design(3,i) = p2[1]/p2[2] * R2(2,i) - R2(1,i);
  }
  design(0,3) = p1[0]/p1[2] * t1[2] - t1[0];
  design(1,3) = p1[1]/p1[2] * t1[2] - t1[1];
  design(2,3) = p2[0]/p2[2] * t2[2] - t2[0];
  design(3,3) = p2[1]/p2[2] * t2[2] - t2[1];
  
  Point3D p;
  Vec4 p_homogenous;
  Nullspace(&design, &p_homogenous);
  for(int i = 0; i < 3; i++){
    p.p[i] = p_homogenous[i]/p_homogenous[3];
  }
  return p;
}

Point3D triangulate_point(const vector<Mat3> &R, const vector<Vec3> &t, const vector<Point> &p){
  int nPoints = R.size();
  Eigen::MatrixXd design(nPoints*2, 4);
  for(int k = 0; k < nPoints; k++){
    for (int i = 0; i < 3; ++i) {
      design(2*k,i) = p[k][0]/p[k][2] * R[k](2,i) - R[k](0,i);
      design(2*k+1,i) = p[k][1]/p[k][2] * R[k](2,i) - R[k](1,i);
    }
    design(2*k,3) = p[k][0]/p[k][2] * t[k][2] - t[k][0];
    design(2*k+1,3) = p[k][1]/p[k][2] * t[k][2] - t[k][1];
  }
  
  Point3D P;
  Vec4 p_homogenous;
  Nullspace(&design, &p_homogenous);
  for(int i = 0; i < 3; i++){
    P.p[i] = p_homogenous[i]/p_homogenous[3];
  }
  return P;
}

vector<Point3D> triangulate_points(const PicturesPoints &points, const vector<Pose> &globalPoses, const Triplets &triplets){
  const int nPictures = globalPoses.size();
  
  vector<Point3D> points3D;
  vector<vector<int>> points3D_idx(points.size());
  for(int i = 0; i < points.size(); i++){
    points3D_idx[i] = vector<int>(points[i].size(), -1);
  }

  // loop on triplets
  for(int i = 0; i < triplets.size(); i++){
    if(triplets[i].type != POINT){continue;}
    
    const int p0 = triplets[i].idx[0];
    const int p1 = triplets[i].idx[1];
    const int p2 = triplets[i].idx[2];
    const int iPic = triplets[i].iPicture;
    
    // merge with a previous 3D line
    if(points3D_idx[iPic-1][p0] != -1){
      int p3D = points3D_idx[iPic-1][p0];
      
      // TODO we suppose here that points are well associated since it is only inliers...
      if(points3D_idx[iPic][p1] != -1){
	if(p3D != points3D_idx[iPic][p1]){cout << "ERROR CODE #2" << endl;}
      }
      else{
	points3D[p3D].cam_ids.push_back(iPic);
	points3D[p3D].proj_ids.push_back(p1);
	points3D_idx[iPic][p1] = p3D;
      }

      points3D[p3D].cam_ids.push_back((iPic+1)%nPictures);
      points3D[p3D].proj_ids.push_back(p2);
      points3D_idx[(iPic+1)%nPictures][p2] = p3D;
    }
    else{
      vector<Mat3> R;
      vector<Vec3> t;
      vector<Point> pts;
      
      for(int k = 0; k < 3; k++){
	R.push_back(globalPoses[(iPic-1+k)%nPictures].first);
	t.push_back(-globalPoses[(iPic-1+k)%nPictures].first*globalPoses[(iPic-1+k)%nPictures].second);
	pts.push_back(points[(iPic-1+k)%nPictures][triplets[i].idx[k]]);
      }
      
      Point3D p = triangulate_point(R, t, pts);
      
      for(int k = 0; k < 3; k++){
	p.cam_ids.push_back((iPic-1+k)%nPictures);
	p.proj_ids.push_back(triplets[i].idx[k]);
	points3D_idx[(iPic-1+k)%nPictures][triplets[i].idx[k]] = points3D.size();
      }
      points3D.push_back(p);
    }
  }
  
  return points3D;
}  

vector<Point3D> triangulate_points(const PicturesPoints &points, const PicturesMatches &matches, const vector<Pose> &globalPoses){
  vector<Point3D> points3D;

  // loop on tracks
  for(int i = 0; i < points.size(); i++){
    PicturePair imPair(i, i+1);
    if(i == points.size()-1){
      continue;
      imPair.first = 0;
      imPair.second = points.size()-1;
    }

    Vec3 t1 = -globalPoses[i].first*globalPoses[i].second;
    Vec3 t2 = -globalPoses[i+1].first*globalPoses[i+1].second;
    
    vector<pair<int, int>> temp;
    PointConstraints pairs = selectPointMatches(points[i], points[i+1], matches.find(imPair)->second, temp);

    for(int j = 0; j < pairs.size(); j++){
      points3D.push_back(triangulate_point(globalPoses[i].first, t1, globalPoses[i+1].first, t2, pairs[j].first, pairs[j].second)); 
    }
  }
  return points3D;
}  

Line3D triangulate_line(const Mat3 &R1, const Vec3 &C1, const Mat3 &R2, const Vec3 &C2, const Segment &l, const Segment &m){
  Mat3 R = R2*R1.transpose();
  Vec3 t = -R2*(C2-C1);

  Vec3 mi = m.line.normalized();
  
  // compute 3D extremities
  float zp1 = -t.dot(mi)/(l.p1.dot(R.transpose()*mi));
  float zp2 = -t.dot(mi)/(l.p2.dot(R.transpose()*mi));
  
  Vec3 p1 = R1.transpose()*(zp1*l.p1)+C1;
  Vec3 p2 = R1.transpose()*(zp2*l.p2)+C1;
    
  return Line3D(p1, p2);
}

Line3D triangulate_line(const vector<Mat3> &R, const vector<Vec3> &C, const vector<Segment> &l){
  int nLines = R.size();
  if(nLines == 2){
    return triangulate_line(R[1], C[1], R[0], C[0], l[1], l[0]);
  }
  else if(nLines == 3){
    Line3D l10 = triangulate_line(R[1], C[1], R[0], C[0], l[1], l[0]);
    Line3D l12 = triangulate_line(R[1], C[1], R[2], C[2], l[1], l[2]);  
    
    Vec3 p1 = 0.5*(l10.p1 + l12.p1);
    Vec3 p2 = 0.5*(l10.p2 + l12.p2);
    return Line3D(p1, p2);
  }
  else{
    cout << "ERROR in triangulate_line function" << endl;
  }
}

void addProjTo3Dline(const int i_cam, const int i_proj, const int i_l3D,
		     PicturesSegments &lines, vector<Line3D> &lines3D, bool cop_only = false){
  // add line reprojection information
  if(!cop_only){
    lines3D[i_l3D].addProjection(i_cam, i_proj);
    lines[i_cam][i_proj].lines3D.push_back(i_l3D);
  }

  // add coplanar cts
  lines3D[i_l3D].addCopCts(lines[i_cam][i_proj]);
}

void boundingBox(Vec3 &minP, Vec3 &maxP, const vector<Pose> &globalPoses){
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
}

bool isInside(const Line3D &l, const Vec3 &minP, const Vec3 &maxP){
  bool inside = true;
  for(int j = 0; j < 3 && inside; j++){
    //inside = inside && min(l.p1[j], l.p2[j]) > minP[j] && max(l.p1[j], l.p2[j]) < maxP[j];
  }
  return inside;
}

void add3Dline(const int* i_cam, const int *i_proj, const int size,
	       PicturesSegments &lines, vector<Line3D> &lines3D,
	       const vector<Pose> &globalPoses, const Vec3 &minP, const Vec3 &maxP){
  // triangulate the 3D line
  vector<Mat3> R;
  vector<Vec3> C;
  vector<Segment> segs;
  
  for(int k = 0; k < size; k++){
    R.push_back(globalPoses[i_cam[k]].first);
    C.push_back(globalPoses[i_cam[k]].second);
    segs.push_back(lines[i_cam[k]][i_proj[k]]);
  }
  
  Line3D l = triangulate_line(R, C, segs);
  
  // check if the line is not too far from the cameras (possibly outlier)
  if(!isInside(l, minP, maxP)){return;}
  
  const int l3D = lines3D.size();
  lines3D.push_back(l);
  for(int k = 0; k < size; k++){
    addProjTo3Dline(i_cam[k], i_proj[k], l3D, lines, lines3D);
  }
}

Line3D compute3Dline(const int* i_cam, const int *i_proj, const int size,
		     const PicturesSegments &lines, const vector<Pose> &globalPoses){
  // triangulate the 3D line
  vector<Mat3> R;
  vector<Vec3> C;
  vector<Segment> segs;
  
  for(int k = 0; k < size; k++){
    R.push_back(globalPoses[i_cam[k]].first);
    C.push_back(globalPoses[i_cam[k]].second);
    segs.push_back(lines[i_cam[k]][i_proj[k]]);
  }
  
  return triangulate_line(R, C, segs);
}

bool addTriplet(const Triplet &tri, vector<Line3D> &lines3D, PicturesSegments &lines,
		const vector<Pose> &globalPoses, const Vec3 &minP, const Vec3 &maxP){
  int i_cam[3];
  i_cam[0] = tri.iPicture-1;
  i_cam[1] = tri.iPicture;
  i_cam[2] = tri.iPicture+1; // TODO BEWARE CLOSE LOOP IS NOT DONE !!!
  
  // check that tracking goes well
  if(lines[i_cam[0]][tri.idx[0]].lines3D.size() > 1 || lines[i_cam[1]][tri.idx[1]].lines3D.size() > 1 || lines[i_cam[2]][tri.idx[2]].lines3D.size() != 0){
    // TODO deal with closure
    cout << "ERROR TRACKING LINES" << endl;
    cout << lines[i_cam[0]][tri.idx[0]].lines3D.size() << ">1? " << lines[i_cam[1]][tri.idx[1]].lines3D.size() << ">1? " << lines[i_cam[2]][tri.idx[2]].lines3D.size() << "!=0?" << endl;
    return false;
  }
  
  // in case l0 already belongs to a 3D line
  if(lines[i_cam[0]][tri.idx[0]].lines3D.size() != 0){
    int l3D = lines[i_cam[0]][tri.idx[0]].lines3D[0];
    
    // if l1 already belongs to a 3D line it should be the same than l0's one
    if(lines[i_cam[1]][tri.idx[1]].lines3D.size() != 0){
      if(l3D != lines[i_cam[1]][tri.idx[1]].lines3D[0]){
	cout << "ERROR CODE #2" << endl;
	return false;
      }
    }
    else{
      addProjTo3Dline(i_cam[1], tri.idx[1], l3D, lines, lines3D);
    }

    // add l2 to the 3D line
    addProjTo3Dline(i_cam[2], tri.idx[2], l3D, lines, lines3D);
  }
  else{
    add3Dline(i_cam, tri.idx, 3, lines, lines3D, globalPoses, minP, maxP);
  }
  return true;
}

bool merge3Dlines(const int i1, const int i2, vector<Line3D> &lines3D, PicturesSegments &lines){  
  double angle_diff = acos(fabs(lines3D[i1].direction.dot(lines3D[i2].direction)))*180/M_PI;
  if(angle_diff > 5){
    cout << "MERGING DIFFERENT 3D LINES !" << endl;
    cout << "angle : " << angle_diff << endl;
    cout << lines3D[i1].p1.transpose() << endl;
    cout << lines3D[i1].p2.transpose() << endl;
    cout << lines3D[i2].p1.transpose() << endl;
    cout << lines3D[i2].p2.transpose() << endl;
    return false;
  }
  
  for(int i = 0; i < lines3D[i2].proj_ids.size(); i++){
    int i_cam = lines3D[i2].cam_ids[i];
    int i_proj = lines3D[i2].proj_ids[i];
    
    // change projections for 3D lines
    for(int j = 0; j < lines3D[i1].proj_ids.size(); j++){
      if(lines3D[i1].cam_ids[j] == i_cam && lines3D[i1].proj_ids[j] == i_proj){
	cout << "ONE SEGMENT BELONG TO 2 3D LINES !!!" << endl;
	return false;
      }
    }
    lines3D[i1].addProjection(i_cam, i_proj);
    
    // change 3D lines for projections
    if(lines[i_cam][i_proj].lines3D.size() > 1){
      cout << "THIS SEGMENT BELONG TO 2 3D LINES !!!" << endl;
      return false;
    }
    lines[i_cam][i_proj].lines3D[0] = i1;
  }
  // clear projections for previous 3D line
  lines3D[i2].cam_ids.clear();
  lines3D[i2].proj_ids.clear();
  
  return true;
}

bool addCopCts(const CopCts &cop, vector<Line3D> &lines3D, PicturesSegments &lines,
		const vector<Pose> &globalPoses, const Vec3 &minP, const Vec3 &maxP){
  // the two lines are triangulated separately
  for(int i = 0; i < 2; i++){
    // test for debug
    if(lines[cop.i_cam[2*i]][cop.i_proj[2*i]].lines3D.size() > 1 && lines[cop.i_cam[2*i+1]][cop.i_proj[2*i+1]].lines3D.size() > 1){
      cout << "ERROR MULTIPLE 3D LINE" << endl;
      return false;
    }
    
    // if no projections belong to a 3D line
    if(lines[cop.i_cam[2*i]][cop.i_proj[2*i]].lines3D.size() == 0 && lines[cop.i_cam[2*i+1]][cop.i_proj[2*i+1]].lines3D.size() == 0){
      add3Dline(&(cop.i_cam[2*i]), &(cop.i_proj[2*i]), 2, lines, lines3D, globalPoses, minP, maxP);
    }
    // if all projections belong to a 3D line
    else if(lines[cop.i_cam[2*i]][cop.i_proj[2*i]].lines3D.size() != 0 && lines[cop.i_cam[2*i+1]][cop.i_proj[2*i+1]].lines3D.size() != 0){
      int l3D = lines[cop.i_cam[2*i]][cop.i_proj[2*i]].lines3D[0];
      if(lines[cop.i_cam[2*i+1]][cop.i_proj[2*i+1]].lines3D[0] != l3D){
	return merge3Dlines(l3D ,lines[cop.i_cam[2*i+1]][cop.i_proj[2*i+1]].lines3D[0], lines3D, lines);
      }
      for(int k = 0; k < 2; k++){
	addProjTo3Dline(cop.i_cam[2*i+k], cop.i_proj[2*i+k], l3D, lines, lines3D, true);
      }
    }
    // if only one of the projection belong to a 3D line
    else{
      bool addProj1 = lines[cop.i_cam[2*i]][cop.i_proj[2*i]].lines3D.size() == 0;
      int l3D = (addProj1)? lines[cop.i_cam[2*i+1]][cop.i_proj[2*i+1]].lines3D[0] : lines[cop.i_cam[2*i]][cop.i_proj[2*i]].lines3D[0];
      int i_cam = (addProj1)? cop.i_cam[2*i] : cop.i_cam[2*i+1];
      int i_proj = (addProj1)? cop.i_proj[2*i] : cop.i_proj[2*i+1];
      addProjTo3Dline(i_cam, i_proj, l3D, lines, lines3D);
    }
  }
  return true;
}

bool sort_pairs(pair<int, double> pi,pair<int, double> pj) { return (pi.second < pj.second);}
vector<CopCts> filterCopCts(PicturesSegments &lines, const PicturesMatches &matches, 
			    const vector<Pose> &globalPoses, const vector<CopCts> &cop_cts){  
  vector<pair<int, double>> cop_cts_distances;
  for(int i = 0; i < cop_cts.size(); i++){
    CopCts cop = cop_cts[i];
    Line3D l1 = compute3Dline(&(cop.i_cam[0]), &(cop.i_proj[0]), 2, lines, globalPoses);
    Line3D l2 = compute3Dline(&(cop.i_cam[2]), &(cop.i_proj[2]), 2, lines, globalPoses);
    Vec3 normal = (CrossProductMatrix(l1.direction)*l2.direction).normalized();
    double d = (l1.mid-l2.mid).dot(normal);
    cop_cts_distances.push_back(pair<int, double>(i, fabs(d)));
  }
  
  sort(cop_cts_distances.begin(), cop_cts_distances.end(), sort_pairs);
  
  vector<CopCts> filtered;
  const int threshold = cop_cts.size()/10;
  for(int i = 0; i < cop_cts_distances.size(); i++){
    int idx = cop_cts_distances[i].first;
    if(i < threshold){
      filtered.push_back(cop_cts[idx]);
    }
    else{
      CopCts cop = cop_cts[idx];
      for(int k = 0; k < 4; k++){
	vector<int>* cc = &(lines[cop.i_cam[k]][cop.i_proj[k]].coplanar_cts);
	vector<int> temp;
	for(int j = 0; j < cc->size(); j++){
	  if((*cc)[j] != idx){
	    temp.push_back((*cc)[j]);
	  }
	}
	lines[cop.i_cam[k]][cop.i_proj[k]].coplanar_cts = temp;
      }
    }
  }
  
  // update lines coplanar indexes
  for(int i = 0; i < filtered.size(); i++){
    CopCts cop = filtered[i];
    int idx = cop_cts_distances[i].first;
    for(int k = 0; k < 4; k++){
      vector<int>* cc = &(lines[cop.i_cam[k]][cop.i_proj[k]].coplanar_cts);
      for(int j = 0; j < cc->size(); j++){
	if((*cc)[j] == idx){
	  (*cc)[j] = i;
	}
      }
    }
  }
  
  return filtered;
}

vector<Line3D> triangulate_lines(PicturesSegments &lines, const PicturesMatches &matches, const vector<Pose> &globalPoses, const Triplets &triplets, 
				 const vector<CopCts> &cop_cts, vector<int> &coplanar_cts){
  const int nPictures = globalPoses.size();
  vector<Line3D> lines3D;
  
  // compute scene bounding box wrt camera positions
  openMVG::Vec3 minP, maxP;
  boundingBox(minP, maxP, globalPoses);
  
  // clear line-line3D indexes
  for(int i = 0; i < lines.size(); i++){
    for(int j = 0; j < lines[i].size(); j++){
      lines[i][j].lines3D.clear();
    }
  }
  
  // loop on triplets
  for(int i = 0; i < triplets.size(); i++){
    if(triplets[i].type != LINE){continue;}
    if(!addTriplet(triplets[i], lines3D, lines, globalPoses, minP, maxP)){
      cout << "ERROR IN ADD TRIPLET FUNCTION" << endl;
      int pause; cin >> pause;
    }
  }
  
  // loop on coplanar only
  for(int i = 0; i < cop_cts.size(); i++){
    if(!addCopCts(cop_cts[i], lines3D, lines, globalPoses, minP, maxP)){
      /*cout << "ERROR IN ADD COPLANAR FUNCTION" << endl;
      int pause; cin >> pause;*/
    }
  }

  vector<int> idxCts(cop_cts.size(), -1);
  coplanar_cts.clear();
  coplanar_cts.reserve(idxCts.size()*2);
  for(int i = 0; i < lines3D.size(); i++){
    set<int>* cop = &(lines3D[i].cop_cts);
    for(set<int>::iterator it = cop->begin(); it != cop->end(); it++){
      int id = idxCts[*it];
      if(id == -1){
	id = coplanar_cts.size();
	idxCts[*it] = id+1;
	coplanar_cts.push_back(i);
	coplanar_cts.push_back(-1);
      }
      else{
	if(coplanar_cts[id] != -1){
	  continue;
	  // TODO test
	  cout << "ERROR IN MERGING COPLANAR CTS" << endl;
	  int pause; cin >> pause;
	}
	coplanar_cts[id] = i;
      }
    }
  }
  
  return lines3D;
}

vector<Line3D> triangulate_lines(const PicturesSegments &lines, const PicturesMatches &matches, const vector<Pose> &globalPoses){
  // compute scene bounding box wrt camera positions
  openMVG::Vec3 minP, maxP;
  boundingBox(minP, maxP, globalPoses);
  
  // expand a bit
  double expansion = 0;
  for(int j = 0; j < 3; j++){
    expansion = max(0.5*(maxP[j] - minP[j]), expansion);
  }
  for(int j = 0; j < 3; j++){
    minP[j] -= expansion;
    maxP[j] += expansion;
  }
  
  vector<Line3D> lines3D;

  // loop on tracks
  for(int i = 0; i < lines.size(); i++){
    PicturePair imPair(i, i+1);
    if(i == lines.size()-1){
      continue;
      imPair.first = 0;
      imPair.second = lines.size()-1;
    }

    const vector<int>* matches_lines = &(matches.find(imPair)->second);
    
    for(int li = 0; li < lines[i].size(); li++){
      int mi = (*matches_lines)[li];
      
      // if the point has no match, discard it
      if(mi == -1){ continue;}

      Line3D l = triangulate_line(globalPoses[i].first, globalPoses[i].second, globalPoses[i+1].first, globalPoses[i+1].second, lines[i][li], lines[i+1][mi]);
      l.cam_ids.push_back(i);
      l.cam_ids.push_back(i+1);
      l.proj_ids.push_back(li);
      l.proj_ids.push_back(mi);
      
      if(isInside(l, minP, maxP)){
	lines3D.push_back(l); 
      }
    }
  }
  return lines3D;
}  

vector<Plane> computePlanes(const vector<int> &coplanar_cts, vector<Line3D> &lines3D, const vector<Pose> &globalPoses){
  // initialize planes
  vector<Plane> planes;
  for(int i = 0; i < coplanar_cts.size()/2; i++){
    int li = coplanar_cts[2*i];
    int lj = coplanar_cts[2*i+1];
    Plane p;
    p.lines3D.insert(li);
    p.lines3D.insert(lj);
    p.computePlane(lines3D);
    
    planes.push_back(p);
  }
  cout << "nb of planes: " << planes.size() << endl;
  
  // compute scene scale with average distance between cameras
  double meanDist = 0;
  for(int i = 0; i < globalPoses.size()-1; i++){
    meanDist += (globalPoses[i].second - globalPoses[i+1].second).norm();
  }
  meanDist /= globalPoses.size()-1;
  
  // merge planes with parallelism and proximity constraints
  const double parallelism_thresh = cos(15*M_PI/180);
  const double proximity_thresh = 0.15*meanDist;
  
  return planes;
}

void computePlanes(PicturesSegments &lines, vector<Line3D> &lines3D, vector<Plane> &planes, const vector<Pose> &globalPoses){
  for(int i = 0; i < planes.size(); i++){
    planes[i].lines3D.clear();
  }

  // add line3D-plane indexes
  for(int i = 0; i < planes.size(); i++){
    int i_pic = planes[i].i_picture;
    for(int j = 0; j < planes[i].proj_ids.size(); j++){
      vector<int>* l3D = &(lines[i_pic][planes[i].proj_ids[j]].lines3D);
      for(int k = 0; k < l3D->size(); k++){
	planes[i].lines3D.insert((*l3D)[k]);
	lines3D[(*l3D)[k]].planes.insert(i);
      }
    }
  }

  // compute plane equations
  for(int i = 0; i < planes.size(); i++){
    if(planes[i].lines3D.size() == 0){continue;}
    planes[i].computePlane(lines3D);
  }  
  
  // compute scene scale with average distance between cameras
  double meanDist = 0;
  for(int i = 0; i < globalPoses.size()-1; i++){
    meanDist += (globalPoses[i].second - globalPoses[i+1].second).norm();
  }
  meanDist /= globalPoses.size()-1;
  
  // plane clean up
  cout << "nb of planes before clean up: " << planes.size() << endl;
  {   
    const double thresh_plane_width = 0.15*meanDist;
    const int thresh_plane = 0;
    // find which planes to delete
    vector<int> newIdx(planes.size(), -1);
    int curIdx = 0;
    for(int i = 0; i < planes.size(); i++){
      if(planes[i].lines3D.size() <= thresh_plane || planes[i].width > thresh_plane_width){continue;}
      newIdx[i] = curIdx;
      curIdx++;
    }
    
    // clean up planes index
    for(int i = 0; i < lines3D.size(); i++){
      set<int>* p3D = &(lines3D[i].planes);
      set<int> newPlanesIdx;
      for(set<int>::iterator it = p3D->begin(); it != p3D->end(); it++){
	int idx = newIdx[*it];
	if(idx == -1){continue;}
	newPlanesIdx.insert(idx);
      }
      lines3D[i].planes = newPlanesIdx;
    }
    
    // clean up planes 
    vector<Plane> newPlanes;
    for(int i = 0; i < planes.size(); i++){
      if(newIdx[i] == -1){continue;}
      newPlanes.push_back(planes[i]);
    }
    planes = newPlanes;
  }
  cout << "nb of planes after clean up: " << planes.size() << endl;
}