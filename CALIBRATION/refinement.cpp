/*----------------------------------------------------------------------------  
  This code is part of the following publication and was subject
  to peer review:
  "Robust and Accurate Line- and/or Point-Based Pose Estimation without Manhattan Assumptions",
  Yohann Salaun, Renaud Marlet, and Pascal Monasse, ECCV 2016
  
  Copyright (c) 2016 Yohann Salaun <yohann.salaun@imagine.enpc.fr>
  
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
#include "refinement.hpp"

#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <boost/concept_check.hpp>

using namespace openMVG;

/*=================== MISCELLANEOUS ===================*/

template <typename T>
void crossProduct(const T* const l0, const T* const l1, T l0Vecl1[3]){
    // compute cross product
    for(unsigned int i = 0; i < 3; i++){
      int i1 = (i+1)%3;
      int i2 = (i+2)%3;

      l0Vecl1[i] = l0[i1]*l1[i2] - l0[i2]*l1[i1];
    }
}

Mat3 root(const Mat3 &R, const double k, const double n){
  double angleAxis[3];
  ceres::RotationMatrixToAngleAxis((const double*)(R.data()), angleAxis);
  for(int i = 0; i < 3; i++){
    angleAxis[i] *= k/n;
  }
  Mat3 nth_root;
  ceres::AngleAxisToRotationMatrix(&angleAxis[0], (nth_root.data()));
  return nth_root;
}

void startCeresPose(ceres::Problem &problem, const bool verbose){
  ceres::Solver::Summary summary;
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.logging_type = ceres::SILENT;
  options.minimizer_progress_to_stdout = true;

  // find global minimum with a near-solution initialization
  // CERES minimization
  ceres::Solve(options, &problem, &summary);
  
  if(verbose){
    cout << summary.FullReport() << endl;
  }
}

void startCeresBA(ceres::Problem &problem, const bool verbose){
  ceres::Solver::Summary summary;
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.preconditioner_type = ceres::JACOBI; 
  options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.max_num_iterations = 500;
  options.parameter_tolerance = 1e-8;
  options.logging_type = ceres::SILENT;
  options.minimizer_progress_to_stdout = true;
  
  // CERES minimization
  ceres::Solve(options, &problem, &summary);
  
  if(verbose){
    cout << summary.FullReport() << endl;
  }
}

/*=================== Plucker/Cayley functions ===================*/
const double tau = 1e-7;
void pluckerToCayley(const Vec3 &m, const Vec3 &l, Vec3 &s, float &w){
  Mat3 Q;
  w = m.norm();
  Vec3 q2, q3;
  if(w < tau){
    if(l[1] == 0 && l[0] == 0){
      q2[0] = l[2];
      q3[1] = fabs(l[2]);
      q2[1] = q3[0] = 0;
      q2[2] = q3[2] = 0;
    }
    else{
      q2[0] = l[1];
      q2[1] = -l[0];
      q2[2] = 0;
      q3 = CrossProductMatrix(l)*q2;
    }
  }
  else{
    q2 = m/w;
    q3 = CrossProductMatrix(l)*m;
    q3.normalize();
  }
  
  for(int i = 0; i < 3; i++){
    Q(i, 0) = l[i];
    Q(i, 1) = q2[i];
    Q(i, 2) = q3[i];
  } 
  Mat3 Scross = (Q - Mat3::Identity())*(Q + Mat3::Identity()).inverse();
  s[0] = Scross(2,1);
  s[1] = Scross(0,2);
  s[2] = Scross(1,0);
}

template <typename T>
void cayleyToPlucker(const T* const sw, T* l, T* m){
  T norm = T(0);
  for(int k = 0; k < 3; k++){
    norm += sw[k]*sw[k];
  }
  
  T den = T(1) / (T(1) + norm);
  l[0] = (T(1) - norm + T(2)*sw[0]*sw[0])*den;
  l[1] = T(2)*( sw[2] + sw[0]*sw[1])*den;
  l[2] = T(2)*(-sw[1] + sw[0]*sw[2])*den;
  
  m[0] = sw[3] * den * T(2) * (-sw[2] + sw[1]*sw[0]);
  m[1] = sw[3] * den * (T(1) - norm + T(2)*sw[1]*sw[1]);
  m[2] = sw[3] * den * T(2) * ( sw[0] + sw[1]*sw[2]);
}

/*=================== CERES conversion ===================*/

Pose ceresToPose(const double rot[3], const double trans[3]){
  Pose pose;
  
  // rotation
  ceres::AngleAxisToRotationMatrix(rot, pose.first.data());
  
  // translation
  for(int k = 0; k < 3; k++){
    pose.second[k] = trans[k];
  }
  pose.second.normalize();
  
  return pose;
}

void poseToCeres(const Pose &pose, double* rot, double* trans){
  // rotation
  ceres::RotationMatrixToAngleAxis((const double*)pose.first.data(), rot);

  // translation
  for(int k = 0; k < 3; k++){
    trans[k] = pose.second[k];
  }
}

void lineToCeres(const Line3D &l, double line_ceres[4]){
  Vec3 s; float w;
  pluckerToCayley(CrossProductMatrix(0.5*(l.p1+l.p2))*l.direction, l.direction, s, w);
  for(int k = 0; k < 3; k++){
    line_ceres[k] = s[k];
  }
  line_ceres[3] = w;
}

void ceresToLine(Line3D &line, const double line_ceres[4]){
  Vec3 p, moment;
  double l[3], m[3];
  cayleyToPlucker(line_ceres, l, m);
  for(int k = 0; k < 3; k++){
    line.direction[k] = l[k];
    moment[k] = m[k];
  }
  p = -CrossProductMatrix(moment)*line.direction;

  line.p1 = p + (line.p1 - p).dot(line.direction)*line.direction;
  line.p2 = p + (line.p2 - p).dot(line.direction)*line.direction; 
}

/*=================== Reprojection functions ===================*/
// compute projection of line R*(m - Cxl) also equal to (C,l) plane normal direction
template <typename T>
void projectLine(const T* const cam_R, const T* const cam_C, 
		 const T* const line_l, const T* const line_m,
		 const Mat3 &cofK, T* proj_l){
    // compute Cxl
    T CxL[3];
    crossProduct(cam_C, line_l, CxL);
    
    // compute m-Cxl
    T mCxL[3], RmCxL[3];
    for(int k = 0; k < 3; k++){
      mCxL[k] = line_m[k] - CxL[k];
    }
    
    // compute R(m-Cxl)
    ceres::AngleAxisRotatePoint(cam_R, mCxL, RmCxL);

    // final projection with K matrix
    for(int k = 0; k < 3; k++){
      proj_l[k] = T(0);
      for(int j = 0; j < 3; j++){
	proj_l[k] += T(cofK(k,j))*RmCxL[j];
      }
    }
}

const double invSqrt2 = 1/sqrt(2.0);
template <typename T>
void residualLine(const T* const cam_R, const T* const cam_C,
		  const T* const line_l, const T* const line_m,
		  const Vec3 &p1, const Vec3 &p2, const Mat3 &cofK,
		  T* out_residuals){
  // project line on picture
  T proj_l[3];
  projectLine(cam_R, cam_C, line_l, line_m, cofK, proj_l);

  // compute distance between segment endpoints and line reprojection
  T dp1 = T(0), dp2 = T(0), norm = T(0);
  for(int k = 0; k < 3 ; k++){
    if(k < 2){norm += proj_l[k]*proj_l[k];}
    dp1 += proj_l[k]*T(p1[k]);
    dp2 += proj_l[k]*T(p2[k]);
  }
  norm = sqrt(norm);

  out_residuals[0] = T(invSqrt2)*dp1/norm;
  out_residuals[1] = T(invSqrt2)*dp2/norm;
}

template <typename T>
void residualPointAngular(const T* const cam_R, const T* const cam_t,
			  const double* p, const double* q,
			  T &out_residuals){
    T P[3], Q[3];
    T RT[3];
    for(int k = 0; k < 3; k++){
      P[k] = T(p[k]);
      Q[k] = T(q[k]);
      RT[k] = -cam_R[k];
    }
    
    // compute (R^T t) x p
    T RTtxp[3], RTt[3];
    ceres::AngleAxisRotatePoint(RT, cam_t, RTt);
    crossProduct(RTt, P, RTtxp); 
    
    // compute (R^T t) x (R^T q)
    T RTtxRTq[3], RTq[3];
    ceres::AngleAxisRotatePoint(RT, Q, RTq);
    crossProduct(RTt, RTq, RTtxRTq);

    // compute [(R^T t) x p] x [(R^T t) x (R^T q)] = 0
    T zero[3];
    crossProduct(RTtxp, RTtxRTq, zero);
    T n1 = T(0), n2 = T(0), d = T(0);
    for(unsigned int i = 0; i < 3; i++){
      n1 += RTtxp[i]*RTtxp[i];
      n2 += RTtxRTq[i]*RTtxRTq[i];
      d += zero[i]*zero[i];
    }
    d = sqrt(d/(n1*n2));

    // residual is d(Ep, q)
    out_residuals = d;
}

template <typename T>
void residualPoint(const T* const cam_R, const T* const cam_C,
		   const double fx, const double fy, 
		   const double px, const double py,
		   const T* const p, const Vec3 &proj, 
		   T* out_residuals){
  T proj_p[3], trans_p[3];
  for(int k = 0; k < 3; k++){
    trans_p[k] = p[k] - cam_C[k];
  }
  
  ceres::AngleAxisRotatePoint(cam_R, trans_p, proj_p);

  proj_p[0] = T(fx)*proj_p[0]/proj_p[2] + T(px);
  proj_p[1] = T(fy)*proj_p[1]/proj_p[2] + T(py);
  proj_p[2] = T(1);
  
  // normal should be close to line proj normal
  out_residuals[0] = proj_p[0] - T(proj[0]); 
  out_residuals[1] = proj_p[1] - T(proj[1]);
}

template <typename T>
void residualCoplanar(const T* const cam_R, const T* const cam_C,
		      const double fx, const double fy, 
		      const double px, const double py,
		      const T* const p1, const T* const p2, 
		      T* out_residuals){
  T proj_p1[3], trans_p1[3];
  T proj_p2[3], trans_p2[3];
  for(int k = 0; k < 3; k++){
    trans_p1[k] = p1[k] - cam_C[k];
    trans_p2[k] = p2[k] - cam_C[k];
  }
  
  ceres::AngleAxisRotatePoint(cam_R, trans_p1, proj_p1);
  ceres::AngleAxisRotatePoint(cam_R, trans_p2, proj_p2);
  
  proj_p1[0] = T(fx)*proj_p1[0]/proj_p1[2] + T(px);
  proj_p1[1] = T(fy)*proj_p1[1]/proj_p1[2] + T(py);
  proj_p2[0] = T(fx)*proj_p2[0]/proj_p2[2] + T(px);
  proj_p2[1] = T(fy)*proj_p2[1]/proj_p2[2] + T(py);
  
  // normal should be close to line proj normal
  out_residuals[0] = proj_p1[0] - proj_p2[0]; 
  out_residuals[1] = proj_p1[1] - proj_p2[1];
}

// PARALLEL PAIR
template <typename T>
void residualPPair(const T* const cam_R, 
		    const double* liXlj, const double* miXmj,
		    T &out_residuals){
    T LiXLj[3], RLiXLj[3], MiXMj[3];
    T R[3];
    for(int k = 0; k < 3; k++){
      LiXLj[k] = T(liXlj[k]);
      MiXMj[k] = T(miXmj[k]);
      R[k]  = cam_R[k];
    }
   
    // compute cross product
    ceres::AngleAxisRotatePoint(R, LiXLj, RLiXLj);
    
    T zero[3];
    crossProduct(MiXMj, RLiXLj, zero);
    T d = T(0);
    for(unsigned int i = 0; i < 3; i++){
      d += zero[i]*zero[i];
    }
    d = sqrt(d);
    
    out_residuals = d;
}

/*=================== Residuals ===================*/
struct Residual_Point_Angular{
  Residual_Point_Angular(const PointConstraint &point){
  for(int k = 0; k < 3; k++){
      p[k] = point.first[k]/point.first[2];
      q[k] = point.second[k]/point.second[2];
    }
  }   

  template <typename T>
  bool operator()(
    const T* const cam_R,
    const T* const cam_t,
    T* out_residuals) const
  {  
    residualPointAngular(cam_R, cam_t, p, q, out_residuals[0]);
    return true;
  }

  // 2D observations of points
  double p[3], q[3];
};

struct Residual_PPair{
  Residual_PPair(const ParallelConstraint &ppair){
    for(int k = 0; k < 3; k++){
      vpi[k] = ppair.first.vp[k];
      vpj[k] = ppair.second.vp[k];
    }
  }   

  /// Compute the residual error of rotation for a parallel pair of lines
  /**
  * @param[in] cam_R: Camera rotation parameters (a, b, c, d = sqrt(1 - a*a - b*b - c*c))
  * @param[out] out_residuals
  */
  template <typename T>
  bool operator()(
    const T* const cam_R,
    T* out_residuals) const
  { 
    residualPPair(cam_R, vpi, vpj, out_residuals[0]);
    return true;
  }

  // 2D observations of VPs
  double vpi[3], vpj[3];
};

struct Residual_Point_Reprojection{
   Residual_Point_Reprojection(const Point &p, const Mat3 &K){
    proj = K*p; proj /= proj[2];

    fx = K(0,0);
    fy = K(1,1);
    px = K(0,2);
    py = K(1,2);
  }   

  template <typename T>
  bool operator()(
    const T* const cam_R,
    const T* const cam_C,
    const T* const p,
    T* out_residuals) const
  {     
    residualPoint(cam_R, cam_C, fx, fy, px, py, p, proj, out_residuals);

    return true;
  }

  Vec3 proj;
  double fx, fy, px, py;
};

struct Residual_Line_Reprojection{
  Residual_Line_Reprojection(const Segment &seg, const Mat3 &K, const Mat3 &cK){
    p1 = K*seg.p1; p1 /= p1[2];
    p2 = K*seg.p2; p2 /= p2[2];

    cofK = cK;
  }   

  template <typename T>
  bool operator()(
    const T* const cam_R,
    const T* const cam_C,
    const T* const l,
    T* out_residuals) const
  {         
    // recover line information from cayley coordinates
    T line_m[3], line_l[3];
    cayleyToPlucker(l, line_l, line_m);
    
    // compute reprojection residual
    residualLine(cam_R, cam_C, line_l, line_m, p1, p2, cofK, out_residuals);
    
    return true;
  }

  Mat3 cofK;
  Vec3 p1, p2;
};

struct Residual_Line_Reprojection_Fixed_Direction{
  Residual_Line_Reprojection_Fixed_Direction(const Segment &seg, const Vec3 &d, const Mat3 &K, const Mat3 &cK){
    p1 = K*seg.p1; p1 /= p1[2];
    p2 = K*seg.p2; p2 /= p2[2];
    dir = d;
    
    cofK = cK;
  }   

  template <typename T>
  bool operator()(
    const T* const cam_R,
    const T* const cam_C,
    const T* const l,
    T* out_residuals) const
  {         
    // recover line information from cayley coordinates
    T line_m[3], line_l[3];
    for(int k = 0; k < 3; k++){
      line_l[k] = T(dir[k]);
    }
    crossProduct(l, line_l, line_m);
    
    // compute reprojection residual
    residualLine(cam_R, cam_C, line_l, line_m, p1, p2, cofK, out_residuals);
    
    return true;
  }

  Mat3 cofK;
  Vec3 p1, p2, dir;
};

struct Residual_Coplanar_Reprojection{
  Residual_Coplanar_Reprojection(const Mat3 &K){    
    fx = K(0,0);
    fy = K(1,1);
    px = K(0,2);
    py = K(1,2);
  }   

  template <typename T>
  bool operator()(
    const T* const cam_R,
    const T* const cam_C,
    const T* const l1,
    const T* const l2,
    T* out_residuals) const
  {     
    // recover line information from cayley coordinates
    T line_m1[3], line_l1[3];
    cayleyToPlucker(l1, line_l1, line_m1);
    T line_m2[3], line_l2[3];
    cayleyToPlucker(l2, line_l2, line_m2);
    
    // compute closest points
    T P12[3], P21[3];
    
    // plane normal
    T normal[3];
    crossProduct(line_l1, line_l2, normal);
    T norm = T(0);
    for(int k = 0; k < 3; k++){
      norm += normal[k]*normal[k];
    }
    
    // point in l1 and l2
    T p1[3], p2[3];
    crossProduct(line_l1, line_m1, p1);
    crossProduct(line_l2, line_m2, p2);
    
    T lambda = T(0), mu = T(0), dp = T(0);
    T l2_ortho[3];
    crossProduct(line_l2, normal, l2_ortho);
    for(int k = 0; k < 3; k++){
      lambda += (p2[k]-p1[k])*l2_ortho[k];
      dp += line_l1[k]*l2_ortho[k];
      mu += (p2[k]-p1[k])*normal[k];
    }
    lambda /= dp;
    
    for(int k = 0; k < 3; k++){
      P12[k] = p1[k] + lambda*line_l1[k];
      P21[k] = P12[k] + mu*normal[k]/norm;
    }
    
    residualCoplanar(cam_R, cam_C, fx, fy, px, py, P12, P21, out_residuals);
    
    return true;
  }

  double fx, fy, px, py;
};

/*=================== RELATIVE POSE REFINEMENT ===================*/
Pose refinedPose(const PointConstraints &points, const ParallelConstraints &ppairs, const Pose &pose, 
		 const vector<FTypeIndex> &inliers, const bool verbose){
  // convert R and t
  double rot[3], trans[3];
  poseToCeres(pose, rot, trans);
  
  // CERES formulation
  ceres::Problem problem;
  // one residual block for each constraint
  for(unsigned int i = 0; i < inliers.size(); i++){
    switch(inliers[i].first){
    case POINT :{
      ceres::CostFunction* cost_function = 
	new ceres::AutoDiffCostFunction<Residual_Point_Angular, 1, 3, 3>(new Residual_Point_Angular(points[inliers[i].second]));
      problem.AddResidualBlock(cost_function, NULL, rot, trans);
      }
      break;
    case PARALLEL_PAIR :{
      ceres::CostFunction* cost_function = 
	new ceres::AutoDiffCostFunction<Residual_PPair, 1, 3>(new Residual_PPair(ppairs[inliers[i].second]));
      problem.AddResidualBlock(cost_function, NULL, rot);
      }
      break;
    }
  }
  
  // CERES options
  startCeresPose(problem, verbose);
  
  // convert back rotation and translation
  return ceresToPose(rot, trans);
}

/*=================== REFINE WHOLE STRUCTURE ===================*/
void pointResidualDistribution(const vector<Point3D> &points3D, const PicturesPoints &points, const vector<double> &points_ceres,
			       const vector<double> &poses, const vector<Mat3> &K){
  const int n = 2;
  vector<double> residual_points;
  
  for(int i = 0; i < points3D.size(); i++){
    for(int k = 0; k < points3D[i].proj_ids.size(); k++){
      int i_cam = points3D[i].cam_ids[k];
      int i_proj = points3D[i].proj_ids[k];

      double residuals[n];
      Residual_Point_Reprojection rpr(points[i_cam][i_proj], K[i_cam]);
      rpr(&poses[6*i_cam], &poses[6*i_cam+3], &points_ceres[3*i], &residuals[0]);
      
      double residual = 0;
      for(int k = 0; k < n; k++){
	residual += residuals[k]*residuals[k];
      }

      residual_points.push_back(sqrt(residual));
    }
  }
  
  sort(residual_points.begin(), residual_points.end());
  double average = 0, sd = 0;
  for(int i = 0; i < residual_points.size(); i++){
    average += residual_points[i];
    sd += residual_points[i]*residual_points[i];
  }
  average /= residual_points.size();
  sd /= residual_points.size();
  sd -= average*average;
  cout << "STATS FOR POINT RESIDUALS: " << endl;
  cout << "min = " << residual_points[0] << endl;
  cout << "mean = " << average << endl;
  cout << "median = " << residual_points[residual_points.size()/2] << endl;
  cout << "max = " << residual_points[residual_points.size()-1] << endl;
  cout << "sd = " << sd << "/" << sqrt(sd) << endl;
}

void bundleAdjustment(vector<Point3D> &points3D, const PicturesPoints &points,
		      vector<Line3D> &lines3D, const PicturesSegments &lines, 
		      vector<Pose> &globalPose, const vector<Mat3> &K,
		      const vector<int> &cop_cts, const bool refine_rotation, const bool verbose){
  const int nViews = globalPose.size();
  
  const bool refine_center = true;
  const bool refine_lines = true;
  const bool refine_points = true;
  
  vector<Mat3> cofK(K.size());
  for(int i = 0; i < K.size(); i++){
    cofK[i] = K[i].determinant()*K[i].inverse().transpose();
  }
  
  vector<bool> usedPose(nViews, false);
  vector<bool> usedLine(lines3D.size(), false);

  // CERES formulation
  ceres::Problem problem;
  ceres::LossFunction * p_LossFunction = NULL;
  
  // convert pose parameters
  vector<double> poses(6*nViews);
  for(int i = 0; i < nViews; i++){
    poseToCeres(globalPose[i], &poses[6*i], &poses[6*i+3]);
  }
  
  // convert point parameters
  vector<double> points_ceres(3*points3D.size(), 0);
  for(int i = 0; i < points3D.size(); i++){
    for(int k = 0; k < 3; k++){
      points_ceres[3*i+k] = points3D[i].p[k];
    }
  }
  
  // converts lines parameters using cayley representation
  vector<double> lines_ceres(4*lines3D.size(), 0);
  vector<int> idx(lines3D.size(), -1);
  for(int i = 0; i < lines3D.size(); i++){
    lineToCeres(lines3D[i], &lines_ceres[4*i]);
  }
  
  // point reprojection constraints
  for(int i = 0; i < points3D.size(); i++){
    for(int k = 0; k < points3D[i].proj_ids.size(); k++){
      int i_cam = points3D[i].cam_ids[k];
      int i_proj = points3D[i].proj_ids[k];
      
      ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<Residual_Point_Reprojection, 2, 3, 3, 3>(
	    new Residual_Point_Reprojection(points[i_cam][i_proj], K[i_cam]));

      problem.AddResidualBlock(cost_function, p_LossFunction, &poses[6*i_cam], &poses[6*i_cam+3], &points_ceres[3*i]);
      
      usedPose[i_cam] = true;
    }
  }
  
  // line reprojection constraints
  for(int i = 0; i < lines3D.size(); i++){        
    // line reprojection constraints for triplets only
    for(int k = 0; k < lines3D[i].proj_ids.size(); k++){
      int i_cam = lines3D[i].cam_ids[k];
      int i_proj = lines3D[i].proj_ids[k];

      ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<Residual_Line_Reprojection, 2, 3, 3, 4>(
	  new Residual_Line_Reprojection(lines[i_cam][i_proj], K[i_cam], cofK[i_cam]));
      
      problem.AddResidualBlock(cost_function, p_LossFunction, &poses[6*i_cam], &poses[6*i_cam+3], &lines_ceres[4*i]);

      usedLine[i] = true;
      usedPose[i_cam] = true;
    }
  }

  // coplanar constraints if any
  for(int i = 0; i < cop_cts.size()/2; i++){
    int li = cop_cts[2*i];
    int lj = cop_cts[2*i+1];
    
    if(li == -1 || lj == -1){continue;}

    for(int k = 0; k < lines3D[li].cam_ids.size(); k++){
      int i_cam = lines3D[li].cam_ids[k];
      int i_proj = lines3D[li].proj_ids[k];
	
      bool found = false;
      int i_proj2;
      int idx;
      for(int j = 0; j < lines3D[lj].cam_ids.size() && !found;j++){
	found = lines3D[lj].cam_ids[j] == i_cam;
	if(found){
	  i_proj2 = lines3D[lj].proj_ids[j];
	  idx = j;
	}
      }
      if(!found){continue;}

      ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<Residual_Coplanar_Reprojection, 2, 3, 3, 4, 4>(
	  new Residual_Coplanar_Reprojection(K[i_cam]));

      problem.AddResidualBlock(cost_function, p_LossFunction, &poses[6*i_cam], &poses[6*i_cam+3], &lines_ceres[4*li], &lines_ceres[4*lj]);

      usedLine[li] = true;
      usedLine[lj] = true;
      usedPose[i_cam] = true;
    }  
  }
  
  if(verbose){
    cout << "nb of 3D points: " << points3D.size() << endl;
    cout << "nb of 3D lines: " << lines3D.size() << endl;
    cout << "nb of coplanar constraints: " << cop_cts.size()/2 << endl;
  }
  
  // define some variable as constants
  {
    // for pose
    for(int i = 0; i < nViews; i++){
      if(!usedPose[i]){continue;}
      
      if(!refine_rotation){
	problem.SetParameterBlockConstant(&poses[6*i]);
      }
      if(!refine_center){
	problem.SetParameterBlockConstant(&poses[6*i+3]);
      }
    }
    // for points
    if(!refine_points){
      for(int i = 0; i < points3D.size(); i++){    
	problem.SetParameterBlockConstant(&points_ceres[3*i]);
      }
    }
    // for lines
    if(!refine_lines){
      for(int i = 0; i < lines3D.size(); i++){
	if(!usedLine[i]){continue;}
	problem.SetParameterBlockConstant(&lines_ceres[4*i]);
      }
    }
  }
  
  // CERES options
  startCeresBA(problem, verbose);
  
  // recover final poses
  for(int i = 0; i < nViews; i++){
    ceres::AngleAxisToRotationMatrix(&poses[6*i], (globalPose[i].first.data()));
    for(int k = 0; k < 3; k++){
      globalPose[i].second[k] = poses[6*i+k+3];
    }
  }
  
  // recover final points
  for(int i = 0; i < points3D.size(); i++){
    for(int k = 0; k < 3; k++){
      points3D[i].p[k] = points_ceres[3*i+k];
    }
  }
  
  // recover final lines
  for(int i = 0; i < lines3D.size(); i++){    
    ceresToLine(lines3D[i], &lines_ceres[4*i]);
  }
  
  if(verbose && refine_rotation && points3D.size() > 0){
    pointResidualDistribution(points3D, points, points_ceres, poses, K);
  }
}

Vec3 updateDir(Vec3 &mainDir, const vector<int> cluster, const vector<Line3D> &lines3D){
  Vec3 direction(0,0,0);
  Vec3 positive = mainDir;
  for(int j = 0; j < cluster.size(); j++){
    int idx = cluster[j];
    if(positive.dot(lines3D[idx].direction) > 0){
      direction += lines3D[idx].direction;
    }
    else{
      direction -= lines3D[idx].direction;
    }
  }
  mainDir = direction.normalized();
}

void manhattanize(vector<Line3D> &lines3D, const PicturesSegments &lines, vector<Plane> &planes,
		  const vector<Pose> &globalPose, const vector<openMVG::Mat3> &K){
  const int nViews = globalPose.size();
  
  // distance threshold computed from camera distances
  double plane_distance_threshold = 0;
  Vec3 centroid(0,0,0);
  for(int i = 0; i < globalPose.size(); i++){
    plane_distance_threshold += (globalPose[i].second - globalPose[(i+1)%nViews].second).norm();
    centroid += globalPose[i].second;
  }
  plane_distance_threshold *= 0.1/nViews;
  centroid /= nViews;
  const double ray = plane_distance_threshold *100;
  
  const bool refine_lines = true;
  
  vector<Mat3> cofK(K.size());
  for(int i = 0; i < K.size(); i++){
    cofK[i] = K[i].determinant()*K[i].inverse().transpose();
  }
  
  vector<bool> usedPose(nViews, false);
  vector<bool> usedLine(lines3D.size(), false);

  // CERES formulation
  ceres::Problem problem;
  ceres::LossFunction * p_LossFunction = NULL;
  
  // convert pose parameters
  vector<double> poses(6*nViews);
  for(int i = 0; i < nViews; i++){
    poseToCeres(globalPose[i], &poses[6*i], &poses[6*i+3]);
  }
  
  // compute main directions
  vector<Vec3> mainDir;
  vector<vector<int>> clusters;
  vector<int> label(lines3D.size(), -1);
  const double same_direction_threshold = cos(10*M_PI/180);
  for(int i = 0; i < lines3D.size(); i++){
    // check if direction already exists
    bool added = false;
    for(int j = 0; j < mainDir.size() && !added; j++){
      if(fabs(mainDir[j].dot(lines3D[i].direction)) > same_direction_threshold){
	added = true;
	clusters[j].push_back(i);
	label[i] = j;
      }
    }
    // add new direction
    if(!added){
      mainDir.push_back(lines3D[i].direction);
      clusters.push_back(vector<int>(1, i));
      label[i] = mainDir.size()-1;
    }
  }
  // recompute main directions from clusters
  for(int i = 0; i < mainDir.size(); i++){
    updateDir(mainDir[i], clusters[i], lines3D);
  }
  // recluster clusters
  while(true){
    bool merged = false;
    for(int i = 0; i < mainDir.size() && !merged; i++){
      for(int j = 0; j < i && !merged; j++){
	if(fabs(mainDir[i].dot(mainDir[j])) > same_direction_threshold){
	  for(int p = 0; p < clusters[i].size(); p++){
	    clusters[j].push_back(clusters[i][p]);
	    label[clusters[i][p]] = j;
	  }
	  updateDir(mainDir[j], clusters[j], lines3D);
	  for(int k = i+1; k < mainDir.size(); k++){
	    mainDir[k-1] = mainDir[k];
	    clusters[k-1] = clusters[k];
	    for(int p = 0; p < clusters[k].size(); p++){
	      label[clusters[k][p]] = k-1;
	    }
	  }
	  mainDir.pop_back();
	  clusters.pop_back();
	  merged = true;
	}
      }
    }
    if(!merged){break;}
  }
  // delete clusters with too few segments
  int counterMainDir = 0;
  const double cluster_size_threshold = 0.05*lines3D.size();
  for(int i = 0; i < mainDir.size(); i++){
    if(clusters[i].size() < cluster_size_threshold){
      for(int k = 0; k < clusters[i].size(); k++){
	label[clusters[i][k]] = -1;
      }
    }
    else{
      counterMainDir++;
    }
  }
  
  // converts lines parameters using cayley representation
  vector<double> lines_ceres(3*lines3D.size(), 0);
  vector<int> idx(lines3D.size(), -1);
  for(int i = 0; i < lines3D.size(); i++){
    for(int k = 0; k < 3; k++){
      lines_ceres[3*i+k] = lines3D[i].p1[k];
    }
  }
    
  // line reprojection constraints
  int counter = 0;
  for(int i = 0; i < lines3D.size(); i++){
    // if no special direction, discard
    int iDir = label[i];
    if(iDir == -1){continue;}
    counter ++;

    // line reprojection constraints 
    for(int k = 0; k < lines3D[i].proj_ids.size(); k++){
      int i_cam = lines3D[i].cam_ids[k];
      int i_proj = lines3D[i].proj_ids[k];

      ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<Residual_Line_Reprojection_Fixed_Direction, 2, 3, 3, 4>(
	  new Residual_Line_Reprojection_Fixed_Direction(lines[i_cam][i_proj], mainDir[iDir], K[i_cam], cofK[i_cam]));
      
      problem.AddResidualBlock(cost_function, p_LossFunction, &poses[6*i_cam], &poses[6*i_cam+3], &lines_ceres[4*i]);

      usedLine[i] = true;
      usedPose[i_cam] = true;
    }
  }
  cout << "nb of fixed direction constraints: " << counter << endl;
  cout << "nb of main directions: " << counterMainDir << "/" << mainDir.size() << endl;
  
  // define some variable as constants
  {
    // for pose
    for(int i = 0; i < nViews; i++){
      if(!usedPose[i]){continue;}
      
      problem.SetParameterBlockConstant(&poses[6*i]);
      problem.SetParameterBlockConstant(&poses[6*i+3]);
    }
  }
  
  // CERES options
  startCeresBA(problem, false); 
  
  // recover final lines
  for(int i = 0; i < lines3D.size(); i++){    
    int iDir = label[i];
    if(iDir == -1){
      lines3D[i].p1 = lines3D[i].p2 = Vec3(0,0,0);
      continue;
    }
    lines3D[i].direction = mainDir[iDir];
    Vec3 p;
    for(int k = 0; k < 3; k++){
      p[k] = lines_ceres[3*i+k];
    }
    double oldDis = (lines3D[i].p1-lines3D[i].p2).norm();
    Vec3 oldP1 = lines3D[i].p1;
    Vec3 oldP2 = lines3D[i].p2;
    lines3D[i].p1 = p + (lines3D[i].p1 - p).dot(lines3D[i].direction)*lines3D[i].direction;
    lines3D[i].p2 = p + (lines3D[i].p2 - p).dot(lines3D[i].direction)*lines3D[i].direction;
    double newDis = (lines3D[i].p1-lines3D[i].p2).norm();
    Vec3 newP1 = lines3D[i].p1;
    Vec3 newP2 = lines3D[i].p2;
    double disMid = max((oldP1-newP1).norm(),(oldP2-newP2).norm());
    if(newDis > 1.2*oldDis || newDis < 0.8*oldDis || disMid > 0.2*oldDis 
      || (lines3D[i].p1 - centroid).norm() > ray || (lines3D[i].p2 - centroid).norm() > ray ){
      lines3D[i].p1 = lines3D[i].p2 = Vec3(0,0,0);
    }
  }
  
  // delete planes that do not correspond to mainDir
  vector<Vec3> allowedDir;
  for(int i = 0; i < mainDir.size(); i++){
    for(int j = 0; j < i; j++){
      allowedDir.push_back((CrossProductMatrix(mainDir[i])*mainDir[j]).normalized());
    }
  }
  for(int i = 0; i < planes.size(); i++){
    bool found = false;
    for(int j = 0; j < allowedDir.size() && !found; j++){
      if(fabs(allowedDir[j].dot(planes[i].normal)) > same_direction_threshold){
	found = true;
	planes[i].normal = allowedDir[j];
      }
    }
    if(!found){
      for(int j = i; j < planes.size()-1; j++){
	planes[i] = planes[i+1];
      }
      planes.pop_back();
    }
  }
  
  // update line information for planes
  const double orthogonal_threshold = cos(75*M_PI/180);
  for(int i = 0; i < planes.size(); i++){
    planes[i].lines3D.clear();
    for(int j = 0; j < lines3D.size(); j++){
      if(lines3D[j].p1 == lines3D[j].p2){continue;}
      
      double dotprod = fabs(lines3D[j].direction.dot(planes[i].normal));
      if(dotprod > orthogonal_threshold){continue;}
      
      double dist = fabs((0.5*(lines3D[j].p1 + lines3D[j].p2) - planes[i].centroid).dot(planes[i].normal));
      if(dist > plane_distance_threshold){continue;}
      
      planes[i].lines3D.insert(j);
    }
    if(planes[i].lines3D.size() < 2){
      for(int j = i+1; j < planes.size(); j++){
	planes[j-1] = planes[j];
      }
      planes.pop_back();
      i--;
    }
  }
  
  // merge planes that are close to each other
  while(true){
    bool merged = false;
    for(int i = 0; i < planes.size() && !merged; i++){
      for(int j = 0; j < i && !merged; j++){
	// search planes with same normal
	if(fabs(planes[i].normal.dot(planes[j].normal)) < 0.95){
	  continue;
	}
	
	// search planes that are close to each other in terms of distance along normal
	double distance = fabs((planes[i].centroid - planes[j].centroid).dot(planes[i].normal));
	if(distance > plane_distance_threshold){
	  continue;
	}
	
	for(set<int>::iterator it = planes[i].lines3D.begin(); it != planes[i].lines3D.end(); it++){
	  planes[j].lines3D.insert(*it);
	}
	planes[j].computeCentroid(lines3D);
	merged = true;
	
	for(int k = i+1; k < planes.size(); k++){
	  planes[k-1] = planes[k];
	}
	planes.pop_back();
      }
    }
    if(!merged){break;}
  }
  
  for(int i = 0; i < planes.size(); i++){
    planes[i].computePlane(lines3D);
  }
}
