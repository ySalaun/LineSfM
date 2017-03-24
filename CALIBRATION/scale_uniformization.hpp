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

#ifndef SCALE_UNIFORMIZATION_HPP
#define SCALE_UNIFORMIZATION_HPP

#include "interface.hpp"
#include "openMVG/numeric/numeric.h"

#define ACRANSAC
#define PLANE_RECO
const double thresh_rsc = 2;
const double coplanar_orthogonal_thresh = cos(89.75*M_PI/180);
const double degenerate_case_thresh = sin(0.25*M_PI/180);

using namespace std;

class TranslationNormAContrario
{
  // constant parameters (mainly for research)
  const bool triplets_points_cts = true;
  const bool triplets_lines_cts = true;
  const bool coplanar_cts = true;
  
  const float epsilon = 1e-20; // to avoid null value inside logarithms
  const int n_neighbours = 10; // number of coplanar neighbours
  
  // parameters for NFA computation
  int nPoints, nLines, nCopPairs, nCopLines, nFeatures;
  map<int, vector<int>> lineToCopPair;
  
  // logNFA precomputation
  double logalpha0_point_line, logalpha0_point_point, logalpha0_point_point_without_K;
  vector<double> logc_n_points, logc_n_lines, logc_n_cop;
  
  // pose parameters
  int i0, i1, i2;
  vector<openMVG::Mat3> K, Ktinv;
  openMVG::Mat3 R01, R12;
  openMVG::Vec3 t01, t12, t10;
  
  // constraints
  vector<ClusterPlane> candidate_planes;
  Lines l01, l12;
  Triplets line_triplets, point_triplets;
  
public:
  bool verbose = true;
  int nConstraints;
  double nb_points, nb_lines, nb_lines_inliers, nb_points_inliers, nb_cop, nb_cop_inliers, nb_cop_pair;
  // initialize class and precompute logNFA and constraints
  TranslationNormAContrario(const PicturesSegments &lines, const PicturesMatches &matches_lines, 
			    const PicturesPoints &points, const PicturesMatches &matches_points,
			    const vector<Pose> &globalPoses, const PicturesRelativePoses &relativePoses, 
			    const int iPicture, const double imDimension, const vector<openMVG::Mat3> &gt_K, const bool closure);
  
  // process the estimation+refinement of translation ratio with a contrario method
  double process(PicturesSegments &lines, const PicturesMatches &matches_lines, 
		 PicturesPoints &points, const PicturesMatches &matches_points,
		 const vector<Pose> &globalPoses, 
		 vector<Plane> &planes, Triplets &final_triplets,  
		 vector<ClusterPlane> &clusters, vector<ClusterPlane> &coplanar_cts,
		 FEATURE_TYPE &chosen_ratio);

private:
  // compute log residuals for constraints
  vector<ErrorFTypeIndex> logNFAcoplanar(const double t_norm, const PicturesSegments &lines, const vector<Pose> &globalPoses, vector<ErrorFTypeIndex> &log_residuals_coppairs, const vector<bool> &cp);
  vector<ErrorFTypeIndex> logNFAlines(const openMVG::Vec3 &C2, const double t_norm, const vector<Pose> &globalPoses, const PicturesSegments &lines, const vector<bool> &lt);
  vector<ErrorFTypeIndex> logNFApoints(const openMVG::Vec3 &C2, const openMVG::Vec3 &C0, const vector<Pose> &globalPoses, const PicturesPoints &points, const vector<bool> &pt);
  
  // estimate the translation ratio
  double computeTranslationRatio(const PicturesSegments &lines, const PicturesPoints &points,
				 const vector<Pose> &globalPoses, vector<FTypeIndex> &inliers, 
				 FEATURE_TYPE &chosen_ratio);
  // refine the translation ratio
  double refineTranslationRatio(double translation_norm, const vector<FTypeIndex> &inliers, 
				PicturesSegments &lines, const PicturesMatches &matches_lines, 
				PicturesPoints &points, const PicturesMatches &matches_points,
				const vector<Pose> &globalPoses, vector<Plane> &planes, vector<ClusterPlane> &clusters);
};

Point3D triangulate_point(const vector<openMVG::Mat3> &R, const vector<openMVG::Vec3> &t, const vector<Point> &p);
Line3D triangulate_line(const vector<openMVG::Mat3> &R, const vector<openMVG::Vec3> &C, const vector<Segment> &l);

vector<Point3D> triangulate_points(const PicturesPoints &points, const vector<Pose> &globalPoses, const Triplets &triplets);
void computePlanes(PicturesSegments &lines, vector<Line3D> &lines3D, vector<Plane> &planes, const vector<Pose> &globalPoses);
vector<Plane> computePlanes(const vector<int> &coplanar_cts, vector<Line3D> &lines3D, const vector<Pose> &globalPoses);
  
// do not take into account inliers
vector<Point3D> triangulate_points(const PicturesPoints &points, const PicturesMatches &matches, const vector<Pose> &globalPoses);
vector<Line3D> triangulate_lines(const PicturesSegments &lines, const PicturesMatches &matches, const vector<Pose> &globalPoses);
vector<Line3D> triangulate_lines(PicturesSegments &lines, const PicturesMatches &matches, const vector<Pose> &globalPoses, const Triplets &triplets, 
				 const vector<CopCts> &cop_cts, vector<int> &coplanar_cts);
vector<CopCts> filterCopCts(PicturesSegments &lines, const PicturesMatches &matches, 
			    const vector<Pose> &globalPoses, const vector<CopCts> &cop_cts);
#endif