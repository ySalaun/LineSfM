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


#ifndef HYBRID_ESSENTIAL_HPP
#define HYBRID_ESSENTIAL_HPP

#include "interface.hpp"
#include "openMVG/numeric/numeric.h"
#include <boost/concept_check.hpp>

using namespace std;

class HybridACRANSAC
{
  // constant 
  const bool separatedNFA = true;
  const int nIterRansac = 5000;
  const float epsilon = 1e-20; // to avoid null value inside logarithms
  
  // parameters for NFA computation
  int nPoints, nLines, nPPairs, nFeatures, nSamples, nSamplesPoints, nSamplesLines;
  map<int, vector<int>> lineToPPair;
  
  // logNFA precomputation
  double loge0, loge0_points, loge0_lines, logalpha0_points;
  std::vector<double> logc_n, logc_k, logc_n_points, logc_k_points, logc_n_lines, logc_k_lines;
  
  // parameters
  bool acransac, verbose;
  double ransac_threshold;
  
public:
  // initialize class and precompute logNFA
  HybridACRANSAC(const PointConstraints &points, const ParallelConstraints &ppairs, const int w, const int h, const double r, const bool v);
  
  // estimate the relative pose
  Pose computeRelativePose(const PointConstraints &points, const ParallelConstraints &ppairs, vector<FTypeIndex> &inliers);
  
private:
  vector<ErrorFTypeIndex> computeLogResiduals(const PointConstraints &points, const ParallelConstraints &ppairs, const Pose &pose, vector<ErrorFTypeIndex> &log_residuals_ppairs);
  void computeLogResiduals(const PointConstraints &points, const ParallelConstraints &ppairs, const Pose &pose, 
			   vector<ErrorFTypeIndex> &log_residuals_ppairs, vector<ErrorFTypeIndex> &log_residuals_points, vector<ErrorFTypeIndex> &log_residuals_lines);
  double bestNFA(const vector<ErrorFTypeIndex> &log_residuals, size_t &kInliers);
  double bestNFA_points(const vector<ErrorFTypeIndex> &log_residuals, size_t &kInliers);
  double bestNFA_lines(const vector<ErrorFTypeIndex> &log_residuals, size_t &kInliers);
};

#endif