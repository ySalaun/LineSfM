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

#include "hybrid_essential.hpp"
#include "refinement.hpp"
#include "miscellaneous.hpp"

using namespace openMVG;

/*=================== NFA PRE-COMPUTATION ===================*/
inline
void addToMap(map<int, vector<int>> &lineToPPair, const int key, const int val){
  map<int, vector<int>>::iterator it = lineToPPair.find(key);
  if(it == lineToPPair.end()){
    lineToPPair.insert(pair<int, vector<int>>(key, vector<int>(1, val)));
  }
  else{
    it->second.push_back(val);
  }
}

void mapLinesToPPairs(const ParallelConstraints &ppairs, map<int, vector<int>> &lineToPPair){  
  for(int i = 0; i < ppairs.size(); i++){
    int li = ppairs[i].first.li;
    int lj = ppairs[i].first.lj;
   
    addToMap(lineToPPair, li, i);
    addToMap(lineToPPair, lj, i);
  }
}

HybridACRANSAC::HybridACRANSAC(const PointConstraints &points, const ParallelConstraints &ppairs, const int w, const int h, const double r, const bool v){
  acransac = (r < 0);
  ransac_threshold = r;
  verbose = v;
  
  nSamples = 6;
  nSamplesPoints = 5;
  nSamplesLines = 4;
  
  // compute nb of features  
  nPoints = points.size();
  nPPairs = ppairs.size();
    
  mapLinesToPPairs(ppairs, lineToPPair);
  nLines = lineToPPair.size();
  
  nFeatures = nPoints + nLines;

  // precompute logNFA
  loge0 = log10(double(nFeatures - nSamples));
  makelogcombi_n(nFeatures, logc_n);
  makelogcombi_k(nSamples, nFeatures, logc_k);
  
  // version for separated NFA
  loge0_points = log10(double(nPoints - nSamplesPoints));
  makelogcombi_n(nPoints, logc_n_points);
  makelogcombi_k(nSamplesPoints, nPoints, logc_k_points);
  loge0_lines = log10(double(nLines - nSamplesLines));
  makelogcombi_n(nLines, logc_n_lines);
  makelogcombi_k(nSamplesLines, nLines, logc_k_lines);

  // for points error
  double D = sqrt(w*w + h*h); // picture diameter
  double A = w*h;             // picture area
  logalpha0_points = log10(2.0*D/A);
}

/*=================== NFA COMPUTATION ===================*/
inline
float costPoints(const PointConstraint &point, const Pose &pose){ 
  // angular distance
  Vec3 nP = (CrossProductMatrix(pose.first*point.first)*pose.second);
  Vec3 nQ = (CrossProductMatrix(point.second)*pose.second);
  
  double dist = fabs(nP.normalized().dot(nQ.normalized()));

  if(dist != dist){
   return 1.0;
  }
  
  // cost is 1 - cos(angle) = 1 - dot_product
  return 1-min(dist,1.0);
}

inline
float costParallelLines(const ParallelConstraint &ppair, const Pose &pose){
  // metric that is null when using the parallel lines R estimator
  Vec3 VP0 = pose.first * ppair.first.vp;
  Vec3 VP1 = ppair.second.vp;

  double dist = fabs(VP0.dot(VP1));
      
  if(dist != dist){
   cout << "nan value in costParallelLines function" << endl;
   int pause;
   cin >> pause;
   return 1.0;
  }
  
  // cost is 1 - cos(angle) = 1 - |dot_product|
  return 1-min(dist,1.0);
}

vector<ErrorFTypeIndex> HybridACRANSAC::computeLogResiduals(const PointConstraints &points, const ParallelConstraints &ppairs, const Pose &pose, vector<ErrorFTypeIndex> &log_residuals_ppairs){
  vector<ErrorFTypeIndex> log_residuals(nFeatures);
  log_residuals_ppairs.resize(nPPairs);
  
  // points residuals
  Mat3 RtCross = CrossProductMatrix(pose.first.transpose()*pose.second);
  for (int i = 0; i < points.size(); ++i)  {
    double error = costPoints(points[i], pose);
    log_residuals[i] = ErrorFTypeIndex(log10(error + epsilon), FTypeIndex(POINT, i));
  }
  
  // parallel lines residuals
  for(int i = 0; i < nPPairs; ++i){    
    double error = costParallelLines(ppairs[i], pose);
    log_residuals_ppairs[i] = ErrorFTypeIndex(log10(error + epsilon) , FTypeIndex(PARALLEL_PAIR, i));  
  }

  // single lines residuals
  int idx = nPoints;
  for(map<int, vector<int>>::iterator it = lineToPPair.begin(); it != lineToPPair.end(); it++, idx++){
    log_residuals[idx] = ErrorFTypeIndex(std::numeric_limits<double>::infinity(), FTypeIndex(LINE, it->first));  
    for(int i = 0; i < it->second.size(); i++){
      log_residuals[idx].first = min(log_residuals[idx].first, log_residuals_ppairs[it->second[i]].first);
    }
  }
  
  // sort log nfa for best nfa computation
  sort(log_residuals.begin(), log_residuals.end());
  sort(log_residuals_ppairs.begin(), log_residuals_ppairs.end());
  
  return log_residuals;
}

void HybridACRANSAC::computeLogResiduals(const PointConstraints &points, const ParallelConstraints &ppairs, const Pose &pose, 
					 vector<ErrorFTypeIndex> &log_residuals_ppairs, vector<ErrorFTypeIndex> &log_residuals_points, vector<ErrorFTypeIndex> &log_residuals_lines){
  log_residuals_points.resize(nPoints);
  log_residuals_lines.resize(nLines);
  log_residuals_ppairs.resize(nPPairs);
  
  // points residuals
  Mat3 RtCross = CrossProductMatrix(pose.first.transpose()*pose.second);
  for (int i = 0; i < points.size(); ++i)  {
    double error = costPoints(points[i], pose);
    log_residuals_points[i] = ErrorFTypeIndex(log10(error + epsilon), FTypeIndex(POINT, i));
  }
  
  // parallel lines residuals
  for(int i = 0; i < nPPairs; ++i){    
    double error = costParallelLines(ppairs[i], pose);
    log_residuals_ppairs[i] = ErrorFTypeIndex(log10(error + epsilon) , FTypeIndex(PARALLEL_PAIR, i));  
  }

  // single lines residuals
  int idx = 0;
  for(map<int, vector<int>>::iterator it = lineToPPair.begin(); it != lineToPPair.end(); it++, idx++){
    log_residuals_lines[idx] = ErrorFTypeIndex(std::numeric_limits<double>::infinity(), FTypeIndex(LINE, it->first));  
    for(int i = 0; i < it->second.size(); i++){
      log_residuals_lines[idx].first = min(log_residuals_lines[idx].first, log_residuals_ppairs[it->second[i]].first);
    }
  }
  
  // sort log nfa for best nfa computation
  sort(log_residuals_points.begin(), log_residuals_points.end());
  sort(log_residuals_lines.begin(), log_residuals_lines.end());
  sort(log_residuals_ppairs.begin(), log_residuals_ppairs.end());
}


// compute best NFA and respective inliers
double HybridACRANSAC::bestNFA(const vector<ErrorFTypeIndex> &log_residuals, size_t &kInliers){
  double bestNFA = std::numeric_limits<double>::infinity();

  for(int k = nSamples+1; k <= nFeatures; k++){
    double NFA = loge0 + logc_n[k] + logc_k[k] + log_residuals[k-1].first*(double)(k-nSamples) ;
    if(NFA < bestNFA){
      bestNFA = NFA;
      kInliers = k;
    }
  }
  return bestNFA;
}

double HybridACRANSAC::bestNFA_points(const vector<ErrorFTypeIndex> &log_residuals, size_t &kInliers){
  double bestNFA = std::numeric_limits<double>::infinity();

  for(int k = nSamplesPoints+1; k <= nPoints; k++){
    double NFA = loge0 + logc_n[k] + logc_k[k] + log_residuals[k-1].first*(double)(k-nSamplesPoints) ;
    if(NFA < bestNFA){
      bestNFA = NFA;
      kInliers = k;
    }
  }
  return bestNFA;
}

double HybridACRANSAC::bestNFA_lines(const vector<ErrorFTypeIndex> &log_residuals, size_t &kInliers){
  double bestNFA = std::numeric_limits<double>::infinity();

  for(int k = nSamplesLines+1; k <= nLines; k++){
    double NFA = loge0 + logc_n[k] + logc_k[k] + log_residuals[k-1].first*(double)(k-nSamplesLines) ;
    if(NFA < bestNFA){
      bestNFA = NFA;
      kInliers = k;
    }
  }
  return bestNFA;
}

/*=================== POSE ESTIMATION ===================*/

inline
bool randomPPair(const ParallelConstraints &ppairs, int &i1, int &i2){
  // first pair is easy to find
  i1 = rand()%(ppairs.size());
  int vp1 = ppairs[i1].first.vp_idx;
  
  // second need to select the pairs belonging to other vps
  // TODO improve this selection it might take too much time...
  vector<int> indexes;
  for(int i = 0; i < ppairs.size(); i++){
    if(ppairs[i].first.vp_idx != vp1){
      indexes.push_back(i);
    }
  }
  
  if(indexes.size() == 0){
    i1 = i2 = -1;
    return false;
  }
  
  i2 = rand()%indexes.size();
  return true;
}

inline
Mat3 rotFrom2vecs(const Vec3* u, const Vec3* v){
  Mat3 B;
  B.setZero();

  for(int i = 0; i < 2; i++){   
    B += v[i]*u[i].transpose();
  }
  
  Eigen::JacobiSVD<Mat> svd(B, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Mat3 Rsvd = svd.matrixU()*svd.matrixV().transpose();
  
  if(Rsvd.determinant() < 0){
    Mat3 S;
    for(unsigned int j = 0; j < 3; j++){
      for(unsigned int k = 0; k < 3; k++){
	S(j,k) = (j==k)? ((j==2)? -1 : 1) : 0; 
      }
    }
    Rsvd = svd.matrixU()*S*svd.matrixV().transpose();
  }
  
  return Rsvd;
}
 
const double thresh_same_vps = sin(10*M_PI/180.0);
bool rotationFromParallelLines(const ParallelConstraint &pp1, const ParallelConstraint &pp2, Mat3 &R){
  Vec3 u[2] = {pp1.first.vp, pp2.first.vp};
  Vec3 v[2] = {pp1.second.vp, pp2.second.vp};
  
  // check if VPs are different enough 
  // TODO try to find a faster way
  if((CrossProductMatrix(u[0])*u[1]).norm() < thresh_same_vps || (CrossProductMatrix(v[0])*v[1]).norm() < thresh_same_vps){
    return false;
  }
  double e0[4] = {1.0,  1.0, -1.0, 1.0};
  double e1[4] = {1.0, -1.0,  1.0, -1.0};
  
  for(unsigned int k = 0; k < 4; k++){
    v[0] *= e0[k];
    v[1] *= e1[k];
    R = rotFrom2vecs(u, v);
    
    if(R.trace() > 1.0){    
      return true;
    }
  }
  return false;
}


bool solveFourParallels(const ParallelConstraints &ppairs, Pose &pose){
  if(ppairs.size() < 2){return false;}
  
  // select randomly two pairs of parallel pairs
  // avoid two pairs with similar VP
  int i1, i2;
  if(!randomPPair(ppairs, i1, i2)){ return false;}
  // compute rotation
  return rotationFromParallelLines(ppairs[i1], ppairs[i2], pose.first);
}


bool solveTwoPoints(const PointConstraints &points, Pose &pose){
  if(points.size() < 2){return false;}
  
  // random sampling for points
  int i0 = rand()%(points.size());
  int i1 = (i0 + 1 + rand()%(points.size()-1))%(points.size());

  // compute translation
  Mat M(2, 3), X(2,1);
  
  Vec3 Rp0 = (pose.first*points[i0].first);
  Vec3 q0 = points[i0].second;
  Vec3 RpCrossQ0 = (CrossProductMatrix(Rp0)*q0).normalized();
  
  Vec3 Rp1 = (pose.first*points[i1].first);
  Vec3 q1 = points[i1].second;
  Vec3 RpCrossQ1 = (CrossProductMatrix(Rp1)*q1).normalized();
  
  pose.second = (CrossProductMatrix(RpCrossQ0)*RpCrossQ1).normalized();
  return true;

  for(int j = 0; j < 3; j++){
    M(0, j) = RpCrossQ0[j];
    M(1, j) = RpCrossQ1[j];
  }
  X.setZero();
  
  // compute SVD to mix information
  Eigen::JacobiSVD<Mat> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);

  pose.second = svd.matrixV().col(2).normalized();

  return true;
}

// compute a random sublist of another list
void getRandList(
  const int p_nb,
  const int p_maxRange,
  vector<int> &o_randList){

    //! Initializations
    o_randList.resize(0);
    o_randList.clear();

    //! list with all possibilities
    vector<int> list(p_maxRange);
    for(int n = 0; n < p_maxRange; n++) {
      list[n] = n;
    }

    //! find the p_nb random numbers
    for(int n = 0; n < p_nb && list.size() > 0; n++){
      const int index = rand() % list.size();
      o_randList.push_back(list[index]);
      list.erase(list.begin() + index);
    }
}

bool solveFivePoints(const PointConstraints &points, Pose &pose){
  const int nSample = 5;
  if(points.size() < 5){return false;}
  
  // randomly select 5 points
  vector<int> random_sample;
  getRandList(nSample, points.size(), random_sample);
  
  // store into matrix form for openMVG function
  Mat xL = Mat(2, nSample);
  Mat xR = Mat(2, nSample);
  for (size_t i = 0; i < nSample; ++i)  {
    Vec3 pL = points[random_sample[i]].first;
    Vec3 pR = points[random_sample[i]].second;
    for(int k = 0; k < 2; k++){
      xL(k, i) = pL[k]/pL[2];
      xR(k, i) = pR[k]/pR[2]; 
    }
  }
  vector<Mat3> essential_matrix;
  openMVG::essential::kernel::FivePointSolver::Solve(xL, xR, &essential_matrix);
  
  // TODO maybe a better way of doing it ?
  Mat3 E = essential_matrix[0];
  if(!estimate_Rt_fromE(xL, xR, E, &pose.first, &pose.second)){return false;}
  if(pose.first.trace() < 1.0){return false;}
  pose.second.normalize();
  return true;
}

Pose HybridACRANSAC::computeRelativePose(const PointConstraints &points, const ParallelConstraints &ppairs, vector<FTypeIndex> &inliers){  
  // Output parameters
  double minNFA = std::numeric_limits<double>::infinity();  double maxInliers = 0;
  bool usedLines;
  Pose finalPose;
  // TODO ADD real AC RSC pipeline with 10% iter for last optim ?

  // AC RANSAC MAIN PIPELINE
  for (size_t iter = 0; iter < nIterRansac; ++iter) {
    Pose poseEstimated;
    
    // Estimate the relative pose with:
    // - 2 pairs of parallel lines and their matches (R) and 2 points and their matches (t)
    // - 5 points and their matches (E)
    if(iter%2 == 1){
      if(!solveFourParallels(ppairs, poseEstimated)){continue;}
      if(!solveTwoPoints(points, poseEstimated)){continue;}
    }
    else{
      if(!solveFivePoints(points, poseEstimated)){continue;}
    }
    // compute logNFA
    vector<ErrorFTypeIndex> log_residuals_ppairs;
    vector<ErrorFTypeIndex> log_residuals, log_residuals_points, log_residuals_lines;
    
    if(separatedNFA){
      computeLogResiduals(points, ppairs, poseEstimated, log_residuals_ppairs, log_residuals_points, log_residuals_lines);
    }
    else{
      log_residuals = computeLogResiduals(points, ppairs, poseEstimated, log_residuals_ppairs);
    }

    // compute best logNFA
    size_t kInliers, kInliersPoints, kInliersLines;
    kInliers = kInliersPoints = kInliersLines = 0;
    double nfa = 0;
    if(separatedNFA){
      if(!acransac){
	const double thresh = log10(1 - cos(ransac_threshold*M_PI/180));
	for(int k = 0; k < log_residuals_points.size(); k++){
	  if(log_residuals_points[k].first > thresh){
	    nfa = 1-k;
	    break;
	  }
	}
	for(int k = 0; k < log_residuals_lines.size(); k++){
	  if(log_residuals_lines[k].first > thresh){
	    nfa += 1-k;
	    break;
	  }
	}
      }
      else{
	nfa += bestNFA_points(log_residuals_points, kInliersPoints); 
	nfa += bestNFA_lines(log_residuals_lines, kInliersLines); 
      }
    }
    else{
      if(!acransac){
	const double thresh = log10(1 - cos(ransac_threshold*M_PI/180));
	for(int k = 0; k < log_residuals.size(); k++){
	  if(log_residuals[k].first > thresh){
	    nfa = 1-k;
	    break;
	  }
	}
      }
      else{
	nfa += bestNFA(log_residuals, kInliers); 
      }
    }
    
    // Update best model if needed
    if (nfa < minNFA ){    
      usedLines = (iter%2 == 1);
      // update nfa
      minNFA = nfa;
      
      // update inliers
      inliers.clear();
      
      if(separatedNFA){
	// -points
	for (size_t i = 0; i < kInliersPoints; ++i){
	  inliers.push_back(log_residuals_points[i].second);
	}

	// -parallel pairs
	const double thresh_log_residual = log_residuals_points[kInliersPoints-1].first;// log_residuals_lines[kInliersLines-1].first;
	for (size_t i = 0; i < log_residuals_ppairs.size(); ++i){
	  if(log_residuals_ppairs[i].first > thresh_log_residual){break;}
	  inliers.push_back(log_residuals_ppairs[i].second);
	}
      }
      else{
	if(!acransac){
	  const double thresh = log10(1 - cos(ransac_threshold*M_PI/180));
	  for(int k = 0; k < log_residuals.size(); k++){
	    if(log_residuals[k].first < thresh){
	      if(log_residuals[k].second.first == POINT){
		inliers.push_back(log_residuals[k].second);
	      }
	    }
	    else{
	      break; 
	    }
	  }
	  for (size_t i = 0; i < log_residuals_ppairs.size(); ++i){
	    if(log_residuals_ppairs[i].first > thresh){break;}
	    inliers.push_back(log_residuals_ppairs[i].second);
	  }
	}
	else{
	  // -points
	  for (size_t i = 0; i < kInliers; ++i){
	    if(log_residuals[i].second.first == POINT){
	      inliers.push_back(log_residuals[i].second);
	    }
	  }

	  // -parallel pairs
	  const double thresh_log_residual = log_residuals[kInliers-1].first;
	  for (size_t i = 0; i < log_residuals_ppairs.size(); ++i){
	    if(log_residuals_ppairs[i].first > thresh_log_residual){break;}
	    inliers.push_back(log_residuals_ppairs[i].second);
	  }
	}
      }
      
      finalPose = poseEstimated;
      
      if(verbose){
	cout << "=== BETTER MODEL FOUND ===" << endl;
	cout << "NFA: " << nfa << endl
	     << "#inliers: " << inliers.size() << endl
	     << " - points: " << kInliersPoints << "/" << nPoints << endl
	     << " - lines: " << kInliersLines << "/" << nLines << endl
	     << " - parallel pairs: " << inliers.size()-kInliersPoints << "/" << ppairs.size() << endl;
      }
    }
  }

  if(verbose){
    cout << "final kept hypothesis: " << ((usedLines)? "LINES" : "POINTS") << endl;
  }
  
  // return after non linear refinement
  return refinedPose(points, ppairs, finalPose, inliers, verbose);
}