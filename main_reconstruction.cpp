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

#include "interface.hpp"
#include "detection.hpp"
#include "line_matching.hpp"
#include "scale_uniformization.hpp"
#include "refinement.hpp"
#include "cmdLine/cmdLine.h"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]){
  // Seed random function
  srand((unsigned int)(time(0)));
  
  // parse arguments
  CmdLine cmd;

  string dirPath;
  string picList;
  
  bool consecutive = true;
  bool close_loop = false;
  bool withRefinedMatching = false;
  bool multiscale = true;
  double segment_length_threshold = 0.01;
  bool verbose = false;
  
  // required
  cmd.add( make_option('d', dirPath, "dirPath") );
  cmd.add( make_option('i', picList, "inputPic") );
    
  // optional
  cmd.add( make_option('c', consecutive, "consecutive") );
  cmd.add( make_option('l', close_loop, "closeLoop") );
  cmd.add( make_option('r', withRefinedMatching, "refinedMatching") );
  cmd.add( make_option('m', multiscale, "multiscale") );
  cmd.add( make_option('t', segment_length_threshold, "threshold") );
  cmd.add( make_option('v', verbose, "verbose") );

  try {
      if (argc == 1) throw std::string("Invalid command line parameter.");
      cmd.process(argc, argv);
  } catch(const std::string& s) {
      std::cerr << "Usage: " << argv[0] << '\n'
      << "[-d|--dirPath] feature path]\n"
      << "[-i|--inputPic] list of pictures] \n"
      << "\n[Optional]\n"
      << "[-c|--consecutive] matches between consecutive pictures (default = TRUE) \n"
      << "[-l|--closeLoop] close the loop (default = FALSE) \n"
      << "[-r|--refinedMatching] refine matches with known relative poses (default = FALSE) \n"
      << "[-m|--multiscale] multiscale option (default = TRUE) [useful only with refined matching]\n"
      << "[-t|--threshold] threshold for segment length (default = 0% of image size) [useful only with refined matching]\n"
      << "[-v|--verbose] enable/disable messages (default = FALSE)\n"
      << endl;
      return EXIT_FAILURE;
  }
  dirPath += "/";
  
  vector<string> picName, picPath;
  readPictureFile(picList, picName, picPath);
  
  const int nPictures = picName.size();
  
  cv::Mat im = imread(picPath[0], CV_LOAD_IMAGE_GRAYSCALE);
  const int wPic = im.cols, hPic = im.rows;
  
  // read K matrix and optionnaly ground truth
  GroundTruth gt(dirPath, picName, consecutive, close_loop, ".png.camera", GLOBAL);
  
  // compute dimension of pictures (assuming all pictures of the same size)
  Mat image = imread(picPath[0], CV_LOAD_IMAGE_GRAYSCALE);
  const int imDimension = 0.5*(image.cols+image.rows);

  const string ext = "_refined";
  
  // read points/lines
  PicturesSegments segments(nPictures);
  PicturesPoints points(nPictures);
  PicturesVPs vpoints(nPictures);
  for(int i = 0; i < nPictures; i++){
    cout << "picture " << i << ": " << picName[i] << endl;

    cout << " - read points" << endl;
    points[i] = readPointsOpenMVG(dirPath, picName[i]);
    
    if(withRefinedMatching){
      Mat im = imread(picPath[i], CV_LOAD_IMAGE_GRAYSCALE);
      
      vector<Mat> imagePyramid = computeImagePyramid(im, multiscale);
      
      cout << " - segment detection" << endl;
      segments[i] = lsd_multiscale(imagePyramid, segment_length_threshold, multiscale);
      saveLines(segments[i], dirPath, picName[i] + ext);
      
      cout << " - compute descriptors" << endl;
      computeDescriptors(imagePyramid, segments[i]);
      saveDescriptors(segments[i], dirPath, picName[i] + ext);
    }
    else{
      cout << " - read segments" << endl;
      segments[i] = readLines(dirPath, picName[i]);
      
      cout << " - read descriptors" << endl;
      readDescriptors(segments[i], dirPath, picName[i]);
    }
  }
  
  // read points/lines matches/relative poses + vp inliers
  PicturesRelativePoses relativePoses;
  PicturesMatches matches_points, matches_lines;
  const double range = 0.6*imDimension;
  for(int i = 0; i < nPictures; i++){
    for(int j = i; j < nPictures; j++){
      if(!isConsecutive(consecutive, close_loop, i, j, nPictures)){ continue;}
      PicturePair imPair(i,j);
      cout << "pictures " << picName[i] << " and " << picName[j] << endl;
      
      cout << " - read relative pose" << endl;
      Pose pose = readPose(dirPath, picName[i], picName[j]);
      relativePoses.insert(PictureRelativePoses(imPair, pose));
      
      cout << " - read point matches" << endl;
      vector<int> mPts = readMatches(dirPath, picName[i], picName[j]+"_inliers", POINT);
      matches_points.insert(PictureMatches(imPair, mPts));
      
      if(withRefinedMatching){
	cout << " - matching pictures " << picName[i] << " and " << picName[j] << endl;
	openMVG::Mat3 F = gt.Kinv[j].transpose()*openMVG::CrossProductMatrix(pose.second)*pose.first*gt.Kinv[i];
	
	vector<int> currentMatch = computeMatches(segments[i], segments[j], range, F, true);
	matches_lines.insert(PictureMatches(imPair, currentMatch));
	saveMatches(currentMatch, dirPath, picName[i], picName[j] + ext, LINE);
      }
      else{
	cout << " - read line matches" << endl;
	vector<int> mLines = readMatches(dirPath, picName[i], picName[j], LINE);
	matches_lines.insert(PictureMatches(imPair, mLines));
      }
    }
  }
  if(close_loop){
    PicturePair imPair(0,nPictures-1);
    PicturePair imPairInvted(nPictures-1, 0);
    
    vector<int> tempMatches = matches_points.find(imPair)->second;
    vector<int> invtedMatchesPoint(points[nPictures-1].size(), -1);
    for(int k = 0; k < tempMatches.size(); k++){
      if(tempMatches[k] == -1){continue;}
      invtedMatchesPoint[tempMatches[k]] = k;
    }
    matches_points.insert(PictureMatches(imPairInvted, invtedMatchesPoint));
    
    tempMatches = matches_lines.find(imPair)->second;
    vector<int> invtedMatchesLines(segments[nPictures-1].size(), -1);
    for(int k = 0; k < tempMatches.size(); k++){
      if(tempMatches[k] == -1){continue;}
      invtedMatchesLines[tempMatches[k]] = k;
    }
    matches_lines.insert(PictureMatches(imPairInvted, invtedMatchesLines));
  }
  clock_t processing_time = clock();
  
  // normalize data
  normalize(points, segments, vpoints, gt.K, gt.Kinv);
  
  // compute global rotations  
  const int nGlobalPoses = (close_loop)? nPictures+1:nPictures;
  if(close_loop){
    Pose pose = relativePoses.find(PicturePair(0, nPictures-1))->second;
    Pose invtedPose;
    invtedPose.first = pose.first.transpose();
    invtedPose.second = invtedPose.first*(pose.second);
    relativePoses.insert(pair<PicturePair, Pose>(PicturePair(nPictures-1,0), invtedPose));
  }
  vector<Pose> globalPoses(nGlobalPoses);
  for(int i = 0; i < nGlobalPoses; i++){
    if(i == 0){
      globalPoses[i].first = openMVG::Mat3::Identity();
    }
    else{
      globalPoses[i].first = relativePoses.find(PicturePair(i-1, i%nPictures))->second.first * globalPoses[i-1].first;
    }
  }

  // initialize camera centers
  globalPoses[0].second = openMVG::Vec3(0,0,0);
  globalPoses[1].second = globalPoses[0].second - globalPoses[1].first.transpose()*relativePoses.find(PicturePair(0, 1))->second.second;
  
  vector<Plane> planes;
  Triplets triplets;
  vector<CopCts> cop_cts;
  for(int i = 1; i < nGlobalPoses-1; i++){
    const bool closure = i==nPictures-1;
    cout << "--------------------------------------" << endl;
    cout << "computing translation norm from picture " << i << " to " << i+1 << endl;
    Mat im = imread(picPath[i], CV_LOAD_IMAGE_COLOR);
    
    // compute translation norm
    TranslationNormAContrario tnac(segments, matches_lines, points, matches_points, globalPoses, relativePoses, i, imDimension, gt.K, closure);

    vector<ClusterPlane> clusters, coplanar_cts;
    FEATURE_TYPE chosen_ratio;
    double t_norm = tnac.process(segments, matches_lines, points, matches_points, globalPoses, planes, triplets, clusters, coplanar_cts, chosen_ratio);
    
    cout << "gt_ratio: seems incorrect " << (gt.centers[i] - gt.centers[i-1]).norm()/(gt.centers[i+1] - gt.centers[i]).norm() << endl;
    cout << "ratio found: " << t_norm*(globalPoses[i].second-globalPoses[i-1].second).norm() << endl;    
    
    // update global poses
    globalPoses[i+1].second = globalPoses[i].second - 1.0 / t_norm * globalPoses[i+1].first.transpose()*relativePoses.find(PicturePair(i, (i+1)%nPictures))->second.second;
    
    vector<int> match01 = matches_lines.find(PicturePair(i-1,i))->second;
    vector<int> match12 = matches_lines.find(PicturePair(i,i+1))->second;

    for(int k = 0; k < coplanar_cts.size(); k++){
      segments[i][coplanar_cts[k].proj_ids[0]].coplanar_cts.push_back(cop_cts.size());
      segments[i][coplanar_cts[k].proj_ids[1]].coplanar_cts.push_back(cop_cts.size());
      
      CopCts cop;
      cop.i_cam[0] = i-1;
      cop.i_cam[1] = cop.i_cam[2] = i;
      cop.i_cam[3] = i+1;
      int ms1 = -1;
      for(int j = 0; j < segments[i-1].size(); j++){
	if(match01[j] == coplanar_cts[k].proj_ids[0]){
	  ms1 = j;
	  break;
	}
      }
      cop.i_proj[0] = ms1;
      cop.i_proj[1] = coplanar_cts[k].proj_ids[0];
      cop.i_proj[2] = coplanar_cts[k].proj_ids[1];
      cop.i_proj[3] = match12[coplanar_cts[k].proj_ids[1]];
      cop_cts.push_back(cop);
    }
    
    saveClustersPicture(clusters, segments[i], im, dirPath, picName[i]);
  }
  
  if(close_loop){
    openMVG::Mat3 Identity = globalPoses[nGlobalPoses-1].first;
    openMVG::Vec3 zero = globalPoses[nGlobalPoses-1].second;
    double norm_sum = 0;
    for(int k = 0; k < globalPoses.size(); k++){
      norm_sum += (globalPoses[k].second).norm();
    }
    double rot_err = 0;
    for(int k = 0; k < 3; k++){
      rot_err += Identity(k,k);
    }
    rot_err = acos(fabs((1-rot_err)/2))*180/M_PI;

    cout << "CLOSURE ERROR" << endl;
    cout << "rotation: " << rot_err << endl;
    cout << "translation: " << zero.norm()/norm_sum*100 << endl;
    globalPoses.pop_back();
  }
  
  vector<Plane> null_planes;
  {
    vector<Point3D> points3D_local = triangulate_points(points, matches_points, globalPoses);
    vector<Line3D> lines3D_local = triangulate_lines(segments, matches_lines, globalPoses);
    saveMesh(points3D_local, lines3D_local, null_planes, globalPoses, dirPath, "dense_first_mesh");
  }
  
  vector<int> coplanar_cts;
  cop_cts = filterCopCts(segments, matches_lines, globalPoses, cop_cts);
  vector<Point3D> points3D = triangulate_points(points, globalPoses, triplets);
  vector<Line3D> lines3D = triangulate_lines(segments, matches_lines, globalPoses, triplets, cop_cts, coplanar_cts);
  
  cout << "######### BEFORE BA" << endl;
  planes = computePlanes(coplanar_cts, lines3D, globalPoses);
  saveMesh(points3D, lines3D, planes, globalPoses, dirPath, "first_mesh");

  cout << "# PLANES: " << planes.size() << endl;
  cout << "# POINTS: " << points3D.size() << endl;
  cout << "# LINES : " << lines3D.size() << endl;
  cout << "PROCESSED IN " << (clock() - processing_time) / float(CLOCKS_PER_SEC) << endl;  
  
  gt.compareGlobalPose(globalPoses, dirPath);
  bundleAdjustment(points3D, points, lines3D, segments, globalPoses, gt.K, coplanar_cts, true, verbose);
  gt.compareGlobalPose(globalPoses, dirPath);
  bundleAdjustment(points3D, points, lines3D, segments, globalPoses, gt.K, coplanar_cts, false, verbose);
  cout << "######### AFTER BA" << endl;
  gt.compareGlobalPose(globalPoses, dirPath);
  
  planes = computePlanes(coplanar_cts, lines3D, globalPoses);
  saveMesh(points3D, lines3D, planes, globalPoses, dirPath, "refined_mesh");
  {
    vector<Point3D> points3D_local = triangulate_points(points, matches_points, globalPoses);
    vector<Line3D> lines3D_local = triangulate_lines(segments, matches_lines, globalPoses);
    saveMesh(points3D_local, lines3D_local, null_planes, globalPoses, dirPath, "dense_refined_mesh");
    manhattanize(lines3D_local, segments, planes, globalPoses, gt.K);
    saveMesh(points3D_local, lines3D_local, planes, globalPoses, dirPath, "manhattanized_mesh");
  }
  
  
  // for Hofer reconstruction
  {
    GroundTruth gtCopy = gt;
    for(int i = 0; i < nPictures; i++){
      gtCopy.rotations[i] = globalPoses[i].first;
      gtCopy.centers[i] = globalPoses[i].second;
    }
    gtCopy.saveComputedPoseHofer(dirPath, picName);
  }
  
  // export results in picture format
  if(withRefinedMatching){
    for(int i = 0; i < nPictures; i++){
      Mat im1 = imread(picPath[i], CV_LOAD_IMAGE_COLOR);
      saveLinesPicture(segments[i], im1, dirPath, picName[i] + ext, false);
      for(int j = i; j < nPictures; j++){
	if(!isConsecutive(consecutive, close_loop, i, j, nPictures)){ continue;}
	Mat im2 = imread(picPath[j], CV_LOAD_IMAGE_COLOR);
	saveMatchesPicture(segments[i], segments[j], matches_lines.find(PicturePair(i,j))->second, im1, im2, dirPath, picName[i], picName[j] + ext, false);
      }
    }
  }
  
  return 0;
}
