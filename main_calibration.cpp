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
#include "hybrid_essential.hpp"
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
  int gt_type = 2;
  bool useOpenMVG = true;
  double ransac = -1;
  bool verbose = false;
  
  // required
  cmd.add( make_option('d', dirPath, "dirPath") );
  cmd.add( make_option('i', picList, "inputPic") );
    
  // optional
  cmd.add( make_option('c', consecutive, "consecutive") );
  cmd.add( make_option('l', close_loop, "closeLoop") );
  cmd.add( make_option('g', gt_type, "gt_type") );
  cmd.add( make_option('o', useOpenMVG, "use_openMVG") );
  cmd.add( make_option('r', ransac, "ransac") );
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
      << "[-g|--gt_type] ground truth type (0 = ONLY_K, 1 = ONLY_RELATIVE, [DEFAULT: 2 = GLOBAL]) \n"
      << "[-o|--use_openMVG] use openMVG point matches (default = TRUE) \n"
      << "[-r|--ransac] ransac threshold (default = -1 (a contrario) ) \n"
      << "[-v|--verbose] enable/disable messages (default = FALSE)\n"
      << endl;
      return EXIT_FAILURE;
  }
  dirPath += "/";
  
  vector<string> picName, picPath;
  readPictureFile(picList, picName, picPath);
  
  const int nPictures = picName.size();
  
  Mat im = imread(picPath[0], CV_LOAD_IMAGE_GRAYSCALE);
  const int wPic = im.cols, hPic = im.rows;
  
  // read K matrix and optionnaly ground truth
  const string ext = (gt_type == ONLY_RELATIVE)? "":".png.camera";
  GroundTruth gt(dirPath, picName, consecutive, close_loop, ext, static_cast<GT_TYPE>(gt_type));
  
  // read points/lines/vanishing_points
  cout << "LOAD DETECTIONS AND MATCHES" << endl;
  PicturesSegments segments(nPictures);
  PicturesPoints points(nPictures);
  PicturesVPs vpoints(nPictures);
  for(int i = 0; i < nPictures; i++){
    if(verbose){cout << "picture " << i << ": " << picName[i] << endl;}

    if(verbose){cout << " - read points" << endl;}
    if(useOpenMVG){
      points[i] = readPointsOpenMVG(dirPath, picName[i]);
    }
    else{
      points[i] = readPoints(dirPath, picName[i]);
    }
    
    if(verbose){cout << " - read segments" << endl;}
    segments[i] = readLines(dirPath, picName[i]);
    
    if(verbose){cout << " - read vanishing points" << endl;}
    vpoints[i] = readVanishingPoints(segments[i], dirPath, picName[i]);
  }
  
  // read points/lines matches
  PicturesMatches matches_points, matches_lines;
  for(int i = 0; i < nPictures; i++){
    for(int j = i; j < nPictures; j++){
      if(!isConsecutive(consecutive, close_loop, i, j, nPictures)){ continue;}
      if(verbose){cout << "pictures " << picName[i] << " and " << picName[j] << endl;}
      PicturePair imPair(i,j);
      
      if(verbose){cout << " - read point matches" << endl;}
      if(useOpenMVG){
	matches_points.insert(PictureMatches(imPair, readMatchesOpenMVG(dirPath, i, j, points[i].size())));
      }
      else{
	matches_points.insert(PictureMatches(PicturePair(i,j), readMatches(dirPath, picName[i], picName[j], POINT)));
      }
      
      if(verbose){cout << " - read line matches" << endl;}
      matches_lines.insert(PictureMatches(imPair, readMatches(dirPath, picName[i], picName[j], LINE)));    
    }
  }

  // normalize data
  cout << "COMPUTE POSES" << endl;
  clock_t processing_time = clock();
  normalize(points, segments, vpoints, gt.K, gt.Kinv);
  
  PicturesRelativePoses poses;
  for(int i = 0; i < nPictures; i++){
    for(int j = i; j < nPictures; j++){
      if(!isConsecutive(consecutive, close_loop, i, j, nPictures)){ continue;}
      PicturePair imPair(i,j);
      
      if(verbose){cout << "pictures " << picName[i] << " and " << picName[j] << endl;}
      
      // select only point matches
      vector<pair<int, int>> point_pairs, line_pairs;
      PointConstraints pointConstraints = selectPointMatches(points[i], points[j], matches_points.find(imPair)->second, point_pairs);
      
      // compute parallel pairs and vanishing points
      ParallelConstraints parConstraints = computeParallelPairs(segments[i], segments[j], matches_lines.find(imPair)->second, line_pairs);
      
      // compute relative pose
      // initialize AC RANSAC
      HybridACRANSAC HAC_RANSAC(pointConstraints, parConstraints, wPic, hPic, ransac, verbose);
      
      // compute pose
      vector<FTypeIndex> inliers;
      Pose pose = HAC_RANSAC.computeRelativePose(pointConstraints, parConstraints, inliers);      
      poses.insert(PictureRelativePoses(imPair, pose));
      savePose(pose, gt.Kinv[i], dirPath, picName[i], picName[j]);
      
      // save point inliers
      saveInliers(inliers, segments[i], segments[j], points[i], point_pairs, line_pairs, 
		  matches_lines.find(imPair)->second, dirPath, picName[i], picName[j]);  
    }
  }
  cout << "PROCESSED IN " << (clock() - processing_time) / float(CLOCKS_PER_SEC) << endl;
  
  if(gt_type != 0){
    gt.compareRelativePose(poses);
  }
  
  return 0;
}
