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
#include "detection.hpp"
#include "line_matching.hpp"
#include "interface.hpp"
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
  bool close_loop = true;
  bool withDetection = true;
  double segment_length_threshold = 0.02;
  bool multiscale = true;
  bool verbose = false;

  // required
  cmd.add( make_option('d', dirPath, "dirPath") );
  cmd.add( make_option('i', picList, "inputPic") );
    
  // optional
  cmd.add( make_option('c', consecutive, "consecutive") );
  cmd.add( make_option('l', close_loop, "closeLoop") );
  cmd.add( make_option('D', withDetection, "detection") );
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
      << "[-l|--closeLoop] close the loop (default = TRUE) \n"
      << "[-m|--multiscale] multiscale option (default = TRUE)\n"
      << "[-D|--detection] LSD detection made before (default = TRUE)\n"
      << "[-t|--threshold] threshold for segment length (default = 0.02% of image size)\n"
      << "[-v|--verbose] enable/disable messages (default = FALSE)\n"
      << endl;
      return EXIT_FAILURE;
  }
  dirPath += "/";

  vector<string> picName, picPath;
  readPictureFile(picList, picName, picPath);
  
  const int nPictures = picName.size();
  
  // compute dimension of pictures (assuming all pictures of the same size)
  Mat image = imread(picPath[0], CV_LOAD_IMAGE_GRAYSCALE);
  const int imDimension = 0.5*(image.cols+image.rows);
  
  // compute descriptors and optionally detect lines and vanishing points
  cout << "DETECT LINES AND COMPUTE DESCRIPTORS" << endl;
  clock_t processing_time = clock();
  PicturesSegments segments(nPictures);
  PicturesVPs vpoints(nPictures);
  for(int i = 0; i < nPictures; i++){
    if(verbose){cout << "picture " << i << ": " << picName[i] << endl;}
    Mat im = imread(picPath[i], CV_LOAD_IMAGE_GRAYSCALE);
    vector<Mat> imagePyramid = computeImagePyramid(im, multiscale);
  
    if(withDetection){
      if(verbose){cout << " - segment detection" << endl;}
      segments[i] = lsd_multiscale(imagePyramid, segment_length_threshold, multiscale);
      saveLines(segments[i], dirPath, picName[i]);
      
      if(verbose){cout << " - vanishing point detection" << endl;}
      vpoints[i] = computeVanishingPoints(im, segments[i]);
      saveVanishingPoints(vpoints[i], dirPath, picName[i]);
    }
    else{
      if(verbose){cout << " - read segments" << endl;}
      segments[i] = readLines(dirPath, picName[i]);
      
      if(verbose){cout << " - read vanishing points" << endl;}
      vpoints[i] = readVanishingPoints(segments[i], dirPath, picName[i]);
    }
      
    if(verbose){cout << " - compute descriptors" << endl;}
    computeDescriptors(imagePyramid, segments[i]);   
    saveDescriptors(segments[i], dirPath, picName[i]);
  }

  // match pictures
  cout << "COMPUTE MATCHES" << endl;
  PicturesMatches matches;
  const double range = 0.4*imDimension;
  for(int i = 0; i < nPictures; i++){
    for(int j = i; j < nPictures; j++){
      if(!isConsecutive(consecutive, close_loop, i, j, nPictures)){ continue;}
      if(verbose){cout << " - matching pictures " << picName[i] << " and " << picName[j] << endl;}
      vector<int> currentMatch = computeMatches(segments[i], segments[j], range);
      matches.insert(PictureMatches(PicturePair(i,j), currentMatch));
      saveMatches(currentMatch, dirPath, picName[i], picName[j], LINE);
    }
  }
  cout << "PROCESSED IN " << (clock() - processing_time) / float(CLOCKS_PER_SEC) << endl;

  // export results in picture format
  cout << "SAVE PICTURES (CTRL+C IF NOT INTERESTED)" << endl;
  for(int i = 0; i < nPictures; i++){
    Mat im1 = imread(picPath[i], CV_LOAD_IMAGE_COLOR);
    if(withDetection){
      saveLinesPicture(segments[i], im1, dirPath, picName[i], false);
    }
    for(int j = i; j < nPictures; j++){
      if(!isConsecutive(consecutive, close_loop, i, j, nPictures)){ continue;}
      Mat im2 = imread(picPath[j], CV_LOAD_IMAGE_COLOR);
      
      // TODO improve
      if(im1.rows != im2.rows){
	if(im1.rows < im2.rows){
	  double scale = double(im2.rows)/im1.rows;
	  Mat temp;
	  resize(im1, temp, im2.size(), scale, scale);
	  for(int k = 0; k < segments[i].size(); k++){
	    segments[i][k].upscale(scale);
	  }
	  im1 = temp;
	}
	else{
	  double scale = double(im1.rows)/im2.rows;
	  Mat temp;
	  resize(im2, temp, im1.size(), scale, scale);
	  for(int k = 0; k < segments[j].size(); k++){
	    segments[j][k].upscale(scale);
	  }
	  im2 = temp;
	}
      }
      
      saveMatchesPicture(segments[i], segments[j], matches.find(PicturePair(i,j))->second, im1, im2, dirPath, picName[i], picName[j], false);
      saveMatchesPicture(segments[i], segments[j], matches.find(PicturePair(i,j))->second, im1, im2, dirPath, picName[i], picName[j], true);
    }
  }
  
  return 0;
}