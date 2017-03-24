/*----------------------------------------------------------------------------  
  This code is part of the following publication and was subject
  to peer review:
  "Multiscale line segment detector for robust and accurate SfM" by
  Yohann Salaun, Renaud Marlet, and Pascal Monasse
  ICPR 2016
  
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

#include "detection.hpp"
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
  bool withDetection = false;
  double segment_length_threshold = 0.02;
  bool multiscale = true;
  bool verbose = false;

  // required
  cmd.add( make_option('d', dirPath, "dirPath") );
  cmd.add( make_option('i', picList, "inputPic") );
    
  // optional
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
      << "[-m|--multiscale] multiscale option (default = TRUE)\n"
      << "[-t|--threshold] threshold for segment length (default = 0.02% of image size)\n"
      << "[-v|--verbose] enable/disable messages (default = FALSE)\n"
      << endl;
      return EXIT_FAILURE;
  }
  dirPath += "/";

  vector<string> picName, picPath;
  readPictureFile(picList, picName, picPath);
  const int nPictures = picName.size();
  
  // detect lines
  cout << "DETECTION" << endl;
  clock_t processing_time = clock();
  for(int i = 0; i < nPictures; i++){
    if(verbose){cout << "picture " << i << ": " << picName[i] << endl;}
    Mat im = imread(picPath[i], CV_LOAD_IMAGE_GRAYSCALE);
    vector<Mat> imagePyramid = computeImagePyramid(im, multiscale);
  
    vector<Segment> segments = lsd_multiscale(imagePyramid, segment_length_threshold, multiscale);
    saveLines(segments, dirPath, picName[i]);
  }
  cout << "PROCESSED IN " << (clock() - processing_time) / float(CLOCKS_PER_SEC) << endl;
  
  // export results in picture format
  cout << "SAVE PICTURES (CTRL+C IF NOT INTERESTED)" << endl;
  for(int i = 0; i < nPictures; i++){
    Mat im = imread(picPath[i], CV_LOAD_IMAGE_COLOR);
    vector<Segment> segments = readLines(dirPath, picName[i]);
    saveLinesPicture(segments, im, dirPath, picName[i], false);
  }
  
  return 0;
}