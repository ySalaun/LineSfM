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
#include "point_matching.hpp"
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
  bool verbose = false;

  // required
  cmd.add( make_option('d', dirPath, "dirPath") );
  cmd.add( make_option('i', picList, "inputPic") );
    
  // optional
  cmd.add( make_option('c', consecutive, "consecutive") );
  cmd.add( make_option('l', close_loop, "closeLoop") );
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
      << "[-k|--KVLD_ONLY] only matches with KVLD (default = TRUE) \n"
      << "[-v|--verbose] enable/disable messages (default = FALSE)\n"
      << endl;
      return EXIT_FAILURE;
  }
  dirPath += "/";
  
  vector<string> picName, picPath;
  readPictureFile(picList, picName, picPath);
  
  const int nPictures = picName.size();
  
  // detect and compute SIFT descriptors 
  cout << "DETECT POINTS" << endl;
  clock_t processing_time = clock();
  vector<Mat> descriptors(nPictures);
  PicturesSifts sift(nPictures);
  for(int i = 0; i < nPictures; i++){
    if(verbose){
      cout << "picture " << i << ": " << picName[i] << endl;
      cout << " - compute descriptors" << endl;
    }
    Mat im = imread(picPath[i], CV_LOAD_IMAGE_GRAYSCALE);
    computeDescriptors(im, sift[i], descriptors[i]);
    savePoints(sift[i], dirPath, picName[i]);
  }
  
  // match pictures
  cout << "COMPUTE MATCHES" << endl;
  PicturesMatches matches;
  for(int i = 0; i < nPictures; i++){
    Mat im1 = imread(picPath[i], CV_LOAD_IMAGE_COLOR);
    for(int j = i; j < nPictures; j++){
      if(!isConsecutive(consecutive, close_loop, i, j, nPictures)){ continue;}
      if(verbose){cout << "matching pictures " << picName[i] << " and " << picName[j] << endl;}
      
      if(verbose){cout << " - brute matching" << endl;}
      vector<int> currentMatch = bruteMatching(descriptors[i], descriptors[j]);
      
      if(verbose){cout << " - kvld filtering" << endl;}
      Mat im2 = imread(picPath[j], CV_LOAD_IMAGE_COLOR);
      
      matches.insert(PictureMatches(PicturePair(i,j), bruteMatching(descriptors[i], descriptors[j])));
      saveMatches(currentMatch, dirPath, picName[i], picName[j], POINT);
    }
  }
  cout << "PROCESSED IN " << (clock() - processing_time) / float(CLOCKS_PER_SEC) << endl;
  
  // export results in picture format
  for(int i = 0; i < nPictures; i++){
    Mat im1 = imread(picPath[i], CV_LOAD_IMAGE_COLOR);
    savePointsPicture(sift[i], im1, dirPath, picName[i], false);
    for(int j = i; j < nPictures; j++){
      if(!isConsecutive(consecutive, close_loop, i, j, nPictures)){ continue;}
      Mat im2 = imread(picPath[j], CV_LOAD_IMAGE_COLOR);
      saveMatchesPicture(sift[i], sift[j], matches.find(PicturePair(i,j))->second, im1, im2, dirPath, picName[i], picName[j]);
    }
  }
  
  return 0;
}
