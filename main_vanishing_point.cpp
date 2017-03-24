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
  
  bool verbose = false;

  // required
  cmd.add( make_option('d', dirPath, "dirPath") );
  cmd.add( make_option('i', picList, "inputPic") );
  cmd.add( make_option('v', verbose, "verbose") );

  try {
      if (argc == 1) throw std::string("Invalid command line parameter.");
      cmd.process(argc, argv);
  } catch(const std::string& s) {
      std::cerr << "Usage: " << argv[0] << '\n'
      << "[-d|--dirPath] feature path]\n"
      << "[-i|--inputPic] list of pictures] \n"
      << "[-v|--verbose] enable/disable messages (default = FALSE)\n"
      << endl;
      return EXIT_FAILURE;
  }
  dirPath += "/";
  
  vector<string> picName, picPath;
  readPictureFile(picList, picName, picPath);
  
  const int nPictures = picName.size();
  
  // compute vanishing points
  cout << "COMPUTE VANISHING POINT" << endl;
  clock_t processing_time = clock();
  PicturesVPs vpoints(nPictures);
  int imDimension;
  for(int i = 0; i < nPictures; i++){
    if(verbose){cout << "picture " << i << ": " << picName[i] << endl;}
    Mat im = imread(picPath[i], CV_LOAD_IMAGE_GRAYSCALE);
    PictureSegments segments = readLines(dirPath, picName[i]);

    if(verbose){cout << " - vanishing point detection" << endl;}
    vpoints[i] = computeVanishingPoints(im, segments);
    saveVanishingPoints(vpoints[i], dirPath, picName[i]);
  }
  cout << "PROCESSED IN " << (clock() - processing_time) / float(CLOCKS_PER_SEC) << endl;
  
  cout << "SAVE PICTURES (CTRL+C IF NOT INTERESTED)" << endl;
  for(int i = 0; i < nPictures; i++){
    Mat im = imread(picPath[i], CV_LOAD_IMAGE_COLOR);
    PictureSegments segments = readLines(dirPath, picName[i]);

    saveVanishingPointsPicture(vpoints[i], segments, im, dirPath, picName[i]);
  }
   
  return 0;
}