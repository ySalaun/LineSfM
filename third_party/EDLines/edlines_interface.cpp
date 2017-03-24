/*=================== ED LINES ===================*/
#include "edlines_interface.hpp"
#include "third_party/EDLines/LS.h"

using namespace cv;

LS *DetectLinesByED(unsigned char *srcImg, int width, int height, int *pNoLines);

PictureSegments edlines(const Mat &im, const float thresh){
  Mat imGray;
  cvtColor(im, imGray, CV_BGR2GRAY );
  
  const float lengthThresh = thresh*(imGray.rows + imGray.cols)*0.5f;

  int nLines;
  const int N = imGray.rows*imGray.cols;
  unsigned char *img = new unsigned char[N];

  for(int j=0;j<imGray.rows;j++) {
    for (int i=0;i<imGray.cols;i++){ 
      img[j*imGray.cols + i] = uchar(imGray.at<float>(j,i));
    }
  }
  
  LS *lines = DetectLinesByED(img, imGray.cols, imGray.rows, &nLines);

  // translate interface
  PictureSegments segments;
  for (int i=0; i<nLines; i++){
    Segment seg(lines[i].sx, lines[i].sy, lines[i].ex, lines[i].ey, 0, 0, 0, 0);
    if(sqrt(seg.qlength()) > lengthThresh){
      segments.push_back(seg);
    }
  }
  delete[] img;
  
  return segments;
}