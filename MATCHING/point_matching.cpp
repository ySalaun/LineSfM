/*----------------------------------------------------------------------------    
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


#include "point_matching.hpp"

// TODO not working
/*#include <opencv/cv.h>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d/features2d.hpp>*/

using namespace cv;

const double sift_matching_criterion = 0.8;

void computeDescriptors(const Mat &imGray, vector<Sift> &points, Mat &descriptors){  
  /*Ptr<cv::xfeatures2d::SiftFeatureDetector> detector = xfeatures2d::SiftFeatureDetector::create(0, 3, 0.01);
  vector<KeyPoint> sift;
  detector->detect(imGray, sift);

  // suppress doublons 
  {
    vector<KeyPoint> tempSIFT;
    float tempX = 0, tempY = -1;
    for(int j = 0; j < sift.size(); j++){
      if(sift[j].pt.x != tempX && sift[j].pt.y != tempY){
	tempSIFT.push_back(sift[j]);
	tempX = sift[j].pt.x;
	tempY = sift[j].pt.y;
      }
    }
    sift = tempSIFT;
  }

  points.resize(sift.size());
  for(int i = 0; i < points.size(); i++){
    points[i].pt = openMVG::Vec3(sift[i].pt.x, sift[i].pt.y, 1);
    points[i].angle = sift[i].angle;
    points[i].scale = sift[i].size;
  }
  
  Ptr<cv::xfeatures2d::SiftDescriptorExtractor> extractor = xfeatures2d::SiftDescriptorExtractor::create();
  extractor->compute(imGray, sift, descriptors);*/
}

vector<int> bruteMatching(const Mat &desc1, const Mat &desc2){
  /*BFMatcher matcher(NORM_L2, false);
  vector<vector<DMatch>> knnmatches;
  matcher.knnMatch(desc1, desc2, knnmatches, 2);*/

  vector<int> matches(desc1.rows, -1);
  /*for (vector<vector<DMatch>>::const_iterator it = knnmatches.begin(); it != knnmatches.end(); it++){
    if (it->at(0).distance < sift_matching_criterion*it->at(1).distance){
      matches[(*it)[0].queryIdx] = (*it)[0].trainIdx;
    }
  } */

  return matches;
}
