/*----------------------------------------------------------------------------    
  Copyright (c) 2012 Lilian Zhang
  An efficient and robust line segment matching approach based on LBD descriptor and pairwise geometric consistency,
  by Lilian Zhang, and Reinhard Koch
  JVCI 24(7), pp:794-805, 2013, DIO: 10.1016/j.jvcir.2013.05.006.

  Copyright (c) 2016-2017 Yohann Salaun <yohann.salaun@imagine.enpc.fr>
  Modifications and library adaptation
  
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

#include "line_matching.hpp"

#include "third_party/arpack++/include/arlsmat.h"
#include "third_party/arpack++/include/arlssym.h"
#include <map>

using namespace cv;

struct NodeLine{
  int left, right;
  float similarity;
  float overlap;
};

bool pointOnSegment(const openMVG::Vec3 &x, const openMVG::Vec3 &p1, const openMVG::Vec3 &p2){
  return ((p1-x).dot(p2-x) < 0);
}
    
float mutualOverlap(const vector<openMVG::Vec3> &colinear_points){
  float overlap = 0.0f;
  if(colinear_points.size() != 4)
    return 0.0f;  
  
  // check if p1-p2 and p3-p4 really overlap
  if(pointOnSegment(colinear_points[0], colinear_points[2], colinear_points[3]) || pointOnSegment(colinear_points[1], colinear_points[2], colinear_points[3])
    || pointOnSegment(colinear_points[2], colinear_points[0], colinear_points[1]) || pointOnSegment(colinear_points[3], colinear_points[0], colinear_points[1])){
    // find outer distance and inner points
    float max_dist = 0.0f;
    size_t outer1 = 0;
    size_t inner1 = 1;
    size_t inner2 = 2;
    size_t outer2 = 3;

    for(size_t i=0; i<3; ++i){
      for(size_t j=i+1; j<4; ++j){
	float dist = (colinear_points[i]-colinear_points[j]).norm();
	if(dist > max_dist){
	  max_dist = dist;
	  outer1 = i;
	  outer2 = j;
	}
      }
    }

    if(max_dist < 1.0f)
      return 0.0f;

    if(outer1 == 0){
      if(outer2 == 1){
	inner1 = 2;
	inner2 = 3;
      }
      else if(outer2 == 2){
	inner1 = 1;
	inner2 = 3;
      }
      else{
	inner1 = 1;
	inner2 = 2;
      }
    }
    else if(outer1 == 1){
      inner1 = 0;
      if(outer2 == 2){
	inner2 = 3;
      }
      else{
	inner2 = 2;
      }
    }
    else{
      inner1 = 0;
      inner2 = 1;
    }
    
    overlap = (colinear_points[inner1]-colinear_points[inner2]).norm()/max_dist;
  }
  
  return overlap;
}

void computeDescriptors(const vector<Mat> &im, vector<Segment> &lines){

  vector<float> gaussCoefL_(widthOfBand_*3), gaussCoefG_(numOfBand_*widthOfBand_);
  {
    double u = (widthOfBand_*3-1)/2;
    double sigma = (widthOfBand_*2+1)/2;
    double invsigma2 = -1/(2*sigma*sigma);
    double dis;
    for(int i=0; i<widthOfBand_*3; i++){
	    dis = i-u;
	    gaussCoefL_[i] = exp(dis*dis*invsigma2);
    }
  
    u = (numOfBand_*widthOfBand_-1)/2;
    sigma = u;
    invsigma2 = -1/(2*sigma*sigma);
    for(int i=0; i<numOfBand_*widthOfBand_; i++){
	    dis = i-u;
	    gaussCoefG_[i] = exp(dis*dis*invsigma2);
    }
  }
  
  vector<Mat> gradX(im.size()), gradY(im.size());
  for(int i = 0; i < im.size(); i++){
    cv::Sobel(im[i], gradX[i], CV_32F, 1, 0, 3);
    cv::Sobel(im[i], gradY[i], CV_32F, 0, 1, 3);
  }
  
  //the default length of the band is the line length.
  short numOfFinalLine = lines.size();
  Vec2f dL;//line direction cos(dir), sin(dir)
  Vec2f dO;//the clockwise orthogonal vector of line direction.
  short heightOfLSP = widthOfBand_*numOfBand_;//the height of line support region;
  //each band, we compute the m( pgdL, ngdL,  pgdO, ngdO) and std( pgdL, ngdL,  pgdO, ngdO);
  float pgdLRowSum;//the summation of {g_dL |g_dL>0 } for each row of the region;
  float ngdLRowSum;//the summation of {g_dL |g_dL<0 } for each row of the region;
  float pgdL2RowSum;//the summation of {g_dL^2 |g_dL>0 } for each row of the region;
  float ngdL2RowSum;//the summation of {g_dL^2 |g_dL<0 } for each row of the region;
  float pgdORowSum;//the summation of {g_dO |g_dO>0 } for each row of the region;
  float ngdORowSum;//the summation of {g_dO |g_dO<0 } for each row of the region;
  float pgdO2RowSum;//the summation of {g_dO^2 |g_dO>0 } for each row of the region;
  float ngdO2RowSum;//the summation of {g_dO^2 |g_dO<0 } for each row of the region;

  vector<float> pgdLBandSum  = vector<float>(numOfBand_, 0.f);//the summation of {g_dL |g_dL>0 } for each band of the region;
  vector<float> ngdLBandSum  = vector<float>(numOfBand_, 0.f);//the summation of {g_dL |g_dL<0 } for each band of the region;
  vector<float> pgdL2BandSum  = vector<float>(numOfBand_, 0.f);//the summation of {g_dL^2 |g_dL>0 } for each band of the region;
  vector<float> ngdL2BandSum  = vector<float>(numOfBand_, 0.f);//the summation of {g_dL^2 |g_dL<0 } for each band of the region;
  vector<float> pgdOBandSum  = vector<float>(numOfBand_, 0.f);//the summation of {g_dO |g_dO>0 } for each band of the region;
  vector<float> ngdOBandSum  = vector<float>(numOfBand_, 0.f);//the summation of {g_dO |g_dO<0 } for each band of the region;
  vector<float> pgdO2BandSum  = vector<float>(numOfBand_, 0.f);//the summation of {g_dO^2 |g_dO>0 } for each band of the region;
  vector<float> ngdO2BandSum  = vector<float>(numOfBand_, 0.f);//the summation of {g_dO^2 |g_dO<0 } for each band of the region;

  short lengthOfLSP; //the length of line support region, varies with lines
  short halfHeight = (heightOfLSP-1)/2;
  short halfWidth;
  short bandID;
  float coefInGaussion;
  float lineMiddlePointX, lineMiddlePointY;
  float sCorX, sCorY,sCorX0, sCorY0;
  short tempCor, xCor, yCor;//pixel coordinates in image plane
  short dx, dy;
  float gDL;//store the gradient projection of pixels in support region along dL vector
  float gDO;//store the gradient projection of pixels in support region along dO vector
  short imageWidth, imageHeight;
  
  for(short li = 0; li<numOfFinalLine; li++){
    Segment curLine = lines[li];
    const int img_index = curLine.scale;
    imageWidth = im[img_index].cols;
    imageHeight = im[img_index].rows;
    //initialization
    fill(pgdLBandSum.begin(), pgdLBandSum.end(), 0.f);
    fill(ngdLBandSum.begin(), ngdLBandSum.end(), 0.f);
    fill(pgdL2BandSum.begin(), pgdL2BandSum.end(), 0.f);
    fill(ngdL2BandSum.begin(), ngdL2BandSum.end(), 0.f);
    fill(pgdOBandSum.begin(), pgdOBandSum.end(), 0.f);
    fill(ngdOBandSum.begin(), ngdOBandSum.end(), 0.f);
    fill(pgdO2BandSum.begin(), pgdO2BandSum.end(), 0.f);
    fill(ngdO2BandSum.begin(), ngdO2BandSum.end(), 0.f);
    float scale_factor = 1.f;
    for(int i = 0; i < im.size() - img_index - 1; i++){
      scale_factor /= 2;
    }
    lengthOfLSP = scale_factor*curLine.length;
    halfWidth   = (lengthOfLSP-1)/2;
    
    lineMiddlePointX = 0.5 * (curLine.x1 + curLine.x2) * scale_factor ;
    lineMiddlePointY = 0.5 * (curLine.y1 + curLine.y2) * scale_factor;
    
    /*1.rotate the local coordinate system to the line direction
    *2.compute the gradient projection of pixels in line support region*/
    dL[0] = cos(curLine.angle);
    dL[1] = sin(curLine.angle);
    dO[0] = -dL[1];
    dO[1] = dL[0];
    sCorX0= -dL[0]*halfWidth + dL[1]*halfHeight + lineMiddlePointX;//hID =0; wID = 0;
    sCorY0= -dL[1]*halfWidth - dL[0]*halfHeight + lineMiddlePointY;
    
    // find gradient information for descriptor
    for(short hID = 0; hID <heightOfLSP; hID++){
      //initialization
      sCorX = sCorX0;
      sCorY = sCorY0;

      pgdLRowSum = 0;
      ngdLRowSum = 0;
      pgdORowSum = 0;
      ngdORowSum = 0;

      for(short wID = 0; wID <lengthOfLSP; wID++){
	tempCor = round(sCorX);
	xCor = (tempCor<0)?0:(tempCor>imageWidth)?imageWidth:tempCor;
	tempCor = round(sCorY);
	yCor = (tempCor<0)?0:(tempCor>imageHeight)?imageHeight:tempCor;
	/* To achieve rotation invariance, each simple gradient is rotated aligned with
	  * the line direction and clockwise orthogonal direction.*/
	dx = gradX[img_index].at<float>(yCor, xCor);
	dy = gradY[img_index].at<float>(yCor, xCor);
	gDL = dx * dL[0] + dy * dL[1];
	gDO = dx * dO[0] + dy * dO[1];
	if(gDL>0){
	  pgdLRowSum  += gDL;
	}else{
	  ngdLRowSum  -= gDL;
	}
	if(gDO>0){
	  pgdORowSum  += gDO;
	}else{
	  ngdORowSum  -= gDO;
	}
	sCorX +=dL[0];
	sCorY +=dL[1];
      }
      sCorX0 -=dL[1];
      sCorY0 +=dL[0];
      coefInGaussion = gaussCoefG_[hID];
      pgdLRowSum = coefInGaussion * pgdLRowSum;
      ngdLRowSum = coefInGaussion * ngdLRowSum;
      pgdL2RowSum = pgdLRowSum * pgdLRowSum;
      ngdL2RowSum = ngdLRowSum * ngdLRowSum;
      pgdORowSum = coefInGaussion * pgdORowSum;
      ngdORowSum = coefInGaussion * ngdORowSum;
      pgdO2RowSum = pgdORowSum * pgdORowSum;
      ngdO2RowSum = ngdORowSum * ngdORowSum;
      //compute {g_dL |g_dL>0 }, {g_dL |g_dL<0 },
      //{g_dO |g_dO>0 }, {g_dO |g_dO<0 } of each band in the line support region
      //first, current row belong to current band;
      bandID = hID/widthOfBand_;
      coefInGaussion = gaussCoefL_[hID%widthOfBand_+widthOfBand_];
      pgdLBandSum[bandID] +=  coefInGaussion * pgdLRowSum;
      ngdLBandSum[bandID] +=  coefInGaussion * ngdLRowSum;
      pgdL2BandSum[bandID] +=  coefInGaussion * coefInGaussion * pgdL2RowSum;
      ngdL2BandSum[bandID] +=  coefInGaussion * coefInGaussion * ngdL2RowSum;
      pgdOBandSum[bandID] +=  coefInGaussion * pgdORowSum;
      ngdOBandSum[bandID] +=  coefInGaussion * ngdORowSum;
      pgdO2BandSum[bandID] +=  coefInGaussion * coefInGaussion * pgdO2RowSum;
      ngdO2BandSum[bandID] +=  coefInGaussion * coefInGaussion * ngdO2RowSum;
      /* In order to reduce boundary effect along the line gradient direction,
	* a row's gradient will contribute not only to its current band, but also
	* to its nearest upper and down band with gaussCoefL_.*/
      bandID--;
      if(bandID>=0){//the band above the current band
	coefInGaussion = gaussCoefL_[hID%widthOfBand_ + 2*widthOfBand_];
	pgdLBandSum[bandID] +=  coefInGaussion * pgdLRowSum;
	ngdLBandSum[bandID] +=  coefInGaussion * ngdLRowSum;
	pgdL2BandSum[bandID] +=  coefInGaussion * coefInGaussion * pgdL2RowSum;
	ngdL2BandSum[bandID] +=  coefInGaussion * coefInGaussion * ngdL2RowSum;
	pgdOBandSum[bandID] +=  coefInGaussion * pgdORowSum;
	ngdOBandSum[bandID] +=  coefInGaussion * ngdORowSum;
	pgdO2BandSum[bandID] +=  coefInGaussion * coefInGaussion * pgdO2RowSum;
	ngdO2BandSum[bandID] +=  coefInGaussion * coefInGaussion * ngdO2RowSum;
      }
      bandID = bandID+2;
      if(bandID<numOfBand_){//the band below the current band
	coefInGaussion = gaussCoefL_[hID%widthOfBand_];
	pgdLBandSum[bandID] +=  coefInGaussion * pgdLRowSum;
	ngdLBandSum[bandID] +=  coefInGaussion * ngdLRowSum;
	pgdL2BandSum[bandID] +=  coefInGaussion * coefInGaussion * pgdL2RowSum;
	ngdL2BandSum[bandID] +=  coefInGaussion * coefInGaussion * ngdL2RowSum;
	pgdOBandSum[bandID] +=  coefInGaussion * pgdORowSum;
	ngdOBandSum[bandID] +=  coefInGaussion * ngdORowSum;
	pgdO2BandSum[bandID] +=  coefInGaussion * coefInGaussion * pgdO2RowSum;
	ngdO2BandSum[bandID] +=  coefInGaussion * coefInGaussion * ngdO2RowSum;
      }
    }
    //construct line descriptor
    vector<float> desVec(descriptorSize);
    short desID;
    /*Note that the first and last bands only have (lengthOfLSP * widthOfBand_ * 2.0) pixels
      * which are counted. */
    float invN2 = 1.0/(widthOfBand_ * 2.0);
    float invN3 = 1.0/(widthOfBand_ * 3.0);
    float invN, temp;
    for(bandID = 0; bandID<numOfBand_; bandID++){
      if(bandID==0||bandID==numOfBand_-1){	
	invN = invN2;
      }
      else{ 
	invN = invN3;
      }
      desID = bandID * 8;
      temp = pgdLBandSum[bandID] * invN;
      desVec[desID]   = temp;//mean value of pgdL;
      desVec[desID+4] = sqrt(pgdL2BandSum[bandID] * invN - temp*temp);//std value of pgdL;
      temp = ngdLBandSum[bandID] * invN;
      desVec[desID+1] = temp;//mean value of ngdL;
      desVec[desID+5] = sqrt(ngdL2BandSum[bandID] * invN - temp*temp);//std value of ngdL;

      temp = pgdOBandSum[bandID] * invN;
      desVec[desID+2] = temp;//mean value of pgdO;
      desVec[desID+6] = sqrt(pgdO2BandSum[bandID] * invN - temp*temp);//std value of pgdO;
      temp = ngdOBandSum[bandID] * invN;
      desVec[desID+3] = temp;//mean value of ngdO;
      desVec[desID+7] = sqrt(ngdO2BandSum[bandID] * invN - temp*temp);//std value of ngdO;
    }
    //normalize;
    float tempM, tempS;
    tempM = 0;
    tempS = 0;
    for(short i=0; i<numOfBand_; i++){
      tempM += desVec[8*i+0] * desVec[8*i+0];
      tempM += desVec[8*i+1] * desVec[8*i+1];
      tempM += desVec[8*i+2] * desVec[8*i+2];
      tempM += desVec[8*i+3] * desVec[8*i+3];
      tempS += desVec[8*i+4] * desVec[8*i+4];
      tempS += desVec[8*i+5] * desVec[8*i+5];
      tempS += desVec[8*i+6] * desVec[8*i+6];
      tempS += desVec[8*i+7] * desVec[8*i+7];
    }
    tempM = 1/sqrt(tempM);
    tempS = 1/sqrt(tempS);
    for(short i=0; i<numOfBand_; i++){
      desVec[8*i] =  desVec[8*i] * tempM;
      desVec[8*i+1] =  desVec[8*i+1] * tempM;
      desVec[8*i+2] =  desVec[8*i+2] * tempM;
      desVec[8*i+3] =  desVec[8*i+3] * tempM;
      desVec[8*i+4] =  desVec[8*i+4] * tempS;
      desVec[8*i+5] =  desVec[8*i+5] * tempS;
      desVec[8*i+6] =  desVec[8*i+6] * tempS;
      desVec[8*i+7] =  desVec[8*i+7] * tempS;
    }
    /*In order to reduce the influence of non-linear illumination,
      *a threshold is used to limit the value of element in the unit feature
      *vector no larger than this threshold. In Z.Wang's work, a value of 0.4 is found
      *empirically to be a proper threshold.*/
    for(short i=0; i<descriptorSize; i++ ){
      if(desVec[i]>0.4){
	desVec[i]=0.4;
      }
    }
    //re-normalize desVec;
    temp = 0;
    for(short i=0; i<descriptorSize; i++){
      temp += desVec[i] * desVec[i];
    }
    temp = 1/sqrt(temp);
    for(short i=0; i<descriptorSize; i++){
      desVec[i] =  desVec[i] * temp;
    }
    lines[li].descriptor = desVec;
  }
}

void homogeneousLine(const Segment &s, double &a, double &b, double &c, double &l){
  a = s.y2 - s.y1;//disY
  b = s.x1 - s.x2;//-disX
  c = (0 - b*s.y1) - a * s.x1;//disX*sy - disY*sx
  l = s.length;  
}

const double Inf = 1e10; //Infinity
void computeGeomCts(const Segment &s1, const Segment &s2, double &iRatio1, double &iRatio2, double &pRatio1, double &pRatio2){
  double a1, b1, c1, length1;
  double a2, b2, c2, length2;
  
  homogeneousLine(s1, a1, b1, c1, length1);
  homogeneousLine(s2, a2, b2, c2, length2);

  double a1b2_a2b1 = a1 * b2 - a2 * b1;
  
  if(fabs(a1b2_a2b1)<0.001){//two lines are almost parallel
    iRatio1 = Inf;
    iRatio2 = Inf;
  }else{
    double interSectionPointX = (c2 * b1 - c1 * b2)/a1b2_a2b1;
    double interSectionPointY = (c1 * a2 - c2 * a1)/a1b2_a2b1;
    //r1 = (s1I*s1e1)/(|s1e1|*|s1e1|)
    double disX = interSectionPointX - s1.x1;
    double disY = interSectionPointY - s1.y1;
    double len  = disY*a1 - disX*b1;
    iRatio1 = len/(length1*length1);
    //r2 = (s2I*s2e2)/(|s2e2|*|s2e2|)
    disX = interSectionPointX - s2.x1;
    disY = interSectionPointY - s2.y1;
    len  = disY*a2 - disX*b2;
    iRatio2 = len/(length2*length2);
  }

  /*project the end points of line1 onto line2 and compute their distances to line2;
    */
  double disS = fabs(a2*s1.x1 + b2*s1.y1 + c2)/length2;
  double disE = fabs(a2*s1.x2 + b2*s1.y2 + c2)/length2;
  pRatio1 = (disS+disE)/length1;

  /*project the end points of line2 onto line1 and compute their distances to line1;
    */
  disS = fabs(a1*s2.x1 + b1*s2.y1 + c1)/length1;
  disE = fabs(a1*s2.x2 + b1*s2.y2 + c1)/length1;
  pRatio2 = (disS+disE)/length2;
}

struct CompareL {
    bool operator() (const double& lhs, const double& rhs) const
    {return lhs>rhs;}
};
bool sort_operator(NodeLine i,NodeLine j) { return (i.similarity < j.similarity);}

inline
bool sidednessCt(const Segment &l1, const Segment &l2, const Segment &m1, const Segment &m2){ 
  Vec3f l1p1(l1.x1, l1.y1, 0.f);
  Vec3f l1p2(l1.x2, l1.y2, 0.f);
  
  Vec3f l2p1(l2.x1, l2.y1, 0.f);
  Vec3f l2p2(l2.x2, l2.y2, 0.f);
  
  Vec3f m1p1(m1.x1, m1.y1, 0.f);
  Vec3f m1p2(m1.x2, m1.y2, 0.f); 
  
  Vec3f m2p1(m2.x1, m2.y1, 0.f);
  Vec3f m2p2(m2.x2, m2.y2, 0.f); 
  
  return ((l1p1 - l2p1).cross(l1p1 - l2p2)).dot((m1p1 - m2p1).cross(m1p1 - m2p2)) > 0 
      && ((l1p2 - l2p1).cross(l1p2 - l2p2)).dot((m1p2 - m2p1).cross(m1p2 - m2p2)) > 0;
}

inline
bool isNeighbour(const Segment &l1, const Segment &l2, const Segment &m1, const Segment &m2, const float range1, const float range2){
  // distance chek
  float dist1 = l1.distTo(l2);
  float dist2 = m1.distTo(m2);
                        
  return dist1 < range1 && dist2 < range2 
         && sidednessCt(l1, l2, m1, m2) 
         && sidednessCt(l2, l1, m2, m1);
                        
}

openMVG::Mat3 blankMat3;
vector<int> computeMatches(const vector<Segment> &linesInLeft, const vector<Segment> &linesInRight, const float range){
  return computeMatches(linesInLeft, linesInRight, range, blankMat3, false);
}

typedef  std::multimap<double,unsigned int,CompareL> EigenMAP;
vector<int> computeMatches(const vector<Segment> &linesInLeft, const vector<Segment> &linesInRight, const float range,
			   const openMVG::Mat3 &E, const bool refined){
  EigenMAP eigenMap_;
  // parameters from LBD
  const double descriptorDifThreshold = refined? 0.3 : 0.5;
  const double overlapThreshold = 0.5;
  const double LengthDifThreshold = 4;
  const double RelativeAngleDifferenceThreshold = 0.7854; // 45 degrees
  const double IntersectionRationDifThreshold = 1;
  const double ProjectionRationDifThreshold = 1;
  const double WeightOfMeanEigenVec = refined? 0.1:0.1;
  const int ResolutionScale = 20;

  const double TwoPI = 2*M_PI;
  const unsigned int numLineLeft  = linesInLeft.size();
  const unsigned int numLineRight = linesInRight.size();

  const unsigned int sizeDescriptor = linesInLeft[0].descriptor.size();

  vector<int> matches(numLineLeft, -1);
  
  double rotationAngle = TwoPI;

  // rotation estimation disabled
  // TODO not sure it is really useful
  if(false)
  {
    const double AcceptableAngleHistogramDifference = 0.49;
    const double AcceptableLengthVectorDifference = 0.4;  
    
    //step 1: compute the angle histogram of lines in the left and right images
    const unsigned int dim = 360/ResolutionScale; //number of the bins of histogram
    unsigned int index;//index in the histogram
    double direction;
    const double scalar = 180/(ResolutionScale*3.1415927);//used when compute the index
    const double angleShift = (ResolutionScale*M_PI)/360;//make sure zero is the middle of the interval

    vector<double> angleHistLeft(dim, 0);
    vector<double> angleHistRight(dim, 0);
    vector<double> lengthLeft(dim, 0);//lengthLeft[i] store the total line length of all the lines in the ith angle bin.
    vector<double> lengthRight(dim, 0);


    for(unsigned int i = 0; i < linesInLeft.size(); i++){
      direction = linesInLeft[i].angle + M_PI + angleShift;
      direction = (direction < TwoPI)? direction : (direction-TwoPI);
      index = floor(direction*scalar);
      angleHistLeft[index] ++;
      lengthLeft[index] += linesInLeft[i].length;
    }
    for(unsigned int i = 0; i < linesInRight.size(); i++){
      direction = linesInRight[i].angle + M_PI + angleShift;
      direction = (direction < TwoPI)? direction : (direction-TwoPI);
      index = floor(direction*scalar);
      angleHistRight[index] ++;
      lengthRight[index] += linesInRight[i].length;
    }
    
    float sumHistLeft = 0, sumHistRight = 0, sumLengthLeft = 0, sumLengthRight = 0;
    for(int i = 0; i < dim; i++){
      sumHistLeft += angleHistLeft[i]*angleHistLeft[i];
      sumHistRight += angleHistRight[i]*angleHistRight[i];
      sumLengthLeft += lengthLeft[i]*lengthLeft[i];
      sumLengthRight += lengthRight[i]*lengthRight[i];
    }

    sumHistLeft = sqrt(sumHistLeft);
    sumHistRight = sqrt(sumHistRight);
    sumLengthLeft = sqrt(sumLengthLeft);
    sumLengthRight = sqrt(sumLengthRight);
      
    for(int i = 0; i < dim; i++){
      angleHistLeft[i] /= sumHistLeft;
      angleHistRight[i] /= sumHistRight;
      lengthLeft[i] /= sumLengthLeft;
      lengthRight[i] /= sumLengthRight;
    }

    //step 2: find shift to decide the approximate global rotation
    vector<double> difVec(dim);//the difference vector between left histogram and shifted right histogram
    double minDif = 10;//the minimal angle histogram difference
    double secondMinDif = 10;//the second minimal histogram difference
    unsigned int minShift;//the shift of right angle histogram when minimal difference achieved
    unsigned int secondMinShift;//the shift of right angle histogram when second minimal difference achieved

    vector<double> lengthDifVec(dim);//the length difference vector between left and right
    double minLenDif = 10;//the minimal length difference
    double secondMinLenDif = 10;//the second minimal length difference
    unsigned int minLenShift;//the shift of right length vector when minimal length difference achieved
    unsigned int secondMinLenShift;//the shift of right length vector when the second minimal length difference achieved

    double normOfVec = 0, normOfVec2 = 0;
    for(unsigned int shift=0; shift<dim; shift++){
      for(unsigned int j=0; j<dim; j++){
	index = j+shift;
	index = (index<dim)? index : index-dim;
	difVec[j] = angleHistLeft[j] - angleHistRight[index];
	normOfVec += difVec[j]*difVec[j];
	lengthDifVec[j] = lengthLeft[j] - lengthRight[index];
	normOfVec2 += lengthDifVec[j]*lengthDifVec[j];
      }
      //find the minShift and secondMinShift for angle histogram
      normOfVec = sqrt(normOfVec);
      normOfVec2 = sqrt(normOfVec2);
      if(normOfVec < secondMinDif){
	if(normOfVec < minDif){
	  secondMinDif   = minDif;
	  secondMinShift = minShift;
	  minDif   = normOfVec;
	  minShift = shift;
	}
	else{
	  secondMinDif    = normOfVec;
	  secondMinShift  = shift;
	}
      }
      //find the minLenShift and secondMinLenShift of length vector
      if(normOfVec2 < secondMinLenDif){
	if(normOfVec2 < minLenDif){
	  secondMinLenDif    = minLenDif;
	  secondMinLenShift  = minLenShift;
	  minLenDif   = normOfVec2;
	  minLenShift = shift;
	}
	else{
	  secondMinLenDif    = normOfVec2;
	  secondMinLenShift  = shift;
	}
      }
    }

    //first check whether there exist an approximate global rotation angle between image pair
    if(minDif < AcceptableAngleHistogramDifference && minLenDif < AcceptableLengthVectorDifference){
      rotationAngle = minShift*ResolutionScale;
      if(rotationAngle>90 && 360-rotationAngle>90){
	//In most case we believe the rotation angle between two image pairs should belong to [-Pi/2, Pi/2]
	rotationAngle = rotationAngle - 180;
      }
      rotationAngle = rotationAngle*M_PI/180;
    }
  }
  
  // find matches with close descriptors and length
  vector<NodeLine> keptMatches;
  float minDis, dis, temp;

  for(int idL=0; idL<numLineLeft; idL++){
    float bestScore = 100;
    for(int idR=0; idR<numLineRight; idR++){
      float dis = 0;
      for(int k = 0; k < sizeDescriptor; k++){
	float temp = linesInLeft[idL].descriptor[k] - linesInRight[idR].descriptor[k];
	dis += temp*temp;
      }
      dis = sqrt(dis);
      float l1 = linesInLeft[idL].length;
      float l2 = linesInRight[idR].length;
      float lengthDif = fabs(l1 - l2)/MIN(l1, l2);

      if(dis > descriptorDifThreshold){ continue;}
      if(lengthDif > LengthDifThreshold){continue;}
            
      NodeLine n;
      n.left = idL;
      n.right = idR;
      n.similarity = dis;
      
      // use estimated pose to better filter matches
      if(refined){
	vector<openMVG::Vec3> points(4);
	
	// compute overlap in left picture
	points[0] = openMVG::CrossProductMatrix(E*linesInLeft[idL].p1)*linesInRight[idR].line;
	points[0] /= points[0][2];
	points[1] = openMVG::CrossProductMatrix(E*linesInLeft[idL].p2)*linesInRight[idR].line;
	points[1] /= points[1][2];
	points[2] = linesInRight[idR].p1;
	points[2] /= points[2][2];
	points[3] = linesInRight[idR].p2;
	points[3] /= points[3][2];
	float overlap_left = mutualOverlap(points);
	
	// compute overlap in right picture
	points[0] = linesInLeft[idL].p1;
	points[0] /= points[0][2];
	points[1] = linesInLeft[idL].p2;
	points[1] /= points[1][2];
	points[2] = openMVG::CrossProductMatrix(E.transpose()*linesInRight[idR].p1)*linesInLeft[idL].line;
	points[2] /= points[2][2];
	points[3] = openMVG::CrossProductMatrix(E.transpose()*linesInRight[idR].p2)*linesInLeft[idL].line;
	points[3] /= points[3][2];
	float overlap_right = mutualOverlap(points);
	
	// if the overlap is too small in one of the pictures, delete the match
	if(min(overlap_left, overlap_right) < overlapThreshold){continue;}
	
	n.overlap = 1 - min(overlap_left, overlap_right);
      }

      keptMatches.push_back(n);
    }
  }
  
  /*Second step, build the adjacency matrix which reflect the geometric constraints between nodes.
    *The matrix is stored in the Compressed Sparse Column(CSC) format.
    */
  
  unsigned int dim = keptMatches.size();// Dimension of the problem.
  int sizeMax = 5000;
  
  if(dim > sizeMax){
    sort(keptMatches.begin(), keptMatches.end(), sort_operator);
    vector<NodeLine> keptMatchesTemp;
    vector<float> hist1 ( numLineLeft, 0 );
    vector<float> hist2 ( numLineRight, 0 );
    const int hMax = 0.3*(numLineLeft+numLineRight)/2;
    for ( unsigned int i = 0; i < keptMatches.size(); i++ ) {
      if(keptMatches.size() == sizeMax) {break;}
      int I = keptMatches[i].left;
      int J = keptMatches[i].right;
      if ( hist1[I] < hMax && hist2[J] < hMax ) {
	hist1[I] ++;
	hist2[J] ++;
	keptMatchesTemp.push_back ( keptMatches[i]);
      }
    }
    keptMatches = keptMatchesTemp;
    dim = sizeMax;
  }
  
  int nnz = 0;// Number of nonzero elements in adjacenceMat.
  /*adjacenceVec only store the lower part of the adjacency matrix which is a symmetric matrix.
    *                    | 0  1  0  2  0 |
    *                    | 1  0  3  0  1 |
    *eg:  adjMatrix =    | 0  3  0  2  0 |
    *                    | 2  0  2  0  3 |
    *                    | 0  1  0  3  0 |
    *     adjacenceVec = [0,1,0,2,0,0,3,0,1,0,2,0,0,3,0]
    */

  vector<double> adjacenceVec(dim*(dim+1)/2, 0);
  /*In order to save computational time, the following variables are used to store
    *the pairwise geometric information which has been computed and will be reused many times
    *latter. The reduction of computational time is at the expenses of memory consumption.
    */
  vector<bool> bComputedLeft(numLineLeft*numLineLeft, false);//flag to show whether the ith pair of left image has already been computed.
  vector<double> intersecRatioLeft(numLineLeft*numLineLeft);//the ratio of intersection point and the line in the left pair
  vector<double> projRatioLeft(numLineLeft*numLineLeft);//the point to line distance divided by the projected length of line in the left pair.

  vector<bool> bComputedRight(numLineRight*numLineRight, false);//flag to show whether the ith pair of right image has already been computed.
  vector<double> intersecRatioRight(numLineRight*numLineRight);//the ratio of intersection point and the line in the right pair
  vector<double> projRatioRight(numLineRight*numLineRight);//the point to line distance divided by the projected length of line in the right pair.

  unsigned int idLeft1, idLeft2;//the id of lines in the left pair
  unsigned int idRight1, idRight2;//the id of lines in the right pair
  double relativeAngleLeft, relativeAngleRight;//the relative angle of each line pair
  double gradientMagRatioLeft, gradientMagRatioRight;//the ratio of gradient magnitude of lines in each pair

  double iRatio1L,iRatio1R,iRatio2L,iRatio2R;
  double pRatio1L,pRatio1R,pRatio2L,pRatio2R;

  double relativeAngleDif, gradientMagRatioDif, iRatioDif, pRatioDif;

  double len;
  double similarity;

  for(unsigned int j=0; j<dim; j++){//column
    idLeft1  = keptMatches[j].left;
    idRight1 = keptMatches[j].right;
    for(unsigned int i=j+1; i<dim; i++){//row
      idLeft2  = keptMatches[i].left;
      idRight2 = keptMatches[i].right;
      if((idLeft1==idLeft2)||(idRight1==idRight2)){
	continue;//not satisfy the one to one match condition
      }
      
      if(!isNeighbour(linesInLeft[idLeft1], linesInLeft[idLeft2], linesInRight[idRight1], linesInRight[idRight2], range, range)){continue;}
      
      //first compute the relative angle between left pair and right pair.
      relativeAngleLeft  = linesInLeft[idLeft1].angle - linesInLeft[idLeft2].angle;
      relativeAngleLeft  = (relativeAngleLeft<M_PI)?relativeAngleLeft:(relativeAngleLeft-TwoPI);
      relativeAngleLeft  = (relativeAngleLeft>(-M_PI))?relativeAngleLeft:(relativeAngleLeft+TwoPI);
      relativeAngleRight = linesInRight[idRight1].angle - linesInRight[idRight2].angle;
      relativeAngleRight = (relativeAngleRight<M_PI)?relativeAngleRight:(relativeAngleRight-TwoPI);
      relativeAngleRight = (relativeAngleRight>(-M_PI))?relativeAngleRight:(relativeAngleRight+TwoPI);
      relativeAngleDif   = fabs(relativeAngleLeft - relativeAngleRight);
      if((TwoPI-relativeAngleDif) > RelativeAngleDifferenceThreshold && relativeAngleDif > RelativeAngleDifferenceThreshold){
	continue;//the relative angle difference is too large;
      }
      else if((TwoPI-relativeAngleDif) < RelativeAngleDifferenceThreshold){
	relativeAngleDif = TwoPI-relativeAngleDif;
      }

      //at last, check the intersect point ratio and point to line distance ratio
      //check whether the geometric information of pairs (idLeft1,idLeft2) and (idRight1,idRight2) have already been computed.
      if(!bComputedLeft[idLeft1*numLineLeft + idLeft2]){//have not been computed yet
	computeGeomCts(linesInLeft[idLeft1], linesInLeft[idLeft2], iRatio1L, iRatio2L, pRatio1L, pRatio2L);
	
	intersecRatioLeft[idLeft1*numLineLeft + idLeft2] = iRatio1L;
	intersecRatioLeft[idLeft2*numLineLeft + idLeft1] = iRatio2L;

	projRatioLeft[idLeft1*numLineLeft + idLeft2] = pRatio1L;
	projRatioLeft[idLeft2*numLineLeft + idLeft1] = pRatio2L;

	//mark them as computed
	bComputedLeft[idLeft1*numLineLeft + idLeft2] = true;
	bComputedLeft[idLeft2*numLineLeft + idLeft1] = true;
      }
      else{//read these information from matrix;
	iRatio1L = intersecRatioLeft[idLeft1*numLineLeft + idLeft2];
	iRatio2L = intersecRatioLeft[idLeft2*numLineLeft + idLeft1];
	pRatio1L = projRatioLeft[idLeft1*numLineLeft + idLeft2];
	pRatio2L = projRatioLeft[idLeft2*numLineLeft + idLeft1];
      }
      if(!bComputedRight[idRight1*numLineRight + idRight2]){//have not been computed yet
	computeGeomCts(linesInRight[idRight1], linesInRight[idRight2], iRatio1R, iRatio2R, pRatio1R, pRatio2R);
	
	intersecRatioRight[idRight1*numLineRight + idRight2] = iRatio1L;
	intersecRatioRight[idRight2*numLineRight + idRight1] = iRatio2L;

	projRatioRight[idRight1*numLineRight + idRight2] = pRatio1R;
	projRatioRight[idRight2*numLineRight + idRight1] = pRatio2R;

	//mark them as computed
	bComputedRight[idRight1*numLineRight + idRight2] = true;
	bComputedRight[idRight2*numLineRight + idRight1] = true;
      }
      else{//read these information from matrix;
	iRatio1R = intersecRatioRight[idRight1*numLineRight + idRight2];
	iRatio2R = intersecRatioRight[idRight2*numLineRight + idRight1];
	pRatio1R = projRatioRight[idRight1*numLineRight + idRight2];
	pRatio2R = projRatioRight[idRight2*numLineRight + idRight1];
      }

      pRatioDif = MIN(fabs(pRatio1L-pRatio1R), fabs(pRatio2L-pRatio2R));
      if(pRatioDif > ProjectionRationDifThreshold){
	continue;//the projection length ratio difference is too large;
      }
      
      if(!refined){
	if((iRatio1L==Inf)||(iRatio2L==Inf)||(iRatio1R==Inf)||(iRatio2R==Inf)){
	  // check conservation of distance between parallel lines
	  float dLeft = min(linesInLeft[idLeft1].distTo(linesInLeft[idLeft2].m), linesInLeft[idLeft2].distTo(linesInLeft[idLeft1].m));
	  float dRight = min(linesInRight[idRight1].distTo(linesInRight[idRight2].m), linesInRight[idRight2].distTo(linesInRight[idRight1].m));
	  if((dLeft < 10 && dRight > 10) || (dLeft > 10 && dRight < 10)){ continue; }
	  //don't consider the intersection length ratio
	  similarity = 4 - keptMatches[j].similarity/descriptorDifThreshold
	  - keptMatches[i].similarity/descriptorDifThreshold
	  - pRatioDif/ProjectionRationDifThreshold
	  - relativeAngleDif/RelativeAngleDifferenceThreshold;
	}
	else{
	  iRatioDif = min(fabs(iRatio1L-iRatio1R), fabs(iRatio2L-iRatio2R));
	  if(iRatioDif>IntersectionRationDifThreshold){
	    continue;//the intersection length ratio difference is too large;
	  }
	  //now compute the similarity score between two line pairs.
	  similarity = 5 - keptMatches[j].similarity/descriptorDifThreshold
	  - keptMatches[i].similarity/descriptorDifThreshold
	  - iRatioDif/IntersectionRationDifThreshold - pRatioDif/ProjectionRationDifThreshold
	  - relativeAngleDif/RelativeAngleDifferenceThreshold;
	}
      }
      else{
	similarity = 4 - 2 * (keptMatches[i].similarity/descriptorDifThreshold + keptMatches[j].similarity/descriptorDifThreshold) 
		       - 0 * (keptMatches[i].overlap/overlapThreshold + keptMatches[i].overlap/overlapThreshold);
      }

      adjacenceVec[(2*dim-j-1)*j/2+i] = similarity;
      nnz++;
      
    }
  }
  
  std::vector<int > matchList;
  
  try{
    // pointer to an array that stores the nonzero elements of Adjacency matrix.
    vector<double> adjacenceMat(nnz, 0);
    // pointer to an array that stores the row indices of the non-zeros in adjacenceMat.
    vector<int> irow(nnz);
    // pointer to an array of pointers to the beginning of each column of adjacenceMat.
    vector<int> pcol(dim+1);
    int idOfNNZ = 0;//the order of none zero element
    pcol[0] = 0;
    unsigned int tempValue;
    
    for(unsigned int j=0; j<dim; j++){//column
      for(unsigned int i=j+1; i<dim; i++){//row
	tempValue = (2*dim-j-1)*j/2+i;
	  
	if(adjacenceVec[tempValue]!=0){
	  adjacenceMat[idOfNNZ] = adjacenceVec[tempValue];
	  irow[idOfNNZ] = i;
	  idOfNNZ++;
	}
      }
      pcol[j+1] = idOfNNZ;
    }
    
    /*Third step, solve the principal eigenvector of the adjacency matrix using Arpack lib.
      */

    ARluSymMatrix<double> arMatrix(dim, nnz, adjacenceMat.data(), irow.data(), pcol.data());
    ARluSymStdEig<double> dprob(2, arMatrix, "LM");// Defining what we need: the first eigenvector of arMatrix with largest magnitude.
    
    // Finding eigenvalues and eigenvectors.
    dprob.FindEigenvectors();

    eigenMap_.clear();

    double meanEigenVec = 0;
    if(dprob.EigenvectorsFound()){
      double value;
      for(unsigned int j=0; j<dim; j++){
	value = fabs(dprob.Eigenvector(1,j));
	meanEigenVec += value;
	eigenMap_.insert(std::make_pair(value,j));
      }
    }
    double minOfEigenVec_ = WeightOfMeanEigenVec*meanEigenVec/dim;
    
    
    double matchScore1 = 0;
    double matchScore2 = 0;
    EigenMAP::iterator iter;
    unsigned int id;
    double sideValueL, sideValueR;
    double pointX,pointY;
    
    /*first try, start from the top element in eigenmap */
    while(true){
      iter = eigenMap_.begin();
      //if the top element in the map has small value, then there is no need to continue find more matching line pairs;
      if(iter->first < minOfEigenVec_){
	break;
      }
      id = iter->second;

      unsigned int idLeft1 = keptMatches[id].left;
      unsigned int idRight1= keptMatches[id].right;
      
      matchList.push_back(idLeft1);
      matchList.push_back(idRight1);
      matchScore1 += iter->first;
      eigenMap_.erase(iter++);

      //remove all potential assignments in conflict with top matched line pair
      double xe_xsLeft = linesInLeft[idLeft1].x2-linesInLeft[idLeft1].x1;
      double ye_ysLeft = linesInLeft[idLeft1].y2-linesInLeft[idLeft1].y1;
      double xe_xsRight = linesInRight[idRight1].x2-linesInRight[idRight1].x1;
      double ye_ysRight = linesInRight[idRight1].y2-linesInRight[idRight1].y1;
      double coefLeft  = sqrt(xe_xsLeft*xe_xsLeft+ye_ysLeft*ye_ysLeft);
      double coefRight = sqrt(xe_xsRight*xe_xsRight+ye_ysRight*ye_ysRight);

      for( ; iter->first >= minOfEigenVec_; ){
	id = iter->second;
	idLeft2 = keptMatches[id].left;
	idRight2= keptMatches[id].right;
	
	//check one to one match condition
	if((idLeft1==idLeft2)||(idRight1==idRight2)){
	  eigenMap_.erase(iter++);
	  continue;//not satisfy the one to one match condition
	}
	
	//check sidedness constraint, the middle point of line2 should lie on the same side of line1.
	//sideValue = (y-ys)*(xe-xs)-(x-xs)*(ye-ys);
	pointX = 0.5*(linesInLeft[idLeft2].x1+linesInLeft[idLeft2].x2);
	pointY = 0.5*(linesInLeft[idLeft2].y1+linesInLeft[idLeft2].y2);
	sideValueL = (pointY-linesInLeft[idLeft1].y1)*xe_xsLeft
	  - (pointX-linesInLeft[idLeft1].x1)*ye_ysLeft;
	sideValueL = sideValueL/coefLeft;
	pointX = 0.5*(linesInRight[idRight2].x1+linesInRight[idRight2].x2);
	pointY = 0.5*(linesInRight[idRight2].y1+linesInRight[idRight2].y2);
	sideValueR = (pointY-linesInRight[idRight1].y1)*xe_xsRight
	  - (pointX-linesInRight[idRight1].x1)*ye_ysRight;
	sideValueR = sideValueR/coefRight;
	if(sideValueL*sideValueR<0&&fabs(sideValueL)>5&&fabs(sideValueR)>5){//have the different sign, conflict happens.
	  eigenMap_.erase(iter++);
	  continue;
	}

	
	//check relative angle difference
	relativeAngleLeft  = linesInLeft[idLeft1].angle - linesInLeft[idLeft2].angle;
	relativeAngleLeft  = (relativeAngleLeft<M_PI)?relativeAngleLeft:(relativeAngleLeft-TwoPI);
	relativeAngleLeft  = (relativeAngleLeft>(-M_PI))?relativeAngleLeft:(relativeAngleLeft+TwoPI);
	relativeAngleRight = linesInRight[idRight1].angle - linesInRight[idRight2].angle;
	relativeAngleRight = (relativeAngleRight<M_PI)?relativeAngleRight:(relativeAngleRight-TwoPI);
	relativeAngleRight = (relativeAngleRight>(-M_PI))?relativeAngleRight:(relativeAngleRight+TwoPI);
	relativeAngleDif   = fabs(relativeAngleLeft - relativeAngleRight);
	if((TwoPI-relativeAngleDif)>RelativeAngleDifferenceThreshold&&relativeAngleDif>RelativeAngleDifferenceThreshold){
	  eigenMap_.erase(iter++);
	  continue;//the relative angle difference is too large;
	}

	iter++;
      }
    }//end while(stillLoop)
  }
  catch(ArpackError e){
    cout << "error in matching" << endl;
    matchList.clear();
  }
  for(int i = 0; i < matchList.size()/2; i++){
    matches[matchList[2*i]] = matchList[2*i+1];
  }
  return matches;
}
/*********************************************************************/