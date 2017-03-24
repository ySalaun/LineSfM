/*----------------------------------------------------------------------------  
  This code is part of the following publication and was subject
  to peer review:
  "Multiscale line segment detector for robust and accurate SfM" by
  Yohann Salaun, Renaud Marlet, and Pascal Monasse
  ICPR 2016
  
  "LSD: a Line Segment Detector" by Rafael Grompone von Gioi,
  Jeremie Jakubowicz, Jean-Michel Morel, and Gregory Randall,
  Image Processing On Line, 2012. DOI:10.5201/ipol.2012.gjmr-lsd
  http://dx.doi.org/10.5201/ipol.2012.gjmr-lsd
  
  Copyright (c) 2016 Yohann Salaun <yohann.salaun@imagine.enpc.fr>
  Copyright (c) 2007-2011 rafael grompone von gioi <grompone@gmail.com>

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

#ifndef DETECTION_HPP
#define DETECTION_HPP

#include "lsd.hpp"
#include "mlsd.hpp"

// NAMESPACES
using namespace std;

/*=================== MULTISCALE LSD INTERFACE ===================*/
// detect the segments into picture given by im
//@imagePyramid is the Gaussian pyramid of scale pictures computed with computeImagePyramid function
//@thresh the process will delete segments of length lower than thresh% of size of the scaled picture 
//	(only for multiscale, allow a faster processing)
//@multiscale enables/disable the multiscale processing
vector<Segment> lsd_multiscale(const vector<cv::Mat> &imagePyramid, const float thresh, const bool multiscale);

// compute the pyramid of image for the multiscale processing
vector<cv::Mat> computeImagePyramid(const cv::Mat &im, const bool multiscale);

#endif