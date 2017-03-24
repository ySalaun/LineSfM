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

#ifndef POINT_MATCHING_HPP
#define POINT_MATCHING_HPP

#include "interface.hpp"

using namespace std;

void computeDescriptors(const cv::Mat &imGray, vector<Sift> &points, cv::Mat &descriptors);

vector<int> bruteMatching(const cv::Mat &desc1, const cv::Mat &desc2);

#endif