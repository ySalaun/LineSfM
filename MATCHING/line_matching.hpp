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

#ifndef LINE_MATCHING_HPP
#define LINE_MATCHING_HPP

#include "interface.hpp"

using namespace std;

void computeDescriptors(const vector<cv::Mat> &im, vector<Segment> &lines);
vector<int> computeMatches(const vector<Segment> &linesInLeft, const vector<Segment> &linesInRight, const float range);
vector<int> computeMatches(const vector<Segment> &linesInLeft, const vector<Segment> &linesInRight, const float range, 
			   const openMVG::Mat3 &E, const bool refined);

#endif