/*----------------------------------------------------------------------------  
  This code is part of the following publication and was subject
  to peer review:
  "Robust and Accurate Line- and/or Point-Based Pose Estimation without Manhattan Assumptions",
  Yohann Salaun, Renaud Marlet, and Pascal Monasse, ECCV 2016
  
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

#ifndef REFINEMENT_HPP
#define REFINEMENT_HPP

#include "interface.hpp"
#include "openMVG/numeric/numeric.h"

//#define ANGULAR

using namespace std;

openMVG::Mat3 root(const openMVG::Mat3 &R, const double k, const double n);

// refine bifocal pose
Pose refinedPose(const PointConstraints &points, const ParallelConstraints &ppairs, const Pose &pose, 
		 const vector<FTypeIndex> &inliers, const bool verbose);

// bundle adjustment on the whole scene
void bundleAdjustment(vector<Point3D> &points3D, const PicturesPoints &points,
		      vector<Line3D> &lines3D, const PicturesSegments &lines, 
		      vector<Pose> &globalPose, const vector<openMVG::Mat3> &K,
		      const vector<int> &cop_cts, const bool refine_rotation, const bool verbose);

void manhattanize(vector<Line3D> &lines3D, const PicturesSegments &lines, vector<Plane> &planes,
		  const vector<Pose> &globalPose, const vector<openMVG::Mat3> &K);

#endif