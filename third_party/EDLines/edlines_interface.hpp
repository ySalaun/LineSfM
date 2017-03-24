#ifndef EDLINES_INTERFACE_HPP
#define EDLINES_INTERFACE_HPP

#include "interface.hpp"

/*=================== ED LINES ===================*/
// interface for line detection with ed Lines
PictureSegments edlines(const cv::Mat &im, const float thresh);

#endif