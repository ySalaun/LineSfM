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

#include "detection.hpp"

using namespace cv;

vector<Segment> LineSegmentDetection(const Mat &im, vector<int> &noisyTexture, const vector<Segment> &rawSegments,
  const double quant, const double ang_th, const double log_eps,
  const double density_th, const int n_bins,
  const bool multiscale, const int i_scale, const float segment_length_threshold)
{
  vector<Segment> returned_lines;

  image_double image, angles, modgrad;
  image_char used;
  
  struct coorlist * list_p;
  void * mem_p;
  struct rect rec;
  struct point * reg;
  int reg_size, min_reg_size, i;
  unsigned int xsize, ysize;
  double rho, reg_angle, prec, p, log_nfa, logNT;

  /* angle tolerance */
  prec = M_PI * ang_th / 180.0;
  p = ang_th / 180.0;
  rho = quant / sin(prec); /* gradient magnitude threshold */

  /* load image and compute angle at each pixel */
  const int N = im.cols*im.rows;
  double* data = new double[N];
  for (int i = 0; i < N; i++){
    data[i] = double(im.data[i]);
  }
  image = new_image_double_ptr((unsigned int)im.cols, (unsigned int)im.rows, data);
  angles = ll_angle(image, rho, &list_p, &mem_p, &modgrad, (unsigned int)n_bins);
  free((void *)image);
  delete[] data;
  
  xsize = angles->xsize;
  ysize = angles->ysize;

  /* Number of Tests - NT

     The theoretical number of tests is Np.(XY)^(5/2)
     where X and Y are number of columns and rows of the image.
     Np corresponds to the number of angle precisions considered.
     As the procedure 'rect_improve' tests 5 times to halve the
     angle precision, and 5 more times after improving other factors,
     11 different precision values are potentially tested. Thus,
     the number of tests is
     11 * (X*Y)^(5/2)
     whose logarithm value is
     log10(11) + 5/2 * (log10(X) + log10(Y)).
     */
  logNT = 5.0 * (log10((double)xsize) + log10((double)ysize)) / 2.0
    + log10(11.0);
  min_reg_size = (int)(-logNT / log10(p)); /* minimal number of points in region that can give a meaningful event */

  used = new_image_char_ini(xsize, ysize, NOTUSED);
  reg = (struct point *) calloc((size_t)(xsize*ysize), sizeof(struct point));

  /* search for line segments with previous scale information */
  vector<Cluster> refinedLines = refineRawSegments(rawSegments, returned_lines, i_scale, angles, modgrad, used, logNT, log_eps);
  /* suppress dense part of gradients (great chance of noisy texture) */  
  if(multiscale){denseGradientFilter(noisyTexture, im, angles, used, xsize, ysize, N);}

  /* classical LSD algorithm
    note that for multiscale algo, some pixels are already set as used
    */
  for (; list_p != NULL; list_p = list_p->next)
    if (used->data[list_p->x + list_p->y * used->xsize] == NOTUSED &&
      angles->data[list_p->x + list_p->y * angles->xsize] != NOTDEF)
      /* there is no risk of double comparison problems here
         because we are only interested in the exact NOTDEF value */
    {
      /* find the region of connected point and ~equal angle */
      region_grow(list_p->x, list_p->y, angles, reg, &reg_size,
        &reg_angle, used, prec);

      /* reject small regions */
      if (reg_size < min_reg_size) continue;

      /* construct rectangular approximation for the region */
      region2rect(reg, reg_size, modgrad, reg_angle, prec, p, &rec);

      /* Check if the rectangle exceeds the minimal density of
         region points. If not, try to improve the region.
         The rectangle will be rejected if the final one does
         not fulfill the minimal density condition.
         This is an addition to the original LSD algorithm published in
         "LSD: A Fast Line Segment Detector with a False Detection Control"
         by R. Grompone von Gioi, J. Jakubowicz, J.M. Morel, and G. Randall.
         The original algorithm is obtained with density_th = 0.0.
         */
      if (!refine(reg, &reg_size, modgrad, reg_angle,
        prec, p, &rec, used, angles, density_th)){
        continue;
      }

      /* compute NFA value */
      log_nfa = rect_improve(&rec, angles, logNT, log_eps);
      if (log_nfa <= log_eps){
        continue;
      }

      /* A New Line Segment was found! */
      if (!multiscale){
        /*
          The gradient was computed with a 2x2 mask, its value corresponds to
          points with an offset of (0.5,0.5), that should be added to output.
          The coordinates origin is at the center of pixel (0,0).
          */
        rec.x1 += 0.5; rec.y1 += 0.5;
        rec.x2 += 0.5; rec.y2 += 0.5;

        /* add line segment found to output */
	Segment seg(rec.x1, rec.y1, rec.x2, rec.y2, rec.width, rec.p, log_nfa, i_scale);
	if(seg.length > segment_length_threshold){
	  returned_lines.push_back(seg);
	}
      }

      refinedLines.push_back(Cluster(angles, modgrad, logNT, reg, reg_size, rec, refinedLines.size(), i_scale));
    }

  if (multiscale){
    mergeSegments(refinedLines, segment_length_threshold, i_scale, angles, modgrad, used, logNT, log_eps);
    
    // convert clusters into segments
    for (int i = 0; i < refinedLines.size(); i++){
      if (refinedLines[i].isMerged()){ continue; }

      /* add line segment found to output */
      returned_lines.push_back(refinedLines[i].toSegment());
    }
  }

  /* free memory */
  /* only the double_image structure should be freed,
                the data pointer was provided to this functions
                and should not be destroyed.                 */
  free_image_double(angles);
  free_image_double(modgrad);
  free_image_char(used);
  free((void *)mem_p);

  return returned_lines;
}

vector<Mat> computeImagePyramid(const Mat &imGray, const bool multiscale){
  // compute nb of scale wrt picture size and multiscale option
  int nScales = scaleNb(imGray, multiscale);

  vector<Mat> imagePyramid(nScales);
  
  // generate gaussian pyramid with gray pictures
  for (int i = 0; i < nScales; i++){
    float scale = pow(scale_step, i - nScales + 1);
    resize(imGray, imagePyramid[i], Size(scale*imGray.cols, scale*imGray.rows), scale, scale);
    GaussianBlur(imagePyramid[i], imagePyramid[i], Size(h_kernel, h_kernel), sigma_scale, sigma_scale);
  }
  
  return imagePyramid;
}

vector<Segment> lsd_multiscale(const vector<Mat> &imagePyramid, const float thresh, const bool multiscale){
  vector<Segment> segments;
  const int nScales = imagePyramid.size();
  vector<int> noisyTexture(0);
  
  if(multiscale){
    for (int i = 0; i < nScales; i++){
      const float lengthThresh = thresh*(imagePyramid[i].rows + imagePyramid[i].cols)*0.5f;
      cout << "scale step: " << i << endl;
      segments = LineSegmentDetection(imagePyramid[i], noisyTexture, segments, 2.f, 22.5f, 0.f, 0.7f, 1024, multiscale, i, lengthThresh);
      cout << "#lines = " << segments.size() << endl;
      // upsize segments
      if (i != nScales - 1){
	for (int j = 0; j < segments.size(); j++){
	  segments[j].upscale(scale_step);
	}     
      }
    }
  }
  else{
    Mat im;
    const float scale = 0.8; // default value in LSD
    resize(imagePyramid[0], im, Size(scale*imagePyramid[0].cols, scale*imagePyramid[0].rows), scale, scale);
    const float lengthThresh = thresh*(im.rows + im.cols)*0.5f;
    segments = LineSegmentDetection(im, noisyTexture, segments, 2.f, 22.5f, 0.f, 0.7f, 1024, multiscale, 0, lengthThresh);
    cout << "#lines = " << segments.size() << endl;
    for (int j = 0; j < segments.size(); j++){
      segments[j].upscale(1/scale);
    }
  }
  
  
  return segments;
}
