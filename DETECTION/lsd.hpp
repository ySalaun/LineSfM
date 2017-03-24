/*----------------------------------------------------------------------------

  LSD - Line Segment Detector on digital images

  This code is part of the following publication and was subject
  to peer review:

  "LSD: a Line Segment Detector" by Rafael Grompone von Gioi,
  Jeremie Jakubowicz, Jean-Michel Morel, and Gregory Randall,
  Image Processing On Line, 2012. DOI:10.5201/ipol.2012.gjmr-lsd
  http://dx.doi.org/10.5201/ipol.2012.gjmr-lsd

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
#ifndef LSD_HPP
#define LSD_HPP

/** ln(10) */
#ifndef M_LN10
#define M_LN10 2.30258509299404568402
#endif /* !M_LN10 */

/** PI */
#ifndef M_PI
#define M_PI   3.14159265358979323846
#endif /* !M_PI */

#ifndef FALSE
#define FALSE 0
#endif /* !FALSE */

#ifndef TRUE
#define TRUE 1
#endif /* !TRUE */

/** Label for pixels with undefined gradient. */
#define NOTDEF -1024.0

/** 3/2 pi */
#define M_3_2_PI 4.71238898038

/** 2 pi */
#define M_2__PI  6.28318530718

/** Label for pixels not used in yet. */
#define NOTUSED 0

/** Label for pixels already used in detection. */
#define USED    1

/*----------------------------------------------------------------------------*/
/*----------------------------- Point structure ------------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** A point (or pixel).
 */
struct point {
  int x, y;
  point(){}
  point(int X, int Y){
    x = X;
    y = Y;
  }
};

/*----------------------------------------------------------------------------*/
/** Chained list of coordinates.
 */
struct coorlist
{
  int x, y;
  struct coorlist * next;
};

/*----------------------------------------------------------------------------*/
/*--------------------------- Rectangle structure ----------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** Rectangle structure: line segment with width.
 */
struct rect
{
  double x1, y1, x2, y2;  /* first and second point of the line segment */
  double width;        /* rectangle width */
  double x, y;          /* center of the rectangle */
  double theta;        /* angle */
  double dx, dy;        /* (dx,dy) is vector oriented as the line segment */
  double prec;         /* tolerance angle */
  double p;            /* probability of a point with angle within 'prec' */
  int n;
};

/*----------------------------------------------------------------------------*/
/** Copy one rectangle structure to another.
 */
void rect_copy(struct rect * in, struct rect * out);

/*----------------------------------------------------------------------------*/
/** Rectangle points iterator.

  The integer coordinates of pixels inside a rectangle are
  iteratively explored. This structure keep track of the process and
  functions ri_ini(), ri_inc(), ri_end(), and ri_del() are used in
  the process. An example of how to use the iterator is as follows:
  \code

  struct rect * rec = XXX; // some rectangle
  rect_iter * i;
  for( i=ri_ini(rec); !ri_end(i); ri_inc(i) )
  {
  // your code, using 'i->x' and 'i->y' as coordinates
  }
  ri_del(i); // delete iterator

  \endcode
  The pixels are explored 'column' by 'column', where we call
  'column' a set of pixels with the same x value that are inside the
  rectangle. The following is an schematic representation of a
  rectangle, the 'column' being explored is marked by colons, and
  the current pixel being explored is 'x,y'.
  \verbatim

  vx[1],vy[1]
  *   *
  *       *
  *           *
  *               ye
  *                :  *
  vx[0],vy[0]           :     *
  *              :        *
  *          x,y          *
  *        :              *
  *     :            vx[2],vy[2]
  *  :                *
  y                     ys              *
  ^                        *           *
  |                           *       *
  |                              *   *
  +---> x                      vx[3],vy[3]

  \endverbatim
  The first 'column' to be explored is the one with the smaller x
  value. Each 'column' is explored starting from the pixel of the
  'column' (inside the rectangle) with the smallest y value.

  The four corners of the rectangle are stored in order that rotates
  around the corners at the arrays 'vx[]' and 'vy[]'. The first
  point is always the one with smaller x value.

  'x' and 'y' are the coordinates of the pixel being explored. 'ys'
  and 'ye' are the start and end values of the current column being
  explored. So, 'ys' < 'ye'.
  */
typedef struct
{
  double vx[4];  /* rectangle's corner X coordinates in circular order */
  double vy[4];  /* rectangle's corner Y coordinates in circular order */
  double ys, ye;  /* start and end Y values of current 'column' */
  int x, y;       /* coordinates of currently explored pixel */
} rect_iter;

/*----------------------------------------------------------------------------*/
/** Create and initialize a rectangle iterator.

  See details in \ref rect_iter
  */
rect_iter * ri_ini(struct rect * r);

/*----------------------------------------------------------------------------*/
/** Check if the iterator finished the full iteration.

  See details in \ref rect_iter
  */
int ri_end(rect_iter * i);

/*----------------------------------------------------------------------------*/
/** Free memory used by a rectangle iterator.
 */
void ri_del(rect_iter * iter);

/*----------------------------------------------------------------------------*/
/** Increment a rectangle iterator.

  See details in \ref rect_iter
  */
void ri_inc(rect_iter * i);

/*----------------------------------------------------------------------------*/
/*----------------------------- Image Data Types -----------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** double image data type

  The pixel value at (x,y) is accessed by:

  image->data[ x + y * image->xsize ]

  with x and y integer.
  */
typedef struct image_double_s
{
  double * data;
  unsigned int xsize, ysize;
} *image_double;

/*----------------------------------------------------------------------------*/
/** Create a new image_double of size 'xsize' times 'ysize'
  with the data pointed by 'data'.
  */
image_double new_image_double_ptr(unsigned int xsize,
  unsigned int ysize, double * data);

/*----------------------------------------------------------------------------*/
/** Free memory used in image_double 'i'.
 */
void free_image_double(image_double i);

/*----------------------------------------------------------------------------*/
/** char image data type

  The pixel value at (x,y) is accessed by:

  image->data[ x + y * image->xsize ]

  with x and y integer.
  */
typedef struct image_char_s
{
  unsigned char * data;
  unsigned int xsize, ysize;
} *image_char;

/*----------------------------------------------------------------------------*/
/** Create a new image_char of size 'xsize' times 'ysize',
  initialized to the value 'fill_value'.
  */
image_char new_image_char_ini(unsigned int xsize, unsigned int ysize,
  unsigned char fill_value);

/*----------------------------------------------------------------------------*/
/** Free memory used in image_char 'i'.
 */
void free_image_char(image_char i);

/*----------------------------------------------------------------------------*/
/*--------------------------------- Gradient ---------------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** Computes the direction of the level line of 'in' at each point.

  The result is:
  - an image_double with the angle at each pixel, or NOTDEF if not defined.
  - the image_double 'modgrad' (a pointer is passed as argument)
  with the gradient magnitude at each point.
  - a list of pixels 'list_p' roughly ordered by decreasing
  gradient magnitude. (The order is made by classifying points
  into bins by gradient magnitude. The parameters 'n_bins' and
  'max_grad' specify the number of bins and the gradient modulus
  at the highest bin. The pixels in the list would be in
  decreasing gradient magnitude, up to a precision of the size of
  the bins.)
  - a pointer 'mem_p' to the memory used by 'list_p' to be able to
  free the memory when it is not used anymore.
  */
image_double ll_angle(image_double in, double threshold,
struct coorlist ** list_p, void ** mem_p,
  image_double * modgrad, unsigned int n_bins);

/*----------------------------------------------------------------------------*/
/*------------------------------------ NFA -----------------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** Computes a rectangle that covers a region of points.
 */
void region2rect(struct point * reg, int reg_size,
  image_double modgrad, double reg_angle,
  double prec, double p, struct rect * rec);

/*----------------------------------------------------------------------------*/
/** Build a region of pixels that share the same angle, up to a
  tolerance 'prec', starting at point (x,y).
  */
void region_grow(int x, int y, image_double angles, struct point * reg,
  int * reg_size, double * reg_angle, image_char used,
  double prec);

/*----------------------------------------------------------------------------*/
/** Compute a rectangle's NFA value.
 */
double rect_nfa(struct rect * rec, image_double angles, double logNT);
double nfa(int n, int k, double p, double logNT);
/*----------------------------------------------------------------------------*/
/** Refine a rectangle.

  For that, an estimation of the angle tolerance is performed by the
  standard deviation of the angle at points near the region's
  starting point. Then, a new region is grown starting from the same
  point, but using the estimated angle tolerance. If this fails to
  produce a rectangle with the right density of region points,
  'reduce_region_radius' is called to try to satisfy this condition.
  */
int refine(struct point * reg, int * reg_size, image_double modgrad,
  double reg_angle, double prec, double p, struct rect * rec,
  image_char used, image_double angles, double density_th);

/*----------------------------------------------------------------------------*/
/** Try some rectangles variations to improve NFA value. Only if the
  rectangle is not meaningful (i.e., log_nfa <= log_eps).
  */
double rect_improve(struct rect * rec, image_double angles,
  double logNT, double log_eps);

/*----------------------------------------------------------------------------*/
/*------------------------- Miscellaneous functions --------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** Is point (x,y) aligned to angle theta, up to precision 'prec'?
 */
int isaligned(int x, int y, image_double angles, double theta, double prec);

/*----------------------------------------------------------------------------*/
/** Absolute value angle difference.
 */
double angle_diff(double a, double b);

#endif