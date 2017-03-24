#include "vpoint.h"
#include "lsd.h"

#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <fstream>

#ifndef FALSE
#define FALSE 0
#endif /* !FALSE */

#ifndef TRUE
#define TRUE 1
#endif /* !TRUE */

using namespace align;
using namespace std;

static void error(char * msg)
{
  fprintf(stderr,"%s\n",msg);
  exit(EXIT_FAILURE);
}

/*----------------------------------------------------------------------------*/
/*------------------------------ PGM image I/O -------------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** Skip white characters and comments in a PGM file.
 */
static void skip_whites_and_comments(FILE * f)
{
  int c;
  do
    {
      while(isspace(c=getc(f))); /* skip spaces */
      if(c=='#') /* skip comments */
        while( c!='\n' && c!='\r' && c!=EOF )
          c=getc(f);
    }
  while( c == '#' || isspace(c) );
  if( c != EOF && ungetc(c,f) == EOF )
    error("Error: unable to 'ungetc' while reading PGM file.");
}

/*----------------------------------------------------------------------------*/
/** Read a ASCII number from a PGM file.
 */
static int get_num(FILE * f)
{
  int num,c;

  while(isspace(c=getc(f)));
  if(!isdigit(c)) error("Error: corrupted PGM file.");
  num = c - '0';
  while( isdigit(c=getc(f)) ) num = 10 * num + c - '0';
  if( c != EOF && ungetc(c,f) == EOF )
    error("Error: unable to 'ungetc' while reading PGM file.");

  return num;
}

/*----------------------------------------------------------------------------*/
/** Read a PGM file into an double image.
    If the name is "-" the file is read from standard input.
 */
static double * read_pgm_image_double(int * X, int * Y, char * name)
{
  FILE * f;
  int c,bin;
  int xsize,ysize,depth,x,y;
  double * image;

  /* open file */
  if( strcmp(name,"-") == 0 ) f = stdin;
  else f = fopen(name,"rb");
  if( f == NULL ) error("Error: unable to open input image file.");

  /* read header */
  if( getc(f) != 'P' ) error("Error: not a PGM file!");
  if( (c=getc(f)) == '2' ) bin = FALSE;
  else if( c == '5' ) bin = TRUE;
  else error("Error: not a PGM file!");
  skip_whites_and_comments(f);
  xsize = get_num(f);            /* X size */
  if(xsize<=0) error("Error: X size <=0, invalid PGM file\n");
  skip_whites_and_comments(f);
  ysize = get_num(f);            /* Y size */
  if(ysize<=0) error("Error: Y size <=0, invalid PGM file\n");
  skip_whites_and_comments(f);
  depth = get_num(f);            /* depth */
  if(depth<=0) fprintf(stderr,"Warning: depth<=0, probably invalid PGM file\n");
  /* white before data */
  if(!isspace(c=getc(f))) error("Error: corrupted PGM file.");

  /* get memory */
  image = (double *) calloc( (size_t) (xsize*ysize), sizeof(double) );
  if( image == NULL ) error("Error: not enough memory.");

  /* read data */
  for(y=0;y<ysize;y++)
    for(x=0;x<xsize;x++)
      image[ x + y * xsize ] = bin ? (double) getc(f)
                                   : (double) get_num(f);

  /* close file if needed */
  if( f != stdin && fclose(f) == EOF )
    error("Error: unable to close file while reading PGM file.");

  /* return image */
  *X = xsize;
  *Y = ysize;
  return image;
}

void lsd_interface(char* name, int &Width, int &Height, vector<Segment> &seg){
  double * image;
  int n, regX,regY;
  double * segs;
  int dim = 7;
  
  /* read input file */
  image = read_pgm_image_double(&Width,&Height,name);

  /* execute LSD */
  segs = LineSegmentDetection( &n, image, Width, Height, 0.8f, 0.6f, 2.f, 22.5f, 0.f, 0.7f, 1024, NULL, &regX, &regY ); 
  
  /* convert double* into vector<Segment> */
  for(int i=0;i<n;i++){
    seg.push_back(Segment(segs[i*dim+0], segs[i*dim+1], segs[i*dim+2], segs[i*dim+3]));
  }
  
}

int main(int argc, char *argv[]) {
  
  // vector containing vanishing points
  vector<Vpoint> vp;
  
  // detect line segments
  int Width, Height;
  vector<Segment> seg;
  lsd_interface(argv[1], Width, Height, seg);
  
  align::Vpoint::detect(vp, seg, Width, Height);
  
  // save VPs segments and display VPs
  ofstream vps((string(string(argv[1]) + "_VPs.txt")).c_str(), std::ios::out | std::ios::trunc);
  vps << vp.size() << endl;
  std::vector<align::Vpoint>::const_iterator it = vp.begin();
  for(int i = 0; it != vp.end(); ++i, ++it) {
    float x, y, z;
    (*it).pos(x, y, z);
    printf("Vanishing point %d pos: %f %f \n", i, x/z, y/z);
    vps << (*it).seg.size() << endl;
    for(unsigned int j = 0; j < (*it).seg.size(); j++){
	vps << (*it).seg[j].x1 << " " << (*it).seg[j].y1 << " " << (*it).seg[j].x2 << " " << (*it).seg[j].y2 << endl;
    }
  }

  
  
  
  
  return 0;
}
