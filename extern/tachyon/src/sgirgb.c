/*
 *  sgirgb.h - This file deals with SGI RGB format image files (reading/writing)
 *
 *  $Id: sgirgb.c,v 1.5 2011/02/07 07:41:51 johns Exp $
 */ 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TACHYON_INTERNAL 1
#include "tachyon.h"
#include "util.h"
#include "imageio.h" /* error codes etc */
#include "sgirgb.h"

static void putbyte(FILE * outf, unsigned char val) {
  unsigned char buf[1];
  buf[0] = val;
  fwrite(buf, 1, 1, outf);
}

static void putshort(FILE * outf, unsigned short val) {
  unsigned char buf[2];
  buf[0] = val >> 8;
  buf[1] = val & 0xff;
  fwrite(buf, 2, 1, outf);
}

static void putint(FILE * outf, unsigned int val) {
  unsigned char buf[4];
  buf[0] = (unsigned char) (val >> 24);
  buf[1] = (unsigned char) (val >> 16);
  buf[2] = (unsigned char) (val >>  8);
  buf[3] = (unsigned char) (val & 0xff);
  fwrite(buf, 4, 1, outf);
}

int writergb(char *name, int xres, int yres, unsigned char *imgdata) {
  FILE * ofp;
  char iname[80];               /* Image name */
  int x, y, i;

  if ((ofp = fopen(name, "wb")) != NULL) {
    putshort(ofp, 474);         /* Magic                       */
    putbyte(ofp, 0);            /* STORAGE is VERBATIM         */
    putbyte(ofp, 1);            /* BPC is 1                    */
    putshort(ofp, 3);           /* DIMENSION is 3              */
    putshort(ofp, (unsigned short) xres);        /* XSIZE      */
    putshort(ofp, (unsigned short) yres);        /* YSIZE      */
    putshort(ofp, 3);           /* ZSIZE                       */
    putint(ofp, 0);             /* PIXMIN is 0                 */
    putint(ofp, 255);           /* PIXMAX is 255               */

    for(i=0; i<4; i++)          /* DUMMY 4 bytes               */
      putbyte(ofp, 0);

    strcpy(iname, "Tachyon Ray Tracer Image");
    fwrite(iname, 80, 1, ofp);  /* IMAGENAME                   */
    putint(ofp, 0);             /* COLORMAP is 0               */

    for(i=0; i<404; i++)        /* DUMMY 404 bytes             */
      putbyte(ofp,0);

    for(i=0; i<3; i++)
      for(y=0; y<yres; y++)
        for(x=0; x<xres; x++)
          fwrite(&imgdata[(y*xres + x)*3 + i], 1, 1, ofp);

    fclose(ofp);
  }

  return IMAGENOERR;
}

