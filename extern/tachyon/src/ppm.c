/*
 *  ppm.c - This file deals with PPM format image files (reading/writing)
 *
 *  $Id: ppm.c,v 1.24 2013/04/21 07:02:23 johns Exp $
 */ 

/* For our puposes, we're interested only in the 3 byte per pixel 24 bit
   truecolor sort of file..  Probably won't implement any decent checking
   at this point, probably choke on things like the # comments.. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TACHYON_INTERNAL 1
#include "tachyon.h"
#include "util.h"
#include "imageio.h" /* error codes etc */
#include "ppm.h"

static int getint(FILE * dfile) {
  char ch[256];
  int i;
  int num;

  num=0; 
  while (num==0) {
    if (fscanf(dfile, "%s", ch) == 1) {
      while (ch[0]=='#') {
        fgets(ch, sizeof(ch), dfile);
      }
    }
    num=sscanf(ch, "%d", &i);
  }
  return i;
}

int readppm(const char * name, int * xres, int * yres, unsigned char **imgdata) {
  char data[256];  
  FILE * ifp;
  int i, bytesread, cnt;
  int datasize;
 
  ifp=fopen(name, "r");  
  if (ifp==NULL) {
    return IMAGEBADFILE; /* couldn't open the file */
  }

  cnt = fscanf(ifp, "%s", data);
 
  if (cnt != 1 || strcmp(data, "P6")) {
    fclose(ifp);
    return IMAGEUNSUP; /* not a format we support */
  }

  *xres=getint(ifp);
  *yres=getint(ifp);
      i=getint(ifp); /* eat the maxval number */

  /* eat the newline */ 
  if (fread(&i, 1, 1, ifp) != 1) {
    fclose(ifp);
    return IMAGEUNSUP; /* not a format we support */
  }

  datasize = 3 * (*xres) * (*yres);

  *imgdata=malloc(datasize); 

  bytesread=fread(*imgdata, 1, datasize, ifp);   

  fclose(ifp);

  if (bytesread != datasize) 
    return IMAGEREADERR;
  
  return IMAGENOERR;
}


int writeppm(const char *name, int xres, int yres, unsigned char *imgdata) {
  FILE * ofp;
  int y, xbytes;
 
  xbytes = 3*xres;

  ofp=fopen(name, "wb");
  if (ofp==NULL)
    return IMAGEBADFILE;

  fprintf(ofp, "P6\n");
  fprintf(ofp, "%d %d\n", xres, yres);
  fprintf(ofp, "255\n"); /* maxval */

  for (y=0; y<yres; y++) {
    if (fwrite(&imgdata[(yres - y - 1)*xbytes], 1, xbytes, ofp) != xbytes) {
      fclose(ofp);
      return IMAGEWRITEERR;
    } 
  }

  fclose(ofp);
  return IMAGENOERR;
}


int writeppm48(const char *name, int xres, int yres, unsigned char *imgdata) {
  FILE * ofp;
  int y, xbytes;

  xbytes = 6*xres;

  ofp=fopen(name, "wb");
  if (ofp==NULL)
    return IMAGEBADFILE;

  fprintf(ofp, "P6\n");
  fprintf(ofp, "%d %d\n", xres, yres);
  fprintf(ofp, "65535\n"); /* maxval */

  for (y=0; y<yres; y++) {
    if (fwrite(&imgdata[(yres - y - 1)*xbytes], 1, xbytes, ofp) != xbytes) {
      fclose(ofp);
      return IMAGEWRITEERR;
    } 
  }

  fclose(ofp);
  return IMAGENOERR;
}


