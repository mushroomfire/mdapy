/*
 *  ppm.h - This file deals with PPM format image files (reading/writing)
 *
 *  $Id: ppm.h,v 1.7 2011/01/13 04:04:38 johns Exp $
 */ 

/* For our puposes, we're interested only in the 3 byte per pixel 24 bit
   truecolor sort of file..  Probably won't implement any decent checking
   at this point, probably choke on things like the # comments.. */

int readppm(const char *name, int *xres, int *yres, unsigned char **imgdata);
int writeppm(const char *name, int xres, int yres, unsigned char *imgdata);
int writeppm48(const char *name, int xres, int yres, unsigned char *imgdata);

