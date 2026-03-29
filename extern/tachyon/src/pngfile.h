/*
 *  pngfile.h - This file deals with PNG format image files (reading/writing)
 *
 *  $Id: pngfile.h,v 1.2 2011/01/13 04:04:38 johns Exp $
 */ 

int readpng(const char *name, int *xres, int *yres, unsigned char **imgdata);
int writepng(const char *name, int xres, int yres, unsigned char *imgdata);
