/*
 *  jpeg.h - This file deals with JPEG format image files (reading/writing)
 *
 *  $Id: jpeg.h,v 1.3 2011/01/13 04:04:38 johns Exp $
 */ 

int readjpeg(const char *name, int *xres, int *yres, unsigned char **imgdata);
int writejpeg(const char *name, int xres, int yres, unsigned char *imgdata);
