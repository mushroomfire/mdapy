/*
 * imageio.h - This file deals with reading/writing image files 
 *
 * (C) Copyright 1994-2022 John E. Stone
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * $Id: imageio.h,v 1.10 2022/02/18 17:55:28 johns Exp $
 *
 */ 

/*
 * For our puposes, we're interested only in the 3 byte per pixel 24 bit
 * truecolor sort of file.. 
 */

#define IMAGENOERR     0  /**< no error */
#define IMAGEBADFILE   1  /**< can't find or can't open the file */
#define IMAGEUNSUP     2  /**< the image file is an unsupported format */
#define IMAGEALLOCERR  3  /**< not enough remaining memory to load this image */
#define IMAGEREADERR   4  /**< failed read, short reads etc */
#define IMAGEWRITEERR  5  /**< failed write, short writes etc */
#define IMAGENULLDATA  6  /**< image to write was a null pointer */

int readimage(rawimage *);
int writeimage(char * name, int xres, int yres, 
               void *imgdata, int imgbufferformat, int fileformat);
void minmax_rgb96f(int xres, int yres, const float *fimg, 
                   float *min, float *max);
void normalize_rgb96f(int xres, int yres, float *fimg);
void gamma_rgb96f(int xres, int yres, float *fimg, float gamma);
unsigned char * image_rgb48be_from_rgb96f(int xres, int yres, float *fimg);
unsigned char * image_rgb48bepl_from_rgb96f(int xres, int yres, float *fimg);
float * image_crop_rgb96f(int xres, int yres, float *fimg,
                          int szx, int szy, int sx, int sy);
unsigned char * image_crop_rgb24(int xres, int yres, unsigned char *img,
                                 int szx, int szy, int sx, int sy);
