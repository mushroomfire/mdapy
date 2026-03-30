/*
 * pngfile.h - This file deals with PNG format image files (reading/writing)
 *
 * (C) Copyright 1994-2022 John E. Stone
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * $Id: pngfile.h,v 1.5 2022/02/18 17:55:28 johns Exp $
 *
 */ 

/* read 24-bit RGB PNG */
int readpng(const char *name, int *xres, int *yres, unsigned char **imgdata);

/* write 24-bit RGB compressed PNG file */
int writepng(const char *name, int xres, int yres, unsigned char *imgdata);

/* write 32-bit RGBA compressed PNG file */
int writepng_alpha(const char *name, int xres, int yres, unsigned char *imgdata);



