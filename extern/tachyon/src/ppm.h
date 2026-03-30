/*
 * ppm.h - This file deals with PPM format image files (reading/writing)
 *
 * (C) Copyright 1994-2022 John E. Stone
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * $Id: ppm.h,v 1.8 2022/02/18 17:55:28 johns Exp $
 *
 */ 

/* 
 * For our puposes, we're interested only in the 3 byte per pixel 24 bit
 * truecolor sort of file..  Probably won't implement any decent checking
 * at this point, probably choke on things like the # comments.. 
 */

int readppm(const char *name, int *xres, int *yres, unsigned char **imgdata);
int writeppm(const char *name, int xres, int yres, unsigned char *imgdata);
int writeppm48(const char *name, int xres, int yres, unsigned char *imgdata);

