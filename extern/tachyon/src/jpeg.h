/*
 * jpeg.h - This file deals with JPEG format image files (reading/writing)
 *
 * (C) Copyright 1994-2022 John E. Stone
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * $Id: jpeg.h,v 1.4 2022/02/18 17:55:28 johns Exp $
 *
 */ 

int readjpeg(const char *name, int *xres, int *yres, unsigned char **imgdata);
int writejpeg(const char *name, int xres, int yres, unsigned char *imgdata);
