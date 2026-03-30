/* 
 * tgafile.h - this file contains fctn prototypes for Targa image parsing
 *
 * (C) Copyright 1994-2022 John E. Stone
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * $Id: tgafile.h,v 1.10 2022/02/18 17:55:28 johns Exp $
 *
 */

/* declare other functions */
int createtgafile(char *, unsigned short, unsigned short);
void * opentgafile(char *);
void writetgaregion(void *, int, int, int, int, unsigned char *);
void closetgafile(void *);
int readtga(char * name, int * xres, int * yres, unsigned char **imgdata);
int writetga(char * name, int xres, int yres, unsigned char *imgdata);
