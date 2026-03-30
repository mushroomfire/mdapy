/*
 * winbmp.h - This file deals with Windows Bitmap image files (reading/writing)
 *
 * (C) Copyright 1994-2022 John E. Stone
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * $Id: winbmp.h,v 1.2 2022/02/18 17:55:28 johns Exp $
 *
 */ 

int writebmp(char * name, int xres, int yres, unsigned char *imgdata);
