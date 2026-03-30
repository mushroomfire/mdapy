/*
 * psd.h - This file deals with Photoshop format image files (reading/writing)
 *
 * (C) Copyright 1994-2022 John E. Stone
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * $Id: psd.h,v 1.3 2022/02/18 17:55:28 johns Exp $
 *
 */ 

int writepsd48(char *name, int xres, int yres, unsigned char *imgdata);


