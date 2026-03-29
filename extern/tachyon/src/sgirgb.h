/*
 *  sgirgb.h - This file deals with SGI RGB format image files (reading/writing)
 *
 *  $Id: sgirgb.h,v 1.1 1999/09/01 16:26:14 johns Exp $
 */ 

int writergb(char *name, int xres, int yres, unsigned char *imgdata);
