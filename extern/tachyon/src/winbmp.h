/*
 *  winbmp.h - This file deals with Windows Bitmap image files 
 *             (reading/writing)
 *
 *  $Id: winbmp.h,v 1.1 2000/08/15 06:26:07 johns Exp $
 */ 

int writebmp(char * name, int xres, int yres, unsigned char *imgdata);
