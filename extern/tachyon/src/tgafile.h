/* 
 * tgafile.h - this file contains defines and structures for tgafile.c
 *
 *  $Id: tgafile.h,v 1.9 2003/08/22 05:05:12 johns Exp $
 */

/* declare other functions */
int createtgafile(char *, unsigned short, unsigned short);
void * opentgafile(char *);
void writetgaregion(void *, int, int, int, int, unsigned char *);
void closetgafile(void *);
int readtga(char * name, int * xres, int * yres, unsigned char **imgdata);
int writetga(char * name, int xres, int yres, unsigned char *imgdata);
