/*
 * imap.h - This file contains defines etc for doing image map type things.  
 *
 *  $Id: imap.h,v 1.13 2009/03/25 03:23:27 johns Exp $
 */

void       ResetImage(void);
void       LoadRawImage(rawimage *);
rawimage * AllocateImageRGB24(const char *, int, int, int, unsigned char *);
rawimage * AllocateImageFile(const char *);
void       DeallocateImage(rawimage *);
void       ResetImages(void);
void       FreeImages(void);
rawimage * DecimateImage(const rawimage *);
mipmap *   LoadMIPMap(const char *, int maxlevels);
mipmap *   CreateMIPMap(rawimage *, int);
void       FreeMIPMap(mipmap * mip);
color      MIPMap(const mipmap *, flt, flt, flt);
color      ImageMap(const rawimage *, flt, flt);
color      VolImageMapNearest(const rawimage *, flt, flt, flt);
color      VolImageMapTrilinear(const rawimage *, flt, flt, flt);
color      VolMIPMap(const mipmap *, flt, flt, flt, flt);
