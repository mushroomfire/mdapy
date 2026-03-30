/*
 * imap.c - This file contains code for doing image map type things.  
 *
 * (C) Copyright 1994-2022 John E. Stone
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * $Id: imap.c,v 1.42 2022/03/25 06:13:10 johns Exp $
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TACHYON_INTERNAL 1
#include "tachyon.h"
#include "imap.h"
#include "global.h" /* XXX this needs to go */
#include "util.h"
#include "parallel.h"
#include "imageio.h"
#include "ui.h"

void ResetImages(void) {
  int i;
  global_numimages=0;
  for (i=0; i<MAXIMGS; i++) {
    global_imagelist[i]=NULL;
  }
}

void FreeImages(void) {
  int i; 
  for (i=0; i<global_numimages; i++) {
    DeallocateImage(global_imagelist[i]);
  }
  ResetImages();
}

void LoadRawImage(rawimage * image) {
  if (!image->loaded) {
    readimage(image);
    image->loaded=1;
  }
}

rawimage * AllocateImageRGB24(const char * filename, int xs, int ys, int zs, unsigned char * rgb) {
  rawimage * newimage = NULL;
  int i, len, intable;

  intable=0;
  if (global_numimages!=0) {
    for (i=0; i<global_numimages; i++) {
      if (!strcmp(filename, global_imagelist[i]->name)) {
        newimage=global_imagelist[i];
        intable=1;
      }
    }
  }

  if (!intable) {
    newimage=malloc(sizeof(rawimage));
    newimage->loaded=1;
    newimage->xres=xs;
    newimage->yres=ys;
    newimage->zres=zs;
    newimage->bpp=3;
    newimage->data=rgb;
    len=strlen(filename);
    if (len > 80) 
      return NULL;
    strcpy(newimage->name, filename);

    global_imagelist[global_numimages]=newimage;  /* add new one to the table       */ 
    global_numimages++;                    /* increment the number of images */
  }
 
  return newimage;
}

rawimage * AllocateImageFile(const char * filename) { 
  rawimage * newimage = NULL;
  int i, len, intable;

  intable=0;
  if (global_numimages!=0) {
    for (i=0; i<global_numimages; i++) {
      if (!strcmp(filename, global_imagelist[i]->name)) {
        newimage=global_imagelist[i];
        intable=1;
      }
    }
  }

  if (!intable) {
    newimage=malloc(sizeof(rawimage));
    newimage->loaded=0;
    newimage->xres=0;
    newimage->yres=0;
    newimage->zres=0;
    newimage->bpp=0;
    newimage->data=NULL;
    len=strlen(filename);
    if (len > 80) 
      return NULL;
    strcpy(newimage->name, filename);

    global_imagelist[global_numimages]=newimage; /* add new tex to table and */ 
    global_numimages++;                          /* increment image count    */
  }
 
  return newimage;
}

rawimage * NewImage(int x, int y, int z) {
  rawimage * newimage = NULL;
  newimage=malloc(sizeof(rawimage));
  if (newimage == NULL)
    return NULL;

  newimage->loaded=1;
  newimage->xres=x;
  newimage->yres=y;
  newimage->zres=z;
  newimage->bpp=0;
  newimage->data=malloc(((long)x)*((long)y)*((long)z)*3L);
  if (newimage->data == NULL) {
    free(newimage);
    return NULL;
  }

  return newimage;
}

void DeallocateImage(rawimage * image) {
  image->loaded=0;
  free(image->data);
  image->data=NULL;
  free(image);
}

void FreeMIPMap(mipmap * mip) {
  int i;

  /* don't free the original image here, FreeImages() will */
  /* get it when all else is completed.                    */
  for (i=1; i<mip->levels; i++) {
    DeallocateImage(mip->images[i]);
  } 
  free(mip->images);
  free(mip);
}

mipmap * LoadMIPMap(const char * filename, int maxlevels) {
  rawimage * img;
  mipmap * mip;

  img = AllocateImageFile(filename);
  if (img == NULL)
    return NULL;

  LoadRawImage(img);

  mip = CreateMIPMap(img, maxlevels); 
  if (mip == NULL) {
    DeallocateImage(img);
    free(mip);
    return NULL;
  }

  return mip;
}

rawimage * DecimateImage(const rawimage * image) {
  rawimage * newimage;
  long x, y, addr, addr2;

  x = (long) image->xres >> 1;
  if (x == 0)
    x = 1;

  y = (long) image->yres >> 1;
  if (y == 0)
    y = 1;

  newimage = NewImage(x, y, 1);

  if (image->xres > 1 && image->yres > 1) {
    for (y=0; y<newimage->yres; y++) {
      for (x=0; x<newimage->xres; x++) {
        addr = (newimage->xres*y + x)*3L;
        addr2 = (image->xres*y + x)*3L*2L;
        newimage->data[addr] = (int)
          (image->data[addr2] + 
           image->data[addr2 + 3] +
           image->data[addr2 + image->xres*3] + 
           image->data[addr2 + (image->xres + 1)*3]) >> 2; 
        addr++;
        addr2++;
        newimage->data[addr] = (int)
          (image->data[addr2] + 
           image->data[addr2 + 3] +
           image->data[addr2 + image->xres*3] + 
           image->data[addr2 + (image->xres + 1)*3]) >> 2; 
        addr++;
        addr2++;
        newimage->data[addr] = (int)
          (image->data[addr2] + 
           image->data[addr2 + 3] +
           image->data[addr2 + image->xres*3] + 
           image->data[addr2 + (image->xres + 1)*3]) >> 2; 
      }
    }
  } else if (image->xres == 1) {
    for (y=0; y<newimage->yres; y++) {
      addr = y*3L;
      addr2 = y*3L*2L;
      newimage->data[addr] = (int)
        (image->data[addr2] + 
         image->data[addr2 + 3]) >> 1;
      addr++;
      addr2++;
      newimage->data[addr] = (int)
        (image->data[addr2] + 
         image->data[addr2 + 3]) >> 1;
      addr++;
      addr2++;
      newimage->data[addr] = (int)
        (image->data[addr2] + 
         image->data[addr2 + 3]) >> 1;
    }
  } else if (image->yres == 1) {
    for (x=0; x<newimage->xres; x++) {
      addr = x*3L;
      addr2 = x*3L*2L;
      newimage->data[addr] = (int)
        (image->data[addr2] + 
         image->data[addr2 + 3]) >> 1;
      addr++;
      addr2++;
      newimage->data[addr] = (int)
        (image->data[addr2] + 
         image->data[addr2 + 3]) >> 1;
      addr++;
      addr2++;
      newimage->data[addr] = (int)
        (image->data[addr2] + 
         image->data[addr2 + 3]) >> 1;
    }
  }

  return newimage;
}

mipmap * CreateMIPMap(rawimage * image, int maxlevels) {
  mipmap * mip;
  int xlevels, ylevels, zlevels, i; 
  
  if (image == NULL) 
    return NULL;

  mip = (mipmap *) malloc(sizeof(mipmap));
  if (mip == NULL)
    return NULL;

  xlevels = 0;  
  i = abs(image->xres);
  while (i) {
    i >>= 1; 
    xlevels++;
  }
  
  ylevels = 0;  
  i = abs(image->yres);
  while (i) {
    i >>= 1; 
    ylevels++;
  }

  zlevels = 0;  
  i = abs(image->zres);
  while (i) {
    i >>= 1; 
    zlevels++;
  }

  mip->levels = (xlevels > ylevels) ? xlevels : ylevels; 
  if (zlevels > mip->levels)
    mip->levels=zlevels;

  /* XXX at present, the decimation routine will not */
  /* handle volumetric images, so if we get one, we  */
  /* have to clamp the maximum MIP levels to 1       */
  if (image->zres > 1) {
    maxlevels = 1;
  }

  if (maxlevels > 0) {
    if (maxlevels < mip->levels)
      mip->levels = maxlevels;
  }

  if (rt_mynode() == 0) {
    char msgtxt[1024];
    sprintf(msgtxt, "Creating MIP Map: xlevels: %d  ylevels: %d  zlevels: %d  levels: %d", xlevels, ylevels, zlevels, mip->levels);
    rt_ui_message(MSG_0, msgtxt);
  }

  mip->images = (rawimage **) malloc(mip->levels * sizeof(rawimage *)); 
  if (mip->images == NULL) {
    free(mip);
    return NULL;
  }

  for (i=0; i<mip->levels; i++) {
    mip->images[i] = NULL;
  } 

  mip->images[0] = image;
  for (i=1; i<mip->levels; i++) {
    mip->images[i] = DecimateImage(mip->images[i - 1]);
  }

  return mip;
}

color MIPMap(const mipmap * mip, flt u, flt v, flt d) {
  int mapindex;
  flt mapflt;
  color col, col1, col2;

  if ((u <= 1.0) && (u >= 0.0) && (v <= 1.0) && (v >= 0.0)) {
    flt t;
    t = (d > 1.0) ? 1.0 : d;
    d = (t < 0.0) ? 0.0 : t;

    mapflt = d * (mip->levels - 0.9999); /* convert range to mapindex        */
    mapindex = (int) mapflt;             /* truncate to nearest integer      */
    mapflt = mapflt - mapindex;          /* fractional part of mip map level */

    /* interpolate between two nearest image maps */
    if (mapindex < (mip->levels - 2)) {
      col1 = ImageMap(mip->images[mapindex    ], u, v);
      col2 = ImageMap(mip->images[mapindex + 1], u, v);
      col.r = col1.r + mapflt*(col2.r - col1.r);
      col.g = col1.g + mapflt*(col2.g - col1.g);
      col.b = col1.b + mapflt*(col2.b - col1.b);
    }
    else {
      /* if mapindex is too high, use the highest,  */
      /* with no MIP-Map interpolation.             */
      col  = ImageMap(mip->images[mip->levels - 1], u, v);
    }
  } 
  else {
    col.r=0.0;
    col.g=0.0;
    col.b=0.0;
  }

  return col;
}


color ImageMap(const rawimage * image, flt u, flt v) {
  color col, colx, colx2;
  flt x, y, px, py;
  int ix, iy, nx, ny;
  unsigned char * ptr;
  const flt texel_inv = 1.0 / 255.0;

  /*
   *  Perform bilinear interpolation between 4 closest pixels.
   */
  nx = (image->xres > 1) ? 3 : 0;
  x = (image->xres - 1.0) * u; /* floating point X location */
  ix = (int) x;                /* integer X location        */
  px = x - ix;                 /* fractional X location     */

  ny = (image->yres > 1) ? image->xres * 3 : 0;
  y = (image->yres - 1.0) * v; /* floating point Y location */
  iy = (int) y;                /* integer Y location        */
  py = y - iy;                 /* fractional Y location     */

  /* pointer to the left lower pixel */
  ptr  = image->data + ((image->xres * iy) + ix) * 3; 

  /* interpolate between left and right lower pixels */
  colx.r = (flt) ((flt)ptr[0] + px*((flt)ptr[nx  ] - (flt) ptr[0])); 
  colx.g = (flt) ((flt)ptr[1] + px*((flt)ptr[nx+1] - (flt) ptr[1])); 
  colx.b = (flt) ((flt)ptr[2] + px*((flt)ptr[nx+2] - (flt) ptr[2])); 

  /* pointer to the left upper pixel */
  ptr  += ny; 

  /* interpolate between left and right upper pixels */
  colx2.r = ((flt)ptr[0] + px*((flt)ptr[nx  ] - (flt)ptr[0])); 
  colx2.g = ((flt)ptr[1] + px*((flt)ptr[nx+1] - (flt)ptr[1])); 
  colx2.b = ((flt)ptr[2] + px*((flt)ptr[nx+2] - (flt)ptr[2])); 

  /* interpolate between upper and lower interpolated pixels */
  col.r = (colx.r + py*(colx2.r - colx.r)) * texel_inv;
  col.g = (colx.g + py*(colx2.g - colx.g)) * texel_inv;
  col.b = (colx.b + py*(colx2.b - colx.b)) * texel_inv;

  return col;
} 


color VolImageMapNearest(const rawimage * img, flt u, flt v, flt w) {
  color col;
  flt x, y, z;
  long ix, iy, iz;
  long addr;

  x = (img->xres - 1.0) * u;  /* floating point X location */
  ix = (long) x;
  y = (img->yres - 1.0) * v;  /* floating point Y location */
  iy = (long) y;
  z = (img->zres - 1.0) * w;  /* floating point Z location */
  iz = (long) z;

  addr = ((iz * img->xres * img->yres) + (iy * img->xres) + ix) * 3; 
  col.r = img->data[addr    ];
  col.g = img->data[addr + 1];
  col.b = img->data[addr + 2];
 
  return col; 
}


color VolImageMapTrilinear(const rawimage * img, flt u, flt v, flt w) {
  color col, colL, colU, colll, colul, colLL, colUL;
  flt x, y, z, px, py, pz;
  long ix, iy, iz, nx, ny, nz;
  unsigned char *llptr, *ulptr, *LLptr, *ULptr;
  long addr;
  const flt texel_inv = 1.0 / 255.0;

  /*
   *  Perform trilinear interpolation between 8 closest pixels.
   */
  nx = (img->xres > 1) ? 3L : 0L;
  x = (img->xres - 1.0) * u;  /* floating point X location */
  ix = (long) x;              /* integer X location        */
  px = x - ix;                /* fractional X location     */

  ny = (img->yres > 1) ? (img->xres * 3L) : 0L;
  y = (img->yres - 1.0) * v;  /* floating point Y location */
  iy = (long) y;              /* integer Y location        */
  py = y - iy;                /* fractional Y location     */

  nz = (img->zres > 1) ? (img->xres * img->yres * 3L) : 0L;
  z = (img->zres - 1.0) * w;  /* floating point Z location */
  iz = (long) z;              /* integer Z location        */
  pz = z - iz;                /* fractional Z location     */

  addr = ((img->xres*img->yres * iz) + (img->xres * iy) + ix) * 3L; 

  /* pointer to the lower left lower pixel (Y  ) */
  llptr = img->data + addr;

  /* pointer to the lower left upper pixel (Y+1) */
  ulptr = llptr + ny;

  /* pointer to the upper left lower pixel (Z+1) (Y  ) */
  LLptr = llptr + nz;
  /* pointer to the upper left upper pixel (Z+1) (Y+1) */
  ULptr = LLptr + ny;

  /* interpolate between left and right lower pixels */
  colll.r = (flt) ((flt)llptr[0] + px*((flt)llptr[nx  ] - (flt) llptr[0])); 
  colll.g = (flt) ((flt)llptr[1] + px*((flt)llptr[nx+1] - (flt) llptr[1])); 
  colll.b = (flt) ((flt)llptr[2] + px*((flt)llptr[nx+2] - (flt) llptr[2])); 

  /* interpolate between left and right upper pixels */
  colul.r = ((flt)ulptr[0] + px*((flt)ulptr[nx  ] - (flt)ulptr[0])); 
  colul.g = ((flt)ulptr[1] + px*((flt)ulptr[nx+1] - (flt)ulptr[1])); 
  colul.b = ((flt)ulptr[2] + px*((flt)ulptr[nx+2] - (flt)ulptr[2])); 

  /* interpolate between left and right lower pixels */
  colLL.r = (flt) ((flt)LLptr[0] + px*((flt)LLptr[nx  ] - (flt) LLptr[0])); 
  colLL.g = (flt) ((flt)LLptr[1] + px*((flt)LLptr[nx+1] - (flt) LLptr[1])); 
  colLL.b = (flt) ((flt)LLptr[2] + px*((flt)LLptr[nx+2] - (flt) LLptr[2])); 

  /* interpolate between left and right upper pixels */
  colUL.r = ((flt)ULptr[0] + px*((flt)ULptr[nx  ] - (flt)ULptr[0])); 
  colUL.g = ((flt)ULptr[1] + px*((flt)ULptr[nx+1] - (flt)ULptr[1])); 
  colUL.b = ((flt)ULptr[2] + px*((flt)ULptr[nx+2] - (flt)ULptr[2])); 

  /* interpolate between upper and lower interpolated pixels */
  colL.r = (colll.r + py*(colul.r - colll.r));
  colL.g = (colll.g + py*(colul.g - colll.g));
  colL.b = (colll.b + py*(colul.b - colll.b));

  /* interpolate between upper and lower interpolated pixels */
  colU.r = (colLL.r + py*(colUL.r - colLL.r));
  colU.g = (colLL.g + py*(colUL.g - colLL.g));
  colU.b = (colLL.b + py*(colUL.b - colLL.b));

  /* interpolate between upper and lower interpolated pixels */
  col.r = (colL.r + pz*(colU.r - colL.r)) * texel_inv;
  col.g = (colL.g + pz*(colU.g - colL.g)) * texel_inv;
  col.b = (colL.b + pz*(colU.b - colL.b)) * texel_inv;

  return col;
}
 

color VolMIPMap(const mipmap * mip, flt u, flt v, flt w, flt d) {
  int mapindex;
  flt mapflt;
  color col, col1, col2;

  if ((u <= 1.0) && (u >= 0.0) && 
      (v <= 1.0) && (v >= 0.0) &&
      (w <= 1.0) && (w >= 0.0)) {
    flt t;
    t = (d > 1.0) ? 1.0 : d;
    d = (t < 0.0) ? 0.0 : t;

    mapflt = d * (mip->levels - 0.9999); /* convert range to mapindex        */
    mapindex = (int) mapflt;             /* truncate to nearest integer      */
    mapflt = mapflt - mapindex;          /* fractional part of mip map level */

    /* interpolate between two nearest image maps */
    if (mapindex < (mip->levels - 2)) {
      col1 = VolImageMapTrilinear(mip->images[mapindex    ], u, v, w);
      col2 = VolImageMapTrilinear(mip->images[mapindex + 1], u, v, w);
      col.r = col1.r + mapflt*(col2.r - col1.r);
      col.g = col1.g + mapflt*(col2.g - col1.g);
      col.b = col1.b + mapflt*(col2.b - col1.b);
    } else {
      /* if mapindex is too high, use the highest,  */
      /* with no MIP-Map interpolation.             */
      col  = VolImageMapTrilinear(mip->images[mip->levels - 1], u, v, w);
    }
  } else {
    col.r=0.0;
    col.g=0.0;
    col.b=0.0;
  }

  return col;
}

