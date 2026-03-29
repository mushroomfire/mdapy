/*
 *  imageio.c - This file deals with reading/writing image files
 *
 *  $Id: imageio.c,v 1.30 2013/04/21 16:58:19 johns Exp $
 */ 

/* For our puposes, we're interested only in the 3 byte per pixel 24 bit
 * truecolor sort of file..
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TACHYON_INTERNAL 1
#include "tachyon.h"
#include "parallel.h"
#include "util.h"
#include "imageio.h"
#include "ppm.h"     /* 24-bit and 48-bit PPM files */
#include "psd.h"     /* 24-bit and 48-bit Photoshop files */
#include "tgafile.h" /* 24-bit Truevision Targa files */
#include "jpeg.h"    /* JPEG files */
#include "pngfile.h" /* PNG files  */
#include "sgirgb.h"  /* 24-bit SGI RGB files */
#include "winbmp.h"  /* 24-bit Windows Bitmap files */
#include "ui.h"      /* UI error messages */

static 
int fakeimage(char * name, int * xres, int * yres, unsigned char ** imgdata) {
  int i, imgsize;

  if (rt_mynode() == 0) {
    char msgtxt[2048];
    sprintf(msgtxt, "Error loading image %s.  Faking it using solid gray.", name);
    rt_ui_message(MSG_0, msgtxt);
  }
 
  *xres = 4;
  *yres = 4;
  imgsize = 3 * (*xres) * (*yres);
  *imgdata = malloc(imgsize);
  for (i=0; i<imgsize; i++) {
    (*imgdata)[i] = 255;
  }

  return IMAGENOERR;
}


int readimage(rawimage * img) {
  int rc;
  int xres, yres, zres;
  unsigned char * imgdata;
  char * name = img->name;
  char msgtxt[2048];
 
  xres=1;
  yres=1;
  zres=1;

  if (strstr(name, ".ppm")) { 
    rc = readppm(name, &xres, &yres, &imgdata);
  } else if (strstr(name, ".tga")) {
    rc = readtga(name, &xres, &yres, &imgdata);
  } else if (strstr(name, ".jpg")) {
    rc = readjpeg(name, &xres, &yres, &imgdata);
  } else if (strstr(name, ".png")) {
    rc = readpng(name, &xres, &yres, &imgdata);
  } else if (strstr(name, ".gif")) {
    rc = IMAGEUNSUP; 
  } else if (strstr(name, ".tiff")) {
    rc = IMAGEUNSUP; 
  } else if (strstr(name, ".rgb")) {
    rc = IMAGEUNSUP; 
  } else if (strstr(name, ".xpm")) {
    rc = IMAGEUNSUP; 
  } else {
    rc = readppm(name, &xres, &yres, &imgdata);
  } 

  switch (rc) {
    case IMAGEREADERR:
      if (rt_mynode() == 0) {
        sprintf(msgtxt, "Short read encountered while loading image %s", name);
        rt_ui_message(MSG_0, msgtxt);
      }
      rc = IMAGENOERR; /* remap to non-fatal error */
      break;

    case IMAGEUNSUP:
      if (rt_mynode() == 0) {
        sprintf(msgtxt, "Cannot read unsupported format for image %s", name);
        rt_ui_message(MSG_0, msgtxt);
      }
      break;
  }    

  /* If the image load failed, create a tiny white colored image to fake it */ 
  /* this allows a scene to render even when a file can't be loaded */
  if (rc != IMAGENOERR) {
    rc = fakeimage(name, &xres, &yres, &imgdata);
  }

  /* If we succeeded in loading the image, return it. */
  if (rc == IMAGENOERR) { 
    img->xres = xres;
    img->yres = yres;
    img->zres = zres;
    img->bpp = 3;  
    img->data = imgdata;
  }

  return rc;
}


void minmax_rgb96f(int xres, int yres, const float *fimg, 
                   float *min, float *max) {
  int i, sz;
  float minval, maxval;

  minval=maxval=fimg[0];

  sz = xres * yres * 3;
  for (i=0; i<sz; i++) {
    if (fimg[i] > maxval)
      maxval=fimg[i];
    if (fimg[i] < minval)
      minval=fimg[i];
  }

  if (min != NULL)
    *min = minval;

  if (max != NULL)
    *max = maxval;
}


void normalize_rgb96f(int xres, int yres, float *fimg) {
  int i, sz;
  float min, max, scale;
  sz = xres * yres * 3;
  minmax_rgb96f(xres, yres, fimg, &min, &max);
  scale = 1.0f / (max-min);
  for (i=0; i<sz; i++)
    fimg[i] = (fimg[i]-min) * scale; 
}


void gamma_rgb96f(int xres, int yres, float *fimg, float gamma) {
  float invgamma = 1.0f / gamma;
  int i, sz;
  sz = xres * yres * 3;
  for (i=0; i<sz; i++)
    fimg[i] = POW(fimg[i], invgamma);
}


unsigned char * image_rgb24_from_rgb96f(int xres, int yres, float *fimg) { 
  unsigned char *img;
  int x, y, R, G, B;
  img = (unsigned char *) malloc(xres * yres * 3);

  for (y=0; y<yres; y++) {
    for (x=0; x<xres; x++) {
      int addr = (xres * y + x) * 3;
      R = (int) (fimg[addr    ] * 255.0f); /* quantize float to integer */
      G = (int) (fimg[addr + 1] * 255.0f); /* quantize float to integer */
      B = (int) (fimg[addr + 2] * 255.0f); /* quantize float to integer */

      if (R > 255) R = 255;       /* clamp pixel value to range 0-255      */
      if (R < 0) R = 0;
      img[addr    ] = (byte) R;   /* Store final pixel to the image buffer */

      if (G > 255) G = 255;       /* clamp pixel value to range 0-255      */
      if (G < 0) G = 0;
      img[addr + 1] = (byte) G;   /* Store final pixel to the image buffer */

      if (B > 255) B = 255;       /* clamp pixel value to range 0-255      */
      if (B < 0) B = 0;
      img[addr + 2] = (byte) B;   /* Store final pixel to the image buffer */
    }
  }

  return img;
}


float * image_crop_rgb96f(int xres, int yres, float *fimg, 
                          int szx, int szy, int sx, int sy) {
  float *cropped;
  int x, y;

  cropped = (float *) malloc(szx * szy * 3 * sizeof(float));
  memset(cropped, 0, szx * szy * 3 * sizeof(float));
  
  for (y=0; y<szy; y++) {
    int oaddr = ((y+sy) * xres + sx) * 3;
    if ((y+sy >= 0) && (y+sy < yres)) {
      for (x=0; x<szx; x++) {
        if ((x+sx >= 0) && (x+sx < xres)) {
          int addr = (szx * y + x) * 3;
          cropped[addr    ] = fimg[oaddr + (x*3)    ];
          cropped[addr + 1] = fimg[oaddr + (x*3) + 1];
          cropped[addr + 2] = fimg[oaddr + (x*3) + 2];
        }
      }
    }
  }

  return cropped;
}


unsigned char * image_crop_rgb24(int xres, int yres, unsigned char *img, 
                                 int szx, int szy, int sx, int sy) {
  unsigned char *cropped;
  int x, y;

  cropped = (unsigned char *) malloc(szx * szy * 3 * sizeof(unsigned char));
  memset(cropped, 0, szx * szy * 3 * sizeof(unsigned char));
  
  for (y=0; y<szy; y++) {
    int oaddr = ((y+sy) * xres + sx) * 3;
    if ((y+sy >= 0) && (y+sy < yres)) {
      for (x=0; x<szx; x++) {
        if ((x+sx >= 0) && (x+sx < xres)) {
          int addr = (szx * y + x) * 3;
          cropped[addr    ] = img[oaddr + (x*3)    ];
          cropped[addr + 1] = img[oaddr + (x*3) + 1];
          cropped[addr + 2] = img[oaddr + (x*3) + 2];
        }
      }
    }
  }

  return cropped;
}


unsigned char * image_rgb48be_from_rgb96f(int xres, int yres, float *fimg) { 
  int x, y, R, G, B;
  unsigned char *img = (unsigned char *) malloc(xres * yres * 6);

  for (y=0; y<yres; y++) {
    for (x=0; x<xres; x++) {
      int faddr = (xres * y + x) * 3;
      int iaddr = faddr *  2;

      R = (int) (fimg[faddr    ] * 65535.0f); /* quantize float to integer */
      G = (int) (fimg[faddr + 1] * 65535.0f); /* quantize float to integer */
      B = (int) (fimg[faddr + 2] * 65535.0f); /* quantize float to integer */

      if (R > 65535) R = 65535;   /* clamp pixel value to range 0-65535    */
      if (R < 0) R = 0;
      img[iaddr    ] = (byte) ((R >> 8) & 0xff);
      img[iaddr + 1] = (byte) (R & 0xff);

      if (G > 65535) G = 65535;   /* clamp pixel value to range 0-65535    */
      if (G < 0) G = 0;
      img[iaddr + 2] = (byte) ((G >> 8) & 0xff);
      img[iaddr + 3] = (byte) (G & 0xff);

      if (B > 65535) B = 65535;   /* clamp pixel value to range 0-65535    */
      if (B < 0) B = 0;
      img[iaddr + 4] = (byte) ((B >> 8) & 0xff);
      img[iaddr + 5] = (byte) (B & 0xff);
    }
  }

  return img;
}


unsigned char * image_rgb48bepl_from_rgb96f(int xres, int yres, float *fimg) { 
  int x, y, R, G, B, sz;
  unsigned char *img = (unsigned char *) malloc(xres * yres * 6);

  sz = xres * yres * 2;
  for (y=0; y<yres; y++) {
    for (x=0; x<xres; x++) {
      int addr = xres * y + x;
      int faddr = addr * 3;
      int iaddr = addr * 2;
      int raddr = iaddr;
      int gaddr = iaddr + sz;
      int baddr = iaddr + (sz * 2);

      R = (int) (fimg[faddr    ] * 65535.0f); /* quantize float to integer */
      G = (int) (fimg[faddr + 1] * 65535.0f); /* quantize float to integer */
      B = (int) (fimg[faddr + 2] * 65535.0f); /* quantize float to integer */

      if (R > 65535) R = 65535;   /* clamp pixel value to range 0-65535    */
      if (R < 0) R = 0;
      img[raddr    ] = (byte) ((R >> 8) & 0xff);
      img[raddr + 1] = (byte) (R & 0xff);

      if (G > 65535) G = 65535;   /* clamp pixel value to range 0-65535    */
      if (G < 0) G = 0;
      img[gaddr    ] = (byte) ((G >> 8) & 0xff);
      img[gaddr + 1] = (byte) (G & 0xff);

      if (B > 65535) B = 65535;   /* clamp pixel value to range 0-65535    */
      if (B < 0) B = 0;
      img[baddr    ] = (byte) ((B >> 8) & 0xff);
      img[baddr + 1] = (byte) (B & 0xff);
    }
  }

  return img;
}


int writeimage(char * name, int xres, int yres, void *img, 
               int imgbufferformat, int fileformat) {
  if (img == NULL) 
    return IMAGENULLDATA;

  if (imgbufferformat == RT_IMAGE_BUFFER_RGB24) {
    unsigned char *imgbuf = (unsigned char *) img;
 
    switch (fileformat) {
      case RT_FORMAT_PPM:
        return writeppm(name, xres, yres, imgbuf);
    
      case RT_FORMAT_SGIRGB:
        return writergb(name, xres, yres, imgbuf);

      case RT_FORMAT_JPEG:
        return writejpeg(name, xres, yres, imgbuf);

      case RT_FORMAT_PNG:
        return writepng(name, xres, yres, imgbuf);

      case RT_FORMAT_WINBMP:
        return writebmp(name, xres, yres, imgbuf);

      case RT_FORMAT_TARGA:
        return writetga(name, xres, yres, imgbuf);       

      default:
        printf("Unsupported image format combination\n");
        return IMAGEUNSUP;
    } 
  } else {
    unsigned char *imgbuf = (unsigned char *) img;
    int rc;

    switch (fileformat) {
      case RT_FORMAT_PPM:
        imgbuf = image_rgb24_from_rgb96f(xres, yres, img);
        rc = writeppm(name, xres, yres, imgbuf);
        free(imgbuf);
        return rc;   
 
      case RT_FORMAT_SGIRGB:
        imgbuf = image_rgb24_from_rgb96f(xres, yres, img);
        rc = writergb(name, xres, yres, imgbuf);
        free(imgbuf);
        return rc;   

      case RT_FORMAT_JPEG:
        imgbuf = image_rgb24_from_rgb96f(xres, yres, img);
        rc = writejpeg(name, xres, yres, imgbuf);
        free(imgbuf);
        return rc;   

      case RT_FORMAT_PNG:
        imgbuf = image_rgb24_from_rgb96f(xres, yres, img);
        rc = writepng(name, xres, yres, imgbuf);
        free(imgbuf);
        return rc;   

      case RT_FORMAT_WINBMP:
        imgbuf = image_rgb24_from_rgb96f(xres, yres, img);
        rc = writebmp(name, xres, yres, imgbuf);
        free(imgbuf);
        return rc;   

      case RT_FORMAT_TARGA:
        imgbuf = image_rgb24_from_rgb96f(xres, yres, img);
        rc = writetga(name, xres, yres, imgbuf);       
        free(imgbuf);
        return rc;   

      case RT_FORMAT_PPM48:
        imgbuf = image_rgb48be_from_rgb96f(xres, yres, img);
        rc = writeppm48(name, xres, yres, imgbuf);
        free(imgbuf);
        return rc;   

      case RT_FORMAT_PSD48:
        imgbuf = image_rgb48bepl_from_rgb96f(xres, yres, img);
        rc = writepsd48(name, xres, yres, imgbuf);
        free(imgbuf);
        return rc;   

      default:
        printf("Unsupported image format combination\n");
        return IMAGEUNSUP;
    } 
  } 
}


