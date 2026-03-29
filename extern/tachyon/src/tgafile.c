/* 
 * tgafile.c - This file contains the code to write 24 bit targa files...
 *
 *  $Id: tgafile.c,v 1.28 2011/02/07 07:41:51 johns Exp $
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TACHYON_INTERNAL 1
#include "tachyon.h"
#include "util.h"
#include "ui.h"
#include "imageio.h"
#include "tgafile.h"

typedef struct {
  unsigned short width;
  unsigned short height;
  FILE * ofp;
} tgahandle;

int createtgafile(char *name, unsigned short width, unsigned short height) {
  int filesize;
  FILE * ofp;

  filesize = 3*width*height + 18 - 10;
  
  if (name==NULL) {
    return IMAGEWRITEERR;
  } else {
    ofp=fopen(name, "w+b");
    if (ofp == NULL) {
      char msgtxt[2048];
      sprintf(msgtxt, "Cannot create %s for output!", name);
      rt_ui_message(MSG_ERR, msgtxt);
      rt_ui_message(MSG_ABORT, "Rendering Aborted.");
      return IMAGEWRITEERR;
    } 

    fputc(0, ofp); /* IdLength      */
    fputc(0, ofp); /* ColorMapType  */
    fputc(2, ofp); /* ImageTypeCode */
    fputc(0, ofp); /* ColorMapOrigin, low byte */
    fputc(0, ofp); /* ColorMapOrigin, high byte */
    fputc(0, ofp); /* ColorMapLength, low byte */
    fputc(0, ofp); /* ColorMapLength, high byte */
    fputc(0, ofp); /* ColorMapEntrySize */
    fputc(0, ofp); /* XOrigin, low byte */
    fputc(0, ofp); /* XOrigin, high byte */
    fputc(0, ofp); /* YOrigin, low byte */
    fputc(0, ofp); /* YOrigin, high byte */
    fputc((width & 0xff),         ofp); /* Width, low byte */
    fputc(((width >> 8) & 0xff),  ofp); /* Width, high byte */
    fputc((height & 0xff),        ofp); /* Height, low byte */
    fputc(((height >> 8) & 0xff), ofp); /* Height, high byte */
    fputc(24, ofp);   /* ImagePixelSize */
    fputc(0x20, ofp); /* ImageDescriptorByte 0x20 == flip vertically */

    fseek(ofp, filesize, 0);
    fprintf(ofp, "9876543210"); 

    fclose(ofp);
  } 

  return IMAGENOERR;
}    

void * opentgafile(char * filename) {
  tgahandle * tga; 
  tga = malloc(sizeof(tgahandle));
  
  tga->ofp=fopen(filename, "r+b");
  if (tga->ofp == NULL) {
    char msgtxt[2048];
    sprintf(msgtxt, "Cannot open %s for output!", filename);
    rt_ui_message(MSG_ERR, msgtxt);
    rt_ui_message(MSG_ABORT, "Rendering Aborted.");
    return NULL;
  } 

  fseek(tga->ofp, 12, 0);  
  tga->width = fgetc(tga->ofp);
  tga->width |= fgetc(tga->ofp) << 8;
  tga->height = fgetc(tga->ofp);
  tga->height |= fgetc(tga->ofp) << 8;

  return tga;
} 

void writetgaregion(void * voidhandle, int startx, int starty, 
                    int stopx, int stopy, unsigned char * buffer) {
  int x, y, totalx, totaly, xbytes, widthbytes, regionstart;
  unsigned char * bufpos;
  int filepos, numbytes;
  tgahandle * tga = (tgahandle *) voidhandle;
  unsigned char * fixbuf; 

  totalx = stopx - startx + 1;
  totaly = stopy - starty + 1;
  xbytes = totalx*3;
  widthbytes = tga->width*3;
  fixbuf = (unsigned char *) malloc(xbytes);
  if (fixbuf == NULL) {
    rt_ui_message(MSG_ERR, "writetgaregion: failed memory allocation!\n");
    return;
  }
 
  regionstart = 18 + (startx-1)*3 + widthbytes*(tga->height-starty-totaly+1);
  if (totalx == tga->width) {
    filepos=regionstart;
    if (filepos >= 18) {
      fseek(tga->ofp, filepos, 0); 
    } else {
      rt_ui_message(MSG_ERR, "writetgaregion: file ptr out of range!!!\n");
      free(fixbuf);
      return;  /* don't try to continue */
    }

    for (y=0; y<totaly; y++) {
      bufpos=buffer + xbytes*(totaly-y-1);
      for (x=0; x<xbytes; x+=3) {
        fixbuf[x    ] = bufpos[x + 2];
        fixbuf[x + 1] = bufpos[x + 1];
        fixbuf[x + 2] = bufpos[x    ];
      }
      numbytes = fwrite(fixbuf, 1, xbytes, tga->ofp);
      if (numbytes != xbytes) {
        char msgtxt[256];
        sprintf(msgtxt, "File write problem, %d bytes written.", numbytes);  
        rt_ui_message(MSG_ERR, msgtxt);
        free(fixbuf);
        return;  /* don't try to continue */
      }
    }
  } else {
    for (y=0; y<totaly; y++) {
      bufpos=buffer + xbytes*(totaly-y-1);
      filepos=regionstart + widthbytes*y;

      if (filepos >= 18) {
        fseek(tga->ofp, filepos, 0); 
  
        for (x=0; x<xbytes; x+=3) {
          fixbuf[x    ] = bufpos[x + 2];
          fixbuf[x + 1] = bufpos[x + 1];
          fixbuf[x + 2] = bufpos[x    ];
        }
 
        numbytes = fwrite(fixbuf, 1, xbytes, tga->ofp);
        if (numbytes != xbytes) {
          char msgtxt[256];
          sprintf(msgtxt, "File write problem, %d bytes written.", numbytes);  
          rt_ui_message(MSG_ERR, msgtxt);
          free(fixbuf);
          return;  /* don't try to continue */
        }
      } else {
        rt_ui_message(MSG_ERR, "writetgaregion: file ptr out of range!!!\n");
        free(fixbuf);
        return;  /* don't try to continue */
      }
    }
  }

  free(fixbuf);
}

void closetgafile(void * voidhandle) {
  tgahandle * tga = (tgahandle *) voidhandle;

  fclose(tga->ofp);
  free(tga);  
}

int readtga(char * name, int * xres, int * yres, unsigned char **imgdata) {
  int format, width, height, w1, w2, h1, h2, depth, flags;
  int imgsize, bytesread, i, tmp;
  FILE * ifp;

  ifp=fopen(name, "r");  
  if (ifp==NULL) {
    return IMAGEBADFILE; /* couldn't open the file */
  }

  /* read the targa header */
  getc(ifp); /* ID length */
  getc(ifp); /* colormap type */
  format = getc(ifp); /* image type */
  getc(ifp); /* color map origin */
  getc(ifp); /* color map origin */
  getc(ifp); /* color map length */
  getc(ifp); /* color map length */
  getc(ifp); /* color map entry size */
  getc(ifp); /* x origin */
  getc(ifp); /* x origin */
  getc(ifp); /* y origin */
  getc(ifp); /* y origin */
  w1 = getc(ifp); /* width (low) */
  w2 = getc(ifp); /* width (hi) */
  h1 = getc(ifp); /* height (low) */
  h2 = getc(ifp); /* height (hi) */
  depth = getc(ifp); /* image pixel size */
  flags = getc(ifp); /* image descriptor byte */

  if ((format != 2) || (depth != 24)) {
    fclose(ifp);
    return IMAGEUNSUP; /* unsupported targa format */
  }
    

  width = ((w2 << 8) | w1);
  height = ((h2 << 8) | h1);

  imgsize = 3 * width * height;
  *imgdata = malloc(imgsize);
  bytesread = fread(*imgdata, 1, imgsize, ifp);
  fclose(ifp);

  /* flip image vertically */
  if (flags == 0x20) {
    int rowsize = 3 * width;
    unsigned char * copytmp;

    copytmp = malloc(rowsize);

    for (i=0; i<height / 2; i++) {
      memcpy(copytmp, &((*imgdata)[rowsize*i]), rowsize);
      memcpy(&(*imgdata)[rowsize*i], &(*imgdata)[rowsize*(height - 1 - i)], rowsize);
      memcpy(&(*imgdata)[rowsize*(height - 1 - i)], copytmp, rowsize);
    }

    free(copytmp);       
  }


  /* convert from BGR order to RGB order */
  for (i=0; i<imgsize; i+=3) {
    tmp = (*imgdata)[i]; /* Blue */
    (*imgdata)[i] = (*imgdata)[i+2]; /* Red */
    (*imgdata)[i+2] = tmp; /* Blue */    
  }

  *xres = width;
  *yres = height;

  if (bytesread != imgsize) 
    return IMAGEREADERR;

  return IMAGENOERR;
}


int writetga(char * name, int xres, int yres, unsigned char *imgdata) {
  void * outfile;
  int rc = IMAGENOERR;

  rc = createtgafile(name, (unsigned short) xres, (unsigned short) yres);
  if (rc == IMAGENOERR) {
    outfile = opentgafile(name);

    if (outfile == NULL) 
      return IMAGEWRITEERR;

    writetgaregion(outfile, 1, 1, xres, yres, imgdata);
    closetgafile(outfile);
  }

  return rc;
}



