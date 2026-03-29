/*
 *  winbmp.c - This file deals with Windows Bitmap image files 
 *             (reading/writing)
 *
 *  $Id: winbmp.c,v 1.4 2011/02/07 07:41:51 johns Exp $
 */ 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TACHYON_INTERNAL 1
#include "tachyon.h"
#include "util.h"
#include "imageio.h" /* error codes etc */
#include "winbmp.h"    /* the protos for this file */

static void write_le_int32(FILE * dfile, int num) {
  fputc((num      ) & 0xFF, dfile);
  fputc((num >> 8 ) & 0xFF, dfile);
  fputc((num >> 16) & 0xFF, dfile);
  fputc((num >> 24) & 0xFF, dfile);
}

static void write_le_int16(FILE * dfile, int num) {
  fputc((num      ) & 0xFF, dfile);
  fputc((num >> 8 ) & 0xFF, dfile);
}

int writebmp(char * filename, int xs, int ys, unsigned char * img) {
  if (img != NULL) {
    FILE * dfile;

    if ((dfile = fopen(filename, "wb")) != NULL) {
      int i, y; /* loop variables */
      int imgdataoffset = 14 + 40;     /* file header + bitmap header size */
      int rowsz = ((xs * 3) + 3) & -4; /* size of one padded row of pixels */
      int imgdatasize = rowsz * ys;    /* size of image data */
      int filesize = imgdataoffset + imgdatasize;
      unsigned char * rowbuf = NULL; 

      /* write out bitmap file header (14 bytes) */
      fputc('B', dfile);
      fputc('M', dfile);
      write_le_int32(dfile, filesize);
      write_le_int16(dfile, 0);
      write_le_int16(dfile, 0);
      write_le_int32(dfile, imgdataoffset);

      /* write out bitmap header (40 bytes) */
      write_le_int32(dfile, 40); /* size of bitmap header structure */
      write_le_int32(dfile, xs); /* size of image in x */
      write_le_int32(dfile, ys); /* size of image in y */
      write_le_int16(dfile, 1);  /* num color planes (only "1" is legal) */
      write_le_int16(dfile, 24); /* bits per pixel */

      /* fields added in Win 3.x */
      write_le_int32(dfile, 0);           /* compression used (0 == none) */
      write_le_int32(dfile, imgdatasize); /* size of bitmap in bytes      */
      write_le_int32(dfile, 11811);       /* X pixels per meter (300dpi)  */
      write_le_int32(dfile, 11811);       /* Y pixels per meter (300dpi)  */
      write_le_int32(dfile, 0);           /* color count (0 for RGB)      */
      write_le_int32(dfile, 0);           /* important colors (0 for RGB) */

      /* write out actual image data */
      rowbuf = (unsigned char *) malloc(rowsz);
      if (rowbuf != NULL) {
        memset(rowbuf, 0, rowsz); /* clear the buffer (and padding) to black */

        for (y=0; y<ys; y++) {
          int addr = xs * 3 * y;

          /* write one row of the image, in reversed RGB -> BGR pixel order */
          /* padding bytes remain 0's, shouldn't have to re-clear them. */
          for (i=0; i<rowsz; i+=3) {
            rowbuf[i    ] = img[addr + i + 2]; /* blue  */
            rowbuf[i + 1] = img[addr + i + 1]; /* green */
            rowbuf[i + 2] = img[addr + i    ]; /* red   */
          }

          fwrite(rowbuf, rowsz, 1, dfile); /* write the whole row of pixels */
        }
        free(rowbuf);
      }
      fclose(dfile);
    }
  }

  return IMAGENOERR;
}




