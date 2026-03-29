/*
 *  jpeg.c - This file deals with JPEG format image files (reading/writing)
 *
 *  $Id: jpeg.c,v 1.9 2011/02/07 07:41:51 johns Exp $
 */ 

/*
 * This code requires support from the Independent JPEG Group's libjpeg.
 * For our purposes, we're interested only in the 3 byte per pixel 24 bit
 * RGB output.  Probably won't implement any decent checking at this point.
 */ 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TACHYON_INTERNAL 1
#include "tachyon.h"
#include "util.h"
#include "imageio.h" /* error codes etc */
#include "jpeg.h"    /* the protos for this file */

#if !defined(USEJPEG)

int readjpeg(const char *name, int *xres, int *yres, unsigned char **imgdata) {
  return IMAGEUNSUP;
}

int writejpeg(const char *name, int xres, int yres, unsigned char *imgdata) {
  return IMAGEUNSUP;
}

#else

#include "jpeglib.h" /* the IJG jpeg library headers */

int readjpeg(const char *name, int *xres, int *yres, unsigned char **imgdata) {
  FILE * ifp;
  struct jpeg_decompress_struct cinfo; /* JPEG decompression struct */
  struct jpeg_error_mgr jerr;          /* JPEG Error handler */
  JSAMPROW row_pointer[1];             /* output row buffer */
  int row_stride;                      /* physical row width in output buf */

  /* open input file before doing any JPEG decompression setup */
  if ((ifp = fopen(name, "rb")) == NULL) 
    return IMAGEBADFILE; /* Could not open image, return error */

  /*
   * Note: The Independent JPEG Group's library does not have a way
   *       of returning errors without the use of setjmp/longjmp.
   *       This is a problem in multi-threaded environment, since setjmp
   *       and longjmp are declared thread-unsafe by many vendors currently.
   *       For now, JPEG decompression errors will result in the "default"
   *       error handling provided by the JPEG library, which is an error
   *       message and a fatal call to exit().  I'll have to work around this
   *       or find a reasonably thread-safe way of doing setjmp/longjmp..
   */

  cinfo.err = jpeg_std_error(&jerr); /* Set JPEG error handler to default */

  jpeg_create_decompress(&cinfo);    /* Create decompression context      */ 
  jpeg_stdio_src(&cinfo, ifp);       /* Set input mechanism to stdio type */
  jpeg_read_header(&cinfo, TRUE);    /* Read the JPEG header for info     */
  jpeg_start_decompress(&cinfo);     /* Prepare for actual decompression  */

  *xres = cinfo.output_width;        /* set returned image width  */
  *yres = cinfo.output_height;       /* set returned image height */

  /* Calculate the size of a row in the image */
  row_stride = cinfo.output_width * cinfo.output_components;

  /* Allocate the image buffer which will be returned to the caller */
  *imgdata = (unsigned char *) malloc(row_stride * cinfo.output_height);

  /* decompress the JPEG, one scanline at a time into the buffer */
  while (cinfo.output_scanline < cinfo.output_height) {
    row_pointer[0] = &((*imgdata)[(cinfo.output_scanline)*row_stride]);
    jpeg_read_scanlines(&cinfo, row_pointer, 1);
  }

  jpeg_finish_decompress(&cinfo);   /* Tell the JPEG library to cleanup   */
  jpeg_destroy_decompress(&cinfo);  /* Destroy JPEG decompression context */

  fclose(ifp); /* Close the input file */

  return IMAGENOERR;  /* No fatal errors */
}


int writejpeg(const char *name, int xres, int yres, unsigned char *imgdata) {
  FILE *ofp;
  struct jpeg_compress_struct cinfo;   /* JPEG compression struct */
  struct jpeg_error_mgr jerr;          /* JPEG error handler */
  JSAMPROW row_pointer[1];             /* output row buffer */
  int row_stride;                      /* physical row width in output buf */

  if ((ofp = fopen(name, "wb")) == NULL) {
    return IMAGEBADFILE;
  }

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);
  jpeg_stdio_dest(&cinfo, ofp);

  cinfo.image_width = xres;
  cinfo.image_height = yres;
  cinfo.input_components = 3;
  cinfo.in_color_space = JCS_RGB;

  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, 95, 0);

  jpeg_start_compress(&cinfo, TRUE);

  /* Calculate the size of a row in the image */
  row_stride = cinfo.image_width * cinfo.input_components;

  /* compress the JPEG, one scanline at a time into the buffer */
  while (cinfo.next_scanline < cinfo.image_height) {
    row_pointer[0] = &(imgdata[(yres - cinfo.next_scanline - 1)*row_stride]);
    jpeg_write_scanlines(&cinfo, row_pointer, 1);
  }

  jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);

  fclose(ofp);

  return IMAGENOERR; /* No fatal errors */
}

#endif
