/*
 *  pngfile.c - This file deals with PNG format image files (reading/writing)
 *
 *  $Id: pngfile.c,v 1.10 2011/02/07 07:41:51 johns Exp $
 */ 

/*
 * This code requires support from libpng, and libz.
 * For our purposes, we're interested only in the 3 byte per pixel 24 bit
 * RGB input/output.  Probably won't implement any decent checking at this 
 * point.
 */ 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TACHYON_INTERNAL 1
#include "tachyon.h"
#include "util.h"
#include "imageio.h" /* error codes etc */
#include "pngfile.h" /* the protos for this file */

#if !defined(USEPNG)

int readpng(const char *name, int *xres, int *yres, unsigned char **imgdata) {
  return IMAGEUNSUP;
}

int writepng(const char *name, int xres, int yres, unsigned char *imgdata) {
  return IMAGEUNSUP;
}

#else

#include "png.h" /* the libpng library headers */

/* The png_jmpbuf() macro, used in error handling, became available in
 * libpng version 1.0.6.  If you want to be able to run your code with older
 * versions of libpng, you must define the macro yourself (but only if it
 * is not already defined by libpng!).
 */
#ifndef png_jmpbuf
#  define png_jmpbuf(png_ptr) ((png_ptr)->jmpbuf)
#endif

int readpng(const char *name, int *xres, int *yres, unsigned char **imgdata) {
  FILE * ifp;
  png_structp png_ptr;
  png_infop info_ptr;
  png_bytep *row_pointers;
  int x, y;

  /* Create and initialize the png_struct with the default error handlers */
  png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (png_ptr == NULL) {
    return IMAGEALLOCERR; /* Could not initialize PNG library, return error */
  }

  /* Allocate/initialize the memory for image information.  REQUIRED. */
  info_ptr = png_create_info_struct(png_ptr);
  if (info_ptr == NULL) {
    png_destroy_read_struct(&png_ptr, (png_infopp)NULL, (png_infopp)NULL);
    return IMAGEALLOCERR; /* Could not initialize PNG library, return error */
  }

  /* open input file before doing any more PNG decompression setup */
  if ((ifp = fopen(name, "rb")) == NULL) 
    return IMAGEBADFILE; /* Could not open image, return error */

  /* Set error handling for setjmp/longjmp method of libpng error handling */
  if (setjmp(png_jmpbuf(png_ptr))) {
    /* Free all of the memory associated with the png_ptr and info_ptr */
    png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)NULL);
    /* If we get here, we had a problem reading the file */
    fclose(ifp);
    return IMAGEBADFILE; /* Could not open image, return error */
  }

  /* Set up the input control if you are using standard C streams */
  png_init_io(png_ptr, ifp);

  /* one-shot call to read the whole PNG file into memory */
  png_read_png(png_ptr, info_ptr,
    PNG_TRANSFORM_STRIP_16 | PNG_TRANSFORM_PACKING | PNG_TRANSFORM_STRIP_ALPHA,
    NULL);
  *xres = png_get_image_width(png_ptr, info_ptr);
  *yres = png_get_image_height(png_ptr, info_ptr);

  /* copy pixel data into our own data structures */
  row_pointers = png_get_rows(png_ptr, info_ptr); 
 
  *imgdata = (unsigned char *) malloc(3 * (*xres) * (*yres));
  if ((*imgdata) == NULL) {
    return IMAGEALLOCERR;
  }

  for (y=0; y<(*yres); y++) { 
    unsigned char *img = &(*imgdata)[(y * (*xres) * 3)];
    for (x=0; x<(*xres); x++) { 
      img[(x*3)    ] = row_pointers[(*yres) - y - 1][x    ]; 
      img[(x*3) + 1] = row_pointers[(*yres) - y - 1][x + 1]; 
      img[(x*3) + 2] = row_pointers[(*yres) - y - 1][x + 2]; 
    }
  }

  /* clean up after the read, and free any memory allocated - REQUIRED */
  png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)NULL);

  fclose(ifp); /* Close the input file */

  return IMAGENOERR;  /* No fatal errors */
}


int writepng(const char *name, int xres, int yres, unsigned char *imgdata) {
  FILE *ofp;
  png_structp png_ptr;
  png_infop info_ptr;
  png_bytep *row_pointers;
  png_textp text_ptr;
  int y;

  /* Create and initialize the png_struct with the default error handlers */
  png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (png_ptr == NULL) {
    return IMAGEALLOCERR; /* Could not initialize PNG library, return error */
  }

  /* Allocate/initialize the memory for image information.  REQUIRED. */
  info_ptr = png_create_info_struct(png_ptr);
  if (info_ptr == NULL) {
    png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
    return IMAGEALLOCERR; /* Could not initialize PNG library, return error */
  }

  /* open output file before doing any more PNG compression setup */
  if ((ofp = fopen(name, "wb")) == NULL) {
    return IMAGEBADFILE;
  }

  /* Set error handling for setjmp/longjmp method of libpng error handling */
  if (setjmp(png_jmpbuf(png_ptr))) {
    /* Free all of the memory associated with the png_ptr and info_ptr */
    png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
    /* If we get here, we had a problem writing the file */
    fclose(ofp);
    return IMAGEBADFILE; /* Could not open image, return error */
  }

  /* Set up the input control if you are using standard C streams */
  png_init_io(png_ptr, ofp);

  png_set_IHDR(png_ptr, info_ptr, xres, yres, 
               8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, 
               PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

  png_set_gAMA(png_ptr, info_ptr, 1.0);

  text_ptr = (png_textp) png_malloc(png_ptr, (png_uint_32)sizeof(png_text) * 2);
   
  text_ptr[0].key = "Description";
  text_ptr[0].text = "A scene rendered by the Tachyon ray tracer";
  text_ptr[0].compression = PNG_TEXT_COMPRESSION_NONE; 
#ifdef PNG_iTXt_SUPPORTED
  text_ptr[0].lang = NULL;
#endif

  text_ptr[1].key = "Software";
  text_ptr[1].text = "Tachyon Parallel/Multiprocessor Ray Tracer";
  text_ptr[1].compression = PNG_TEXT_COMPRESSION_NONE; 
#ifdef PNG_iTXt_SUPPORTED
  text_ptr[1].lang = NULL;
#endif
  png_set_text(png_ptr, info_ptr, text_ptr, 1);

  row_pointers = png_malloc(png_ptr, yres*sizeof(png_bytep));
  for (y=0; y<yres; y++) {
    row_pointers[yres - y - 1] = &imgdata[y * xres * 3];
  }

  png_set_rows(png_ptr, info_ptr, row_pointers); 

  /* one-shot call to write the whole PNG file into memory */
  png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);

  png_free(png_ptr, row_pointers);
  png_free(png_ptr, text_ptr);
 
  /* clean up after the write and free any memory allocated - REQUIRED */
  png_destroy_write_struct(&png_ptr, (png_infopp)NULL);

  fclose(ofp); /* close the output file */

  return IMAGENOERR; /* No fatal errors */
}

#endif
