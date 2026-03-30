/* 
 * image.h - Image Data Structures and definitions
 *
 * (C) Copyright 1994-2022 John E. Stone
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * $Id: image.h,v 1.4 2022/02/18 17:55:28 johns Exp $
 *
 */

/*
 * One day, this file will define a common structure
 * for all image / animation / volume type data structures.
 * This will allow me to design and implement a library of
 * pixel processing routines for pre and post processing of
 * image data in the rendering process.  Good examples will be
 * image map filtering, animated image maps, DCT/IDCT algorithms,
 * scalar volume data, a common set of file format readers and converters.
 */

typedef struct {
  int ID;                      /**< frame number */
  unsigned int info;           /**< bitmapped flags and values        */
                               /**< YUV, RGB, DCT, image, animation,  */
                               /**< volume, etc.                      */ 
  unsigned int loaded;         /**< memory residency information */
  unsigned int xs;             /**< pels in x dimension   */
  unsigned int ys;             /**< pels in y dimension   */
  unsigned int zs;             /**< pels in z dimension   */
  unsigned char * data;        /**< raw image/volume data */
  char filename[FILENAME_MAX]; /**< filename or remote access identifier */
} Frame;


