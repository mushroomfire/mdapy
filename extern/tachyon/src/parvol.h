/*
 * parvol.h - Volume rendering definitions etc.
 *
 * (C) Copyright 1994-2022 John E. Stone
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * $Id: parvol.h,v 1.6 2022/02/18 17:55:28 johns Exp $
 *
 */

typedef struct {
  RT_OBJECT_HEAD
  vector min;      /**< minimum axis-aligned box coordinate */
  vector max;      /**< maximum axis-aligned box coordinate */
  flt ambient;     /**< ambient lighting coefficient */
  flt diffuse;     /**< diffuse lighting coefficient */
  flt opacity;     /**< transmissive surface factor */
  int samples;     /**< number of volumetric samples to take */
  flt (* evaluator)(flt, flt, flt); /**< sample fctn pointer */
} parvol;

parvol * newparvol();
color par_volume_texture(vector *, texture *, ray *);


