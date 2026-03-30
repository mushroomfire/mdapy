/*
 * vol.h - Volume rendering definitions etc.
 *
 * (C) Copyright 1994-2022 John E. Stone
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * $Id: extvol.h,v 1.13 2022/02/18 17:55:28 johns Exp $
 *
 */

typedef struct {
  RT_OBJECT_HEAD
  vector min;       /**< minimum box vertex coordinate                */
  vector max;       /**< maximum box vertex coordinate                */
  flt ambient;      /**< ambient lighting coefficient                 */
  flt diffuse;      /**< diffuse lighting coefficient                 */
  flt opacity;      /**< surface transmission factor                  */
  int samples;      /**< number of samples to take through volume     */
  flt (* evaluator)(flt, flt, flt); /**< user-defined sample fctn ptr */
} extvol;

extvol * newextvol(void * voidtex, vector min, vector max, 
                   int samples, flt (* evaluator)(flt, flt, flt));
color ext_volume_texture(const vector *, const texture *, ray *);


