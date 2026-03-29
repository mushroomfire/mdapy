/*
 * parvol.h - Volume rendering definitions etc.
 *
 *
 *  $Id: parvol.h,v 1.5 2011/02/05 08:10:11 johns Exp $
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


