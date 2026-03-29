/* 
 * box.h - This file contains the defines for boxes etc.
 *
 *  $Id: box.h,v 1.9 2011/02/05 08:10:11 johns Exp $
 */

/**
 * axis-aligned box definition
 */ 
typedef struct {
  RT_OBJECT_HEAD
  vector min;     /**< minimum vertex coordinate */
  vector max;     /**< maximum vertex coordinate */
} box; 


box * newbox(void * tex, vector min, vector max);
void box_intersect(const box *, ray *);
void box_normal(const box *, const vector *, const ray *, vector *);
