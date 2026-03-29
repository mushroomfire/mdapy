/* 
 * cylinder.h - This file contains the defines for cylinders etc.
 *
 *  $Id: cylinder.h,v 1.11 2011/02/05 08:10:11 johns Exp $
 */

object * newcylinder(void *, vector, vector, flt);
object * newfcylinder(void *, vector, vector, flt);

#ifdef CYLINDER_PRIVATE

/**
 * Types for cylinder objects
 */
typedef struct {
  RT_OBJECT_HEAD
  vector ctr;       /**< starting endpoint of cylinder */
  vector axis;      /**< cylinder axis                 */
  flt rad;
} cylinder;

static void cylinder_intersect(const cylinder *, ray *);
static void fcylinder_intersect(const cylinder *, ray *);

static int cylinder_bbox(void * obj, vector * min, vector * max);
static int fcylinder_bbox(void * obj, vector * min, vector * max);

static void cylinder_normal(const cylinder *, const vector *, const ray *, vector *);
#endif

