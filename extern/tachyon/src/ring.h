/* 
 * ring.h - This file contains the defines for rings etc.
 *
 *  $Id: ring.h,v 1.11 2011/02/05 08:10:11 johns Exp $
 */

object * newring(void * tex, vector ctr, vector norm, flt in, flt out);

#ifdef RING_PRIVATE 
typedef struct {
  RT_OBJECT_HEAD
  vector ctr;       /**< center of ring */
  vector norm;      /**< surface normal */
  flt inrad;        /**< inner ring radius (0.0 for disk) */
  flt outrad;       /**< outer ring raidus */
} ring; 

static int ring_bbox(void * obj, vector * min, vector * max);
static void ring_intersect(const ring *, ray *);
static void ring_normal(const ring *, const vector *, const ray * incident, vector *);
#endif

