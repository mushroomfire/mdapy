/* 
 * cone.h - This file contains the defines for cone primitives.
 *
 *  Added by Alexander Stukowski.
 */

object * newcone(void *, vector, vector, flt);

#ifdef CONE_PRIVATE

/**
 * Types for cone objects
 */
typedef struct {
  RT_OBJECT_HEAD
  vector ctr;       /**< starting endpoint of cone */
  vector axis;      /**< cone axis                 */
  flt rad;			/**< cone radius at end point. */
  flt height;		/**< cone height. */
  flt cos_angle;	/**< cosine of cone angle.     */
  flt sin_angle;	/**< sine of cone angle.     */
} cone;

static void cone_intersect(const cone *, ray *);

static int cone_bbox(void * obj, vector * min, vector * max);

static void cone_normal(const cone *, const vector *, const ray *, vector *);
#endif

