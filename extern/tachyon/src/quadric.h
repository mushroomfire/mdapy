/* 
 * quadric.h - This file contains the defines for quadrics.
 *
 *  $Id: quadric.h,v 1.12 2011/02/05 08:10:11 johns Exp $
 */

typedef struct {
  flt a; flt b; flt c;
  flt d; flt e; flt f;
  flt g; flt h; flt i; flt j;
} quadmatrix;

 
typedef struct {
  RT_OBJECT_HEAD
  vector ctr;      /**< center of quadric object            */
  flt bbox;        /**< Size of user-defined bounding box   */
  quadmatrix mat;  /**< quadric function coefficient matrix */
} quadric; 


quadric * newquadric(void);
void quadric_intersect(const quadric *, ray *);
void quadric_normal(const quadric *, const vector *, const ray *, vector *);



