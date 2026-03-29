/* 
 * triangle.h - This file contains the defines for triangles etc.
 *
 *  $Id: triangle.h,v 1.21 2011/02/05 08:10:11 johns Exp $
 */

object * newtri(void *, vector, vector, vector);
object * newstri(void *, vector, vector, vector, vector, vector, vector);
void stri_normal_fixup(object *, int mode);
object * newvcstri(void *, vector, vector, vector, vector, vector, vector,
                   color, color, color);
void vcstri_normal_fixup(object *, int mode);
color vcstri_color(const vector * hit, const texture * tex, const ray * incident);

#ifdef TRIANGLE_PRIVATE

#define TRIXMAJOR 0
#define TRIYMAJOR 1
#define TRIZMAJOR 2
 
typedef struct {
  RT_OBJECT_HEAD
  vector edge2;    /**< edge vector between v0 and v2 */
  vector edge1;    /**< edge vector between v0 and v1 */
  vector v0;       /**< triangle vertex v0            */
} tri; 

typedef struct {
  RT_OBJECT_HEAD
  vector edge2;    /**< edge vector between v0 and v2 */
  vector edge1;    /**< edge vector between v0 and v1 */
  vector v0;       /**< triangle vertex v0            */
  vector n0;       /**< surface normal for v0         */
  vector n1;       /**< surface normal for v1         */
  vector n2;       /**< surface normal for v2         */
} stri; 

typedef struct {
  RT_OBJECT_HEAD
  vector edge2;    /**< edge vector between v0 and v2 */
  vector edge1;    /**< edge vector between v0 and v1 */
  vector v0;       /**< triangle vertex v0            */
  vector n0;       /**< surface normal for v0         */
  vector n1;       /**< surface normal for v1         */
  vector n2;       /**< surface normal for v2         */
  color  c0;       /**< surface color for v0          */
  color  c1;       /**< surface color for v1          */
  color  c2;       /**< surface color for v2          */
} vcstri; 

static int tri_bbox(void * obj, vector * min, vector * max);

static void tri_intersect(const tri *, ray *);

static void tri_normal(const tri *, const vector *, const ray *, vector *);
static void stri_normal(const stri *, const vector *, const ray *, vector *);
static void stri_normal_reverse(const stri *, const vector *, const ray *, vector *);
static void stri_normal_guess(const stri *, const vector *, const ray *, vector *);

#endif


