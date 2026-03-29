/*
 * coordsys.h - defines for coordinate system routines.
 *  
 *  $Id: coordsys.h,v 1.3 1998/05/27 16:32:08 johns Exp $
 */

void xytopolar(flt, flt, flt, flt *, flt *);
void xyztocyl(vector, flt, flt *, flt *);
void xyztospr(vector, flt *, flt *);
