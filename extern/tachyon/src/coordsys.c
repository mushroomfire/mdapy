/*
 * coordsys.c -  Routines to translate from one coordinate system to another.
 *
 *  $Id: coordsys.c,v 1.5 2012/10/17 04:25:57 johns Exp $
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TACHYON_INTERNAL 1
#include "tachyon.h"
#include "coordsys.h"

void xytopolar(flt x, flt y, flt rad, flt * u, flt * v) {
  flt r1;
  r1=x*x + y*y;  
  *v=SQRT(r1 / (rad*rad));
  if (y<0.0) 
    *u=1.0 - ACOS(x/SQRT(r1))/TWOPI;
  else 
    *u= ACOS(x/SQRT(r1))/TWOPI; 
}

void xyztocyl(vector pnt, flt height, flt * u, flt * v) {
  flt r1;

  r1=pnt.x*pnt.x + pnt.y*pnt.y;

  *v=pnt.z / height;
  if (pnt.y<0.0) 
    *u=1.0 - ACOS(pnt.x/SQRT(r1))/TWOPI;
  else 
    *u=ACOS(pnt.x/SQRT(r1))/TWOPI;
}

void xyztospr(vector pnt, flt * u, flt * v) {
  flt r1, phi, theta;
 
  r1=SQRT(pnt.x*pnt.x + pnt.y*pnt.y + pnt.z*pnt.z);

  phi=ACOS(-pnt.y/r1);   
  *v=phi/3.1415926;

  theta=ACOS((pnt.x/r1)/SIN(phi))/TWOPI;

  if (pnt.z > 0.0) 
    *u = theta;
  else 
    *u = 1 - theta; 
}


