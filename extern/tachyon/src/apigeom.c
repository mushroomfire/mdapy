/*
 * apigeom.c - This file contains all of the API calls that are defined for
 *         external driver code to use.  
 * 
 *  $Id: apigeom.c,v 1.14 2012/10/17 04:25:57 johns Exp $
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tachyon.h"

void VNorm(apivector *); /* hack proto, should fix later */

void rt_polycylinder(SceneHandle scene, void * tex, apivector * points, int numpts, flt rad) {
  apivector a;
  int i;

  if ((points == NULL) || (numpts == 0)) {
    return;
  }

  if (numpts > 0) {
    rt_sphere(scene, tex, points[0], rad);
    
    if (numpts > 1) {
      for (i=1; i<numpts; i++) {
        a.x = points[i].x - points[i-1].x;
        a.y = points[i].y - points[i-1].y;
        a.z = points[i].z - points[i-1].z;
        
        rt_fcylinder(scene, tex, points[i-1], a, rad);
        rt_sphere(scene, tex, points[i], rad);
      }
    }
  }
}

void rt_polycylinder3fv(SceneHandle scene, void * tex,
                        const float *points, int numpts, float rad) {
  float a[3];
  int i;

  if ((points == NULL) || (numpts == 0)) {
    return;
  }

  if (numpts > 0) {
    rt_sphere3fv(scene, tex, points, rad);
    
    if (numpts > 1) {
      for (i=1; i<numpts; i++) {
        int addr = i*3;
        a[0] = points[addr    ] - points[addr - 3];
        a[1] = points[addr + 1] - points[addr - 2];
        a[2] = points[addr + 2] - points[addr - 1];
        
        rt_fcylinder3fv(scene, tex, &points[addr-3], a, rad);
        rt_sphere3fv(scene, tex, &points[addr], rad);
      }
    }
  }
}

void rt_heightfield(SceneHandle scene, void * tex, apivector ctr, int m, int n, 
                    flt * field, flt wx, flt wy) {
  int xx,yy; 
  apivector v0, v1, v2; 
  flt xoff, yoff, zoff;

  xoff=ctr.x - (wx / 2.0);
  yoff=ctr.z - (wy / 2.0);
  zoff=ctr.y;

  for (yy=0; yy<(n-1); yy++) { 
    for (xx=0; xx<(m-1); xx++) {
      v0.x=wx*(xx    )/(m*1.0) + xoff; 
      v0.y=field[(yy    )*m + (xx    )] + zoff;
      v0.z=wy*(yy    )/(n*1.0) + yoff;

      v1.x=wx*(xx + 1)/(m*1.0) + xoff; 
      v1.y=field[(yy    )*m + (xx + 1)] + zoff;
      v1.z=wy*(yy    )/(n*1.0) + yoff;

      v2.x=wx*(xx + 1)/(m*1.0) + xoff; 
      v2.y=field[(yy + 1)*m + (xx + 1)] + zoff;
      v2.z=wy*(yy + 1)/(n*1.0) + yoff;

      rt_tri(scene, tex, v1, v0, v2);

      v0.x=wx*(xx    )/(m*1.0) + xoff;
      v0.y=field[(yy    )*m + (xx    )] + zoff;
      v0.z=wy*(yy    )/(n*1.0) + yoff;

      v1.x=wx*(xx    )/(m*1.0) + xoff;
      v1.y=field[(yy + 1)*m + (xx    )] + zoff;
      v1.z=wy*(yy + 1)/(n*1.0) + yoff;

      v2.x=wx*(xx + 1)/(m*1.0) + xoff;
      v2.y=field[(yy + 1)*m + (xx + 1)] + zoff;
      v2.z=wy*(yy + 1)/(n*1.0) + yoff;
 
      rt_tri(scene, tex, v0, v1, v2);
    }
  } 
} /* end of heightfield */


static void rt_sheightfield(SceneHandle scene, void * tex, apivector ctr, int m, int n, 
                    flt * field, flt wx, flt wy) {
  apivector * vertices;
  apivector * normals;
  apivector offset;
  flt xinc, yinc;
  int x, y, addr; 
   
  vertices = (apivector *) malloc(m*n*sizeof(apivector));
  normals = (apivector *) malloc(m*n*sizeof(apivector));

  offset.x = ctr.x - (wx / 2.0);
  offset.y = ctr.z - (wy / 2.0);
  offset.z = ctr.y;

  xinc = wx / ((flt) m-1);
  yinc = wy / ((flt) n-1);

  /* build vertex list */
  for (y=0; y<n; y++) { 
    for (x=0; x<m; x++) {
      addr = y*m + x;
      vertices[addr] = rt_vector(
        x * xinc + offset.x,
        field[addr] + offset.z,
        y * yinc + offset.y);
    }
  }

  /* build normals from vertex list */
  for (x=1; x<m; x++) {
    normals[x] = normals[(n - 1)*m + x] = rt_vector(0.0, 1.0, 0.0);
  }
  for (y=1; y<n; y++) {
    normals[y*m] = normals[y*m + (m-1)] = rt_vector(0.0, 1.0, 0.0);
  }
  for (y=1; y<(n-1); y++) {
    for (x=1; x<(m-1); x++) {
      addr = y*m + x;

      normals[addr] = rt_vector(
        -(field[addr + 1] - field[addr - 1]) / (2.0 * xinc), 
        1.0, 
        -(field[addr + m] - field[addr - m]) / (2.0 * yinc));

      VNorm(&normals[addr]);
    }
  }    

  /* generate actual triangles */
  for (y=0; y<(n-1); y++) {
    for (x=0; x<(m-1); x++) {
      addr = y*m + x;

      rt_stri(scene, tex, vertices[addr], vertices[addr + 1 + m], vertices[addr + 1],
                   normals[addr], normals[addr + 1 + m], normals[addr + 1]);
      rt_stri(scene, tex, vertices[addr], vertices[addr + m], vertices[addr + 1 + m],
                   normals[addr], normals[addr + m], normals[addr + 1 + m]);
    }
  }

  free(normals);
  free(vertices);
} /* end of smoothed heightfield */


static void adjust(flt *base, int xres, int yres, flt wx, flt wy, 
                   int xa, int ya, int x, int y, int xb, int yb, 
                   unsigned int *rndval) {
  flt d, v, len;
 
  len = SQRT(wx * wx + wy * wy);
 
  if (base[x + (xres*y)]==0.0) { 
    d=(abs(xa - xb) / (xres * 1.0))*wx + (abs(ya - yb) / (yres * 1.0))*wy; 

    v= (base[xa + (xres*ya)] + base[xb + (xres*yb)]) / 2.0 +
       ((((rt_rand(rndval) / RT_RAND_MAX) - 0.5)*d) / 4.0) * (len / 16.0);

    if (v < 0.0) v=0.0; 
    if (v > (xres + yres)) v=(xres + yres);
    base[x + (xres * y)]=v; 
 } 
}

static void subdivide(flt *base, int xres, int yres, flt wx, flt wy,
                      int x1, int y1, int x2, int y2, unsigned int *rndval) {
  int x, y;

  if (((x2 - x1) < 2) && ((y2 - y1) < 2)) { return; }

  x=(x1 + x2) / 2;
  y=(y1 + y2) / 2;

  adjust(base, xres, yres, wx, wy, x1, y1, x, y1, x2, y1, rndval);
  adjust(base, xres, yres, wx, wy, x2, y1, x2, y, x2, y2, rndval);
  adjust(base, xres, yres, wx, wy, x1, y2, x, y2, x2, y2, rndval);
  adjust(base, xres, yres, wx, wy, x1, y1, x1, y, x1, y2, rndval);

 
  if (base[x + xres*y]==0.0) {
    base[x + (xres * y)]=(base[x1 + xres*y1] + base[x2 + xres*y1] +
                          base[x2 + xres*y2] + base[x1 + xres*y2]   )/4.0;
  }
 
  subdivide(base, xres, yres, wx, wy, x1, y1 ,x ,y, rndval);
  subdivide(base, xres, yres, wx, wy, x, y1, x2, y, rndval);
  subdivide(base, xres, yres, wx, wy, x, y, x2, y2, rndval);
  subdivide(base, xres, yres, wx, wy, x1, y, x, y2, rndval);
}

void rt_landscape(SceneHandle scene, void * tex, int m, int n, 
                  apivector ctr, flt wx, flt wy) {
  int totalsize, x, y;
  flt * field; 
  unsigned int rndval; /* random number.. */

  totalsize=m*n;
  rndval = totalsize;
 
  field=(flt *) malloc(totalsize*sizeof(flt));

  for (y=0; y<n; y++) {
    for (x=0; x<m; x++) {
       field[x + y*m]=0.0;
    }
  }

  field[0 + 0            ] = rt_rand(&rndval) / RT_RAND_MAX;
  field[m - 1            ] = rt_rand(&rndval) / RT_RAND_MAX;
  field[0     + m*(n - 1)] = rt_rand(&rndval) / RT_RAND_MAX;
  field[m - 1 + m*(n - 1)] = rt_rand(&rndval) / RT_RAND_MAX;

  subdivide(field, m, n, wx, wy, 0, 0, m-1, n-1, &rndval);

  rt_sheightfield(scene, tex, ctr, m, n, field, wx, wy);

  free(field);
}

