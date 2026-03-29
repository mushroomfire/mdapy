/*
 * grid.c - spatial subdivision efficiency structures
 *
 * $Id: grid.c,v 1.60 2011/02/07 07:41:51 johns Exp $
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TACHYON_INTERNAL 1
#include "tachyon.h"
#include "macros.h"
#include "vector.h"
#include "intersect.h"
#include "util.h"
#include "ui.h"
#include "parallel.h"

#define GRID_PRIVATE
#include "grid.h"

#ifndef cbrt
#define     cbrt(x)     ((x) > 0.0 ? pow((double)(x), 1.0/3.0) : \
                          ((x) < 0.0 ? -pow((double)-(x), 1.0/3.0) : 0.0))
#endif

static object_methods grid_methods = {
  (void (*)(const void *, void *))(grid_intersect),
  (void (*)(const void *, const void *, const void *, void *))(NULL),
  grid_bbox, 
  grid_free 
};

object * newgrid(scenedef * scene, int xsize, int ysize, int zsize, vector min, vector max) {
  grid * g;
  int numcells;

  g = (grid *) malloc(sizeof(grid));
  memset(g, 0, sizeof(grid));  

  g->methods = &grid_methods;
  g->id = new_objectid(scene);

  g->xsize = xsize;
  g->ysize = ysize;
  g->zsize = zsize;

  numcells = xsize * ysize * zsize;

  g->min = min;
  g->max = max;

  VSub(&g->max, &g->min, &g->voxsize);
  g->voxsize.x /= (flt) g->xsize; 
  g->voxsize.y /= (flt) g->ysize; 
  g->voxsize.z /= (flt) g->zsize; 

  g->cells = (objectlist **) malloc(numcells * sizeof(objectlist *));
  memset(g->cells, 0, numcells * sizeof(objectlist *));

  return (object *) g;
}

static int grid_bbox(void * obj, vector * min, vector * max) {
  grid * g = (grid *) obj;
 
  *min = g->min;
  *max = g->max;

  return 1;
}

static void grid_free(void * v) {
  int i, numvoxels;
  grid * g = (grid *) v;

  /* loop through all voxels and free the object lists */
  numvoxels = g->xsize * g->ysize * g->zsize; 
  for (i=0; i<numvoxels; i++) {
    objectlist * lcur;
    objectlist * lnext;

    lcur = g->cells[i];
    while (lcur != NULL) {
      lnext = lcur->next;
      free(lcur);
      lcur = lnext;
    }
  }

  /* free the grid cells */ 
  free(g->cells);

  /* free all objects on the grid object list */
  free_objects(g->objects);   

  free(g);
}

static void globalbound(object ** rootlist, vector * gmin, vector * gmax) {
  vector min, max;
  object * cur;

  if (*rootlist == NULL)  /* don't bound non-existant objects */
    return;

  gmin->x =  FHUGE;   gmin->y =  FHUGE;   gmin->z =  FHUGE;
  gmax->x = -FHUGE;   gmax->y = -FHUGE;   gmax->z = -FHUGE;

  cur=*rootlist;
  while (cur != NULL)  {  /* Go! */
    min.x = -FHUGE; min.y = -FHUGE; min.z = -FHUGE;
    max.x =  FHUGE; max.y =  FHUGE; max.z =  FHUGE;

    if (cur->methods->bbox((void *) cur, &min, &max)) {
      gmin->x = MYMIN( gmin->x , min.x);
      gmin->y = MYMIN( gmin->y , min.y);
      gmin->z = MYMIN( gmin->z , min.z);

      gmax->x = MYMAX( gmax->x , max.x);
      gmax->y = MYMAX( gmax->y , max.y);
      gmax->z = MYMAX( gmax->z , max.z);
    }

    cur=cur->nextobj;
  }
}


static int cellbound(const grid *g, const gridindex *index, vector * cmin, vector * cmax) {
  vector min, max, cellmin, cellmax;
  objectlist * cur;
  int numinbounds = 0;

  cur = g->cells[index->z*g->xsize*g->ysize + index->y*g->xsize + index->x]; 

  if (cur == NULL)  /* don't bound non-existant objects */
    return 0;

  cellmin.x = voxel2x(g, index->x); 
  cellmin.y = voxel2y(g, index->y); 
  cellmin.z = voxel2z(g, index->z); 

  cellmax.x = cellmin.x + g->voxsize.x;
  cellmax.y = cellmin.y + g->voxsize.y;
  cellmax.z = cellmin.z + g->voxsize.z;

  cmin->x =  FHUGE;   cmin->y =  FHUGE;   cmin->z =  FHUGE;
  cmax->x = -FHUGE;   cmax->y = -FHUGE;   cmax->z = -FHUGE;

  while (cur != NULL)  {  /* Go! */
    min.x = -FHUGE; min.y = -FHUGE; min.z = -FHUGE;
    max.x =  FHUGE; max.y =  FHUGE; max.z =  FHUGE;

    if (cur->obj->methods->bbox((void *) cur->obj, &min, &max)) {
      if ((min.x >= cellmin.x) && (max.x <= cellmax.x) &&
          (min.y >= cellmin.y) && (max.y <= cellmax.y) &&
          (min.z >= cellmin.z) && (max.z <= cellmax.z)) {
      
        cmin->x = MYMIN( cmin->x , min.x);
        cmin->y = MYMIN( cmin->y , min.y);
        cmin->z = MYMIN( cmin->z , min.z);

        cmax->x = MYMAX( cmax->x , max.x);
        cmax->y = MYMAX( cmax->y , max.y);
        cmax->z = MYMAX( cmax->z , max.z);
      
        numinbounds++;
      }
    }

    cur=cur->next;
  }
 
  /* in case we get a 0.0 sized axis on the cell bounds, we'll */
  /* use the original cell bounds */
  if ((cmax->x - cmin->x) < EPSILON) {
    cmax->x += EPSILON;
    cmin->x -= EPSILON;
  }
  if ((cmax->y - cmin->y) < EPSILON) {
    cmax->y += EPSILON;
    cmin->y -= EPSILON;
  }
  if ((cmax->z - cmin->z) < EPSILON) {
    cmax->z += EPSILON;
    cmin->z -= EPSILON;
  }

  return numinbounds;
}

static int countobj(object * root) {
  object * cur;     /* counts the number of objects on a list */
  int numobj;

  numobj=0;
  cur=root;

  while (cur != NULL) {
    cur=cur->nextobj;
    numobj++;
  }
  return numobj;
}

static void gridstats(int xs, int ys, int zs, int numobj) {
  char t[256]; /* msgtxt */
  int numcells = xs*ys*zs; 
  sprintf(t, "Grid:  X:%3d  Y:%3d  Z:%3d  Cells:%9d  Obj:%9d  Obj/Cell: %7.3f",
          xs, ys, zs, numcells, numobj, ((float) numobj) / ((float) numcells));
  rt_ui_message(MSG_0, t);
}

int engrid_scene(scenedef * scene, int boundthresh) {
  grid * g;
  int numobj, numcbrt;
  vector gmin={0,0,0};
  vector gmax={0,0,0};
  gridindex index;
  char msgtxt[128];
  int numsucceeded; 
  if (scene->objgroup.boundedobj == NULL)
    return 0;

  numobj = countobj(scene->objgroup.boundedobj);

  if (scene->mynode == 0) {
    sprintf(msgtxt, "Scene contains %d objects.", numobj);
    rt_ui_message(MSG_0, msgtxt);
  }

  if (numobj > boundthresh) {
    numcbrt = (int) cbrt(4*numobj);
    
    globalbound(&scene->objgroup.boundedobj, &gmin, &gmax);
    if (scene->verbosemode && scene->mynode == 0) {
      char t[256]; /* msgtxt */
      sprintf(t, "Global bounds: %g %g %g -> %g %g %g", 
              gmin.x, gmin.y, gmin.z, gmax.x, gmax.y, gmax.z);  
      rt_ui_message(MSG_0, t);

      sprintf(t, "Creating top level grid: X:%d Y:%d Z:%d", 
              numcbrt, numcbrt, numcbrt);
      rt_ui_message(MSG_0, t);
    }

    g = (grid *) newgrid(scene, numcbrt, numcbrt, numcbrt, gmin, gmax);
    numsucceeded = engrid_objlist(g, &scene->objgroup.boundedobj);
    if (scene->verbosemode && scene->mynode == 0)
      gridstats(numcbrt, numcbrt, numcbrt, numsucceeded); 

    if (scene->verbosemode && scene->mynode == 0) {
      char t[256]; /* msgtxt */
      numobj = countobj(scene->objgroup.boundedobj);
      sprintf(t, "Scene contains %d non-gridded objects\n", numobj);
      rt_ui_message(MSG_0, t);
    } 

    /* add this grid to the bounded object list removing the objects */
    /* now contained and managed by the grid                         */
    g->nextobj = scene->objgroup.boundedobj;
    scene->objgroup.boundedobj = (object *) g;

    /* create subgrids for overfull cell in the top level grid...    */
    for (index.z=0; index.z<g->zsize; index.z++) {
      for (index.y=0; index.y<g->ysize; index.y++) {
        for (index.x=0; index.x<g->xsize; index.x++) {
          engrid_cell(scene, boundthresh, g, &index);
        }
      }
    } 
  }

  return 1;
}


static int engrid_objlist(grid * g, object ** list) {
  object * cur, * next, **prev;
  int numsucceeded = 0;

  if (*list == NULL) 
    return 0;
  
  prev = list; 
  cur = *list;

  while (cur != NULL) {
    next = cur->nextobj;

    if (engrid_object(g, cur, 1)) {
      *prev = next;
      numsucceeded++;
    } else {
      prev = (object **) &cur->nextobj;
    }

    cur = next;
  } 

  return numsucceeded;
}


static int engrid_cell(scenedef * scene, int boundthresh, grid * gold, gridindex *index) {
  vector gmin, gmax, gsize;
  flt len;
  int numobj, numcbrt, xs, ys, zs;
  grid * g;
  objectlist **list;
  objectlist * newobj;
  int numsucceeded;

  list = &gold->cells[index->z*gold->xsize*gold->ysize + 
                     index->y*gold->xsize  + index->x];

  if (*list == NULL)
    return 0;

  numobj =  cellbound(gold, index, &gmin, &gmax);

  VSub(&gmax, &gmin, &gsize);
  len = 1.0 / (MYMAX( MYMAX(gsize.x, gsize.y), gsize.z ));
  gsize.x *= len;  
  gsize.y *= len;  
  gsize.z *= len;  

  if (numobj > boundthresh) {
    numcbrt = (int) cbrt(2*numobj); 
    
    xs = (int) ((flt) numcbrt * gsize.x);
    if (xs < 1) xs = 1;
    ys = (int) ((flt) numcbrt * gsize.y);
    if (ys < 1) ys = 1;
    zs = (int) ((flt) numcbrt * gsize.z);
    if (zs < 1) zs = 1;

    g = (grid *) newgrid(scene, xs, ys, zs, gmin, gmax);
    numsucceeded = engrid_objectlist(g, list);

    if (scene->verbosemode && scene->mynode == 0)
      gridstats(xs, ys, zs, numsucceeded); 

    newobj = (objectlist *) malloc(sizeof(objectlist));    
    newobj->obj = (object *) g;
    newobj->next = *list;
    *list = newobj;

    g->nextobj = gold->objects;
    gold->objects = (object *) g;
  }

  return 1;
}

static int engrid_objectlist(grid * g, objectlist ** list) {
  objectlist * cur, * next, **prev;
  int numsucceeded = 0; 

  if (*list == NULL) 
    return 0;
  
  prev = list; 
  cur = *list;

  while (cur != NULL) {
    next = cur->next;

    if (engrid_object(g, cur->obj, 0)) {
      *prev = next;
      free(cur);
      numsucceeded++;
    } else {
      prev = &cur->next;
    }

    cur = next;
  } 

  return numsucceeded;
}


static int engrid_object(grid * g, object * obj, int addtolist) {
  vector omin, omax; 
  gridindex low, high;
  int x, y, z, zindex, yindex, voxindex;
  objectlist * tmp;
 
  if (obj->methods->bbox(obj, &omin, &omax)) { 
    if (!pos2grid(g, &omin, &low) || !pos2grid(g, &omax, &high)) {
      return 0; /* object is not wholly contained in the grid, don't engrid */
    }
  } else {
    return 0;   /* object is unbounded, don't engrid this object */
  }

#if 0
  /* test grid cell occupancy size to see if this object would   */
  /* consume a huge number of grid cells (thus causing problems) */
  /* in the special case of a top level grid, we could count the */
  /* number of problematic objects, and if they exceed a maximum */
  /* percentage or absolute number of these, we should cancel    */
  /* filling the top level grid and rebuild a coarser top level  */
  /* grid instead, to prevent an explosion of memory use.        */
  {
  int voxeloccupancy = (high.x - low.x) * (high.y - low.y) * (high.z - low.z);
  if (voxeloccupancy > 22000) {
    return 0; /* don't engrid this object */
  }
  }
#endif

  /* add the object to the complete list of objects in the grid */
  if (addtolist) { 
    obj->nextobj = g->objects;
    g->objects = obj;
  }

  /* add this object to all voxels it inhabits */
  for (z=low.z; z<=high.z; z++) {
    zindex = z * g->xsize * g->ysize;
    for (y=low.y; y<=high.y; y++) {
      yindex = y * g->xsize;
      for (x=low.x; x<=high.x; x++) {
        voxindex = x + yindex + zindex; 
        tmp = (objectlist *) malloc(sizeof(objectlist));
        tmp->next = g->cells[voxindex];
        tmp->obj = obj;
        g->cells[voxindex] = tmp;
      }
    }
  }
 
  return 1;
}


static int pos2grid(grid * g, vector * pos, gridindex * index) {
  index->x = (int) ((flt) (pos->x - g->min.x) / g->voxsize.x);
  index->y = (int) ((flt) (pos->y - g->min.y) / g->voxsize.y);
  index->z = (int) ((flt) (pos->z - g->min.z) / g->voxsize.z);
  
  if (index->x == g->xsize)
    index->x--;
  if (index->y == g->ysize)
    index->y--;
  if (index->z == g->zsize)
    index->z--;

  if (index->x < 0 || index->x > g->xsize ||
      index->y < 0 || index->y > g->ysize ||
      index->z < 0 || index->z > g->zsize) 
    return 0;

  if (pos->x < g->min.x || pos->x > g->max.x ||
      pos->y < g->min.y || pos->y > g->max.y ||
      pos->z < g->min.z || pos->z > g->max.z) 
    return 0; 

  return 1;
}


/* the real thing */
static void grid_intersect(const grid * g, ray * ry) {
  flt tnear, tfar;
  vector curpos, tmax, tdelta;
  gridindex curvox, step, out; 
  int voxindex, SY, SZ;
  unsigned long serial;
#if !defined(DISABLEMBOX)
  unsigned long * mbox;
#endif
  objectlist * cur;

  if (ry->flags & RT_RAY_FINISHED)
    return;

  if (!grid_bounds_intersect(g, ry, &tnear, &tfar))
    return;
 
  if (ry->maxdist < tnear)
    return;
  
  serial=ry->serial;
#if !defined(DISABLEMBOX)
  mbox=ry->mbox;
#endif

  /* find the entry point in the grid from the near hit */ 
  curpos.x = ry->o.x + (ry->d.x * tnear);
  curpos.y = ry->o.y + (ry->d.y * tnear);
  curpos.z = ry->o.z + (ry->d.z * tnear);

  /* map the entry point to its nearest voxel */
  curvox.x = (int) ((flt) (curpos.x - g->min.x) / g->voxsize.x);
  curvox.y = (int) ((flt) (curpos.y - g->min.y) / g->voxsize.y);
  curvox.z = (int) ((flt) (curpos.z - g->min.z) / g->voxsize.z);
  if (curvox.x == g->xsize) curvox.x--;
  if (curvox.y == g->ysize) curvox.y--;
  if (curvox.z == g->zsize) curvox.z--;

  /* Setup X iterator stuff */
  if (ry->d.x < -EPSILON) {
    tmax.x = tnear + ((voxel2x(g, curvox.x) - curpos.x) / ry->d.x); 
    tdelta.x = g->voxsize.x / - ry->d.x;
    step.x = -1;
    out.x = -1;
  } else if (ry->d.x > EPSILON) {
    tmax.x = tnear + ((voxel2x(g, curvox.x + 1) - curpos.x) / ry->d.x);
    tdelta.x = g->voxsize.x / ry->d.x;
    step.x = 1;
    out.x = g->xsize;
  } else {
    tmax.x = FHUGE;
    tdelta.x = 0.0;
    step.x = 0;
    out.x = 0; /* never goes out of bounds on this axis */
  }

  /* Setup Y iterator stuff */
  if (ry->d.y < -EPSILON) {
    tmax.y = tnear + ((voxel2y(g, curvox.y) - curpos.y) / ry->d.y);
    tdelta.y = g->voxsize.y / - ry->d.y;
    step.y = -1;
    out.y = -1;
  } else if (ry->d.y > EPSILON) {
    tmax.y = tnear + ((voxel2y(g, curvox.y + 1) - curpos.y) / ry->d.y);
    tdelta.y = g->voxsize.y / ry->d.y;
    step.y = 1;
    out.y = g->ysize;
  } else {
    tmax.y = FHUGE;
    tdelta.y = 0.0; 
    step.y = 0;
    out.y = 0; /* never goes out of bounds on this axis */
  }

  /* Setup Z iterator stuff */
  if (ry->d.z < -EPSILON) {
    tmax.z = tnear + ((voxel2z(g, curvox.z) - curpos.z) / ry->d.z);
    tdelta.z = g->voxsize.z / - ry->d.z;
    step.z = -1;
    out.z = -1;
  } else if (ry->d.z > EPSILON) {
    tmax.z = tnear + ((voxel2z(g, curvox.z + 1) - curpos.z) / ry->d.z);
    tdelta.z = g->voxsize.z / ry->d.z;
    step.z = 1;
    out.z = g->zsize;
  } else {
    tmax.z = FHUGE;
    tdelta.z = 0.0; 
    step.z = 0;
    out.z = 0; /* never goes out of bounds on this axis */
  }

  /* pre-calculate row/column/plane offsets for stepping through grid */
  SY = step.y * g->xsize;
  SZ = step.z * g->xsize * g->ysize;

  /* first cell we'll be testing */
  voxindex = curvox.z*g->xsize*g->ysize + curvox.y*g->xsize + curvox.x; 

  /* Unrolled while loop by one... */
  /* Test all objects in the current cell for intersection */
  cur = g->cells[voxindex];
  while (cur != NULL) {
#if !defined(DISABLEMBOX)
    if (mbox[cur->obj->id] != serial) {
      mbox[cur->obj->id] = serial; 
      cur->obj->methods->intersect(cur->obj, ry);
    }
#else
    cur->obj->methods->intersect(cur->obj, ry);
#endif
    cur = cur->next;
  }

  /* Loop through grid cells until we're done */
  while (!(ry->flags & RT_RAY_FINISHED)) {
    /* Walk to next cell */
    if (tmax.x < tmax.y && tmax.x < tmax.z) {
      curvox.x += step.x;
      if (ry->maxdist < tmax.x || curvox.x == out.x) 
        break; 
      tmax.x += tdelta.x;
      voxindex += step.x;
    }
    else if (tmax.z < tmax.y) {
      curvox.z += step.z;
      if (ry->maxdist < tmax.z || curvox.z == out.z) 
        break;
      tmax.z += tdelta.z;
      voxindex += SZ;
    }
    else {
      curvox.y += step.y;
      if (ry->maxdist < tmax.y || curvox.y == out.y) 
        break;
      tmax.y += tdelta.y;
      voxindex += SY;
    }

    /* Test all objects in the current cell for intersection */
    cur = g->cells[voxindex];
    while (cur != NULL) {
#if !defined(DISABLEMBOX)
      if (mbox[cur->obj->id] != serial) {
        mbox[cur->obj->id] = serial; 
        cur->obj->methods->intersect(cur->obj, ry);
      }
#else
      cur->obj->methods->intersect(cur->obj, ry);
#endif
      cur = cur->next;
    }
  }
}



static int grid_bounds_intersect(const grid * g, const ray * ry, flt *hitnear, flt *hitfar) {
  flt a, tx1, tx2, ty1, ty2, tz1, tz2;
  flt tnear, tfar;

  tnear= -FHUGE;
  tfar= FHUGE;

  if (ry->d.x == 0.0) {
    if ((ry->o.x < g->min.x) || (ry->o.x > g->max.x)) return 0;
  } else {
    tx1 = (g->min.x - ry->o.x) / ry->d.x;
    tx2 = (g->max.x - ry->o.x) / ry->d.x;
    if (tx1 > tx2) { a=tx1; tx1=tx2; tx2=a; }
    if (tx1 > tnear) tnear=tx1;
    if (tx2 < tfar)   tfar=tx2;
  }
  if (tnear > tfar) return 0;
  if (tfar < 0.0) return 0;

  if (ry->d.y == 0.0) {
    if ((ry->o.y < g->min.y) || (ry->o.y > g->max.y)) return 0;
  } else {
    ty1 = (g->min.y - ry->o.y) / ry->d.y;
    ty2 = (g->max.y - ry->o.y) / ry->d.y;
    if (ty1 > ty2) { a=ty1; ty1=ty2; ty2=a; }
    if (ty1 > tnear) tnear=ty1;
    if (ty2 < tfar)   tfar=ty2;
  }
  if (tnear > tfar) return 0;
  if (tfar < 0.0) return 0;

  if (ry->d.z == 0.0) {
    if ((ry->o.z < g->min.z) || (ry->o.z > g->max.z)) return 0;
  } else {
    tz1 = (g->min.z - ry->o.z) / ry->d.z;
    tz2 = (g->max.z - ry->o.z) / ry->d.z;
    if (tz1 > tz2) { a=tz1; tz1=tz2; tz2=a; }
    if (tz1 > tnear) tnear=tz1;
    if (tz2 < tfar)   tfar=tz2;
  }
  if (tnear > tfar) return 0;
  if (tfar < 0.0) return 0;

  if (tnear < 0.0) {
    *hitnear = 0.0;
  } else {
    *hitnear = tnear;
  }

  *hitfar = tfar; 
  return 1;
}




