/*
 * camera.c - This file contains all of the functions for doing camera work.
 *
 *  $Id: camera.c,v 1.61 2012/10/17 04:25:57 johns Exp $
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TACHYON_INTERNAL 1
#include "tachyon.h"
#include "macros.h"
#include "vector.h"
#include "camera.h"
#include "util.h"
#include "intersect.h"

/* 
 * camera_init()
 *   take camera parameters stored in scene definition and do all 
 *   necessary initialization and whatever pre-calculation can be done. 
 */
void camera_init(scenedef *scene) {
  flt sx, sy;
  vector newupvec;
  vector newviewvec;
  vector newrightvec;

  /* recompute the camera vectors */
  VCross(&scene->camera.upvec, &scene->camera.viewvec, &newrightvec);
  VNorm(&newrightvec);

  VCross(&scene->camera.viewvec, &newrightvec, &newupvec);
  VNorm(&newupvec);

  newviewvec=scene->camera.viewvec;
  VNorm(&newviewvec);
  scene->camera.rightvec=newrightvec;
  scene->camera.upvec=newupvec;

  sx = (flt) scene->hres; 
  sy = (flt) scene->vres;

  /* calculate the width and height of the image plane in world coords */
  /* given the aspect ratio, image resolution, and zoom factor */
  scene->camera.px=((sx / sy) / scene->aspectratio) / scene->camera.camzoom;
  scene->camera.py=1.0 / scene->camera.camzoom;
  scene->camera.psx = scene->camera.px / sx;
  scene->camera.psy = scene->camera.py / sy;

  if (scene->camera.frustumcalc == RT_CAMERA_FRUSTUM_AUTO) {
    scene->camera.left   = -0.5 * scene->camera.px;
    scene->camera.right  =  0.5 * scene->camera.px;
    scene->camera.bottom = -0.5 * scene->camera.py;
    scene->camera.top    =  0.5 * scene->camera.py;
  }
  
  /* setup function pointer for camera ray generation */
  switch (scene->camera.projection) {
    case RT_PROJECTION_PERSPECTIVE:
      if (scene->antialiasing > 0) {
        scene->camera.cam_ray = (colora (*)(void *,flt,flt)) cam_aa_perspective_ray;
      } else {
        scene->camera.cam_ray = (colora (*)(void *,flt,flt)) cam_perspective_ray;
      }
      break;

    case RT_PROJECTION_PERSPECTIVE_DOF:
      scene->camera.cam_ray = (colora (*)(void *,flt,flt)) cam_aa_dof_ray;
      break;

    case RT_PROJECTION_ORTHOGRAPHIC:
      if (scene->antialiasing > 0) {
        scene->camera.cam_ray = (colora (*)(void *,flt,flt)) cam_aa_orthographic_ray;
      } else {
        scene->camera.cam_ray = (colora (*)(void *,flt,flt)) cam_orthographic_ray;
      }
      break;

    case RT_PROJECTION_FISHEYE:
      if (scene->antialiasing > 0) {
        scene->camera.cam_ray = (colora (*)(void *,flt,flt)) cam_aa_fisheye_ray;
      } else {
        scene->camera.cam_ray = (colora (*)(void *,flt,flt)) cam_fisheye_ray;
      }
      break;
  }


  /* assuming viewvec is a unit vector, then the center of the */
  /* image plane is the camera center + vievec                 */
  switch (scene->camera.projection) { 
    case RT_PROJECTION_ORTHOGRAPHIC:
      scene->camera.projcent = scene->camera.center;

      /* assuming viewvec is a unit vector, then the lower left    */
      /* corner of the image plane is calculated below             */
      scene->camera.lowleft.x = scene->camera.projcent.x +
        (scene->camera.left   * scene->camera.rightvec.x) +
        (scene->camera.bottom * scene->camera.upvec.x);
      scene->camera.lowleft.y = scene->camera.projcent.y +
        (scene->camera.left   * scene->camera.rightvec.y) +
        (scene->camera.bottom * scene->camera.upvec.y);
      scene->camera.lowleft.z = scene->camera.projcent.z +
        (scene->camera.left   * scene->camera.rightvec.z) +
        (scene->camera.bottom * scene->camera.upvec.z);
      break;
  
    case RT_PROJECTION_PERSPECTIVE_DOF:
      scene->camera.projcent.x = scene->camera.center.x + 
                        (scene->camera.focallength * scene->camera.viewvec.x);
      scene->camera.projcent.y = scene->camera.center.y + 
                        (scene->camera.focallength * scene->camera.viewvec.y);
      scene->camera.projcent.z = scene->camera.center.z + 
                        (scene->camera.focallength * scene->camera.viewvec.z);

      /* assuming viewvec is a unit vector, then the lower left    */
      /* corner of the image plane is calculated below             */
      scene->camera.lowleft.x = scene->camera.projcent.x +
        (scene->camera.left   * scene->camera.rightvec.x) +
        (scene->camera.bottom * scene->camera.upvec.x);
      scene->camera.lowleft.y = scene->camera.projcent.y +
        (scene->camera.left   * scene->camera.rightvec.y) +
        (scene->camera.bottom * scene->camera.upvec.y);
      scene->camera.lowleft.z = scene->camera.projcent.z +
        (scene->camera.left   * scene->camera.rightvec.z) +
        (scene->camera.bottom * scene->camera.upvec.z);
      break;

    case RT_PROJECTION_FISHEYE:
      scene->camera.projcent.x = scene->camera.center.x + 
                        (scene->camera.focallength * scene->camera.viewvec.x);
      scene->camera.projcent.y = scene->camera.center.y + 
                        (scene->camera.focallength * scene->camera.viewvec.y);
      scene->camera.projcent.z = scene->camera.center.z + 
                        (scene->camera.focallength * scene->camera.viewvec.z);
      break;

    case RT_PROJECTION_PERSPECTIVE:
    default:
      scene->camera.projcent.x = scene->camera.center.x + 
                        (scene->camera.focallength * scene->camera.viewvec.x);
      scene->camera.projcent.y = scene->camera.center.y + 
                        (scene->camera.focallength * scene->camera.viewvec.y);
      scene->camera.projcent.z = scene->camera.center.z + 
                        (scene->camera.focallength * scene->camera.viewvec.z);

      /* assuming viewvec is a unit vector, then the lower left    */
      /* corner of the image plane is calculated below             */
      /* for normal perspective rays, we are really storing the    */
      /* direction to the lower left, not the lower left itself,   */
      /* since this allows us to eliminate a subtraction per pixel */
      scene->camera.lowleft.x = scene->camera.projcent.x +
        (scene->camera.left   * scene->camera.rightvec.x) +
        (scene->camera.bottom * scene->camera.upvec.x)
        - scene->camera.center.x;
      scene->camera.lowleft.y = scene->camera.projcent.y +
        (scene->camera.left   * scene->camera.rightvec.y) +
        (scene->camera.bottom * scene->camera.upvec.y)
        - scene->camera.center.y;
      scene->camera.lowleft.z = scene->camera.projcent.z +
        (scene->camera.left   * scene->camera.rightvec.z) +
        (scene->camera.bottom * scene->camera.upvec.z)
        - scene->camera.center.z;
      break;
  }

  /* size of image plane */
  scene->camera.px = scene->camera.right - scene->camera.left; 
  scene->camera.py = scene->camera.top - scene->camera.bottom; 
  scene->camera.psx = scene->camera.px / scene->hres;
  scene->camera.psy = scene->camera.py / scene->vres;

  scene->camera.iplaneright.x = scene->camera.px * scene->camera.rightvec.x / sx;
  scene->camera.iplaneright.y = scene->camera.px * scene->camera.rightvec.y / sx;
  scene->camera.iplaneright.z = scene->camera.px * scene->camera.rightvec.z / sx;
  
  scene->camera.iplaneup.x = scene->camera.py * scene->camera.upvec.x / sy;
  scene->camera.iplaneup.y = scene->camera.py * scene->camera.upvec.y / sy;
  scene->camera.iplaneup.z = scene->camera.py * scene->camera.upvec.z / sy;
}


/* 
 * camray_init() 
 *   initializes a camera ray which will be reused over and over
 *   by the current worker thread.  This includes attaching thread-specific
 *   data to this ray.
 */
void camray_init(scenedef *scene, ray *primary, unsigned long serial, 
                 unsigned long * mbox, unsigned int randval) {
  /* setup the right function pointer depending on what features are in use */
  if (scene->flags & RT_SHADE_CLIPPING) {
    primary->add_intersection = add_clipped_intersection;
  } else {
    primary->add_intersection = add_regular_intersection;
  }

  primary->serial = serial;
  primary->mbox = mbox;
  primary->scene = scene;
  primary->depth = scene->raydepth;      /* set to max ray depth      */
  primary->transcnt = scene->transcount; /* set to max trans surf cnt */
  primary->randval = randval;            /* random number seed */
  rng_frand_init(&primary->frng);        /* seed 32-bit FP RNG */

  /* orthographic ray direction is always coaxial with view direction */
  primary->d = scene->camera.viewvec; 

  /* for perspective rendering without depth of field */
  primary->o = scene->camera.center;
}


#if 0
/*
 * cam_aa_dof_ray() 
 *  Generate a perspective camera ray incorporating
 *  antialiasing and depth-of-field.
 */
colora cam_aa_dof_ray(ray * ry, flt x, flt y) {
  color col, sample;
  color colsum, colsumsq, colvar;
  int samples, samplegroup; 
  scenedef * scene=ry->scene;
  flt dx, dy, rnsamples, ftmp;
  flt maxvar;

  sample=cam_dof_ray(ry, x, y);     /* generate primary ray */
  colsum = sample;                  /* accumulate first sample */
  colsumsq.r = colsum.r * colsum.r; /* accumulate first squared sample */
  colsumsq.g = colsum.g * colsum.g;
  colsumsq.b = colsum.b * colsum.b;

  /* perform antialiasing if enabled.                           */
  /* samples are run through a very simple box filter averaging */
  /* each of the sample pixel colors to produce a final result  */
  /* No special weighting is done based on the jitter values in */
  /* the circle of confusion nor for the jitter within the      */
  /* pixel in the image plane.                                  */ 

  samples = 1; /* only one ray cast so far */
  while (samples < scene->antialiasing) {
#if 0
    samplegroup = scene->antialiasing;
#else
    if (samples > 32) {
      samplegroup = samples + 1;
    } else {
      samplegroup = samples + 32;
    }
#endif

    for (; samples <= samplegroup; samples++) {
      /* calculate random eye aperture offset */
      float jxy[2];
      jitter_offset2f(&ry->randval, jxy);
      dx = jxy[0] * ry->scene->camera.aperture * ry->scene->hres; 
      dy = jxy[1] * ry->scene->camera.aperture * ry->scene->vres; 

      /* perturb the eye center by the random aperture offset */
      ry->o.x = ry->scene->camera.center.x + 
                dx * ry->scene->camera.iplaneright.x +
                dy * ry->scene->camera.iplaneup.x;
      ry->o.y = ry->scene->camera.center.y + 
                dx * ry->scene->camera.iplaneright.y +
                dy * ry->scene->camera.iplaneup.y;
      ry->o.z = ry->scene->camera.center.z + 
                dx * ry->scene->camera.iplaneright.z +
                dy * ry->scene->camera.iplaneup.z;

      /* shoot the ray, jittering the pixel position in the image plane */
      jitter_offset2f(&ry->randval, jxy);
      sample=cam_dof_ray(ry, x + jxy[0], y + jxy[1]);
  
      colsum.r += sample.r;               /* accumulate samples */
      colsum.g += sample.g;
      colsum.b += sample.b;

      colsumsq.r += sample.r * sample.r;  /* accumulate squared samples */
      colsumsq.g += sample.g * sample.g;
      colsumsq.b += sample.b * sample.b;
    }

    /* calculate the variance for the color samples we've taken so far */
    rnsamples = 1.0 / samples;
    ftmp = colsum.r * rnsamples;
    colvar.r = ((colsumsq.r * rnsamples) - ftmp*ftmp) * rnsamples;
    ftmp = colsum.g * rnsamples;
    colvar.g = ((colsumsq.g * rnsamples) - ftmp*ftmp) * rnsamples;
    ftmp = colsum.b * rnsamples;
    colvar.b = ((colsumsq.b * rnsamples) - ftmp*ftmp) * rnsamples;

    maxvar = 0.002; /* default maximum color variance to accept */
 
    /* early exit antialiasing if we're below maximum allowed variance */
    if ((colvar.r < maxvar) && (colvar.g < maxvar) && (colvar.b < maxvar)) {
      break; /* no more samples should be needed, we are happy now */
    } 
  }

  /* average sample colors, back to range 0.0 - 1.0 */ 
  col.r = colsum.r * rnsamples;
  col.g = colsum.g * rnsamples;
  col.b = colsum.b * rnsamples;

  return col;
}


#else

/*
 * cam_aa_dof_ray() 
 *  Generate a perspective camera ray incorporating
 *  antialiasing and depth-of-field.
 */
colora cam_aa_dof_ray(ray * ry, flt x, flt y) {
  colora col, avcol;
  int alias; 
  scenedef * scene=ry->scene;
  float scale;
  flt dx, dy;

  col=cam_dof_ray(ry, x, y);   /* generate ray */

  /* perform antialiasing if enabled.                           */
  /* samples are run through a very simple box filter averaging */
  /* each of the sample pixel colors to produce a final result  */
  /* No special weighting is done based on the jitter values in */
  /* the circle of confusion nor for the jitter within the      */
  /* pixel in the image plane.                                  */ 
  for (alias=1; alias <= scene->antialiasing; alias++) {
    float jxy[2];
    jitter_offset2f(&ry->randval, jxy);
    dx = jxy[0] * ry->scene->camera.aperture * ry->scene->hres; 
    dy = jxy[1] * ry->scene->camera.aperture * ry->scene->vres; 

    /* perturb the eye center by the random aperture offset */
    ry->o.x = ry->scene->camera.center.x + 
              dx * ry->scene->camera.iplaneright.x +
              dy * ry->scene->camera.iplaneup.x;
    ry->o.y = ry->scene->camera.center.y + 
              dx * ry->scene->camera.iplaneright.y +
              dy * ry->scene->camera.iplaneup.y;
    ry->o.z = ry->scene->camera.center.z + 
              dx * ry->scene->camera.iplaneright.z +
              dy * ry->scene->camera.iplaneup.z;

    /* shoot the ray, jittering the pixel position in the image plane */
    jitter_offset2f(&ry->randval, jxy);
    avcol=cam_dof_ray(ry, x + jxy[0], y + jxy[1]);

    col.r += avcol.r;       /* accumulate antialiasing samples */
    col.g += avcol.g;
    col.b += avcol.b;
    col.a += avcol.a;
  }

  /* average sample colors, back to range 0.0 - 1.0 */ 
  scale = 1.0f / (scene->antialiasing + 1.0f); 
  col.r *= scale;
  col.g *= scale;
  col.b *= scale;
  col.a *= scale;

  return col;
}
#endif

/*
 * cam_dof_ray() 
 *  Generate a perspective camera ray for depth-of-field rendering
 */
colora cam_dof_ray(ray * ry, flt x, flt y) {
  flt rdx, rdy, rdz, len;
  scenedef * scene=ry->scene;

  /* starting from the lower left corner of the image plane, we move the   */
  /* center of the pel we're calculating:                       */ 
  /* lowerleft + (rightvec * X_distance) + (upvec * Y_distance) */
  /* rdx/y/z are the ray directions (unnormalized)              */
  rdx = scene->camera.lowleft.x + 
                (x * scene->camera.iplaneright.x) + 
                (y * scene->camera.iplaneup.x) - ry->o.x;

  rdy = scene->camera.lowleft.y + 
                (x * scene->camera.iplaneright.y) + 
                (y * scene->camera.iplaneup.y) - ry->o.y;

  rdz = scene->camera.lowleft.z + 
                (x * scene->camera.iplaneright.z) + 
                (y * scene->camera.iplaneup.z) - ry->o.z;

  /* normalize the ray direction vector */
  len = 1.0 / SQRT(rdx*rdx + rdy*rdy + rdz*rdz);
  ry->d.x = rdx * len;
  ry->d.y = rdy * len;
  ry->d.z = rdz * len;

  /* initialize ray attributes for a primary ray */
  ry->maxdist = FHUGE;         /* unbounded ray */
  ry->opticdist = 0.0;         /* ray is just starting */

  /* camera only generates primary rays */
  ry->flags = RT_RAY_PRIMARY | RT_RAY_REGULAR;  

  ry->serial++;                /* increment the ray serial number */
  intersect_objects(ry);       /* trace the ray */
  return scene->shader(ry);    /* shade the hit point */
}


/*
 * cam_aa_perspective_ray() 
 *  Generate a perspective camera ray incorporating antialiasing.
 */
colora cam_aa_perspective_ray(ray * ry, flt x, flt y) {
  colora col, avcol;
  int alias; 
  scenedef * scene=ry->scene;
  float scale;
  col=cam_perspective_ray(ry, x, y);   /* generate ray */

  /* perform antialiasing if enabled.                           */
  /* samples are run through a very simple box filter averaging */
  /* each of the sample pixel colors to produce a final result  */
  for (alias=1; alias <= scene->antialiasing; alias++) {
    float jxy[2];
    jitter_offset2f(&ry->randval, jxy);
    avcol=cam_perspective_ray(ry, x + jxy[0], y + jxy[1]);

    col.r += avcol.r;       /* accumulate antialiasing samples */
    col.g += avcol.g;
    col.b += avcol.b;
    col.a += avcol.a;
  }

  /* average sample colors, back to range 0.0 - 1.0 */ 
  scale = 1.0f / (scene->antialiasing + 1.0f); 
  col.r *= scale;
  col.g *= scale;
  col.b *= scale;
  col.a *= scale;

  return col;
}


/*
 * cam_perspective_ray() 
 *  Generate a perspective camera ray, no antialiasing
 */
colora cam_perspective_ray(ray * ry, flt x, flt y) {
  flt rdx, rdy, rdz, len;
  scenedef * scene=ry->scene;

  /* starting from the lower left corner of the image plane, we move the   */
  /* center of the pel we're calculating:                       */ 
  /* lowerleft + (rightvec * X_distance) + (upvec * Y_distance) */
  /* rdx/y/z are the ray directions (unnormalized)              */
  rdx = scene->camera.lowleft.x + 
                (x * scene->camera.iplaneright.x) + 
                (y * scene->camera.iplaneup.x);

  rdy = scene->camera.lowleft.y + 
                (x * scene->camera.iplaneright.y) + 
                (y * scene->camera.iplaneup.y);

  rdz = scene->camera.lowleft.z + 
                (x * scene->camera.iplaneright.z) + 
                (y * scene->camera.iplaneup.z);

  /* normalize the ray direction vector */
  len = 1.0 / SQRT(rdx*rdx + rdy*rdy + rdz*rdz);
  ry->d.x = rdx * len;
  ry->d.y = rdy * len;
  ry->d.z = rdz * len;

  /* initialize ray attributes for a primary ray */
  ry->maxdist = FHUGE;         /* unbounded ray */
  ry->opticdist = 0.0;         /* ray is just starting */

  /* camera only generates primary rays */
  ry->flags = RT_RAY_PRIMARY | RT_RAY_REGULAR;  

  ry->serial++;                /* increment the ray serial number */
  intersect_objects(ry);       /* trace the ray */
  return scene->shader(ry);    /* shade the hit point */
}


/*
 * cam_aa_orthographic_ray() 
 *  Generate an orthographic camera ray, potentially incorporating
 *  antialiasing.
 */
colora cam_aa_orthographic_ray(ray * ry, flt x, flt y) {
  colora col, avcol;
  int alias; 
  scenedef * scene=ry->scene;
  float scale;

  col=cam_orthographic_ray(ry, x, y);   /* generate ray */

  /* perform antialiasing if enabled.                           */
  /* samples are run through a very simple box filter averaging */
  /* each of the sample pixel colors to produce a final result  */
  for (alias=1; alias <= scene->antialiasing; alias++) {
    float jxy[2];
    jitter_offset2f(&ry->randval, jxy);
    avcol=cam_orthographic_ray(ry, x + jxy[0], y + jxy[1]);

    col.r += avcol.r;       /* accumulate antialiasing samples */
    col.g += avcol.g;
    col.b += avcol.b;
    col.a += avcol.a;
  }

  /* average sample colors, back to range 0.0 - 1.0 */ 
  scale = 1.0f / (scene->antialiasing + 1.0f); 
  col.r *= scale;
  col.g *= scale;
  col.b *= scale;
  col.a *= scale;

  return col;
}

/*
 * cam_orthographic_ray() 
 *  Generate an orthographic camera ray, no antialiasing
 */
colora cam_orthographic_ray(ray * ry, flt x, flt y) {
  scenedef * scene=ry->scene;

  /* starting from the lower left corner of the image plane, we move the   */
  /* center of the pel we're calculating:                       */ 
  /* lowerleft + (rightvec * X_distance) + (upvec * Y_distance) */
  ry->o.x = scene->camera.lowleft.x + 
                (x * scene->camera.iplaneright.x) + 
                (y * scene->camera.iplaneup.x);

  ry->o.y = scene->camera.lowleft.y + 
                (x * scene->camera.iplaneright.y) + 
                (y * scene->camera.iplaneup.y);

  ry->o.z = scene->camera.lowleft.z + 
                (x * scene->camera.iplaneright.z) + 
                (y * scene->camera.iplaneup.z);

  /* initialize ray attributes for a primary ray */
  ry->maxdist = FHUGE;         /* unbounded ray */
  ry->opticdist = 0.0;         /* ray is just starting */

  /* camera only generates primary rays */
  ry->flags = RT_RAY_PRIMARY | RT_RAY_REGULAR;  

  ry->serial++;                /* increment the ray serial number */
  intersect_objects(ry);       /* trace the ray */
  return scene->shader(ry);    /* shade the hit point */
}

/*
 * cam_fisheye_ray() 
 *  Generate a perspective camera ray, no antialiasing
 */
colora cam_fisheye_ray(ray * ry, flt x, flt y) {
  flt ax, ay;
  scenedef * scene=ry->scene;

  ax = scene->camera.left   + x * scene->camera.psx;
  ay = scene->camera.bottom + y * scene->camera.psy;

  ry->d.x = COS(ay) * (COS(ax) * scene->camera.viewvec.x + 
                       SIN(ax) * scene->camera.rightvec.x) +
            SIN(ay) * scene->camera.upvec.x;

  ry->d.y = COS(ay) * (COS(ax) * scene->camera.viewvec.y +
                       SIN(ax) * scene->camera.rightvec.y) +
            SIN(ay) * scene->camera.upvec.y;

  ry->d.z = COS(ay) * (COS(ax) * scene->camera.viewvec.z +
                       SIN(ax) * scene->camera.rightvec.z) +
            SIN(ay) * scene->camera.upvec.z;
        
  /* initialize ray attributes for a primary ray */
  ry->maxdist = FHUGE;         /* unbounded ray */
  ry->opticdist = 0.0;         /* ray is just starting */

  /* camera only generates primary rays */
  ry->flags = RT_RAY_PRIMARY | RT_RAY_REGULAR;  

  ry->serial++;                /* increment the ray serial number */
  intersect_objects(ry);       /* trace the ray */
  return scene->shader(ry);    /* shade the hit point */
}

/*
 * cam_aa_fisheye_ray() 
 *  Generate a fisheye camera ray, potentially incorporating
 *  antialiasing.
 */
colora cam_aa_fisheye_ray(ray * ry, flt x, flt y) {
  colora col, avcol;
  int alias; 
  scenedef * scene=ry->scene;
  float scale;

  col=cam_fisheye_ray(ry, x, y);   /* generate ray */

  /* perform antialiasing if enabled.                           */
  /* samples are run through a very simple box filter averaging */
  /* each of the sample pixel colors to produce a final result  */
  for (alias=1; alias <= scene->antialiasing; alias++) {
    float jxy[2];
    jitter_offset2f(&ry->randval, jxy);
    avcol=cam_fisheye_ray(ry, x + jxy[0], y + jxy[1]);

    col.r += avcol.r;       /* accumulate antialiasing samples */
    col.g += avcol.g;
    col.b += avcol.b;
    col.a += avcol.a;
  }

  /* average sample colors, back to range 0.0 - 1.0 */ 
  scale = 1.0f / (scene->antialiasing + 1.0f); 
  col.r *= scale;
  col.g *= scale;
  col.b *= scale;
  col.a *= scale;

  return col;
}


void cameraprojection(camdef * camera, int mode) {
  camera->projection=mode;    
}


/**
 * When the user directly specifies the world coordinates of the 
 * view frustum, it overrides the normal calculations involving zoom factor,
 * aspect ratio, etc.  The caller must therefore be responsible for making
 * sure that all of these factors work out.  We wash our hands of all of the
 * usual automatic computations and use these factors as-is.
 */
void camerafrustum(camdef * camera, flt left, flt right, flt bottom, flt top) {
  camera->frustumcalc = RT_CAMERA_FRUSTUM_USER;
  camera->left = left;    
  camera->right = right;    
  camera->bottom = bottom;    
  camera->top = top;    
}


void cameradof(camdef * camera, flt focallength, flt aperture) {
  camera->focallength=focallength;    
  camera->aperture=aperture;
}


/*
 * When the user specifies zoom factor, we are implicitly expecting to
 * auto-calculate the resulting view frustum.
 */
void camerazoom(camdef * camera, flt zoom) {
  camera->frustumcalc = RT_CAMERA_FRUSTUM_AUTO;
  camera->camzoom = zoom;
}

void cameradefault(camdef * camera) {
  camerazoom(camera, 1.0);
  camera->focallength = 1.0;
}


void cameraposition(camdef * camera, vector center, vector viewvec, 
                    vector upvec) {
  vector newupvec;
  vector newviewvec;
  vector newrightvec;

  /* recompute the camera vectors */
  VCross(&upvec, &viewvec, &newrightvec);
  VNorm(&newrightvec);

  VCross(&viewvec, &newrightvec, &newupvec);
  VNorm(&newupvec);

  newviewvec=viewvec;
  VNorm(&newviewvec);

  camera->center=center;
  camera->viewvec=newviewvec;
  camera->rightvec=newrightvec;
  camera->upvec=newupvec;
}


void getcameraposition(camdef * camera, vector * center, vector * viewvec, 
                       vector * upvec, vector * rightvec) {
  *center = camera->center;
  *viewvec = camera->viewvec;
  *upvec = camera->upvec;
  *rightvec = camera->rightvec;
}


