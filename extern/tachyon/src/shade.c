/* 
 * shade.c - This file contains the functions that perform surface shading.
 *
 *  $Id: shade.c,v 1.115 2012/10/17 04:25:57 johns Exp $
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TACHYON_INTERNAL 1
#include "tachyon.h"
#include "macros.h"
#include "threads.h"
#include "light.h"
#include "intersect.h"
#include "vector.h"
#include "trace.h"
#include "shade.h"


/*
 * Lowest Quality Shader - Returns the raw color of an object.
 *
 */

colora lowest_shader(ray * incident) {
  int numints;
  object const * obj;
  flt t = FHUGE;
  colora col;

  numints=closest_intersection(&t, &obj, incident);
                /* find the number of intersections */
                /* and return the closest one.      */

  if (numints < 1) {
    /* if there weren't any object intersections then return the */
    /* black for the pixel color.                                */
    col.r = 0.0;
    col.g = 0.0;
    col.b = 0.0;
    col.a = 0.0;

    return col;
  }

  col.r = 1.0;
  col.g = 1.0;
  col.b = 1.0;
  col.a = 1.0;

  return col;
}


/*
 * Low Quality Shader - Returns raw texture color of objects hit, nothing else.
 *
 */

colora low_shader(ray * incident) {
  int numints;
  object const * obj;
  vector hit;
  flt t = FHUGE;

  numints=closest_intersection(&t, &obj, incident);
                /* find the number of intersections */
                /* and return the closest one.      */

  if (numints < 1) {
    /* if there weren't any object intersections then return the */
    /* background texture for the pixel color.                   */
    return incident->scene->bgtexfunc(incident);
  }

  RAYPNT(hit, (*incident), t) /* find the point of intersection from t */
  incident->opticdist = FHUGE; 
  return tocolora(obj->tex->texfunc(&hit, obj->tex, incident));
}



/*
 * Medium Quality Shader - Includes a subset of the rendering features
 *
 */

colora medium_shader(ray * incident) {
  colora cola;
  color col, diffuse, phongcol;
  shadedata shadevars;
  flt inten;
  flt t = FHUGE;
  object const * obj;
  int numints;
  list * cur;

  numints=closest_intersection(&t, &obj, incident);  
		/* find the number of intersections */
                /* and return the closest one.      */

  if (numints < 1) {         
    /* if there weren't any object intersections then return the */
    /* background texture for the pixel color.                   */
    cola=incident->scene->bgtexfunc(incident);

    /* Fog overrides the background color if we're using         */
    /* Tachyon radial fog, but not for OpenGL style fog.         */
    if (incident->scene->fog.type == RT_FOG_NORMAL &&
        incident->scene->fog.fog_fctn != NULL) {
      cola = tocolora(fog_color(incident, col, t));
    }

    return cola;
  }

  RAYPNT(shadevars.hit, (*incident), t) /* find point of intersection from t */ 
  incident->opticdist += t;
  obj->methods->normal(obj, &shadevars.hit, incident, &shadevars.N);  /* find the surface normal */

  /* don't render transparent surfaces if we've reached the max count */
  if ((obj->tex->opacity < 1.0) && (incident->transcnt < 1)) {      
    /* spawn transmission rays / refraction */
    /* note: this will overwrite the old intersection list */
    return tocolora(shade_transmission(incident, &shadevars, 1.0));
  }

  /* execute the object's texture function */
  col = obj->tex->texfunc(&shadevars.hit, obj->tex, incident); 

  if (obj->tex->flags & RT_TEXTURE_ISLIGHT) {  
                  /* if the current object is a light, then we  */
    return tocolora(col);   /* will only use the object's base color      */
  }

  diffuse.r = 0.0; 
  diffuse.g = 0.0; 
  diffuse.b = 0.0; 
  phongcol = diffuse;

  if ((obj->tex->diffuse > MINCONTRIB) || (obj->tex->phong > MINCONTRIB)) {  
    flt light_scale = incident->scene->light_scale;
    cur = incident->scene->lightlist;
    while (cur != NULL) {              /* loop for light contributions */
      light * li=(light *) cur->item;  /* set li=to the current light  */
      inten = light_scale * li->shade_diffuse(li, &shadevars);

      /* add in diffuse lighting for this light if we're facing it */ 
      if (inten > MINCONTRIB) {            
        /* calculate diffuse lighting component */
        ColorAddS(&diffuse, &((standard_texture *)li->tex)->col, inten);

        /* phong type specular highlights */
        if (obj->tex->phong > MINCONTRIB) {
          flt phongval = light_scale * incident->scene->phongfunc(incident, &shadevars, obj->tex->phongexp);
          if (obj->tex->phongtype == RT_PHONG_METAL) 
            ColorAddS(&phongcol, &col, phongval * obj->tex->phong);
          else
            ColorAddS(&phongcol, &((standard_texture *)li->tex)->col, phongval * obj->tex->phong);
        }
      }  

      cur = cur->next;
    } 
  }

  if (obj->tex->outline > 0.0) {
    flt outlinefactor;
    flt edgefactor = VDot(&shadevars.N, &incident->d);
    edgefactor *= edgefactor;
    edgefactor = 1.0 - edgefactor;
    edgefactor = 1.0 - POW(edgefactor, (1.0 - obj->tex->outlinewidth) * 32.0);
    outlinefactor = (1.0-obj->tex->outline) + (edgefactor * obj->tex->outline);
    ColorScale(&diffuse, obj->tex->diffuse * outlinefactor);
  } else {
    ColorScale(&diffuse, obj->tex->diffuse);
  }

  col.r *= (diffuse.r + obj->tex->ambient); /* do a product of the     */
  col.g *= (diffuse.g + obj->tex->ambient); /* diffuse intensity with  */
  col.b *= (diffuse.b + obj->tex->ambient); /* object color + ambient  */

  if (obj->tex->phong > MINCONTRIB) {
    ColorAccum(&col, &phongcol);
  }

  /* spawn reflection rays if necessary */
  /* note: this will overwrite the old intersection list */
  if (obj->tex->specular > MINCONTRIB) {    
    color specol;
    specol = shade_reflection(incident, &shadevars, obj->tex->specular);
    ColorAccum(&col, &specol);
  }

  /* spawn transmission rays / refraction */
  /* note: this will overwrite the old intersection list */
  if (obj->tex->opacity < (1.0 - MINCONTRIB)) {      
    color transcol;
    float alpha = obj->tex->opacity;

    /* Emulate Raster3D's angle-dependent surface opacity if enabled */    
    if ((incident->scene->transmode | obj->tex->transmode) & RT_TRANS_RASTER3D) {
      alpha = 1.0 + COS(3.1415926 * (1.0-alpha) * VDot(&shadevars.N, &incident->d));
      alpha = alpha*alpha * 0.25;
    }

    transcol = shade_transmission(incident, &shadevars, 1.0 - alpha);
    if (incident->scene->transmode & RT_TRANS_VMD) 
      ColorScale(&col, alpha);

    ColorAccum(&col, &transcol);
  }

  /* calculate fog effects */
  if (incident->scene->fog.fog_fctn != NULL) {
    col = fog_color(incident, col, t);
  }

  return tocolora(col);    /* return the color of the shaded pixel... */
}



/*
 * Full Quality Shader - Includes all possible rendering features
 *
 */

colora full_shader(ray * incident) {
  colora cola;
  color col, diffuse, ambocccol, phongcol;
  shadedata shadevars;
  ray shadowray;
  flt inten;
  flt t = FHUGE;
  object const * obj;
  int numints;
  list * cur;

  numints=closest_intersection(&t, &obj, incident);  
		/* find the number of intersections */
                /* and return the closest one.      */

  if (numints < 1) {         
    /* if there weren't any object intersections then return the */
    /* background texture for the pixel color.                   */
    cola=incident->scene->bgtexfunc(incident);

    /* Fog overrides the background color if we're using         */
    /* Tachyon radial fog, but not for OpenGL style fog.         */
    if (incident->scene->fog.type == RT_FOG_NORMAL &&
        incident->scene->fog.fog_fctn != NULL) {
      cola = tocolora(fog_color(incident, col, t));
    }

    return cola;
  }

  RAYPNT(shadevars.hit, (*incident), t) /* find point of intersection from t */ 
  incident->opticdist += t;
  obj->methods->normal(obj, &shadevars.hit, incident, &shadevars.N);  /* find the surface normal */

  /* don't render transparent surfaces if we've reached the max count */
  if ((obj->tex->opacity < 1.0) && (incident->transcnt < 1)) {      
    /* spawn transmission rays / refraction */
    /* note: this will overwrite the old intersection list */
    return tocolora(shade_transmission(incident, &shadevars, 1.0));
  }

  /* execute the object's texture function */
  col = obj->tex->texfunc(&shadevars.hit, obj->tex, incident); 

  if (obj->tex->flags & RT_TEXTURE_ISLIGHT) {  
                  /* if the current object is a light, then we  */
    return tocolora(col);   /* will only use the object's base color      */
  }

  diffuse.r = 0.0; 
  diffuse.g = 0.0; 
  diffuse.b = 0.0; 
  ambocccol = diffuse;
  phongcol = diffuse;
  if ((obj->tex->diffuse > MINCONTRIB) || (obj->tex->phong > MINCONTRIB)) {  
    flt light_scale = incident->scene->light_scale;
    cur = incident->scene->lightlist;

    if (incident->scene->flags & RT_SHADE_CLIPPING) {
      shadowray.add_intersection = add_clipped_shadow_intersection;
    } else {
      shadowray.add_intersection = add_shadow_intersection;
    }
    shadowray.serial = incident->serial + 1; /* track ray serial number */
    shadowray.mbox = incident->mbox;
    shadowray.scene = incident->scene;

    while (cur != NULL) {              /* loop for light contributions */
      light * li=(light *) cur->item;  /* set li=to the current light  */
      inten = light_scale * li->shade_diffuse(li, &shadevars);

      /* add in diffuse lighting for this light if we're facing it */ 
      if (inten > MINCONTRIB) {            
        /* test for a shadow */
        shadowray.o   = shadevars.hit;
        shadowray.d   = shadevars.L;      
        shadowray.maxdist = shadevars.Llen;
        shadowray.flags = RT_RAY_SHADOW;
        shadowray.serial++;
        intersect_objects(&shadowray); /* trace the shadow ray */

        if (!shadow_intersection(&shadowray)) {
          /* If the light isn't occluded, then we modulate it by any */
          /* transparent surfaces the shadow ray encountered, and    */
          /* proceed with illumination calculations                  */
          inten *= shadowray.intstruct.shadowfilter;

          /* calculate diffuse lighting component */
          ColorAddS(&diffuse, &((standard_texture *)li->tex)->col, inten);

          /* phong type specular highlights */
          if (obj->tex->phong > MINCONTRIB) {
            flt phongval = light_scale * incident->scene->phongfunc(incident, &shadevars, obj->tex->phongexp); 
            if (obj->tex->phongtype == RT_PHONG_METAL) 
              ColorAddS(&phongcol, &col, phongval * obj->tex->phong);
            else
              ColorAddS(&phongcol, &((standard_texture *)li->tex)->col, phongval * obj->tex->phong);
          }
        }
      }  

      cur = cur->next;
    } 
    incident->serial = shadowray.serial; /* track ray serial number */

    /* add ambient occlusion lighting, if enabled */
    if (incident->scene->ambocc.numsamples > 0) { 
      ambocccol = shade_ambient_occlusion(incident, &shadevars);
    }
  }

  /* accumulate diffuse and ambient occlusion together */
  diffuse.r += ambocccol.r;
  diffuse.g += ambocccol.g;
  diffuse.b += ambocccol.b;

  if (obj->tex->outline > 0.0) {
    flt outlinefactor;
    flt edgefactor = VDot(&shadevars.N, &incident->d);
    edgefactor *= edgefactor;
    edgefactor = 1.0 - edgefactor;
    edgefactor = 1.0 - POW(edgefactor, (1.0 - obj->tex->outlinewidth) * 32.0);
    outlinefactor = (1.0-obj->tex->outline) + (edgefactor * obj->tex->outline);
    ColorScale(&diffuse, obj->tex->diffuse * outlinefactor);
  } else {
    ColorScale(&diffuse, obj->tex->diffuse);
  }

  /* do a product of the diffuse intensity with object color + ambient */
  col.r *= (diffuse.r + obj->tex->ambient);
  col.g *= (diffuse.g + obj->tex->ambient);
  col.b *= (diffuse.b + obj->tex->ambient);

  if (obj->tex->phong > MINCONTRIB) {
    ColorAccum(&col, &phongcol);
  }

  /* spawn reflection rays if necessary */
  /* note: this will overwrite the old intersection list */
  if (obj->tex->specular > MINCONTRIB) {    
    color specol;
    specol = shade_reflection(incident, &shadevars, obj->tex->specular);
    ColorAccum(&col, &specol);
  }

  /* spawn transmission rays / refraction */
  /* note: this will overwrite the old intersection list */
  if (obj->tex->opacity < (1.0 - MINCONTRIB)) {      
    color transcol;
    float alpha = obj->tex->opacity;

    /* Emulate Raster3D's angle-dependent surface opacity if enabled */    
    if ((incident->scene->transmode | obj->tex->transmode) & RT_TRANS_RASTER3D) {
      alpha = 1.0 + COS(3.1415926 * (1.0-alpha) * VDot(&shadevars.N, &incident->d));
      alpha = alpha*alpha * 0.25;
    }

    transcol = shade_transmission(incident, &shadevars, 1.0 - alpha);
    if (incident->scene->transmode & RT_TRANS_VMD) 
      ColorScale(&col, alpha);

    ColorAccum(&col, &transcol);
  }

  /* calculate fog effects */
  if (incident->scene->fog.fog_fctn != NULL) {
    col = fog_color(incident, col, t);
  }

  return tocolora(col);    /* return the color of the shaded pixel... */
}


/*
 * Ambient Occlusion shader, implements a simple global illumination-like
 * lighting model which does well for highly diffuse natural lighting 
 * scenes.
 */
color shade_ambient_occlusion(ray * incident, const shadedata * shadevars) {
  ray ambray;
  color ambcol;
  int i;
  flt ndotambl;
  flt inten = 0.0;

  /* The integrated hemisphere for an unweighted non-importance-sampled  */
  /* ambient occlusion case has a maximum sum (when uniformly sampled)   */
  /* of 0.5 relative to an similar direct illumination case oriented     */
  /* exactly with the surface normal.  So, we prescale the normalization */
  /* for ambient occlusion by a factor of 2.0.  This will have to change */
  /* when importance sampling is implemented.  If a small number of      */
  /* occlusion samples are taken, and they coincidentally end up facing  */
  /* with the surface normal, we could exceed the expected normalization */
  /* factor, but the results will be correctly clamped by the rest of    */
  /* shading code, so we don't worry about it here.                      */
  flt lightscale = 2.0 / incident->scene->ambocc.numsamples;

  ambray.o=shadevars->hit;
  ambray.d=shadevars->N;
  ambray.o=Raypnt(&ambray, EPSILON);    /* avoid numerical precision bugs */
  ambray.serial = incident->serial + 1; /* next serial number */
  ambray.randval=incident->randval;     /* random number seed */
  ambray.frng=incident->frng;           /* 32-bit FP RNG handle */
  if (incident->scene->flags & RT_SHADE_CLIPPING) {
    ambray.add_intersection = add_clipped_shadow_intersection;
  } else {
    ambray.add_intersection = add_shadow_intersection;
  }
  ambray.mbox = incident->mbox; 
  ambray.scene=incident->scene;         /* global scenedef info */

  for (i=0; i<incident->scene->ambocc.numsamples; i++) {
    float dir[3];
    ambray.maxdist = FHUGE;         /* take any intersection */
    ambray.flags = RT_RAY_SHADOW;   /* shadow ray */
    ambray.serial++;

    /* generate a randomly oriented ray */
    jitter_sphere3f(&ambray.frng, dir);
    ambray.d.x = dir[0];
    ambray.d.y = dir[1];
    ambray.d.z = dir[2];

    /* flip the ray so it's in the same hemisphere as the surface normal */
    ndotambl = VDot(&ambray.d, &shadevars->N);
    if (ndotambl < 0) {
      ndotambl   = -ndotambl;
      ambray.d.x = -ambray.d.x;
      ambray.d.y = -ambray.d.y;
      ambray.d.z = -ambray.d.z;
    }

    intersect_objects(&ambray); /* trace the shadow ray */

    /* if no objects were hit, add an ambient contribution */
    if (!shadow_intersection(&ambray)) {
      /* If the light isn't occluded, then we modulate it by any */
      /* transparent surfaces the shadow ray encountered, and    */
      /* proceed with illumination calculations                  */
      ndotambl *= ambray.intstruct.shadowfilter;

      inten += ndotambl;
    }
  }
  ambcol.r = lightscale * inten * incident->scene->ambocc.col.r;
  ambcol.g = lightscale * inten * incident->scene->ambocc.col.g;
  ambcol.b = lightscale * inten * incident->scene->ambocc.col.b;

  incident->serial = ambray.serial + 1;     /* update the serial number */
  incident->frng = ambray.frng;             /* update AO RNG state      */

  return ambcol; 
}


color shade_reflection(ray * incident, const shadedata * shadevars, flt specular) {
  ray specray;
  color col;
  colora cola;
  vector R;

  /* Do recursion depth test immediately to early-exit ASAP */
  if (incident->depth <= 1) {
    /* if ray is truncated, return the background texture as its color */
	cola = incident->scene->bgtexfunc(incident);
	col.r = cola.r;
	col.g = cola.g;
	col.b = cola.b;
    return col;
  }
  specray.depth=incident->depth - 1;     /* go up a level in recursion depth */
  specray.transcnt=incident->transcnt;   /* maintain trans surface count */
 
  VAddS(-2.0 * (incident->d.x * shadevars->N.x + 
                incident->d.y * shadevars->N.y + 
                incident->d.z * shadevars->N.z), &shadevars->N, &incident->d, &R);

  specray.o=shadevars->hit; 
  specray.d=R;			         /* reflect incident ray about normal */
  specray.o=Raypnt(&specray, EPSILON);   /* avoid numerical precision bugs */
  specray.maxdist = FHUGE;               /* take any intersection */
  specray.opticdist = incident->opticdist;
  specray.add_intersection=incident->add_intersection; /* inherit ray type  */
  specray.flags = RT_RAY_REGULAR;        /* infinite ray, to start with */
  specray.serial = incident->serial + 1; /* next serial number */
  specray.mbox = incident->mbox; 
  specray.scene=incident->scene;         /* global scenedef info */
  specray.randval=incident->randval;     /* random number seed */
  specray.frng=incident->frng;           /* 32-bit FP RNG handle */

  /* inlined code from trace() to eliminate one level of recursion */
  intersect_objects(&specray);           /* trace specular reflection ray */
  cola=specray.scene->shader(&specray);
  col.r = cola.r;
  col.g = cola.g;
  col.b = cola.b;

  incident->serial = specray.serial;     /* update the serial number */
  incident->frng = specray.frng;         /* update AO RNG state      */

  ColorScale(&col, specular);

  return col;
}


color shade_transmission(ray * incident, const shadedata * shadevars, flt trans) {
  ray transray;
  color col;
  colora cola;

  /* Do recursion depth test immediately to early-exit ASAP */
  if (incident->depth <= 1) {
    /* if ray is truncated, return the background texture as its color */
    cola = incident->scene->bgtexfunc(incident);
	col.r = cola.r;
	col.g = cola.g;
	col.b = cola.b;
    return col;
  }
  transray.o=shadevars->hit; 
  transray.d=incident->d;                 /* ray continues on incident path */
  transray.o=Raypnt(&transray, EPSILON);  /* avoid numerical precision bugs */
  transray.maxdist = FHUGE;               /* take any intersection */
  transray.opticdist = incident->opticdist;
  transray.add_intersection=incident->add_intersection; /* inherit ray type  */
  transray.depth=incident->depth - 1;     /* track recursion depth           */
  transray.transcnt=incident->transcnt-1; /* maintain trans surface count    */
  transray.flags = RT_RAY_REGULAR;        /* infinite ray, to start with */
  transray.serial = incident->serial + 1; /* update serial number */
  transray.mbox = incident->mbox;
  transray.scene=incident->scene;         /* global scenedef info */
  transray.randval=incident->randval;     /* random number seed */
  transray.frng=incident->frng;           /* 32-bit FP RNG handle */

  /* inlined code from trace() to eliminate one level of recursion */
  intersect_objects(&transray);           /* trace transmission ray */
  cola=transray.scene->shader(&transray);
  col.r = cola.r;
  col.g = cola.g;
  col.b = cola.b;

  incident->serial = transray.serial;     /* update the serial number */
  incident->frng = transray.frng;         /* update AO RNG state      */

  ColorScale(&col, trans);

  return col;
}


/*
 * Phong shader, always returns 0.0, used for testing
 */
flt shade_nullphong(const ray * incident, const shadedata * shadevars, flt specpower) {
  return 0.0;
}

/*
 * Phong shader, implements specular highlight model
 * using Blinn's halfway vector dotted with the surface normal.
 * This is also the shading model used by OpenGL and Direct3D.
 */
flt shade_blinn(const ray * incident, const shadedata * shadevars, flt specpower) {
  vector H;   /* Blinn's halfway vector */
  flt inten;  /* calculated intensity   */

  /* since incident ray is negated direction to viewer, we subtract... */
  /* sub. incoming ray dir. from light direction */
  H.x = shadevars->L.x - incident->d.x; 
  H.y = shadevars->L.y - incident->d.y;
  H.z = shadevars->L.z - incident->d.z;

  inten = shadevars->N.x * H.x + shadevars->N.y * H.y + shadevars->N.z * H.z;
  if (inten > MINCONTRIB) {
    /* normalize the previous dot product */
    inten /= SQRT(H.x * H.x + H.y * H.y + H.z * H.z);

    /* calculate specular exponent */
    inten = POW(inten, specpower);
  } else {
    inten = 0.0;
  }

  return inten;
}



/*
 * Phong shader, implements specular highlight model
 * using Blinn's halfway vector dotted with the surface normal.
 * This is also the shading model used by OpenGL and Direct3D.
 * Uses Graphics Gems IV chapter VI.1 algorithm for phong exponent
 * instead of the usual call to pow(). 
 */
flt shade_blinn_fast(const ray * incident, const shadedata * shadevars, flt specpower) {
  vector H;   /* Blinn's halfway vector */
  flt inten;  /* calculated intensity   */

  /* since incident ray is negated direction to viewer, we subtract... */
  /* sub. incoming ray dir. from light direction */
  H.x = shadevars->L.x - incident->d.x; 
  H.y = shadevars->L.y - incident->d.y;
  H.z = shadevars->L.z - incident->d.z;

  inten = shadevars->N.x * H.x + shadevars->N.y * H.y + shadevars->N.z * H.z;
  if (inten > 0.0) {
    /* normalize the previous dot product */
    inten /= SQRT(H.x * H.x + H.y * H.y + H.z * H.z);

    /* replace specular exponent with a simple approximation */
    inten = inten / (specpower - (specpower * inten) + inten);
  } else {
    inten = 0.0;
  }

  return inten;
} 


/*
 * Phong shader, implements a Phong specular highlight model
 * using the reflection vector about the surface normal, dotted
 * with the direction to the viewer.  This is the "classic" phong
 */
flt shade_phong(const ray * incident, const shadedata * shadevars, flt specpower) {
  vector R;   /* reflection vector      */
  vector V;   /* direction to viewpoint */
  vector LL;  /* reverse direction to light     */
  flt inten;  /* calculated intensity   */

  LL = shadevars->L;
  VScale(&LL, -1.0);
  VAddS(-2.0 * (LL.x * shadevars->N.x + 
                LL.y * shadevars->N.y + 
                LL.z * shadevars->N.z), &shadevars->N, &LL, &R);

  V = incident->d;
  VScale(&V, -1.0);
  VNorm(&R);            /* normalize reflection vector */
  inten = VDot(&V, &R); /* dot product of halfway vector and surface normal */
  if (inten > 0.0)     
    inten = POW(inten, specpower);
  else 
    inten = 0.0;

  return inten;
} 


/*
 * Fog functions (can be used for both radial or planar fog implementations)
 */

/**
 * Compute the fot color, given the active fogging function and
 * fog parameters.
 */
color fog_color(const ray * incident, color col, flt t) {
  struct fogdata_t * fog = &incident->scene->fog;
  float fogcoord = t; /* radial fog by default */

  if (fog->type == RT_FOG_OPENGL) {
    /* Compute planar fog (e.g. to match OpenGL) by projecting t value onto  */
    /* the camera view direction vector to yield a planar a depth value.     */
    flt hitz = VDot(&incident->d, &incident->scene->camera.viewvec) * t;

    /* use the Z-depth for primary rays, radial distance otherwise */
    fogcoord = (incident->flags & RT_RAY_PRIMARY) ? hitz : t;
  }

  return incident->scene->fog.fog_fctn(fog, col, fogcoord);
}


/**
 * OpenGL-like linear fog
 */
color fog_color_linear(struct fogdata_t * fog, color col, flt r) {
  color c; 
  flt f;

  f = (fog->end - r) / (fog->end - fog->start);
  if (f > 1.0) {
    f = 1.0;
  } else if (f < 0.0) {
    f = 0.0;
  } 

  c.r = (f * col.r) + ((1 - f) * fog->col.r);
  c.g = (f * col.g) + ((1 - f) * fog->col.g);
  c.b = (f * col.b) + ((1 - f) * fog->col.b);

  return c;
}


/**
 * OpenGL-like exponential fog
 */
color fog_color_exp(struct fogdata_t * fog, color col, flt r) {
  color c; 
  flt f, v;

  v = fog->density * (r - fog->start);
  f = EXP(-v);
  if (f > 1.0) {
    f = 1.0;
  } else if (f < 0.0) {
    f = 0.0;
  } 

  c.r = (f * col.r) + ((1 - f) * fog->col.r);
  c.g = (f * col.g) + ((1 - f) * fog->col.g);
  c.b = (f * col.b) + ((1 - f) * fog->col.b);

  return c;
}


/**
 * OpenGL-like exponential-squared fog
 */
color fog_color_exp2(struct fogdata_t * fog, color col, flt r) {
  color c; 
  flt f, v;
  
  v = fog->density * (r - fog->start);
  f = EXP(-v*v);
  if (f > 1.0) {
    f = 1.0;
  } else if (f < 0.0) {
    f = 0.0;
  } 

  c.r = (f * col.r) + ((1 - f) * fog->col.r);
  c.g = (f * col.g) + ((1 - f) * fog->col.g);
  c.b = (f * col.b) + ((1 - f) * fog->col.b);

  return c;
}

