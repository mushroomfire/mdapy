/* 
 * texture.c - This file contains functions for implementing textures.
 * 
 * (C) Copyright 1994-2022 John E. Stone
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * $Id: texture.c,v 1.42 2022/03/25 06:13:10 johns Exp $ 
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TACHYON_INTERNAL 1
#include "tachyon.h"
#include "macros.h"
#include "texture.h"
#include "coordsys.h"
#include "imap.h"
#include "vector.h"
#include "box.h"
#include "util.h"

static texture_methods normal_methods = {
  free
};

static texture_methods standard_methods = {
  free_standard_texture
};

static texture_methods vcstri_methods = {
  free
};

texture * new_texture(void) {
  texture * tex;
  tex = (texture *) malloc(sizeof(texture));
  tex->methods = &normal_methods;
  return tex;
}

texture * new_standard_texture(void) {
  standard_texture * tex;
  tex = (standard_texture *) malloc(sizeof(standard_texture));
  tex->methods = &standard_methods;
  return (texture *) tex;
}

texture * new_vcstri_texture(void) {
  vcstri_texture * tex;
  tex = (vcstri_texture *) malloc(sizeof(vcstri_texture));
  tex->methods = &vcstri_methods;
  return (texture *) tex;
}

void free_standard_texture(void * voidtex) {
  standard_texture * tex = (standard_texture *) voidtex;
  if (tex->img != NULL) {
    if ((tex->texfunc == (color (*)(const void *, const void *, void *)) image_plane_texture) ||
        (tex->texfunc == (color (*)(const void *, const void *, void *)) image_cyl_texture) ||
        (tex->texfunc == (color (*)(const void *, const void *, void *)) image_sphere_texture) ||
        (tex->texfunc == (color (*)(const void *, const void *, void *)) image_volume_texture)) {
      FreeMIPMap(tex->img);
      tex->img = NULL;
    } else {
printf("XXX Doh, unrecognized image map type!\n");
    }
  }
  free(tex);
}


/* standard solid background texture */
color solid_background_texture(ray *ry) {
  return ry->scene->bgtex.bg_color;
}


/* sky sphere background texture, linear mapping */
color sky_sphere_background_texture(ray *ry) {
  color col;
  flt tmp;
  flt IdotG = VDot(&ry->d, &ry->scene->bgtex.bg_grad_updir);
  flt val = (IdotG - ry->scene->bgtex.bg_grad_botval) * 
            ry->scene->bgtex.bg_grad_invrange;

  /*
   * Compute and add random noise to the background gradient to
   * avoid banding artifacts, particularly in compressed video.
   * Noise RNG depends only on pixel index, with no sample/subframe
   * contribution, so that dither pattern won't average out.
   */
  unsigned int randval = tea4(ry->idx, ry->idx);
  flt u = rt_rand(&randval) * RT_RAND_MAX_INV;
  flt noise = ry->scene->bgtex.bg_grad_noisemag * (u - 0.5); 
  val += noise;

  tmp = (val > 1.0) ? 1.0 : val;
  val = (tmp < 0.0) ? 0.0 : tmp;

  col.r = ry->scene->bgtex.bg_grad_top.r * val + 
          ry->scene->bgtex.bg_grad_bot.r * (1.0 - val);
  col.g = ry->scene->bgtex.bg_grad_top.g * val + 
          ry->scene->bgtex.bg_grad_bot.g * (1.0 - val);
  col.b = ry->scene->bgtex.bg_grad_top.b * val + 
          ry->scene->bgtex.bg_grad_bot.b * (1.0 - val);

  return col;
}


/* sky orthographic plane background texture, linear mapping */
color sky_plane_background_texture(ray *ry) {
  color col;
  flt tmp;
  flt IdotG = VDot(&ry->o, &ry->scene->bgtex.bg_grad_updir);
  flt val = (IdotG - ry->scene->bgtex.bg_grad_botval) * 
            ry->scene->bgtex.bg_grad_invrange;

  /*
   * Compute and add random noise to the background gradient to
   * avoid banding artifacts, particularly in compressed video.
   * Noise RNG depends only on pixel index, with no sample/subframe
   * contribution, so that dither pattern won't average out.
   */
  unsigned int randval = tea4(ry->idx, ry->idx);
  flt u = rt_rand(&randval) * RT_RAND_MAX_INV;
  flt noise = ry->scene->bgtex.bg_grad_noisemag * (u - 0.5); 
  val += noise;

  tmp = (val > 1.0) ? 1.0 : val;
  val = (tmp < 0.0) ? 0.0 : tmp;

  col.r = ry->scene->bgtex.bg_grad_top.r * val + 
          ry->scene->bgtex.bg_grad_bot.r * (1.0 - val);
  col.g = ry->scene->bgtex.bg_grad_top.g * val + 
          ry->scene->bgtex.bg_grad_bot.g * (1.0 - val);
  col.b = ry->scene->bgtex.bg_grad_top.b * val + 
          ry->scene->bgtex.bg_grad_bot.b * (1.0 - val);

  return col;
}


/* plain vanilla texture solely based on object color */
color constant_texture(const vector * hit, const texture * tx, const ray * ry) {
  standard_texture * tex = (standard_texture *) tx;
  return tex->col;
}

/* cylindrical image map */
color image_cyl_texture(const vector * hit, const texture * tx, const ray * ry) {
  vector rh;
  flt u, v, miprad, maxscale, cyrad;
  standard_texture * tex = (standard_texture *) tx;
 
  rh.x=hit->x - tex->ctr.x;
  rh.z=hit->y - tex->ctr.y;
  rh.y=hit->z - tex->ctr.z;
 
  xyztocyl(rh, 1.0, &u, &v);

  u = u * tex->scale.x;  
  u = u + tex->rot.x;
  u = u - ((int) u);
  if (u < 0.0) u+=1.0; 

  v = v * tex->scale.y; 
  v = v + tex->rot.y;
  v = v - ((int) v);
  if (v < 0.0) v+=1.0; 

  cyrad = EPSILON + 8.0 * SQRT(rh.x*rh.x + rh.y*rh.y + rh.z*rh.z);

  maxscale = (FABS(tex->scale.x) > FABS(tex->scale.y)) ? 
             tex->scale.x : tex->scale.y;

  miprad = (0.05 * ry->opticdist * FABS(maxscale)) / cyrad;

  return MIPMap(tex->img, u, v, miprad);
}  

/* spherical image map */
color image_sphere_texture(const vector * hit, const texture * tx, const ray * ry) {
  vector rh;
  flt u, v, miprad, maxscale, sprad;
  standard_texture * tex = (standard_texture *) tx;
 
  rh.x=hit->x - tex->ctr.x;
  rh.y=hit->y - tex->ctr.y;
  rh.z=hit->z - tex->ctr.z;
 
  xyztospr(rh, &u, &v);

  u = u * tex->scale.x;
  u = u + tex->rot.x;
  u = u - ((int) u);
  if (u < 0.0) u+=1.0;
 
  v = v * tex->scale.y;
  v = v + tex->rot.y;
  v = v - ((int) v);
  if (v < 0.0) v+=1.0;

  sprad = EPSILON + 8.0 * SQRT(rh.x*rh.x + rh.y*rh.y + rh.z*rh.z);

  maxscale = (FABS(tex->scale.x) > FABS(tex->scale.y)) ? 
             tex->scale.x : tex->scale.y;

  miprad = (0.05 * ry->opticdist * FABS(maxscale)) / sprad;
 
  return MIPMap(tex->img, u, v, miprad);
}

/* planar image map */
color image_plane_texture(const vector * hit, const texture * tx, const ray * ry) {
  vector pnt;
  flt u, v, miprad, maxscale;
  standard_texture * tex = (standard_texture *) tx;
 
  pnt.x=hit->x - tex->ctr.x;
  pnt.y=hit->y - tex->ctr.y;
  pnt.z=hit->z - tex->ctr.z;

  VDOT(u, tex->uaxs, pnt);
  VDOT(v, tex->vaxs, pnt); 

  u = u * tex->scale.x;
  u = u + tex->rot.x;
  u = u - ((int) u);
  if (u < 0.0) u += 1.0;

  v = v * tex->scale.y;
  v = v + tex->rot.y;
  v = v - ((int) v);
  if (v < 0.0) v += 1.0;

  
  maxscale = (FABS(tex->scale.x) > FABS(tex->scale.y)) ? 
             tex->scale.x : tex->scale.y;

  miprad = 0.05 * ry->opticdist * FABS(maxscale);

  return MIPMap(tex->img, u, v, miprad);
}


/* volumetric texture map (applied to surface geometry etc) */
color image_volume_texture(const vector * hit, const texture * tx, const ray * ry) {
  vector pnt;
  flt u, v, w, miprad, maxscale;
  standard_texture * tex = (standard_texture *) tx;
 
  pnt.x=hit->x - tex->ctr.x;
  pnt.y=hit->y - tex->ctr.y;
  pnt.z=hit->z - tex->ctr.z;

  VDOT(u, tex->uaxs, pnt);
  VDOT(v, tex->vaxs, pnt); 
  VDOT(w, tex->waxs, pnt); 

  u = u * tex->scale.x;
  u = u + tex->rot.x;
  u = u - ((int) u);
  if (u < 0.0) u += 1.0;

  v = v * tex->scale.y;
  v = v + tex->rot.y;
  v = v - ((int) v);
  if (v < 0.0) v += 1.0;

  w = w * tex->scale.z;
  w = w + tex->rot.z;
  w = w - ((int) w);
  if (w < 0.0) w += 1.0;

  maxscale = (FABS(tex->scale.x) > FABS(tex->scale.y)) ? 
             tex->scale.x : tex->scale.y;
  if (FABS(tex->scale.z) > FABS(maxscale))
    maxscale = tex->scale.z;

  miprad = 0.05 * ry->opticdist * FABS(maxscale);

  return VolMIPMap(tex->img, u, v, w, miprad);
}


color grit_texture(const vector * hit, const texture * tx, const ray * ry) {
  int rnum;
  flt fnum;
  color col;
  standard_texture * tex = (standard_texture *) tx;

  rnum=rand() % 4096;
  fnum=(rnum / 4096.0 * 0.2) + 0.8;

  col.r=tex->col.r * fnum;
  col.g=tex->col.g * fnum;
  col.b=tex->col.b * fnum;

  return col;
}

color checker_texture(const vector * hit, const texture * tx, const ray * ry) {
  long x,y,z;
  flt xh,yh,zh;
  color col;
  standard_texture * tex = (standard_texture *) tx;

  xh=hit->x - tex->ctr.x; 
  x=(long) ((FABS(xh) * 3) + 0.5);
  x=x % 2;
  yh=hit->y - tex->ctr.y;
  y=(long) ((FABS(yh) * 3) + 0.5);
  y=y % 2;
  zh=hit->z - tex->ctr.z;
  z=(long) ((FABS(zh) * 3) + 0.5);
  z=z % 2;

  if (((x + y + z) % 2)==1) {
    col.r=1.0;
    col.g=0.2;
    col.b=0.0;
  }
  else {
    col.r=0.0;
    col.g=0.2;
    col.b=1.0;
  }

  return col;
}

color cyl_checker_texture(const vector * hit, const texture * tx, const ray * ry) {
  long x,y;
  vector rh;
  flt u,v;
  color col;
  standard_texture * tex = (standard_texture *) tx;
 
  rh.x=hit->x - tex->ctr.x;
  rh.y=hit->y - tex->ctr.y;
  rh.z=hit->z - tex->ctr.z;

  xyztocyl(rh, 1.0, &u, &v); 

  x=(long) (FABS(u) * 18.0);
  x=x % 2;
  y=(long) (FABS(v) * 10.0);
  y=y % 2;
 
  if (((x + y) % 2)==1) {
    col.r=1.0;
    col.g=0.2;
    col.b=0.0;
  }
  else {
    col.r=0.0;
    col.g=0.2;
    col.b=1.0;
  }
 
  return col;
}


color wood_texture(const vector * hit, const texture * tx, const ray * ry) {
  flt radius, angle;
  int grain;
  color col;
  flt x,y,z;
  standard_texture * tex = (standard_texture *) tx;

  x=(hit->x - tex->ctr.x) / tex->scale.x;
  y=(hit->y - tex->ctr.y) / tex->scale.y;
  z=(hit->z - tex->ctr.z) / tex->scale.z;

  radius=sqrt(x*x + z*z);
  if (z == 0.0) 
    angle=3.1415926/2.0;
  else 
    angle=atan(x / z);

  radius=radius + 3.0 * SIN(20 * angle + y / 150.0);
  grain=((int) (radius + 0.5)) % 60;
  if (grain < 40) {
    col.r=0.8;
    col.g=1.0;
    col.b=0.2;
  }
  else {
    col.r=0.0;
    col.g=0.0;
    col.b=0.0;
  }     

  return col;
} 



#define NMAX 28
short int NoiseMatrix[NMAX][NMAX][NMAX];

void InitNoise(void) {
  byte x,y,z,i,j,k;
  unsigned int rndval = 1234567; /* pathetic random number seed */

  for (x=0; x<NMAX; x++) {
    for (y=0; y<NMAX; y++) {
      for (z=0; z<NMAX; z++) {
        NoiseMatrix[x][y][z]=(short int) ((rt_rand(&rndval) * RT_RAND_MAX_INV) * 12000.0);

        if (x==NMAX-1) i=0; 
        else i=x;

        if (y==NMAX-1) j=0;
        else j=y;

        if (z==NMAX-1) k=0;
        else k=z;

        NoiseMatrix[x][y][z]=NoiseMatrix[i][j][k];
      }
    }
  }
}

int Noise(flt x, flt y, flt z) {
  byte ix, iy, iz;
  flt ox, oy, oz;
  int p000, p001, p010, p011;
  int p100, p101, p110, p111;
  int p00, p01, p10, p11;
  int p0, p1;
  int d00, d01, d10, d11;
  int d0, d1, d;

  x=FABS(x);
  y=FABS(y);
  z=FABS(z);

  ix=((int) x) % NMAX;
  iy=((int) y) % NMAX;
  iz=((int) z) % NMAX;

  ox=(x - ((int) x));
  oy=(y - ((int) y));
  oz=(z - ((int) z));

  p000=NoiseMatrix[ix][iy][iz];
  p001=NoiseMatrix[ix][iy][iz+1];
  p010=NoiseMatrix[ix][iy+1][iz];
  p011=NoiseMatrix[ix][iy+1][iz+1];
  p100=NoiseMatrix[ix+1][iy][iz];
  p101=NoiseMatrix[ix+1][iy][iz+1];
  p110=NoiseMatrix[ix+1][iy+1][iz];
  p111=NoiseMatrix[ix+1][iy+1][iz+1];

  d00=p100-p000;
  d01=p101-p001;
  d10=p110-p010;
  d11=p111-p011;

  p00=(int) ((int) d00*ox) + p000;
  p01=(int) ((int) d01*ox) + p001;
  p10=(int) ((int) d10*ox) + p010;
  p11=(int) ((int) d11*ox) + p011;
  d0=p10-p00;
  d1=p11-p01;
  p0=(int) ((int) d0*oy) + p00;
  p1=(int) ((int) d1*oy) + p01;
  d=p1-p0;

  return (int) ((int) d*oz) + p0;
}

color marble_texture(const vector * hit, const texture * tx, const ray * ry) {
  flt i,d,t;
  flt x,y,z;
  color col;
/*
  standard_texture * tex = (standard_texture *) tx;
*/
 
  x=hit->x;
  y=hit->y; 
  z=hit->z;

  x=x * 1.0;

  d=x + 0.0006 * Noise(x, (y * 1.0), (z * 1.0));
  d=d*(((int) d) % 25);
  i=0.0 + 0.10 * FABS(d - 10.0f - 20.0f * ((int) d * 0.05f));

  t = (i > 1.0) ? 1.0 : i;
  i = (t < 0.0) ? 0.0 : t;

/*
  col.r=i * tex->col.r;
  col.g=i * tex->col.g;
  col.b=i * tex->col.b;
*/

  col.r = (1.0 + SIN(i * 6.28)) / 2.0;
  col.g = (1.0 + SIN(i * 16.28)) / 2.0;
  col.b = (1.0 + COS(i * 30.28)) / 2.0;

  return col;      
}


color gnoise_texture(const vector * hit, const texture * tx, const ray * ry) {
  color col;
  flt f, t;
  standard_texture * tex = (standard_texture *) tx;

  f=Noise((hit->x - tex->ctr.x), 
          (hit->y - tex->ctr.y), 
	  (hit->z - tex->ctr.z));

  t = (f > 1.0) ? 1.0 : f;
  f = (t < 0.01) ? 0.01 : t;

  col.r=tex->col.r * f;
  col.g=tex->col.g * f;
  col.b=tex->col.b * f;

  return col;
}

void InitTextures(void) {
  InitNoise();
  ResetImages();
}

void FreeTextures(void) {
  FreeImages();
}

