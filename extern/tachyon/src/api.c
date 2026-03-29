/*
 * api.c - This file contains all of the API calls that are defined for
 *         external driver code to use.  
 * 
 *  $Id: api.c,v 1.193 2011/02/18 06:01:46 johns Exp $
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TACHYON_INTERNAL 1
#include "tachyon.h"
#include "macros.h"

#include "parallel.h"
#include "threads.h"

#include "box.h"
#include "cone.h"
#include "cylinder.h"
#include "plane.h"
#include "quadric.h"
#include "ring.h"
#include "sphere.h"
#include "triangle.h"
#include "vol.h"
#include "extvol.h"

#include "texture.h"
#include "light.h"
#include "render.h"
#include "trace.h"
#include "camera.h"
#include "vector.h"
#include "intersect.h"
#include "shade.h"
#include "util.h"
#include "imap.h"
#include "global.h"
#include "ui.h"
#include "shade.h"

apivector rt_vector(flt x, flt y, flt z) {
  apivector v;

  v.x = x;
  v.y = y;
  v.z = z;

  return v;
}

apicolor rt_color(flt r, flt g, flt b) {
  apicolor c;
  
  c.r = r;
  c.g = g;
  c.b = b;
  
  return c;
}

colora tocolora(apicolor c) {
	colora ca = { c.r, c.g, c.b, 1 };
	return ca;
}

int rt_initialize(int * argc, char ***argv) {
  InitTextures();

  if (!parinitted) {
    rt_par_init(argc, argv);
    parinitted=1;
  }

  return rt_mynode(); /* return our node id */ 
}

void rt_finalize(void) {
  FreeTextures();
  rt_par_finish();
}

void rt_renderscene(SceneHandle voidscene) {
  scenedef * scene = (scenedef *) voidscene;
  renderscene(scene);
}

void rt_normal_fixup_mode(SceneHandle voidscene, int mode) {
  scenedef * scene = (scenedef *) voidscene;
  switch (mode) {
    /* RT_NORMAL_FIXUP_MODE_GUESS */
    case 2:
      scene->normalfixupmode = 2; /* accept any normal/winding order combo   */
                                  /* and suffer the consequences, since this */
                                  /* leaves an unhandled case where surface  */
                                  /* normals on poorly tessellated objects   */
                                  /* will cause black edges.                 */
      break;

    /* RT_NORMAL_FIXUP_MODE_FLIP */
    case 1:
      scene->normalfixupmode = 1; /* reverse the surface normal     */
      break;

    /* RT_NORMAL_FIXUP_MODE_OFF */
    case 0:
    default: 
      scene->normalfixupmode = 0; /* use strict winding order rules */
      break;
  }
}

void rt_aa_maxsamples(SceneHandle voidscene, int maxsamples) {
  scenedef * scene = (scenedef *) voidscene;

  if (maxsamples >= 0)
    scene->antialiasing=maxsamples;
  else  
    scene->antialiasing=0;
}

void rt_shadow_filtering(SceneHandle voidscene, int onoff) {
  scenedef * scene = (scenedef *) voidscene;
  scene->shadowfilter = onoff; 
}

void rt_trans_max_surfaces(SceneHandle voidscene, int count) {
  scenedef * scene = (scenedef *) voidscene;
  scene->transcount= count; 
}

void rt_camera_setup(SceneHandle voidscene, flt zoom, flt aspectratio, 
	             int antialiasing, int raydepth, 
                     apivector camcent, apivector viewvec, apivector upvec) {
  scenedef * scene = (scenedef *) voidscene;

  cameradefault(&scene->camera);

  rt_camera_zoom(voidscene, zoom);

  rt_camera_position(voidscene, camcent, viewvec, upvec);

  rt_aspectratio(voidscene, aspectratio);

  rt_aa_maxsamples(voidscene, antialiasing);

  rt_camera_raydepth(voidscene, raydepth);
}

void rt_camera_projection(SceneHandle voidscene, int mode) {
  scenedef * scene = (scenedef *) voidscene;
  cameraprojection(&scene->camera, mode);
}

void rt_camera_position(SceneHandle voidscene, apivector camcent, 
                        apivector viewvec, apivector upvec) {
  scenedef * scene = (scenedef *) voidscene;
  cameraposition(&scene->camera, camcent, viewvec, upvec);
}

void rt_camera_position3fv(SceneHandle voidscene, const float *camcent, 
                           const float *viewvec, const float *upvec) {
  scenedef * scene = (scenedef *) voidscene;
  vector vctr, vview, vup;
  vctr.x = camcent[0];  vctr.y = camcent[1];  vctr.z = camcent[2];
  vview.x = viewvec[0]; vview.y = viewvec[1]; vview.z = viewvec[2];
  vup.x = upvec[0];     vup.y = upvec[1];     vup.z = upvec[2];
  cameraposition(&scene->camera, vctr, vview, vup);
}


void rt_get_camera_position(SceneHandle voidscene, apivector * camcent, 
                            apivector * viewvec, apivector * upvec, 
                            apivector * rightvec) {
  scenedef * scene = (scenedef *) voidscene;
  getcameraposition(&scene->camera, camcent, viewvec, upvec, rightvec);
}

void rt_get_camera_position3fv(SceneHandle voidscene, float *camcent, 
                               float *viewvec, float *upvec, float *rightvec) {
  scenedef * scene = (scenedef *) voidscene;
  vector ctr, view, up, right;
  getcameraposition(&scene->camera, &ctr, &view, &up, &right);
  camcent[0] = ctr.x; camcent[1] = ctr.y; camcent[2] = ctr.z;
  viewvec[0] = view.x; viewvec[1] = view.y; viewvec[2] = view.z;
  upvec[0] = up.x; upvec[1] = up.y; upvec[2] = up.z;
  rightvec[0] = right.x; rightvec[1] = right.y; rightvec[2] = right.z;
}


void rt_camera_raydepth(SceneHandle voidscene, int maxdepth) {
  scenedef * scene = (scenedef *) voidscene;
  scene->raydepth=maxdepth; 
}

void rt_camera_zoom(SceneHandle voidscene, flt zoom) {
  scenedef * scene = (scenedef *) voidscene;
  camerazoom(&scene->camera, zoom);
}

flt rt_get_camera_zoom(SceneHandle voidscene) {
  scenedef * scene = (scenedef *) voidscene;
  return scene->camera.camzoom;
}

void rt_camera_vfov(SceneHandle voidscene, flt vfov) {
  flt zoom = 1.0 / tan((vfov/360.0)*TWOPI/2.0);
  rt_camera_zoom(voidscene, zoom);
}

flt rt_get_camera_vfov(SceneHandle voidscene) {
  scenedef * scene = (scenedef *) voidscene;
  flt vfov = 90.0 * 2.0 * (atan(1.0 / scene->camera.camzoom) / (TWOPI/4.0));
  return vfov;
}


void rt_camera_frustum(SceneHandle voidscene, flt left, flt right, flt bottom, flt top) {
  scenedef * scene = (scenedef *) voidscene;
  camerafrustum(&scene->camera, left, right, bottom, top);
}

void rt_outputfile(SceneHandle voidscene, const char * outname) {
  scenedef * scene = (scenedef *) voidscene;
  if (strlen(outname) > 0) {
    strcpy((char *) &scene->outfilename, outname);
    scene->writeimagefile = 1;
  }
  else {
    scene->writeimagefile = 0;
  }
}

void rt_camera_dof(SceneHandle voidscene, flt focallength, flt aperture) {
  scenedef * scene = (scenedef *) voidscene;
  cameradof(&scene->camera, focallength, aperture);
}


void rt_outputformat(SceneHandle voidscene, int format) {
  scenedef * scene = (scenedef *) voidscene;
  scene->imgfileformat = format; 
}

void rt_resolution(SceneHandle voidscene, int hres, int vres) {
  scenedef * scene = (scenedef *) voidscene;
  scene->hres=hres;
  scene->vres=vres;
  scene->scenecheck = 1;
}

void rt_get_resolution(SceneHandle voidscene, int *hres, int *vres) {
  scenedef * scene = (scenedef *) voidscene;
  *hres = scene->hres;
  *vres = scene->vres;
}

void rt_aspectratio(SceneHandle voidscene, float aspectratio) {
  scenedef * scene = (scenedef *) voidscene;
  scene->aspectratio=aspectratio;
  scene->scenecheck = 1;
}

void rt_get_aspectratio(SceneHandle voidscene, float *aspectratio) {
  scenedef * scene = (scenedef *) voidscene;
  *aspectratio = scene->aspectratio;
}

void rt_crop_disable(SceneHandle voidscene) {
  scenedef * scene = (scenedef *) voidscene;
  scene->imgcrop.cropmode = RT_CROP_DISABLED;
  scene->imgcrop.xres = 0;
  scene->imgcrop.yres = 0;
  scene->imgcrop.xstart = 0;
  scene->imgcrop.ystart = 0;
}

void rt_crop_output(SceneHandle voidscene, int hres, int vres, int sx, int sy) {
  scenedef * scene = (scenedef *) voidscene;
  scene->imgcrop.cropmode = RT_CROP_ENABLED;
  scene->imgcrop.xres = hres;
  scene->imgcrop.yres = vres;
  scene->imgcrop.xstart = sx;
  scene->imgcrop.ystart = sy;
}

void rt_verbose(SceneHandle voidscene, int v) {
  scenedef * scene = (scenedef *) voidscene;
  scene->verbosemode = v;
}

void rt_rawimage_rgb24(SceneHandle voidscene, unsigned char *img) {
  scenedef * scene = (scenedef *) voidscene;
  scene->img = (void *) img;
  scene->imginternal = 0;  /* image was allocated by the caller */
  scene->imgbufformat = RT_IMAGE_BUFFER_RGB24;
  scene->scenecheck = 1;
}

void rt_rawimage_rgba32(SceneHandle voidscene, unsigned char *img) {
  scenedef * scene = (scenedef *) voidscene;
  scene->img = (void *) img;
  scene->imginternal = 0;  /* image was allocated by the caller */
  scene->imgbufformat = RT_IMAGE_BUFFER_RGBA32;
  scene->scenecheck = 1;
}

void rt_rawimage_rgb96f(SceneHandle voidscene, float *img) {
  scenedef * scene = (scenedef *) voidscene;
  scene->img = (void *) img;
  scene->imginternal = 0;  /* image was allocated by the caller */
  scene->imgbufformat = RT_IMAGE_BUFFER_RGB96F;
  scene->scenecheck = 1;
}


void rt_image_clamp(SceneHandle voidscene) {
  scenedef * scene = (scenedef *) voidscene;
  scene->imgprocess = RT_IMAGE_CLAMP;
}


void rt_image_normalize(SceneHandle voidscene) {
  scenedef * scene = (scenedef *) voidscene;
  scene->imgprocess = RT_IMAGE_NORMALIZE;
}


void rt_image_gamma(SceneHandle voidscene, float gamma) {
  scenedef * scene = (scenedef *) voidscene;
  scene->imggamma = gamma; 
  scene->imgprocess = RT_IMAGE_NORMALIZE | RT_IMAGE_GAMMA;
}


void rt_set_numthreads(SceneHandle voidscene, int numthreads) {
  scenedef * scene = (scenedef *) voidscene;
#ifdef THR
  if (numthreads > 0) {
    scene->numthreads = numthreads;
  }
  else {
    scene->numthreads = rt_thread_numprocessors();
  }

  /* force set of # kernel threads  */
  rt_thread_setconcurrency(scene->numthreads);

#else
  scene->numthreads = 1;
#endif
  scene->scenecheck = 1;
}

void rt_background(SceneHandle voidscene, colora col) {
  scenedef * scene = (scenedef *) voidscene;
  scene->bgtex.background.r = col.r;
  scene->bgtex.background.g = col.g;
  scene->bgtex.background.b = col.b;
  scene->bgtex.background.a = col.a;
}

void rt_background_gradient(SceneHandle voidscene, 
  apivector up,
  flt topval, flt botval, apicolor topcol, apicolor botcol) {
  scenedef * scene = (scenedef *) voidscene;

  scene->bgtex.gradient = up;

  scene->bgtex.gradtopval = topval;
  scene->bgtex.gradbotval = botval;

  scene->bgtex.backgroundtop.r = topcol.r;  
  scene->bgtex.backgroundtop.g = topcol.g;  
  scene->bgtex.backgroundtop.b = topcol.b;

  scene->bgtex.backgroundbot.r = botcol.r;  
  scene->bgtex.backgroundbot.g = botcol.g;  
  scene->bgtex.backgroundbot.b = botcol.b;
}

void rt_background_sky_sphere(SceneHandle voidscene, apivector up, flt topval, 
                              flt botval, apicolor topcol, apicolor botcol) {
  rt_background_gradient(voidscene, up, topval, botval, topcol, botcol);
}

void rt_background_mode(SceneHandle voidscene, int mode) {
  scenedef * scene = (scenedef *) voidscene;
  switch (mode) {
    case RT_BACKGROUND_TEXTURE_SKY_SPHERE:
      scene->bgtexfunc=sky_sphere_background_texture;
      break;

    case RT_BACKGROUND_TEXTURE_SKY_ORTHO_PLANE:
      scene->bgtexfunc=sky_plane_background_texture;
      break;

    case RT_BACKGROUND_TEXTURE_SOLID:
    default:
      scene->bgtexfunc=solid_background_texture;
      break;
  }
}


void rt_ambient_occlusion(SceneHandle voidscene, int numsamples, apicolor col) {
  scenedef * scene = (scenedef *) voidscene;
  scene->ambocc.numsamples = numsamples; 
  scene->ambocc.col.r = col.r;
  scene->ambocc.col.g = col.g;
  scene->ambocc.col.b = col.b;
}

void rt_fog_parms(SceneHandle voidscene, apicolor col, flt start, flt end, flt density) {
  scenedef * scene = (scenedef *) voidscene;
  scene->fog.col = col;
  scene->fog.start = start;
  scene->fog.end = end;
  scene->fog.density = density;
}

void rt_fog_rendering_mode(SceneHandle voidscene, int mode) {
  scenedef * scene = (scenedef *) voidscene;
  switch (mode) {
    /* RT_FOG_VMD is currently a synonym for RT_FOG_OPENGL */
    case RT_FOG_OPENGL:
      scene->fog.type = RT_FOG_OPENGL;
      break;

    case RT_FOG_NORMAL:
    default:
      scene->fog.type = RT_FOG_NORMAL;
      break;
  }
}

void rt_fog_mode(SceneHandle voidscene, int mode) {
  scenedef * scene = (scenedef *) voidscene;

  switch (mode) {
    case RT_FOG_LINEAR:
      scene->fog.fog_fctn = fog_color_linear;
      break;

    case RT_FOG_EXP:
      scene->fog.fog_fctn = fog_color_exp;
      break;

    case RT_FOG_EXP2:
      scene->fog.fog_fctn = fog_color_exp2;
      break;

    case RT_FOG_NONE: 
    default:
      scene->fog.fog_fctn = NULL;
      break;
  }
}

void rt_trans_mode(SceneHandle voidscene, int mode) {
  scenedef * scene = (scenedef *) voidscene;
  scene->transmode = mode; 
}

void rt_boundmode(SceneHandle voidscene, int mode) {
  scenedef * scene = (scenedef *) voidscene;
  scene->boundmode = mode;
  scene->scenecheck = 1;
}

void rt_boundthresh(SceneHandle voidscene, int threshold) {
  scenedef * scene = (scenedef *) voidscene;
 
  if (threshold > 1) {
    scene->boundthresh = threshold;
  }
  else {
    if (rt_mynode() == 0) {
      rt_ui_message(MSG_0, "Out-of-range automatic bounding threshold.\n");
      rt_ui_message(MSG_0, "Automatic bounding threshold reset to default.\n");
    }
    scene->boundthresh = BOUNDTHRESH;
  }
  scene->scenecheck = 1;
}

void rt_shadermode(SceneHandle voidscene, int mode) {
  scenedef * scene = (scenedef *) voidscene;

  /* Main shader used for whole scene */
  switch (mode) {
    case RT_SHADER_LOWEST:
      scene->shader = (colora (*)(void *)) lowest_shader;
      break;
    case RT_SHADER_LOW:
      scene->shader = (colora (*)(void *)) low_shader;
      break;
    case RT_SHADER_MEDIUM:
      scene->shader = (colora (*)(void *)) medium_shader;
      break;
    case RT_SHADER_HIGH:
      scene->shader = (colora (*)(void *)) full_shader;
      break;
    case RT_SHADER_FULL:
      scene->shader = (colora (*)(void *)) full_shader;
      break;
    case RT_SHADER_AUTO:
    default:
      scene->shader = NULL;
      break;
  }
}

void rt_rescale_lights(SceneHandle voidscene, flt lightscale) {
  scenedef * scene = (scenedef *) voidscene;
  scene->light_scale = lightscale;
}

void rt_phong_shader(SceneHandle voidscene, int mode) {
  scenedef * scene = (scenedef *) voidscene;
  switch (mode) {
    case RT_SHADER_NULL_PHONG:
      scene->phongfunc = shade_nullphong;
      break;
    case RT_SHADER_BLINN_FAST:
      scene->phongfunc = shade_blinn_fast;
      break;
    case RT_SHADER_BLINN:
      scene->phongfunc = shade_blinn;
      break;
    default: 
    case RT_SHADER_PHONG:
      scene->phongfunc = shade_phong;
      break;
  }
}

/* allocate and initialize a scene with default parameters */
SceneHandle rt_newscene(void) {
  scenedef * scene;
  SceneHandle voidscene;
  color bgcolor = rt_color(0.0, 0.0, 0.0);
  apicolor ambcolor = rt_color(1.0, 1.0, 1.0);

  scene = (scenedef *) malloc(sizeof(scenedef));
  memset(scene, 0, sizeof(scenedef));             /* clear all valuas to 0  */

  voidscene = (SceneHandle) scene;

  rt_outputfile(voidscene, "/tmp/outfile.tga");   /* default output file    */
  rt_crop_disable(voidscene);                     /* disable cropping */
  rt_outputformat(voidscene, RT_FORMAT_TARGA);    /* default iamge format   */
  rt_resolution(voidscene, 512, 512);             /* 512x512 resolution     */
  rt_verbose(voidscene, 0);                       /* verbose messages off   */

  rt_image_gamma(voidscene, 2.2f);                /* set default gamma */
  rt_image_clamp(voidscene);                      /* clamp image colors */

#if 1
  rt_rawimage_rgb96f(voidscene, NULL);            /* raw image output off   */
#else
  rt_rawimage_rgb24(voidscene, NULL);             /* raw image output off   */
#endif

  rt_boundmode(voidscene, RT_BOUNDING_ENABLED);   /* spatial subdivision on */
  rt_boundthresh(voidscene, BOUNDTHRESH);         /* default threshold      */
  rt_camera_setup(voidscene, 1.0, 1.0, 0, 6,
                  rt_vector(0.0, 0.0, 0.0),
                  rt_vector(0.0, 0.0, 1.0),
                  rt_vector(0.0, 1.0, 0.0));
  rt_camera_dof(voidscene, 1.0, 0.0);
  rt_shadermode(voidscene, RT_SHADER_AUTO);
  rt_rescale_lights(voidscene, 1.0);
  rt_phong_shader(voidscene, RT_SHADER_BLINN);

  rt_background(voidscene, tocolora(bgcolor));
  rt_background_sky_sphere(voidscene, rt_vector(0.0, 1.0, 0.0), 0.3, 0, 
                           rt_color(0.0, 0.0, 0.0), rt_color(0.0, 0.0, 0.5));
  rt_background_mode(voidscene, RT_BACKGROUND_TEXTURE_SOLID);

  rt_ambient_occlusion(voidscene, 0, ambcolor);    /* disable AO by default  */
  rt_fog_rendering_mode(voidscene, RT_FOG_NORMAL); /* radial fog by default  */
  rt_fog_mode(voidscene, RT_FOG_NONE);             /* disable fog by default */
  rt_fog_parms(voidscene, bgcolor, 0.0, 1.0, 1.0);

  /* use max positive integer for max transparent surface limit by default */
  rt_trans_max_surfaces(voidscene,((((int)1) << ((sizeof(int) * 8) - 2))-1)*2);

  rt_trans_mode(voidscene, RT_TRANS_ORIG);         /* set transparency mode  */
  rt_normal_fixup_mode(voidscene, 0);              /* disable normal fixup   */
  rt_shadow_filtering(voidscene, 1);               /* shadow filtering on    */

  scene->objgroup.boundedobj = NULL;
  scene->objgroup.unboundedobj = NULL;
  scene->objgroup.numobjects = 0;

  scene->texlist = NULL;
  scene->lightlist = NULL;
  scene->cliplist = NULL;
  scene->numlights = 0;
  scene->scenecheck = 1;
  scene->parbuf = NULL;
  scene->threads = NULL;
  scene->threadparms = NULL;
  scene->flags = RT_SHADE_NOFLAGS;
 
  rt_set_numthreads(voidscene, -1);         /* auto determine num threads */ 

  /* number of distributed memory nodes, fills in array of node/cpu info */
  scene->nodes = rt_getcpuinfo(&scene->cpuinfo);
  scene->mynode = rt_mynode();

  return scene;
}



void rt_deletescene(SceneHandle voidscene) {
  scenedef * scene = (scenedef *) voidscene;
  list * cur, * next;

  if (scene != NULL) {
    if (scene->imginternal) {
      free(scene->img);
    }

    /* tear down and deallocate persistent rendering threads */
    destroy_render_threads(scene);

    /* tear down and deallocate persistent scanline receives */
    if (scene->parbuf != NULL)
      rt_delete_scanlinereceives(scene->parbuf);

    /* free all lights */
    cur = scene->lightlist;
    while (cur != NULL) {
      next = cur->next;

      /* free lights that have special data, or aren't freed */
      /* as part of the object list deallocation loop.       */
      free_light_special(cur->item);
      free(cur); 
      cur = next;
    }    

    /* free all textures */
    cur = scene->texlist;
    while (cur != NULL) {
      next = cur->next;
      ((texture *) cur->item)->methods->freetex(cur->item); /* free texture */
      free(cur); /* free list entry */
      cur = next;
    }

    /* free all clipping planes */
    cur = scene->cliplist;
    while (cur != NULL) {
      next = cur->next;
      free(((clip_group *) cur->item)->planes); /* free array of clip planes */
      free(cur->item);                          /* free clip group struct    */
      free(cur);                                /* free list entry           */
      cur = next;
    }    

    /* free all other textures, MIP Maps, and images */
    FreeTextures();
    
    free(scene->cpuinfo);
    free_objects(scene->objgroup.boundedobj);
    free_objects(scene->objgroup.unboundedobj);

    free(scene);
  }
}

void apitextotex(apitexture * apitex, texture * tx) {
  standard_texture * tex = (standard_texture *) tx;
  tex->img = NULL;
 
  switch(apitex->texturefunc) {
    case RT_TEXTURE_3D_CHECKER: 
      tex->texfunc=(color(*)(const void *, const void *, void *))(checker_texture);
      break;

    case RT_TEXTURE_GRIT: 
      tex->texfunc=(color(*)(const void *, const void *, void *))(grit_texture);
      break;

    case RT_TEXTURE_MARBLE: 
      tex->texfunc=(color(*)(const void *, const void *, void *))(marble_texture);
      break;

    case RT_TEXTURE_WOOD: 
      tex->texfunc=(color(*)(const void *, const void *, void *))(wood_texture);
      break;

    case RT_TEXTURE_GRADIENT: 
      tex->texfunc=(color(*)(const void *, const void *, void *))(gnoise_texture);
      break;
	
    case RT_TEXTURE_CYLINDRICAL_CHECKER: 
      tex->texfunc=(color(*)(const void *, const void *, void *))(cyl_checker_texture);
      break;

    case RT_TEXTURE_CYLINDRICAL_IMAGE: 
      tex->texfunc=(color(*)(const void *, const void *, void *))(image_cyl_texture);
      tex->img=LoadMIPMap(apitex->imap, 0);
      break;

    case RT_TEXTURE_SPHERICAL_IMAGE: 
      tex->texfunc=(color(*)(const void *, const void *, void *))(image_sphere_texture);
      tex->img=LoadMIPMap(apitex->imap, 0);
      break;

    case RT_TEXTURE_PLANAR_IMAGE: 
      tex->texfunc=(color(*)(const void *, const void *, void *))(image_plane_texture);
      tex->img=LoadMIPMap(apitex->imap, 0);
      break;

    case RT_TEXTURE_VOLUME_IMAGE: 
      tex->texfunc=(color(*)(const void *, const void *, void *))(image_volume_texture);
      tex->img=LoadMIPMap(apitex->imap, 0);
      break;

    case RT_TEXTURE_CONSTANT: 
    default: 
      tex->texfunc=(color(*)(const void *, const void *, void *))(constant_texture);
      break;
  }

       tex->ctr = apitex->ctr;
       tex->rot = apitex->rot;
     tex->scale = apitex->scale;
      tex->uaxs = apitex->uaxs;
      tex->vaxs = apitex->vaxs;
      tex->waxs = apitex->waxs;
   tex->ambient = apitex->ambient;
   tex->diffuse = apitex->diffuse;
  tex->specular = apitex->specular;
   tex->opacity = apitex->opacity;
       tex->col = apitex->col; 

  /* initialize texture flags */
  tex->flags = RT_TEXTURE_NOFLAGS;

  /* anything less than an entirely opaque object will modulate */
  /* the light intensity rather than completly occluding it     */
  if (apitex->opacity >= 0.99999)
    tex->flags = RT_TEXTURE_SHADOWCAST;

  tex->phong = 0.0;
  tex->phongexp = 0.0;
  tex->phongtype = 0;

  tex->transmode = RT_TRANS_ORIG;

  tex->outline = 0.0;
  tex->outlinewidth = 0.0;
}

void * rt_texture(SceneHandle sc, apitexture * apitex) {
  scenedef * scene = (scenedef *) sc;
  texture * tex;
  list * lst;

  tex = new_standard_texture();
  apitextotex(apitex, tex); 

  /* add texture to the scene texture list */
  lst = (list *) malloc(sizeof(list));
  lst->item = (void *) tex;
  lst->next = scene->texlist;
  scene->texlist = lst;

  return(tex);
}


void rt_define_teximage_rgb24(const char *name, int xs, int ys, int zs, unsigned char *rgb) {
  AllocateImageRGB24(name, xs, ys, zs, rgb); 
}

/* deprecated version used in old revs of VMD */
void rt_define_image(const char *name, int xs, int ys, int zs, unsigned char *rgb) {
  AllocateImageRGB24(name, xs, ys, zs, rgb); 
}


/* this is a gross hack that needs to be eliminated    */
/* by writing a new mesh triangle object that doesn't  */
/* need multiple instantiations of texture objects for */
/* correct operation.  Ideally we'd store the object   */
/* pointer in the intersection record so the texture   */
/* needn't store this itself                           */
void * rt_texture_copy_standard(SceneHandle sc, void *oldtex) {
  texture *newtex;
  newtex = new_standard_texture();
  memcpy(newtex, oldtex, sizeof(standard_texture)); 
  return newtex;
}
void * rt_texture_copy_vcstri(SceneHandle sc, void *oldvoidtex) {
  texture *oldtex = (texture *) oldvoidtex;
  texture *newtex = new_vcstri_texture();

  /* copy in all of the texture components common to both tex types */
  newtex->flags = oldtex->flags;
  newtex->ambient = oldtex->ambient;
  newtex->diffuse = oldtex->diffuse;
  newtex->phong = oldtex->phong;
  newtex->phongexp = oldtex->phongexp;
  newtex->phongtype = oldtex->phongtype;
  newtex->specular = oldtex->specular;
  newtex->opacity = oldtex->opacity;
  newtex->transmode =  oldtex->transmode;
  newtex->outline =  oldtex->outline;
  newtex->outlinewidth =  oldtex->outlinewidth;
   
  return newtex;
}


void rt_tex_phong(void * voidtex, flt phong, flt phongexp, int type) {
  texture * tex = (texture *) voidtex;
  tex->phong = phong;
  tex->phongexp = phongexp;
  tex->phongtype = type;
}


void rt_tex_transmode(void * voidtex, int transmode) {
  texture * tex = (texture *) voidtex;
  tex->transmode = transmode;
}


void rt_tex_outline(void * voidtex, flt outline, flt outlinewidth) {
  texture * tex = (texture *) voidtex;
  tex->outline = outline;
  tex->outlinewidth = outlinewidth;
}


static void add_bounded_object(scenedef * scene, object * obj) {
  object * objtemp;

  if (obj == NULL)
    return;

  obj->id = new_objectid(scene);
  objtemp = scene->objgroup.boundedobj;
  scene->objgroup.boundedobj = obj;
  obj->nextobj = objtemp;
  obj->clip = scene->curclipgroup;

  /* XXX Clipping ought to be applied to objects before they */
  /*     are even added to the internal data structures, so  */
  /*     they aren't even considered during rendering.       */
  
  scene->scenecheck = 1;
}

static void add_unbounded_object(scenedef * scene, object * obj) {
  object * objtemp;

  if (obj == NULL)
    return;

  obj->id = new_objectid(scene);
  objtemp = scene->objgroup.unboundedobj;
  scene->objgroup.unboundedobj = obj;
  obj->nextobj = objtemp;
  obj->clip = scene->curclipgroup;
  scene->scenecheck = 1;
}


void * rt_light(SceneHandle voidscene, void * tex, apivector ctr, flt rad) {
  point_light * li;
  scenedef * scene = (scenedef *) voidscene;
  list * lst;

  li=newpointlight(tex, ctr, rad);

  /* add light to the scene lightlist */
  lst = (list *) malloc(sizeof(list));
  lst->item = (void *) li;
  lst->next = scene->lightlist;
  scene->lightlist = lst;
  scene->numlights++;

  /* add light as an object as well... */
  add_bounded_object((scenedef *) scene, (object *)li);

  return li;
}

void * rt_light3fv(SceneHandle voidscene, void * tex,
                   const float *ctr, float rad) {
  vector vctr;
  vctr.x = ctr[0]; vctr.y = ctr[1]; vctr.z = ctr[2];
  return rt_light(voidscene, tex, vctr, rad);
}


void * rt_directional_light(SceneHandle voidscene, void * tex, apivector dir) {
  directional_light * li;
  scenedef * scene = (scenedef *) voidscene;
  list * lst;

  VNorm(&dir);
  li=newdirectionallight(tex, dir);

  /* add light to the scene lightlist */
  lst = (list *) malloc(sizeof(list));
  lst->item = (void *) li;
  lst->next = scene->lightlist;
  scene->lightlist = lst;
  scene->numlights++;

  /* don't add to the object list since it's at infinity */
  /* XXX must loop over light list and deallocate these  */
  /*     specially since they aren't in the obj list.    */
 
  return li;
}

void * rt_directional_light3fv(SceneHandle voidscene, void * tex,
                               const float *dir) {
  vector vdir;
  vdir.x = dir[0]; vdir.y = dir[1]; vdir.z = dir[2];
  return rt_directional_light(voidscene, tex, vdir);
}


void * rt_spotlight(SceneHandle voidscene, void * tex, apivector ctr, flt rad,
                    apivector dir, flt start, flt end) {
  flt fallstart, fallend;
  point_light * li;
  scenedef * scene = (scenedef *) voidscene;
  list * lst;

  fallstart = start * 3.1415926 / 180.0;
  fallend   = end   * 3.1415926 / 180.0;
  VNorm(&dir);
  li = newspotlight(tex, ctr, rad, dir, fallstart, fallend);

  /* add light to the scene lightlist */
  lst = (list *) malloc(sizeof(list));
  lst->item = (void *) li;
  lst->next = scene->lightlist;
  scene->lightlist = lst;
  scene->numlights++;
 
  /* add light as an object as well... */
  add_bounded_object(scene, (object *) li);

  return li;
}

void * rt_spotlight3fv(SceneHandle voidscene, void * tex,
                       const float *ctr, float rad,
                       const float *dir, float start, float end) {
  vector vctr, vdir;
  vctr.x = ctr[0]; vctr.y = ctr[1]; vctr.z = ctr[2];
  vdir.x = dir[0]; vdir.y = dir[1]; vdir.z = dir[2];
  return rt_spotlight(voidscene, tex, vctr, rad, vdir, start, end);
}


void rt_light_attenuation(void * vli, flt Kc, flt Kl, flt Kq) {
  light_set_attenuation((point_light *) vli, Kc, Kl, Kq);
}

void rt_scalarvol(SceneHandle scene, void * tex, apivector min, apivector max,
	int xs, int ys, int zs, const char * fname, void * voidvol) {
  scalarvol * invol = (scalarvol *) voidvol; 
  add_bounded_object((scenedef *) scene, (object *) newscalarvol(tex, min, max, xs, ys, zs, fname, invol));
}

void rt_extvol(SceneHandle scene, void * tex, apivector min, apivector max, int samples, flt (* evaluator)(flt, flt, flt)) {
  add_bounded_object((scenedef *) scene, (object *) newextvol(tex, min, max, samples, evaluator));
}

void rt_box(SceneHandle scene, void * tex, apivector min, apivector max) {
  add_bounded_object((scenedef *) scene, (object *) newbox(tex, min, max));
} 

void rt_cylinder(SceneHandle scene, void * tex, apivector ctr, apivector axis, flt rad) {
  add_unbounded_object((scenedef *) scene, newcylinder(tex, ctr, axis, rad));
}

void rt_cylinder3fv(SceneHandle scene, void * tex,
                    const float *ctr, const float *axis, float rad) {
  vector vctr, vaxis;
  vctr.x = ctr[0];   vctr.y = ctr[1];   vctr.z = ctr[2];
  vaxis.x = axis[0]; vaxis.y = axis[1]; vaxis.z = axis[2];
  add_bounded_object((scenedef *) scene, newcylinder(tex, vctr, vaxis, rad));
}


void rt_fcylinder(SceneHandle scene, void * tex, apivector ctr, apivector axis, flt rad) {
  add_bounded_object((scenedef *) scene, newfcylinder(tex, ctr, axis, rad));
}

void rt_fcylinder3fv(SceneHandle scene, void * tex, 
                     const float *ctr, const float *axis, float rad) {
  vector vctr, vaxis;
  vctr.x = ctr[0];   vctr.y = ctr[1];   vctr.z = ctr[2];
  vaxis.x = axis[0]; vaxis.y = axis[1]; vaxis.z = axis[2];
  add_bounded_object((scenedef *) scene, newfcylinder(tex, vctr, vaxis, rad));
}

void rt_cone(SceneHandle scene, void * tex, apivector ctr, apivector axis, flt rad) {
  add_bounded_object((scenedef *) scene, newcone(tex, ctr, axis, rad));
}

void rt_cone3fv(SceneHandle scene, void * tex,
                     const float *ctr, const float *axis, float rad) {
  vector vctr, vaxis;
  vctr.x = ctr[0];   vctr.y = ctr[1];   vctr.z = ctr[2];
  vaxis.x = axis[0]; vaxis.y = axis[1]; vaxis.z = axis[2];
  add_bounded_object((scenedef *) scene, newcone(tex, vctr, vaxis, rad));
}

void rt_plane(SceneHandle scene, void * tex, apivector ctr, apivector norm) {
  add_unbounded_object((scenedef *) scene, newplane(tex, ctr, norm));
} 

void rt_plane3fv(SceneHandle scene, void * tex,
                 const float *ctr, const float *norm) {
  vector vctr, vnorm;
  vctr.x = ctr[0];   vctr.y = ctr[1];   vctr.z = ctr[2];
  vnorm.x = norm[0]; vnorm.y = norm[1]; vnorm.z = norm[2];
  add_unbounded_object((scenedef *) scene, newplane(tex, vctr, vnorm));
} 


void rt_ring(SceneHandle scene, void * tex, apivector ctr, apivector norm, flt inner, flt outer) {
  add_bounded_object((scenedef *) scene, newring(tex, ctr, norm, inner, outer));
} 

void rt_ring3fv(SceneHandle scene, void * tex,
                const float *ctr, const float *norm, float inner, float outer) {
  vector vctr, vnorm;
  vctr.x = ctr[0];   vctr.y = ctr[1];   vctr.z = ctr[2];
  vnorm.x = norm[0]; vnorm.y = norm[1]; vnorm.z = norm[2];
  add_bounded_object((scenedef *) scene, newring(tex, vctr, vnorm, inner, outer));
} 


void rt_sphere(SceneHandle scene, void * tex, apivector ctr, flt rad) {
  add_bounded_object((scenedef *) scene, newsphere(tex, ctr, rad));
}

void rt_sphere3fv(SceneHandle scene, void * tex, 
                  const float *ctr, float rad) {
  vector vctr;
  vctr.x = ctr[0]; vctr.y = ctr[1]; vctr.z = ctr[2];
  add_bounded_object((scenedef *) scene, newsphere(tex, vctr, rad));
}


void rt_tri(SceneHandle voidscene, void * tex, apivector v0, apivector v1, apivector v2) {
  scenedef * scene = (scenedef *) voidscene;
  object * o = newtri(tex, v0, v1, v2);
  /* don't add degenerate triangles */
  if (o != NULL) {
    add_bounded_object(scene, o);
  }
} 


void rt_tri3fv(SceneHandle voidscene, void * tex,
               const float *v0, const float *v1, const float *v2) {
  scenedef * scene = (scenedef *) voidscene;
  vector vv0, vv1, vv2;
  object * o;

  vv0.x = v0[0]; vv0.y = v0[1]; vv0.z = v0[2];
  vv1.x = v1[0]; vv1.y = v1[1]; vv1.z = v1[2]; 
  vv2.x = v2[0]; vv2.y = v2[1]; vv2.z = v2[2];

  o = newtri(tex, vv0, vv1, vv2);
  /* don't add degenerate triangles */
  if (o != NULL) {
    add_bounded_object(scene, o);
  }
}


void rt_stri(SceneHandle voidscene, void * tex, apivector v0, apivector v1, apivector v2, apivector n0, apivector n1, apivector n2) {
  scenedef * scene = (scenedef *) voidscene;
  object * o = newstri(tex, v0, v1, v2, n0, n1, n2);
  /* don't add degenerate triangles */
  if (o != NULL) {
    if (scene->normalfixupmode)
      stri_normal_fixup(o, scene->normalfixupmode);
    add_bounded_object(scene, o);
  }
} 


void rt_stri3fv(SceneHandle voidscene, void * tex, 
                const float *v0, const float *v1, const float *v2, 
                const float *n0, const float *n1, const float *n2) {
  scenedef * scene = (scenedef *) voidscene;
  vector vv0, vv1, vv2, vn0, vn1, vn2;
  object * o;

  vv0.x = v0[0]; vv0.y = v0[1]; vv0.z = v0[2];
  vv1.x = v1[0]; vv1.y = v1[1]; vv1.z = v1[2]; 
  vv2.x = v2[0]; vv2.y = v2[1]; vv2.z = v2[2];
  vn0.x = n0[0]; vn0.y = n0[1]; vn0.z = n0[2];
  vn1.x = n1[0]; vn1.y = n1[1]; vn1.z = n1[2]; 
  vn2.x = n2[0]; vn2.y = n2[1]; vn2.z = n2[2];

  o = newstri(tex, vv0, vv1, vv2, vn0, vn1, vn2);
  /* don't add degenerate triangles */
  if (o != NULL) {
    if (scene->normalfixupmode)
      stri_normal_fixup(o, scene->normalfixupmode);
    add_bounded_object(scene, o);
  }
} 


void rt_vcstri(SceneHandle voidscene, void * tex, 
               apivector v0, apivector v1, apivector v2, 
               apivector n0, apivector n1, apivector n2, 
               apicolor c0, apicolor c1, apicolor c2) {
  scenedef * scene = (scenedef *) voidscene;
  object * o = newvcstri(tex, v0, v1, v2, n0, n1, n2, c0, c1, c2);
  /* don't add degenerate triangles */
  if (o != NULL) {
    if (scene->normalfixupmode)
      vcstri_normal_fixup(o, scene->normalfixupmode);
    add_bounded_object(scene, o);
  }
} 


void rt_vcstri3fv(SceneHandle voidscene, void * tex, 
                  const float *v0, const float *v1, const float *v2, 
                  const float *n0, const float *n1, const float *n2, 
                  const float *c0, const float *c1, const float *c2) {
  scenedef * scene = (scenedef *) voidscene;
  vector vv0, vv1, vv2, vn0, vn1, vn2;
  color cc0, cc1, cc2;
  object * o;

  vv0.x = v0[0]; vv0.y = v0[1]; vv0.z = v0[2];
  vv1.x = v1[0]; vv1.y = v1[1]; vv1.z = v1[2]; 
  vv2.x = v2[0]; vv2.y = v2[1]; vv2.z = v2[2];
  vn0.x = n0[0]; vn0.y = n0[1]; vn0.z = n0[2];
  vn1.x = n1[0]; vn1.y = n1[1]; vn1.z = n1[2]; 
  vn2.x = n2[0]; vn2.y = n2[1]; vn2.z = n2[2];
  cc0.r = c0[0]; cc0.g = c0[1]; cc0.b = c0[2];
  cc1.r = c1[0]; cc1.g = c1[1]; cc1.b = c1[2]; 
  cc2.r = c2[0]; cc2.g = c2[1]; cc2.b = c2[2];

  o = newvcstri(tex, vv0, vv1, vv2, vn0, vn1, vn2, cc0, cc1, cc2);
  /* don't add degenerate triangles */
  if (o != NULL) {
    if (scene->normalfixupmode)
      vcstri_normal_fixup(o, scene->normalfixupmode);
    add_bounded_object(scene, o);
  }
} 


void rt_tristripscnv3fv(SceneHandle voidscene, void * tex,
                        int numverts, const float * cnv, int numstrips,
                        const int *vertsperstrip, const int *facets) {
  int strip, t, v;
  int stripaddr[2][3] = { {0, 1, 2}, {1, 0, 2} };
  scenedef * scene = (scenedef *) voidscene;

  /* render triangle strips one triangle at a time 
   * triangle winding order is:
   *   v0, v1, v2, then v2, v1, v3, then v2, v3, v4, etc.
   */
  /* loop over all of the triangle strips */
  for (strip=0, v=0; strip < numstrips; strip++) {
    /* loop over all triangles in this triangle strip */
    for (t=0; t < (vertsperstrip[strip] - 2); t++) {
      apivector v0, v1, v2;
      apivector n0, n1, n2;
      apicolor  c0, c1, c2;
      int a0, a1, a2;
      list * lst;
      object * o;

      /* copy the original input texture to each of the triangles... */
      /* converting to a vcstri texture if it isn't already          */
      vcstri_texture * newtex=rt_texture_copy_vcstri(scene, tex);
  
      /* add texture to the scene texture list */
      lst = (list *) malloc(sizeof(list));
      lst->item = (void *) newtex;
      lst->next = scene->texlist;
      scene->texlist = lst;

      /* render one triangle, using lookup table to fix winding order */
      a0 = facets[v + (stripaddr[t & 0x01][0])] * 10;
      a1 = facets[v + (stripaddr[t & 0x01][1])] * 10;
      a2 = facets[v + (stripaddr[t & 0x01][2])] * 10;

      c0.r = cnv[a0 + 0];
      c0.g = cnv[a0 + 1];
      c0.b = cnv[a0 + 2];
      n0.x = cnv[a0 + 4];
      n0.y = cnv[a0 + 5];
      n0.z = cnv[a0 + 6];
      v0.x = cnv[a0 + 7];
      v0.y = cnv[a0 + 8];
      v0.z = cnv[a0 + 9];

      c1.r = cnv[a1 + 0];
      c1.g = cnv[a1 + 1];
      c1.b = cnv[a1 + 2];
      n1.x = cnv[a1 + 4];
      n1.y = cnv[a1 + 5];
      n1.z = cnv[a1 + 6];
      v1.x = cnv[a1 + 7];
      v1.y = cnv[a1 + 8];
      v1.z = cnv[a1 + 9];

      c2.r = cnv[a2 + 0];
      c2.g = cnv[a2 + 1];
      c2.b = cnv[a2 + 2];
      n2.x = cnv[a2 + 4];
      n2.y = cnv[a2 + 5];
      n2.z = cnv[a2 + 6];
      v2.x = cnv[a2 + 7];
      v2.y = cnv[a2 + 8];
      v2.z = cnv[a2 + 9];

      o = newvcstri(newtex, v0, v1, v2, n0, n1, n2, c0, c1, c2); 
      if (scene->normalfixupmode)
        vcstri_normal_fixup(o, scene->normalfixupmode);
      add_bounded_object((scenedef *) scene, o);
      v++; /* move on to next vertex */
    }
    v+=2; /* last two vertices are already used by last triangle */
  }
}


void rt_quadsphere(SceneHandle scene, void * tex, apivector ctr, flt rad) {
  quadric * q;
  flt factor;
  q=(quadric *) newquadric();
  factor= 1.0 / (rad*rad);
  q->tex=tex;
  q->ctr=ctr;
 
  q->mat.a=factor;
  q->mat.b=0.0;
  q->mat.c=0.0;
  q->mat.d=0.0;
  q->mat.e=factor;
  q->mat.f=0.0;
  q->mat.g=0.0;
  q->mat.h=factor;
  q->mat.i=0.0;
  q->mat.j=-1.0;
 
  add_unbounded_object((scenedef *) scene, (object *)q);
}

void rt_quadric(SceneHandle scene, void * tex, apivector ctr, flt a, flt b, flt c, flt d, flt e, flt f, flt g, flt h, flt i, flt j, flt bbox) {
  quadric * q;
  q=(quadric *) newquadric();
  q->tex=tex;
  q->ctr=ctr;
  q->bbox = bbox;

  q->mat.a=a;
  q->mat.b=b;
  q->mat.c=c;
  q->mat.d=d;
  q->mat.e=e;
  q->mat.f=f;
  q->mat.g=g;
  q->mat.h=h;
  q->mat.i=i;
  q->mat.j=j;

  if(bbox <= 0.0)
	  add_unbounded_object((scenedef *) scene, (object *)q);
  else
	  add_bounded_object((scenedef *) scene, (object *)q);
}


void rt_clip_fv(SceneHandle voidscene, int numplanes, const float *planes) {
  list * lst;
  clip_group * clip; 
  int i;
  scenedef * scene = (scenedef *) voidscene;

  /* copy clipping info into structure */
  clip = (clip_group *) malloc(sizeof(clip_group));
  clip->numplanes = numplanes;
  clip->planes = (flt *) malloc(numplanes * sizeof(flt) * 4);
  for (i=0; i<(numplanes*4); i++) {
    clip->planes[i] = planes[i];
  }  

  /* add clipping info to the scene clip list */
  lst = (list *) malloc(sizeof(list));
  lst->item = (void *) clip;
  lst->next = scene->cliplist;
  scene->cliplist = lst;

  /* all objects added from this point on are added with this clip group */
  scene->curclipgroup = clip;
}


void rt_clip_dv(SceneHandle voidscene, int numplanes, const double *planes) {
  list * lst;
  clip_group * clip; 
  int i;
  scenedef * scene = (scenedef *) voidscene;

  /* copy clipping info into structure */
  clip = (clip_group *) malloc(sizeof(clip_group));
  clip->numplanes = numplanes;
  clip->planes = (flt *) malloc(numplanes * sizeof(flt) * 4);
  for (i=0; i<(numplanes*4); i++) {
    clip->planes[i] = planes[i];
  }  

  /* add clipping info to the scene clip list */
  lst = (list *) malloc(sizeof(list));
  lst->item = (void *) clip;
  lst->next = scene->cliplist;
  scene->cliplist = lst;

  /* all objects added from this point on are added with this clip group */
  scene->curclipgroup = clip;
}


void rt_clip_off(SceneHandle voidscene) {
  scenedef * scene = (scenedef *) voidscene;

  /* all objects added from this point on are added without clipping */
  scene->curclipgroup = NULL;
}


