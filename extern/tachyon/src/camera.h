/*
 * camera.h - This file contains the defines for camera routines etc.
 *
 *  $Id: camera.h,v 1.22 2011/02/14 05:38:48 johns Exp $
 */

void camera_init(scenedef *);
void camray_init(scenedef *, ray *, unsigned long, unsigned long *, unsigned int);

void cameradefault(camdef *);
void cameraprojection(camdef *, int);
void cameradof(camdef *, flt focallength, flt aperture);
void camerafrustum(camdef *, flt l, flt r, flt b, flt t);
void camerazoom(camdef *, flt zoom);
void cameraposition(camdef * camera, vector center, vector viewvec, 
                    vector upvec);
void getcameraposition(camdef * camera, vector * center, vector * viewvec, 
                       vector * upvec, vector *rightvec);

colora cam_aa_perspective_ray(ray *, flt, flt);
colora cam_perspective_ray(ray *, flt, flt);
colora cam_aa_dof_ray(ray *, flt, flt);
colora cam_dof_ray(ray *, flt, flt);
colora cam_aa_orthographic_ray(ray *, flt, flt);
colora cam_orthographic_ray(ray *, flt, flt);
colora cam_fisheye_ray(ray *, flt, flt);
colora cam_aa_fisheye_ray(ray *, flt, flt);


