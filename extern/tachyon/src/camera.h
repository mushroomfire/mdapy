/*
 * camera.h - This file contains the defines for camera routines etc.
 *
 * (C) Copyright 1994-2022 John E. Stone
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * $Id: camera.h,v 1.29 2022/02/18 17:55:28 johns Exp $
 *
 */

void camera_init(scenedef *);
void camray_init(scenedef *, ray *, unsigned long, unsigned long *, 
                 unsigned int, unsigned int);

void cameradefault(camdef *);
void cameraprojection(camdef *, int);
void cameradof(camdef *, flt focaldist, flt aperture);
void camerafrustum(camdef *, flt l, flt r, flt b, flt t);
void camerazoom(camdef *, flt zoom);
void cameraposition(camdef * camera, vector center, vector viewvec, 
                    vector upvec);
void getcameraposition(camdef * camera, vector * center, vector * viewvec, 
                       vector * upvec, vector *rightvec);


void cam_prep_perspective_ray(ray *, flt, flt);
color cam_perspective_aa_dof_ray(ray *, flt, flt);
color cam_perspective_aa_ray(ray *, flt, flt);
color cam_perspective_ray(ray *, flt, flt);
color cam_perspective_dof_ray(ray *, flt, flt);

color cam_orthographic_aa_dof_ray(ray *, flt, flt);
color cam_orthographic_aa_ray(ray *, flt, flt);
color cam_orthographic_ray(ray *, flt, flt);
color cam_orthographic_dof_ray(ray *, flt, flt);

color cam_equirectangular_aa_ray(ray *, flt, flt);
color cam_equirectangular_ray(ray *, flt, flt);

color cam_equirectangular_aa_stereo_ray(ray *, flt, flt);
color cam_equirectangular_stereo_ray(ray *, flt, flt);

color cam_fisheye_aa_ray(ray *, flt, flt);
color cam_fisheye_ray(ray *, flt, flt);


