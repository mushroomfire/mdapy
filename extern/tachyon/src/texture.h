/*
 * texture.h - This file contains prototypes for the texture 
 *             mapping part of the shader code.
 *
 * (C) Copyright 1994-2022 John E. Stone
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * $Id: texture.h,v 1.17 2022/03/13 23:30:01 johns Exp $
 *
 */

void InitTextures(void);

/* background texturing routines */
color solid_background_texture(ray *ry);
color sky_sphere_background_texture(ray *ry);
color sky_plane_background_texture(ray *ry);

/* object texturing routines */
color     constant_texture(const vector *, const texture *, const ray *);
color    image_cyl_texture(const vector *, const texture *, const ray *);
color image_sphere_texture(const vector *, const texture *, const ray *);
color  image_plane_texture(const vector *, const texture *, const ray *);
color image_volume_texture(const vector *, const texture *, const ray *);
color      checker_texture(const vector *, const texture *, const ray *);
color  cyl_checker_texture(const vector *, const texture *, const ray *);
color         grit_texture(const vector *, const texture *, const ray *);
color         wood_texture(const vector *, const texture *, const ray *);
color       marble_texture(const vector *, const texture *, const ray *);
color       gnoise_texture(const vector *, const texture *, const ray *);
int Noise(flt, flt, flt);
void InitTextures(void);
void FreeTextures(void);

texture * new_texture(void);
texture * new_standard_texture(void);
texture * new_vcstri_texture(void);
void free_standard_texture(void * voidtex);

