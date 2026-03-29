/*
 * texture.h This file contains all of the includes and defines for the texture 
 * mapping part of the shader.
 *
 *  $Id: texture.h,v 1.15 2011/02/15 20:27:58 johns Exp $
 */

void InitTextures(void);

/* background texturing routines */
colora solid_background_texture(const ray *ry);
colora sky_sphere_background_texture(const ray *ry);
colora sky_plane_background_texture(const ray *ry);

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

