/*
 * tachyon_dep.h - Deprecated Tachyon APIs that have been replaced by
 *                 newer APIs or improved functionality.
 *                 Existing applications should be updated to avoid using
 *                 these APIs as they will be removed in a future version.
 *
 * $Id: tachyon_dep.h,v 1.2 2011/02/15 20:27:58 johns Exp $
 *
 */


#if !defined(TACHYON_NO_DEPRECATED)

#if !defined(TACHYON_DEP_H)
#define TACHYON_DEP_H 1

#ifdef  __cplusplus
extern "C" {
#endif



/**
 * Define a camera for a perspective projection, given the specified
 * zoom factor, aspect ratio, antialiasing sample count,
 * maximum ray recursion depth, and
 * camera center, view direction, and up direction, in a left-handed
 * coordinate system.
 */
void rt_camera_setup(SceneHandle, flt zoom, flt aspect,
                     int alias, int maxdepth,
                     apivector ctr, apivector viewdir, apivector updir);

/**
 * Defines a named 1-D, 2-D, or 3-D texture image with a
 * 24-bit RGB image buffer, without any file references.
 * This allows an application to send Tachyon images for texture mapping
 * without having to touch the filesystem.
 */
void rt_define_image(const char *name, int xsize, int ysize, int zsize,
                     unsigned char *rgb24data);

/** Set parameters for sky sphere background texturing.  */
void rt_background_sky_sphere(SceneHandle, apivector up,
                              flt topval, flt botval,
                              apicolor topcolor, apicolor botcolor);


#ifdef  __cplusplus
}
#endif
#endif

#endif

