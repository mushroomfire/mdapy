/*
 * tachyon.h - The declarations and prototypes needed so that 3rd party     
 *   driver code can run the raytracer.  Third party driver code should       
 *   only use the functions in this header file to interface with the 
 *   rendering engine.                                            
 *
 * $Id: tachyon.h,v 1.121 2013/04/09 16:44:41 johns Exp $
 *
 */

#if !defined(TACHYON_H)
#define TACHYON_H 1

#ifdef  __cplusplus
extern "C" {
#endif

#include "util.h"    /* rt_timer_xxx() and rt_rand() */
#include "hash.h"    /* rt_hash_xxx() */

/******************************************************************/
/* Constants and types defined for use with the Tachyon API calls */
/******************************************************************/

/*
 * Tachyon version strings for feature detection and compatibility testing.
 */
#define TACHYON_VERSION_STRING      "0.99"    /**< string version info  */
#define TACHYON_MAJOR_VERSION       0         /**< major version number */
#define TACHYON_MINOR_VERSION       99        /**< minor version number */
#define TACHYON_PATCH_VERSION       0         /**< patch version number */

/*
 * Build Tachyon and its interfaces using either double- or single-precision
 * floating point types, based on compile-time defition of the
 * USESINGLEFLT macro.
 */
#ifdef USESINGLEFLT
typedef float flt;   /**< generic floating point number, using float  */
#else
typedef double flt;  /**< generic floating point number, using double */
#endif
typedef flt apiflt;  /**< for backward compatibility */

typedef void * SceneHandle;

typedef struct {
   flt x;
   flt y;
   flt z;
} apivector;

typedef struct {
   float r;
   float g;
   float b;
} apicolor;

typedef struct {
   float r;
   float g;
   float b;
   float a;
} colora;

colora tocolora(apicolor c);

typedef struct {
  int texturefunc; /**< which texture function to use */
  apicolor col;    /**< base object color */
  int shadowcast;  /**< does the object cast a shadow */
  flt ambient;     /**< ambient lighting */
  flt diffuse;     /**< diffuse reflection */
  flt specular;    /**< specular reflection */
  flt opacity;     /**< how opaque the object is */ 
  apivector ctr;   /**< origin of texture */
  apivector rot;   /**< rotation of texture around origin */
  apivector scale; /**< scale of texture in x,y,z */ 
  apivector uaxs;  /**< planar map u axis */
  apivector vaxs;  /**< planar map v axis */
  apivector waxs;  /**< volume map W axis */
  char imap[96];   /**< name of image map */ 
} apitexture;

/********************************************/
/* Functions implemented to provide the API */
/********************************************/

/** Helper function to make vectors.  */
apivector rt_vector(flt x, flt y, flt z);

/** Helper function to make colors.   */
apicolor  rt_color(flt r, flt g, flt b);  /**< helper to make colors */

/** Set function pointer for user interface output callbacks.  */
void rt_set_ui_message(void (* func) (int, char *)); 

/** Set function pointer for user interface progress callbacks.  */
void rt_set_ui_progress(void (* func) (int));

/**
 * Initialize ray tracing library, must be first Tachyon API called.
 * Takes pointer to argument count, and pointer to argument array
 * 1. resets and initializes the raytracing system
 * 2. initializes internal parallel processing facilities, and tests
 *    inter-node connectivity.
 * 3. deallocates previously allocated internal data structures
 * 4. returns the id of this computational node on success, -1 on failure.
 */ 
int rt_initialize(int *, char ***); 

/**
 * Shutdown the ray tracing library for good, at final use before
 * program termination.  The ray tracer may not be used after rt_finalize
 * has been called.
 */ 
void rt_finalize(void); 

/** Allocate, initialize, and return a handle for a new scene.  */
SceneHandle rt_newscene(void); 

/** Destroy and deallocate the specified scene.  */
void rt_deletescene(SceneHandle);

/** Render the current scene.  */
void rt_renderscene(SceneHandle);

/** Set the filename for the output image for the specified scene.  */
void rt_outputfile(SceneHandle, const char * outname); 

/* 
 * 24-bit color image formats
 */
#define RT_FORMAT_TARGA                 0  /**< 24-bit Targa file          */
#define RT_FORMAT_PPM                   1  /**< 24-bit NetPBM PPM file     */
#define RT_FORMAT_SGIRGB                2  /**< 24-bit SGI RGB file        */
#define RT_FORMAT_JPEG                  3  /**< 24-bit JPEG file           */
#define RT_FORMAT_WINBMP                4  /**< 24-bit Windows BMP file    */
#define RT_FORMAT_PNG                   5  /**< 24-bit PNG file            */

/*
 * 48-bit deep-color image formats
 */
#define RT_FORMAT_PPM48                 6  /**< 48-bit NetPBM PPM file     */
#define RT_FORMAT_PSD48                 7  /**< 48-bit Photoshop PSD file  */

/** Set the format of the output image(s).  */
void rt_outputformat(SceneHandle, int format);

/**
 * Set the horizontal and vertical resolution (in pixels)
 * for the specified scene.
 */
void rt_resolution(SceneHandle, int hres, int vres);

/**
 * Get the horizontal and vertical resolution (in pixels)
 * for the specified scene.
 */
void rt_get_resolution(SceneHandle, int *hres, int *vres);

/** Set the view frustum aspect ratio (width/height) */
void rt_aspectratio(SceneHandle voidscene, float aspectratio);

/** Get the view frustum aspect ratio (width/height) */
void rt_get_aspectratio(SceneHandle voidscene, float *aspectratio);

/*
 * Image cropping modes
 */
#define RT_CROP_DISABLED                0  /**< Image cropping disabled     */
#define RT_CROP_ENABLED                 1  /**< Image cropping enabled      */

/** 
 * Crop the output image to the specified size, 
 * intended only for use in SPECMPI benchmarking.
 */
void rt_crop_output(SceneHandle, int hres, int vres, int lx, int ly);

/** Disable output image cropping.  */
void rt_crop_disable(SceneHandle);

/** Sets the maximum number of supersamples to take for any pixel.  */
void rt_aa_maxsamples(SceneHandle, int maxsamples);

/**
 * Enables or Disables verbose messages from the ray tracing library
 * during rendering. (a zero value means off, non-zero means on)
 */
void rt_verbose(SceneHandle, int v);

/*
 * Surface normal and winding order fixup mode constants used
 * to optionally auto-correct triangles with interpolate normals
 */
#define RT_NORMAL_FIXUP_OFF   0  /**< surface normals and winding order agree */
#define RT_NORMAL_FIXUP_FLIP  1  /**< flip normals to agree with winding order*/
#define RT_NORMAL_FIXUP_GUESS 2  /**< random normal/winding, use best guess   */

/**
 * Set the surface normal and polygon winding order fixup mode to use
 * when generating triangles with interpolated surface normals.
 */
void rt_normal_fixup_mode(SceneHandle, int mode);

/**
 * Enable clamping of pixel values to the range [0 1)
 * (rather than renormalizing) prior to output.
 * This mode is useful for improved rendering performance.
 */
void rt_image_clamp(SceneHandle voidscene);

/**
 * Enable renormalization of pixel values to the range [0 1)
 * (rather than clamping) prior to output.
 */
void rt_image_normalize(SceneHandle voidscene);

/** Apply gamma correction to the pixel values after normalization.  */
void rt_image_gamma(SceneHandle voidscene, float gamma);

/**
 * Have the ray tracer save the output image in the specified
 * memory area, in raw 24-bit, packed, pixel interleaved, unsigned
 * RGB bytes.  The caller is responsible for making sure that there
 * is enough space in the memory area for the entire image.
 */
void rt_rawimage_rgb24(SceneHandle, unsigned char *rawimage);
void rt_rawimage_rgba32(SceneHandle, unsigned char *rawimage);

/**
 * Request Tachyon to save the output image in the specified
 * memory area, in raw 96-bit, packed, pixel interleaved, 32-bit float
 * RGB bytes.  The caller is responsible for making sure that there
 * is enough space in the memory area for the entire image.
 */
void rt_rawimage_rgb96f(SceneHandle, float *rawimage);

/** Explicitly set the number of worker threads Tachyon will use.  */
void rt_set_numthreads(SceneHandle, int);

/** Set the background color of the specified scene.  */
void rt_background(SceneHandle, colora);

/** 
 * Set parameters for gradient (sky plane or sphere) 
 * background texturing.  The "up" vector defines the direction
 * of the "top" color.  The top and bottom values give maximum and
 * minimum projection values for the dot product between the 
 * the incident ray directon or original (sphere or plane respectively)
 * and the "up" vector.  The final resulting scaled and clamped
 * value is used as the interpolation factor between the top and bottom
 * gradient colors.
 */
void rt_background_gradient(SceneHandle, apivector up, 
                            flt topval, flt botval,
                            apicolor topcolor, apicolor botcolor);

/**
 * Background texture modes for rt_background_mode, 
 * determines behavior to use when rays don't hit any objects.
 */
#define RT_BACKGROUND_TEXTURE_SOLID             0
#define RT_BACKGROUND_TEXTURE_SKY_SPHERE        1
#define RT_BACKGROUND_TEXTURE_SKY_ORTHO_PLANE   2

/**
 * Set the background texturing mode to use.
 * When the solid texture mode is used, any ray that does not hit an object
 * and does not achieve 100% fog density will be assigned the solid 
 * background color.  
 *
 * When the sky sphere mode is active, the background color is computed
 * by interpolating between a top/bottom color pair. The sky sphere
 * color interpolation is performed by computing a dot product between
 * the incident ray direction and the "up" color gradient direction,
 * and the projected direction component is normalized and clamped against
 * the top and bottom values.  The sky sphere background mode is 
 * apropriate for any of the perspective or fisheye style camera 
 * projections, but not for orthographic projections.
 *
 * The sky plane background mode is intended for use with orthographic
 * projections.  The sky plane mode operates by interpolating similarly
 * to the sky sphere, except that instead of projecting the incident ray
 * direction vector onto the "up" direction vector, the direction component
 * is computed by projecting the incident ray origin onto the "up" vector,
 * since in the orthographic projection, all camera rays have identical
 * direction vectors.
 */
void rt_background_mode(SceneHandle, int mode);

/*
 * Fog modes for rt_fog_rendering_mode()
 */
#define RT_FOG_NORMAL     0  /**< radial fog                        */
#define RT_FOG_OPENGL     1  /**< planar OpenGL-like fog            */
#define RT_FOG_VMD        1  /**< planar OpenGL-like fog            */

/**
 * Set fog rendering mode, either radial fog (native Tachyon behavior),
 * or an OpenGL- or VMD-like planar fog.  The Tachyon-native radial
 * fog implementation uses the distance along the ray to the point of 
 * intersection as the fog coordinate.  This gives more natural results in 
 * mirror reflections.  The Tachyon-native radial fog implementation 
 * also applies fog to the background color, unlike OpenGL.
 * The OpenGL- or VMD-style fog implmentation computes
 * the fog coordinate by determining the its depth in the plane normal to the
 * view direction, at the intersection point.  Another difference in behavior
 * is that OpenGL fog does not affect the background color.  OpenGL-style
 * fog is only applied to rendered geometry, not to the background color.  
 */
void rt_fog_rendering_mode(SceneHandle, int);

/*
 * Fog type parameters.
 */
#define RT_FOG_NONE       0  /**< no fog                            */
#define RT_FOG_LINEAR     1  /**< linear fog                        */
#define RT_FOG_EXP        2  /**< exponential fog                   */
#define RT_FOG_EXP2       3  /**< exponential-squared fog           */

/** Set fog style (linear, exponential, exponential-squared).  */
void rt_fog_mode(SceneHandle, int);

/** Set fog rendering parameters.  */
void rt_fog_parms(SceneHandle, apicolor col, 
                  flt start, flt end, flt density);

/** Set the maximum number of transparent surfaces that will be rendered.  */
void rt_trans_max_surfaces(SceneHandle, int maxsurfaces);

/*
 * Transparency mode flags for rt_trans_mode()
 */
#define RT_TRANS_ORIG     0  /**< original transparency mode             */
#define RT_TRANS_VMD      1  /**< mult shaded color by opacity, for VMD  */
#define RT_TRANS_RASTER3D 2  /**< angle-dependent opacity modulation     */

/** Set transparency rendering mode.  */
void rt_trans_mode(SceneHandle, int mode);

/**
 * Control whether or not transparent surfaces modulate incident light or not
 */
void rt_shadow_filtering(SceneHandle, int mode);


/*
 * Parameter values for rt_boundmode()
 */
#define RT_BOUNDING_DISABLED 0  /**< Disable spatial subdivision/bounding  */
#define RT_BOUNDING_ENABLED  1  /**< Enable spatial subdivision/bounding   */

/**
 * Enables or disable automatic generation and use of ray tracing 
 * acceleration data structures. 
 */
void rt_boundmode(SceneHandle, int mode);

/** 
 * Set the threshold to be used when automatic generation of ray tracing
 * acceleration structures is to be used.  The threshold represents the 
 * minimum number of objects which must be present in an area of space 
 * before an automatic acceleration system will consider optimizing the
 * objects using spatial subdivision or automatic bounds generation methods.
 */
void rt_boundthresh(SceneHandle, int threshold);


/**************************/
/* Camera definition APIs */
/**************************/

/*
 * Parameter values for rt_camera_projection()
 */
#define RT_PROJECTION_PERSPECTIVE      0  /**< Perspective projection mode  */
#define RT_PROJECTION_ORTHOGRAPHIC     1  /**< Orthographic projection mode */
#define RT_PROJECTION_PERSPECTIVE_DOF  2  /**< Perspective projection mode  */
#define RT_PROJECTION_FISHEYE          3  /**< Perspective projection mode  */

/** Set camera projection mode.  */
void rt_camera_projection(SceneHandle, int mode);

/** Set camera position and orientation.  */ 
void rt_camera_position(SceneHandle, apivector center, apivector viewdir, 
                        apivector updir);
/** Set camera position and orientation.  */ 
void rt_camera_position3fv(SceneHandle, const float *center, 
                           const float *viewdir, const float *updir);

/** Get camera position and orientation.  */ 
void rt_get_camera_position(SceneHandle, apivector *center, apivector *viewdir,
                            apivector *updir, apivector *rightdir);
/** Get camera position and orientation.  */ 
void rt_get_camera_position3fv(SceneHandle, float *center, float *viewdir,
                              float *updir, float *rightdir);

/**
 * Camera maximum ray recursion depth (i.e. number of levels of 
 * reflection and transmission rays traced). 
 */
void rt_camera_raydepth(SceneHandle, int maxdepth);

/**
 * Set camera "zoom" factor.
 * At a "zoom" factor of 1.0 for a perspective camera, Tachyon defines 
 * the height of the image plane as 1.0, at a distance of 1.0 from the 
 * camera center, yielding a 90 degree vertical field of view for a 
 * normal perspective camera.
 * Zooming to a factor of 2.0 cuts the vertical height of the image 
 * plane in half, giving a correspondingly reduced vertical field 
 * of view of 53 degrees.  For other types of cameras the zoom factor
 * adjusts the projected image plane size accordingly, though the 
 * specific field of view .
 */
void rt_camera_zoom(SceneHandle, flt zoom);

/** Return current camera "zoom" factor. */
flt rt_get_camera_zoom(SceneHandle);


/**
 * Set vertical field of view (in degrees) for a perspective camera.
 * This API won't have the intended effect on other types of cameras.
 */
void rt_camera_vfov(SceneHandle, flt vfov);

/** 
 * Return vertical field of view (in degrees) for a perspective camera.
 * This API won't have the intended effect on other types of cameras.
 */
flt rt_get_camera_vfov(SceneHandle);


/**
 * Set view frustum for active camera.  
 * This routine is best used by experts.  The center of the image plane is 
 * defined at the camera center, translated one unit length in the view
 * direction.  Given this, by defining the left, right, bottom, and top
 * edges of the image plane, one can easily render a very high resolution
 * image in multiple passes (a tile at a time), or one can use the precise
 * view frustum definition to control the field of view more conveniently
 * when matching vs. OpenGL, etc.
 */
void rt_camera_frustum(SceneHandle, flt left, flt right, flt bottom, flt top);

/** Set depth-of-field rendering options.  */
void rt_camera_dof(SceneHandle voidscene, flt focallength, flt aperture);


/***********************/
/*Texture mapping APIs */
/***********************/

/*
 * Object texture mapping functions.
 */
#define RT_TEXTURE_CONSTANT             0  /**< solid color                 */
#define RT_TEXTURE_3D_CHECKER           1  /**< checkerboard texture        */
#define RT_TEXTURE_GRIT                 2  /**< "grit" procedural texture   */
#define RT_TEXTURE_MARBLE               3  /**< "marble" procedural texture */
#define RT_TEXTURE_WOOD                 4  /**< "wood" procedural texture   */
#define RT_TEXTURE_GRADIENT             5  /**< gradient noise procedural texture*/
#define RT_TEXTURE_CYLINDRICAL_CHECKER  6  /**< cylindrical checkerboard    */
#define RT_TEXTURE_CYLINDRICAL_IMAGE    7  /**< cylindrical image map       */
#define RT_TEXTURE_SPHERICAL_IMAGE      8  /**< spherical image map         */
#define RT_TEXTURE_PLANAR_IMAGE         9  /**< planar image map            */
#define RT_TEXTURE_VOLUME_IMAGE        10  /**< volumetric image map        */

/**
 * translates a texture definition into the internal format used
 * by the ray tracing system, and returns an opaque pointer to the
 * internally used structure, which should be passed to object creation
 * routines.
 *
 * NOTE: This API should be deprecated, but a suitable replacement has not 
 *       been written yet.
 */
void * rt_texture(SceneHandle, apitexture *);

/**
 * Defines a named 1-D, 2-D, or 3-D texture image with a 
 * 24-bit RGB image buffer, without any file references.
 * This allows an application to send Tachyon images for texture mapping
 * without having to touch the filesystem.
 */
void rt_define_teximage_rgb24(const char *name, int xsize, int ysize, int zsize,
                              unsigned char *rgb24data);

/** 
 * Do not use this unless you know what you're doing, this is a 
 * short-term workaround until new object types have been created.
 */
void * rt_texture_copy_standard(SceneHandle, void *oldtex);

/** 
 * Do not use this unless you know what you're doing, this is a 
 * short-term workaround until new object types have been created.
 */
void * rt_texture_copy_vcstri(SceneHandle, void *oldtex);


/*****************************/
/* Shading and lighting APIs */
/*****************************/

/*
 * Shader modes settings for rt_shadermode()
 * These are sorted from lowest quality (and fastest execution)
 * to highest quality (and slowest execution)
 */
#define RT_SHADER_AUTO    0  /**< Automatically determine shader needed */
#define RT_SHADER_LOWEST  1  /**< lowest quality shading available      */
#define RT_SHADER_LOW     2  /**< low quality shading                   */
#define RT_SHADER_MEDIUM  3  /**< Medium quality shading                */
#define RT_SHADER_HIGH    4  /**< High quality shading                  */
#define RT_SHADER_FULL    5  /**< Highest quality shading available     */

/** 
 * Set the shading mode for the specified scene. 
 * Modes are sorted from lowest quality (and fastest execution)
 * to highest quality (and slowest execution)
 */
void rt_shadermode(SceneHandle voidscene, int mode);


/*
 * Shader modes for rt_phong_shader()
 */
#define RT_SHADER_NULL_PHONG 0 /**< Disable Phong contributions               */
#define RT_SHADER_BLINN_FAST 1 /**< Fast version of Blinn's equation          */
#define RT_SHADER_BLINN      2 /**< Blinn's specular highlights, as in OpenGL */
#define RT_SHADER_PHONG      3 /**< Phong specular highlights                 */

/** Set the equation used for rendering specular highlights */
void rt_phong_shader(SceneHandle voidscene, int mode);

/*
 * Phong types
 */
#define RT_PHONG_PLASTIC                0  /**< Dielectric Phong highlight  */
#define RT_PHONG_METAL                  1  /**< Metallic Phong highlight    */

/** Set Phong shading parameters for an existing texture.  */
void rt_tex_phong(void * voidtex, flt phong, flt phongexp, int type); 

/**
 * Set transparent surface shading parameters for an existing texture,
 * enabling or disabling angle-modulated transparency.
 */
void rt_tex_transmode(void * voidtex, int transmode);

/** Set edge cueing outline shading parameters for an existing texture. */
void rt_tex_outline(void * voidtex, flt outline, flt outlinewidth); 


/** Rescale all light sources in the scene by factor lightscale.  */
void rt_rescale_lights(SceneHandle, flt lightscale);


/** Define a point light source with associated texture, center, and radius. */
void * rt_light(SceneHandle, void *tex, apivector center, flt radius);     
/** Define a point light source with associated texture, center, and radius. */
void * rt_light3fv(SceneHandle, void *tex, const float *center, float radius);


/**
 * Define a directional light source with associated texture, 
 * center, and direction.
 */
void * rt_directional_light(SceneHandle, void *tex, apivector direction);     
/**
 * Define a directional light source with associated texture, 
 * center, and direction.
 */
void * rt_directional_light3fv(SceneHandle, void *tex, const float *direction);


/**
 * Define a spotlight with associated texture, center, radius, direction,
 * falloff start, and falloff end parameters.
 */
void * rt_spotlight(SceneHandle, void *tex, apivector center, flt radius,
                    apivector direction, flt fallstart, flt fallend);     
/**
 * Define a spotlight with associated texture, center, radius, direction,
 * falloff start, and falloff end parameters.
 */
void * rt_spotlight3fv(SceneHandle, void *tex, const float *center, 
                       float radius, const float *direction, 
                       float fallstart, float fallend);     


/** Set light attenuation parameters for an existing light.  */
void rt_light_attenuation(void *light, flt constfactor, 
                          flt linearfactor, flt quadfactor);

/**
 * Ambient occlusion lighting, with monte carlo sampling of 
 * omnidirectional "sky" light.
 */
void rt_ambient_occlusion(void *scene, int numsamples, apicolor col);


/************************/
/* Object Creation APIs */
/************************/
/** Enable or update a clipping plane group.  */
void rt_clip_fv(SceneHandle, int numplanes, const float * planes);

/** Enable or update a clipping plane group.  */
void rt_clip_dv(SceneHandle, int numplanes, const double * planes);

/** Disable active clipping plane group.  */
void rt_clip_off(SceneHandle);


/** Define an infinite cylinder.  */
void rt_cylinder(SceneHandle, void *tex, apivector center, 
                 apivector axis, flt radius);
/** Define an infinite cylinder.  */
void rt_cylinder3fv(SceneHandle, void *tex, const float *center, 
                    const float *axis, float radius);


/** Define a finite-length cylinder.  */
void rt_fcylinder(SceneHandle, void *tex, apivector center, 
                  apivector axis, flt radius);
/** Define a finite-length cylinder.  */
void rt_fcylinder3fv(SceneHandle, void *tex, const float *center, 
                     const float *axis, float radius);


/** Define a sequence of connected cylinders.  */
void rt_polycylinder(SceneHandle, void *tex, apivector *points, 
                     int numpoints, flt radius);
/** Define a sequence of connected cylinders.  */
void rt_polycylinder3fv(SceneHandle, void *tex, const float *points, 
                        int numpoints, float radius);

/** Define a cone.  */
void rt_cone(SceneHandle, void *tex, apivector center,
                  apivector axis, flt radius);
/** Define a cone.  */
void rt_cone3fv(SceneHandle, void *tex, const float *center,
                     const float *axis, float radius);

/** Define a sphere with associated texture, center, and radius.  */
void rt_sphere(SceneHandle, void *tex, apivector center, flt radius);
/** Define a sphere with associated texture, center, and radius.  */
void rt_sphere3fv(SceneHandle, void *tex, const float *center, float radius);


/** Define a plane.  */
void rt_plane(SceneHandle, void *tex, apivector center, apivector normal);
/** Define a plane.  */
void rt_plane3fv(SceneHandle, void *tex, const float *center, 
                 const float *normal);


/** Define an annular ring.  */
void rt_ring(SceneHandle, void *tex, apivector center, apivector mormal, 
             flt innerrad, flt outerrad); 
/** Define an annular ring.  */
void rt_ring3fv(SceneHandle, void *tex, const float *center, 
                const float *normal, float innerrad, float outerrad); 


/** Define a flat-shaded triangle.  */
void rt_tri(SceneHandle, void *tex, apivector v0, apivector v1, apivector v2);  
/** Define a flat-shaded triangle.  */
void rt_tri3fv(SceneHandle, void *tex,
               const float *v0, const float *v1, const float *v2);  


/** Define a smooth-shaded triangle using interpolated vertex normals.  */
void rt_stri(SceneHandle, void *, apivector v0, apivector v1, apivector v2, 
             apivector n0, apivector n1, apivector n2); 
/** Define a smooth-shaded triangle using interpolated vertex normals.  */
void rt_stri3fv(SceneHandle, void *, 
                const float *v0, const float *v1, const float *v2, 
                const float *n0, const float *n1, const float *n2); 


/**
 * Define a smooth-shaded triangle using interpolated 
 * vertex normals and per-vertex colors.
 */
void rt_vcstri(SceneHandle, void *tex, apivector v0, apivector v1, apivector v2,
               apivector n0, apivector n1, apivector n2,
               apicolor c0, apicolor c1, apicolor c2); 
/**
 * Define a smooth-shaded triangle using interpolated 
 * vertex normals and per-vertex colors.
 */
void rt_vcstri3fv(SceneHandle, void *tex, 
                  const float *v0, const float *v1, const float *v2,
                  const float *n0, const float *n1, const float *n2,
                  const float *c0, const float *c1, const float *c2); 


/**
 * Define smooth-shaded triangle strips using interpolated vertex normals,
 * and per-vertex colors.  All vertex data is stored in a single packed array
 * of 32-bit floating point values formatted with each vertex consisting 
 * of colors, normals, and vertices, e.g. CrCgCbNxNyNzVxVyVz.  One or 
 * multiple triangle strips are defined with a list of facet arrays, where
 * each facet array contains a list of vertex indices.
 */
void rt_tristripscnv3fv(SceneHandle scene, void * tex,
                        int numverts, const float * cnv,
                        int numstrips, const int *vertsperstrip, 
                        const int *facets);


/**
 * Define an axis-aligned volumetric data set, with a user-defined
 *  sample evaluation callback function.
 */
void rt_extvol(SceneHandle, void *tex, 
               apivector mincoord, apivector maxcoord, 
               int samples, flt (* evaluator)(flt, flt, flt)); 

/**
 * Define an axis-aligned scalar volumetric data set, loaded from a file.
 */
void rt_scalarvol(SceneHandle, void *tex, 
                  apivector mincoord, apivector maxcoord,
                  int xsize, int ysize, int zsize, 
                  const char *filename, void *invol); 


/** Define an axis-aligned height field.  */
void rt_heightfield(SceneHandle, void *tex, apivector center, 
                    int m, int n, flt *field, flt wx, flt wy);

/** Define an auto-generated height field.  */
void rt_landscape(SceneHandle, void *tex, int m, int n, 
                  apivector center,  flt wx, flt wy);


/** Define an axis-aligned box.  */
void rt_box(SceneHandle, void *tex, apivector mincoord, apivector maxcoord);  


/**
 * Define a quadric sphere, normally used only for testing and benchmarking.
 */
void rt_quadsphere(SceneHandle, void *tex, apivector center, flt rad);

/**
 * Define a quadric.
 */
void rt_quadric(SceneHandle, void *tex, apivector center, flt a, flt b, flt c, flt d, flt e, flt f, flt g, flt h, flt i, flt j, flt bbox);


/*
 * Include now-deprecated Tachyon APIs, unless the user has told us not to
 */
#if !defined(TACHYON_NO_DEPRECATED)
#include "tachyon_dep.h"
#endif


/*
 * Internal Tachyon APIs and data structures.
 * Application developers should not be using anything below this point
 * in the header file.
 */
#if defined(TACHYON_INTERNAL)

#ifdef USESINGLEFLT
/* All floating point types will be based on "float" */
#define SPEPSILON   0.0001f     /**< amount to crawl down a ray           */
#define EPSILON     0.0001f     /**< amount to crawl down a ray           */
#define FHUGE       1e18f       /**< biggest fp number we care about      */
#define TWOPI       6.28318531f /**< Two times Pi                         */
#define MINCONTRIB  0.001959f   /**< 1.0 / 512.0, smallest contribution   */
                                /**< to overall pixel color we care about */
                                /**< XXX this must change for HDR images  */
#else
/* All floating point types will be based on "double" */
#define SPEPSILON   0.000000001 /**< amount to crawl down a ray           */
#define EPSILON     0.000000001 /**< amount to crawl down a ray           */
#define FHUGE       1e18        /**< biggest fp number we care about      */
#define TWOPI       6.28318531  /**< Two times Pi                         */
#define MINCONTRIB  0.001959    /**< 1.0 / 512.0, smallest contribution   */
                                /**< to overall pixel color we care about */
                                /**< XXX this must change for HDR images  */
#endif

#define BOUNDTHRESH 16          /**< spatial subdiv. object count threshold */


/* 
 * Maximum internal table sizes 
 * Use prime numbers for best memory system performance
 * (helps avoid cache aliasing..)
 */
#define MAXIMGS   39            /**< maxiumum number of distinct images   */


/* 
 * Ray flags 
 *
 * These are used in order to skip calculations which are only
 * needed some of the time.  For example, when shooting shadow
 * rays, we only have to find *one* intersection that's valid, 
 * if we find even one, we can quit early, thus saving lots of work.
 */
#define RT_RAY_PRIMARY   1  /**< A primary ray */
#define RT_RAY_REGULAR   2  /**< A regular ray, fewer shorcuts available    */
#define RT_RAY_SHADOW    4  /**< A shadow ray, we can early-exit asap       */
#define RT_RAY_FINISHED  8  /**< We've found what we're looking for already */
                            /**< early-exit at soonest opportunity..        */


/**
 * Shader capability flags - sorted by relative execution cost.
 * Used to automatically setup the fastest shader that supports
 * all of the capabilities used in a given scene.
 * Ideally, we use the shader that just has the features we need,
 * and nothing more, but its impractical to have that many seperate
 * shaders, each optimized for an exact set of features, but we
 * do the best we can with a reasonable amount of code.
 */
#define RT_SHADE_NOFLAGS                0  /**< clear feature flags          */
#define RT_SHADE_LIGHTING               1  /**< need lighting                */
#define RT_SHADE_PHONG                  2  /**< need phong shading           */
#define RT_SHADE_TEXTURE_MAPS           4  /**< need texture mapping         */
#define RT_SHADE_MIPMAP                 8  /**< need mip-maps                */
#define RT_SHADE_REFLECTION            16  /**< need reflections             */
#define RT_SHADE_REFRACTION            32  /**< need refraction              */
#define RT_SHADE_SHADOWS               64  /**< need shadows                 */
#define RT_SHADE_VOLUMETRIC           128  /**< need volume rendering        */
#define RT_SHADE_ANTIALIASING         256  /**< need antialiasing            */
#define RT_SHADE_DEPTH_OF_FIELD       512  /**< need depth of field          */
#define RT_SHADE_SOFT_SHADOW         1024  /**< need soft-shadows/penumbra   */
#define RT_SHADE_VOLUMETRIC_SHADOW   2048  /**< need volumetric shadows      */
#define RT_SHADE_CLIPPING            4096  /**< need clipping logic enabled  */
#define RT_SHADE_AMBIENTOCCLUSION    8192  /**< need ambient occlusion       */


/* 
 * Texture flags
 * 
 * These are used in order to skip calculations that are only needed
 * some of the time.
 */
#define RT_TEXTURE_NOFLAGS      0 /**< No special behavior        */
#define RT_TEXTURE_SHADOWCAST   1 /**< This object casts a shadow */ 
#define RT_TEXTURE_ISLIGHT      2 /**< This object is a light     */


/*
 * Image buffer format flags
 */
#define RT_IMAGE_BUFFER_RGB24   0 /**< 24-bit color, unsigned char RGB */
#define RT_IMAGE_BUFFER_RGB96F  1 /**< 96-bit color, 32-bit float RGB  */
#define RT_IMAGE_BUFFER_RGBA32  2 /**< 32-bit color, unsigned char RGBA */


/*
 * Image post-processing flags
 */
#define RT_IMAGE_CLAMP          0 /**< clamp pixel values [0 to 1)     */
#define RT_IMAGE_NORMALIZE      1 /**< normalize pixel values [0 to 1) */
#define RT_IMAGE_GAMMA          2 /**< gamma correction                */


struct ray_t;
typedef unsigned char byte; /* 1 byte */
typedef apivector vector;
typedef apicolor color;


typedef struct {         /**< Raw 24 bit RGB image structure */
  int loaded;            /**< image memory residence flag    */
  int xres;              /**< image X axis size              */
  int yres;              /**< image Y axis size              */
  int zres;              /**< image Z axis size              */
  int bpp;               /**< image bits per pixel           */
  char name[96];         /**< image filename (with path)     */
  unsigned char * data;  /**< pointer to raw byte image data */
} rawimage;


typedef struct {
  int levels;
  rawimage ** images;
} mipmap;


typedef struct {         /**< Scalar Volume Data */
  int loaded;            /**< Volume data memory residence flag */
  int xres;		 /**< volume X axis size                */
  int yres;		 /**< volume Y axis size                */
  int zres;		 /**< volume Z axis size                */
  flt opacity;		 /**< opacity per unit length           */
  char name[96];         /**< Volume data filename              */
  unsigned char * data;  /**< pointer to raw byte volume data   */
} scalarvol;


/*
 * Background texture data structure
 */
typedef struct {
  colora background;      /**< solid background color     */
  vector gradient;       /**< gradient direction vector for "up"  */
  flt gradtopval;        /**< texture dot product max parameter for top  */
  flt gradbotval;        /**< texture dot product min parameter for bot  */
  color backgroundtop;   /**< gradient background top    */ 
  color backgroundbot;   /**< gradient background bottom */
} background_texture;

/*
 * Object texture data structures
 */
typedef struct {
  void (* freetex)(void *);   /**< free the texture */
} texture_methods;

#define RT_TEXTURE_HEAD \
  color (* texfunc)(const void *, const void *, void *);                   \
  texture_methods * methods;  /**< this texture's methods */               \
  unsigned int flags;         /**< texturing/lighting flags */             \
  float ambient;              /**< ambient lighting */                     \
  float diffuse;              /**< diffuse reflection */                   \
  float phong;                /**< phong specular highlights */            \
  float phongexp;             /**< phong exponent/shininess factor */      \
  int phongtype;              /**< 0 == dielectric, nonzero == metal */    \
  float specular;             /**< specular reflection */                  \
  float opacity;              /**< how opaque the object is */             \
  int transmode;              /**< transparency modulation mode */         \
  float outline;              /**< edge outline shading */                 \
  float outlinewidth;         /**< edge outline width */

typedef struct {
  RT_TEXTURE_HEAD
} texture;

typedef struct {
  RT_TEXTURE_HEAD
  color  col;         /**< base object color */
  vector ctr;         /**< origin of texture */
  vector rot;         /**< rotation of texture about origin */
  vector scale;       /**< scale of texture in x,y,z */
  vector uaxs;	      /**< planar/volume map U axis */
  vector vaxs;	      /**< planar/volume map V axis */
  vector waxs;	      /**< volumetric map W axis */
  void * img;         /**< pointer to image or volume texture */
  void * obj;         /**< object ptr, hack for vol shaders */
} standard_texture;

typedef struct {
  RT_TEXTURE_HEAD
  void * obj;         /**< object ptr, hack for vcstri for now */
  color c0;           /**< color for vertex 0 */
  color c1;           /**< color for vertex 1 */
  color c2;           /**< color for vertex 2 */
} vcstri_texture;


/*
 * Object data structures
 */
typedef struct {
  void (* intersect)(const void *, void *);      /**< intersection func ptr  */
  void (* normal)(const void *, const void *, const void *, void *); /**< normal function ptr    */
  int (* bbox)(void *, vector *, vector *);      /**< return the object bbox */
  void (* freeobj)(void *);                      /**< free the object        */
} object_methods;


/*
 * Clipping plane data structure
 */
typedef struct {
  int numplanes;             /**< number of clipping planes */
  flt * planes;              /**< 4 plane eq coefficients per plane */
} clip_group;
 

#define RT_OBJECT_HEAD \
  unsigned int id;           /**< Unique Object serial number    */ \
  void * nextobj;            /**< pointer to next object in list */ \
  object_methods * methods;  /**< this object's methods          */ \
  clip_group * clip;         /**< this object's clip group       */ \
  texture * tex;             /**< object texture                 */ 


typedef struct {
  RT_OBJECT_HEAD
} object; 


typedef struct {
  const object * obj;        /**< to object we hit                        */ 
  flt t;                     /**< distance along the ray to the hit point */
} intersection;


typedef struct {
  int num;                   /**< number of intersections    */
  intersection closest;      /**< closest intersection > 0.0 */
  flt shadowfilter;          /**< modulation by transparent surfaces */
} intersectstruct;


/* camera related defines etc */
#define RT_CAMERA_FRUSTUM_AUTO 0   /**< compute frustum automatically     */
#define RT_CAMERA_FRUSTUM_USER 1   /**< use user-specified frustum bounds */

typedef struct {
  int frustumcalc;           /**< auto-calc or user-defined frustum       */
  int projection;            /**< camera projection mode                  */
  vector center;             /**< center of the camera in world coords    */
  vector viewvec;            /**< view direction of the camera  (Z axis)  */
  vector rightvec;           /**< right axis for the camera     (X axis)  */
  vector upvec;              /**< up axis for the camera        (Y axis)  */
  flt camzoom;               /**< zoom factor for the camera              */
  flt px;                    /**< width of image plane in world coords    */
  flt py;                    /**< height of image plane in world coords   */
  flt psx;                   /**< width of pixel in world coords          */
  flt psy;                   /**< height of pixel in world coords         */
  flt focallength;           /**< distance from eye to focal plane        */
  flt left;                  /**< left side of perspective frustum        */
  flt right;                 /**< right side of perspective frustum       */
  flt top;                   /**< top side of perspective frustum         */
  flt bottom;                /**< bottom side of perspective frustum      */
  flt aperture;              /**< depth of field aperture                 */
  vector projcent;           /**< center of image plane in world coords   */
  colora (* cam_ray)(void *, flt, flt);   /**< camera ray generator fctn   */
  vector lowleft;            /**< lower left corner of image plane        */
  vector iplaneright;        /**< image plane right vector                */
  vector iplaneup;           /**< image plane up    vector                */
} camdef;

typedef struct fogdata_t {
  color (* fog_fctn)(struct fogdata_t *, color, flt);   /**< fog function */
  int type;                  /**< radial, planer, etc                     */
  color col;                 /**< fog color                               */
  flt start;                 /**< fog start parameter                     */
  flt end;                   /**< fog end parameter                       */
  flt density;               /**< fog density parameter                   */
} fogdata;

typedef struct amboccdata_t {
  int numsamples;            /**< number of samples for ambient occlusion */
  color col;                 /**< color of ambient occlusion light        */
} amboccludedata;

typedef struct {
  int numcpus;               /**< number of processors on this node       */
  flt cpuspeed;              /**< relative speed of cpus on this node     */
  flt nodespeed;             /**< relative speed index for this node      */
  char machname[512];        /**< machine/node name                       */
} nodeinfo;

typedef struct list {
  void * item;
  struct list * next;
} list;

typedef struct {
  vector hit;  /**< ray object intersection hit point */
  vector N;    /**< surface normal at the hit point */
  vector L;    /**< vector point in the direction from hit point to the light */
  flt    Llen; /**< distance from hit point to the light (if any) */
} shadedata;

typedef struct {
  int cropmode; /**< output image cropping mode */
  int xres;     /**< cropped image x resolution in pixels */
  int yres;     /**< cropped image y resolution in pixels */
  int xstart;   /**< starting pixel in x (left side) */
  int ystart;   /**< starting pixel in y (top size) */ 
} cropinfo;

typedef struct {
  object * boundedobj;       /**< bounded object list, starts out empty   */
  object * unboundedobj;     /**< unbounded object list, starts out empty */
  int numobjects;            /**< number of objects in group              */
} displist;
 
typedef struct {
  char outfilename[256];     /**< name of the output image                */
  int writeimagefile;        /**< enable/disable writing of image to disk */
  void * img;                /**< pointer to a raw rgb image to be stored */
  int imginternal;           /**< image was allocated by the library      */
  int imgprocess;            /**< image post processing flags             */
  float imggamma;            /**< image gamma correction value            */
  int imgbufformat;          /**< pixel format for image buffer           */
  int imgfileformat;         /**< output format for final image           */
  cropinfo imgcrop;          /**< image output cropping for SPEC MPI      */
  int numthreads;            /**< user controlled number of threads       */
  int nodes;                 /**< number of distributed memory nodes      */
  int mynode;                /**< my distributed memory node number       */
  nodeinfo * cpuinfo;        /**< overall cpu/node/threads info           */
  int hres;                  /**< horizontal output image resolution      */
  int vres;                  /**< vertical output image resolution        */
  flt aspectratio;           /**< aspect ratio of output image            */
  int raydepth;              /**< maximum recursion depth                 */
  int transcount;            /**< maximum # transparent surfaces shown    */
  int shadowfilter;          /**< whether trans. surfaces filter lights   */
  int antialiasing;          /**< number of antialiasing rays to fire     */
  int verbosemode;           /**< verbose reporting flag                  */
  int boundmode;             /**< automatic spatial subdivision flag      */
  int boundthresh;           /**< threshold number of subobjects          */
  list * texlist;            /**< linked list of texture objects          */
  list * cliplist;           /**< linked list of clipping plane groups    */
  unsigned int flags;        /**< scene feature requirement flags         */
  camdef camera;             /**< camera definition                       */
  colora (* shader)(void *);  /**< main shader used for the whole scene    */
  flt (* phongfunc)(const struct ray_t * incident, const shadedata * shadevars, flt specpower);              /**< phong shader used for whole scene       */ 
  int transmode;             /**< transparency mode flags                 */
  background_texture bgtex;  /**< background texture parameters           */
  colora (* bgtexfunc)(const struct ray_t * incident); /**< background texturing function ptr  */
  fogdata fog;               /**< fog parameters                          */
  displist objgroup;         /**< objects in the scene                    */
  list * lightlist;          /**< linked list of lights in the scene      */
  flt light_scale;           /**< global scaling factor for direct lights */
  int numlights;             /**< number of lights in the scene           */
  amboccludedata ambocc;     /**< ambient occlusion data                  */
  int scenecheck;            /**< re-check scene for changes              */
  void * parbuf;             /**< parallel message passing handle         */
  void * threads;            /**< thread handles                          */
  void * threadparms;        /**< thread parameters                       */
  clip_group * curclipgroup; /**< current clipping group, during parsing  */
  int normalfixupmode;       /**< normal/winding order fixup for stri     */
} scenedef;


typedef struct ray_t {
  vector o;              /**< origin of the ray X,Y,Z                        */
  vector d;              /**< normalized direction of the ray                */
  flt maxdist;           /**< maximum distance to search for intersections   */
  flt opticdist;         /**< total distance traveled from camera so far     */
  void (* add_intersection)(flt, const object *, struct ray_t *); 
  intersectstruct intstruct; /**< ptr to thread's intersection data          */ 
  unsigned int depth;    /**< levels left to recurse.. (maxdepth - curdepth) */
  int transcnt;          /**< transparent surfaces left to show              */
  unsigned int flags;    /**< ray flags, any special treatment needed etc    */
  unsigned long serial;  /**< serial number of the ray                       */
  unsigned long * mbox;  /**< mailbox array for optimizing intersections     */
  scenedef * scene;      /**< pointer to the scene, for global parms such as */
                         /**< background colors etc                          */
  unsigned int randval;  /**< random number seed                             */
  rng_frand_handle frng; /**< 32-bit FP random number generator handle       */
} ray;


#endif

#ifdef  __cplusplus
}
#endif
#endif
