/* 
 * render.c - This file contains the main program and driver for the raytracer.
 *
 *  $Id: render.c,v 1.110 2013/04/21 08:28:14 johns Exp $
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TACHYON_INTERNAL 1
#include "tachyon.h"
#include "macros.h"
#include "threads.h"
#include "parallel.h"
#include "imageio.h"
#include "trace.h"
#include "render.h"
#include "util.h"
#include "shade.h"
#include "ui.h"
#include "grid.h"
#include "camera.h"
#include "intersect.h"

/*
 * Determine which shader to use based on the list of capabilities
 * needed to render the scene at full quality.  Ideally we'll avoid
 * using anything more sophisticated than is actually needed to render
 * a scene.
 */
static void rt_autoshader(scenedef * scene) {
  /* 
   * If the user has already specified a particular shader
   * then we use what they asked for, otherwise we determine
   * which shader to use ourselves.
   */
  if (scene->shader == NULL) {
    /* No logic yet, just use max quality */
    scene->shader = (colora (*)(void *)) full_shader;
  }
}


/*
 * All of the threads in the pool wait on a barrier until
 * they are told to wake up and do some work.  At present,
 * the only actions they can take are to render the scene 
 * or to terminate be returning to the master.
 */
void * thread_worker(void * voidparms) {
  thr_parms * parms = (thr_parms *) voidparms;

#if defined(USECPUAFFINITY)
  /* Optionally set CPU affinity mask for each thread */
  int cpuaffinity = -1;

#if defined(__MIC__)
  /* On the MIC platform, we want 4 threads per CPU, with a hard-coded  */
  /* mapping that puts neighboring workers on neighboring CPUs with the */
  /* hope of better L1/L2 cache sharing                                 */
  cpuaffinity = parms->tid / 4;
#endif

  if (cpuaffinity > 0) {
    rt_thread_set_self_cpuaffinity(cpuaffinity); 
#if 0
    printf("Thread[%d] setting affinity to %d\n", parms->tid, parms->tid / 4);
#endif
  }
#endif

  while (rt_thread_barrier(parms->runbar, 0)) {
    thread_trace(parms);
  }
  return NULL;
}


/* 
 * Create the pool of rendering threads, initialize all of the
 * state variables they need, and start them waiting on the barrier.
 */
void create_render_threads(scenedef * scene) {
  thr_parms * parms;
  rt_thread_t * threads;
  rt_barrier_t * bar;
#if defined(MPI) && defined(THR)
  int row, numrowbars;
  rt_atomic_int_t * rowbars;
  rt_atomic_int_t * rowsdone;
#endif
  int thr;

  /* allocate and initialize thread parameter buffers */
  threads = (rt_thread_t *) malloc(scene->numthreads * sizeof(rt_thread_t));
  parms = (thr_parms *) malloc(scene->numthreads * sizeof(thr_parms));

  bar = rt_thread_barrier_init(scene->numthreads);

#if defined(MPI) && defined(THR)
  /* initialize row barriers for MPI builds */
  numrowbars = scene->vres;
  rowbars = (rt_atomic_int_t *) calloc(1, numrowbars * sizeof(rt_atomic_int_t));
  rowsdone = (rt_atomic_int_t *) calloc(1, sizeof(rt_atomic_int_t));
  for (row=0; row<numrowbars; row++) {
    rt_atomic_int_init(&rowbars[row], 0);
  }
  rt_atomic_int_init(rowsdone, 0);
#endif

  for (thr=0; thr<scene->numthreads; thr++) {
    parms[thr].tid=thr;
    parms[thr].nthr=scene->numthreads;
    parms[thr].scene=scene;

    /* the sizes of these arrays are padded to avoid cache aliasing */
    /* and false sharing between threads.                           */
    parms[thr].local_mbox = 
#if !defined(DISABLEMBOX)
      (unsigned long *) calloc(sizeof(unsigned long)*scene->objgroup.numobjects + 32, 1);
#else
      NULL;
#endif

    parms[thr].serialno = 1;
    parms[thr].runbar = bar;

    /* For a threads-only build (or MPI nodes == 1), we distribute  */
    /* work round-robin by scanlines.  For MPI-only builds, we also */
    /* distribute by scanlines.  For mixed MPI+threads builds, we   */
    /* distribute work to nodes by scanline, and to the threads     */
    /* within a node on a pixel-by-pixel basis.                     */
    if (scene->nodes == 1) {
      parms[thr].startx = 1;
      parms[thr].stopx  = scene->hres;
      parms[thr].xinc   = 1;
      parms[thr].starty = thr + 1;
      parms[thr].stopy  = scene->vres;
      parms[thr].yinc   = scene->numthreads;
    } else {
      parms[thr].startx = thr + 1;
      parms[thr].stopx  = scene->hres;
      parms[thr].xinc   = scene->numthreads;
      parms[thr].starty = scene->mynode + 1;
      parms[thr].stopy  = scene->vres;
      parms[thr].yinc   = scene->nodes;
    }
#if defined(MPI) && defined(THR)
    parms[thr].numrowbars = numrowbars;
    parms[thr].rowbars = rowbars;
    parms[thr].rowsdone = rowsdone;
#endif
  }

  scene->threadparms = (void *) parms;
  scene->threads = (void *) threads;

  for (thr=1; thr < scene->numthreads; thr++) 
    rt_thread_create(&threads[thr], thread_worker, (void *) (&parms[thr]));

}


/*
 * Shutdown all of the worker threads and free up their resources
 */
void destroy_render_threads(scenedef * scene) {
  thr_parms * parms = (thr_parms *) scene->threadparms;
  rt_thread_t * threads = (rt_thread_t *) scene->threads;
  int thr;
#if defined(MPI) && defined(THR)
  int row;
#endif

  if (scene->threads != NULL) {
    /* wake up sleepers and tell them to exit */
    rt_thread_barrier(parms[0].runbar, 0); 

    /* wait for all sleepers to exit */
    for (thr=1; thr<parms[0].nthr; thr++) 
      rt_thread_join(threads[thr], NULL);
  
    /* destroy the thread barrier */
    rt_thread_barrier_destroy(parms[0].runbar);

    free(scene->threads);
  }

  if (scene->threadparms != NULL) {
    /* deallocate thread parameter buffers 
     * NOTE: This has to use the remembered number of threads stored in the
     *       thread parameter area for thread 0, since the one in the scene
     *       may have changed on us.
     */
    for (thr=0; thr < parms[0].nthr; thr++) {
      if (parms[thr].local_mbox != NULL) 
        free(parms[thr].local_mbox);
    }

#if defined(MPI) && defined(THR)
    /* destroy and free row barriers for MPI builds */
    for (row=0; row<parms[0].numrowbars; row++) {
      rt_atomic_int_destroy(&parms[0].rowbars[row]);
    }  
    rt_atomic_int_destroy(parms[0].rowsdone);
    free(parms[0].rowbars);
    free(parms[0].rowsdone);
#endif

    free(scene->threadparms);
  }

  scene->threads = NULL;
  scene->threadparms = NULL;
}



/*
 * Check the scene to determine whether or not any parameters that affect
 * the thread pool, the persistent message passing primitives, or other
 * infrastructure needs to be reconfigured before rendering commences.
 */
void rendercheck(scenedef * scene) {
  flt runtime;
  rt_timerhandle stth; /* setup time timer handle */

  if (scene->verbosemode && scene->mynode == 0) {
    char msgtxt[1024];
    int i, totalcpus;
    flt totalspeed;

    rt_ui_message(MSG_0, "CPU Information:");

    totalspeed = 0.0;
    totalcpus = 0;
    for (i=0; i<scene->nodes; i++) {
      sprintf(msgtxt,
            "  Node %4d: %2d CPUs, CPU Speed %4.2f, Node Speed %6.2f Name: %s",
            i, scene->cpuinfo[i].numcpus, scene->cpuinfo[i].cpuspeed,
            scene->cpuinfo[i].nodespeed, scene->cpuinfo[i].machname);
      rt_ui_message(MSG_0, msgtxt);

      totalcpus += scene->cpuinfo[i].numcpus;
      totalspeed += scene->cpuinfo[i].nodespeed;
    }

    sprintf(msgtxt, "  Total CPUs: %d", totalcpus);
    rt_ui_message(MSG_0, msgtxt);
    sprintf(msgtxt, "  Total Speed: %f\n", totalspeed);
    rt_ui_message(MSG_0, msgtxt);
  }

  rt_barrier_sync();     /* synchronize all nodes at this point             */
  stth=rt_timer_create();
  rt_timer_start(stth);  /* Time the preprocessing of the scene database    */
  rt_autoshader(scene);  /* Adapt to the shading features needed at runtime */

  /* Hierarchical grid ray tracing acceleration scheme */
  if (scene->boundmode == RT_BOUNDING_ENABLED) 
    engrid_scene(scene, scene->boundthresh); 

  /* if any clipping groups exist, we have to use appropriate */
  /* intersection testing logic                               */
  if (scene->cliplist != NULL) {
    scene->flags |= RT_SHADE_CLIPPING;
  }

  /* if there was a preexisting image, free it before continuing */
  if (scene->imginternal && (scene->img != NULL)) {
    free(scene->img);
    scene->img = NULL;
  }

  /* Allocate a new image buffer if necessary */
  if (scene->img == NULL) {
    scene->imginternal = 1;
    if (scene->verbosemode && scene->mynode == 0) { 
      rt_ui_message(MSG_0, "Allocating Image Buffer."); 
    }

    /* allocate the image buffer accordinate to pixel format */
    if (scene->imgbufformat == RT_IMAGE_BUFFER_RGB24) {
      scene->img = malloc(scene->hres * scene->vres * 3);
    } else if (scene->imgbufformat == RT_IMAGE_BUFFER_RGBA32) {
        scene->img = malloc(scene->hres * scene->vres * 4);
    } else if (scene->imgbufformat == RT_IMAGE_BUFFER_RGB96F) {
      scene->img = malloc(sizeof(float) * scene->hres * scene->vres * 3);
    } else {
      rt_ui_message(MSG_0, "Illegal image buffer format specifier!"); 
    }

    if (scene->img == NULL) {
      scene->imginternal = 0;
      rt_ui_message(MSG_0, "Warning: Failed To Allocate Image Buffer!"); 
    } 
  }

  /* if any threads are leftover from a previous scene, and the  */
  /* scene has changed significantly, we have to collect, and    */
  /* respawn the worker threads, since lots of things may have   */
  /* changed which would affect them.                            */
  destroy_render_threads(scene);
  create_render_threads(scene);

  /* allocate and initialize persistent scanline receive buffers */
  /* which are used by the parallel message passing code.        */
  scene->parbuf = rt_init_scanlinereceives(scene);

  /* the scene has been successfully prepared for rendering      */
  /* unless it gets modified in certain ways, we don't need to   */
  /* pre-process it ever again.                                  */
  scene->scenecheck = 0;

  rt_timer_stop(stth); /* Preprocessing is finished, stop timing */
  runtime=rt_timer_time(stth);   
  rt_timer_destroy(stth);

  /* Print out relevent timing info */
  if (scene->mynode == 0) {
    char msgtxt[256];
    sprintf(msgtxt, "Preprocessing Time: %10.4f seconds",runtime);
    rt_ui_message(MSG_0, msgtxt);
  }
}


/*
 * Save the rendered image to disk.
 */
static void renderio(scenedef * scene) {
  flt iotime;
  char msgtxt[256];
  rt_timerhandle ioth; /* I/O timer handle */

  ioth=rt_timer_create();
  rt_timer_start(ioth);

  if (scene->imgbufformat == RT_IMAGE_BUFFER_RGB96F) {
    if (scene->imgprocess & RT_IMAGE_NORMALIZE) {
      normalize_rgb96f(scene->hres, scene->vres, (float *) scene->img);
      rt_ui_message(MSG_0, "Post-processing: normalizing pixel values.");
    }

    if (scene->imgprocess & RT_IMAGE_GAMMA) {
      gamma_rgb96f(scene->hres, scene->vres, (float *) scene->img, 
                   scene->imggamma);
      rt_ui_message(MSG_0, "Post-processing: gamma correcting pixel values.");
    }
  } else if (scene->imgbufformat == RT_IMAGE_BUFFER_RGB24) {
    if (scene->imgprocess & (RT_IMAGE_NORMALIZE | RT_IMAGE_GAMMA))
      rt_ui_message(MSG_0, "Can't post-process 24-bit integer image data");
  }

  /* support cropping of output images for SPECMPI benchmarks */
  if (scene->imgcrop.cropmode == RT_CROP_DISABLED) {
    writeimage(scene->outfilename, scene->hres, scene->vres, 
               scene->img, scene->imgbufformat, scene->imgfileformat);
  } else {
    /* crop image before writing if necessary */
    if (scene->imgbufformat == RT_IMAGE_BUFFER_RGB96F) {
      float *imgcrop;
      imgcrop = image_crop_rgb96f(scene->hres, scene->vres, scene->img,
                                  scene->imgcrop.xres, scene->imgcrop.yres, 
                                  scene->imgcrop.xstart, scene->imgcrop.ystart);
      writeimage(scene->outfilename, scene->imgcrop.xres, scene->imgcrop.yres,
                 imgcrop, scene->imgbufformat, scene->imgfileformat);
      free(imgcrop);
    } else if (scene->imgbufformat == RT_IMAGE_BUFFER_RGB24) {
      unsigned char *imgcrop;
      imgcrop = image_crop_rgb24(scene->hres, scene->vres, scene->img,
                                 scene->imgcrop.xres, scene->imgcrop.yres, 
                                 scene->imgcrop.xstart, scene->imgcrop.ystart);
      writeimage(scene->outfilename, scene->imgcrop.xres, scene->imgcrop.yres,
                 imgcrop, scene->imgbufformat, scene->imgfileformat);
      free(imgcrop);
    }
  }

  rt_timer_stop(ioth);
  iotime = rt_timer_time(ioth);
  rt_timer_destroy(ioth);

  sprintf(msgtxt, "    Image I/O Time: %10.4f seconds", iotime);
  rt_ui_message(MSG_0, msgtxt);
}


/*
 * Render the scene
 */
void renderscene(scenedef * scene) {
  flt runtime;
  rt_timerhandle rtth; /* render time timer handle */

  /* if certain key aspects of the scene parameters have been changed */
  /* since the last frame rendered, or when rendering the scene the   */
  /* first time, various setup, initialization and memory allocation  */
  /* routines need to be run in order to prepare for rendering.       */
  if (scene->scenecheck)
    rendercheck(scene);

  if (scene->mynode == 0) 
    rt_ui_progress(0);     /* print 0% progress at start of rendering */


  /* 
   * Core Ray Tracing Code
   *
   * Ideally, as little as possible other than this code should be
   * executed for rendering a frame.  Most if not all memory allocations
   * should be done outside of the core code, and all setup should be
   * done outside of here.  This will give the best speed when rendering
   * walk-throughs and similar things.  
   */

  rtth=rt_timer_create();  /* create/init rendering timer              */
  rt_timer_start(rtth);    /* start ray tracing timer                  */

  camera_init(scene);      /* Initialize all aspects of camera system  */

#if defined(MPI) && defined(THR)
  /* reset the rows counter for this frame */
  rt_atomic_int_set(((thr_parms *) scene->threadparms)[0].rowsdone, 0);
#endif

#ifdef THR
  /* if using threads, wake up the child threads...  */
  rt_thread_barrier(((thr_parms *) scene->threadparms)[0].runbar, 1);
#endif

#ifdef MPI
  /* if using message passing, start persistent receives */
  rt_start_scanlinereceives(scene->parbuf); /* start scanline receives */
#endif

  /* Actually Ray Trace The Image */
  thread_trace(&((thr_parms *) scene->threadparms)[0]);

#ifdef MPI
  rt_waitscanlines(scene->parbuf);  /* wait for all scanlines to recv/send  */
#endif

  rt_timer_stop(rtth);              /* stop timer for ray tracing runtime   */
  runtime=rt_timer_time(rtth);
  rt_timer_destroy(rtth);

  /*
   * End of Core Ray Tracing Code
   *
   * Anything after here should be UI, tear-down, or reset code 
   *
   */

  if (scene->mynode == 0) {
    char msgtxt[256];

    rt_ui_progress(100); /* print 100% progress when finished rendering */

    sprintf(msgtxt, "\n  Ray Tracing Time: %10.4f seconds", runtime);
    rt_ui_message(MSG_0, msgtxt);
 
    if (scene->writeimagefile) 
      renderio(scene);
  }
} /* end of renderscene() */

