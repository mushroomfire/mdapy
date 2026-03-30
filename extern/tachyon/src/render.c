/* 
 * render.c - This file contains the main program and driver for the raytracer.
 *
 * (C) Copyright 1994-2022 John E. Stone
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * $Id: render.c,v 1.123 2022/02/18 17:55:28 johns Exp $
 *
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
    scene->shader = (color (*)(void *)) full_shader;
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

#if 0 && defined(_ARCH_PPC64) 
  /* On POWER7/8 platforms, CPUs are numbered sequentially including    */
  /* indices for all per-core SMT threads which may be enabled/disabled */
  /* in a consecutive sequence.                                         */
  cpuaffinity = parms->tid;
#endif

  if (cpuaffinity > 0) {
    rt_thread_set_self_cpuaffinity(cpuaffinity); 
#if 0
    if (scene->verbosemode && scene->mynode == 0) {
      printf("Thread[%d] setting affinity to %d\n", parms->tid, cpuaffinity);
    }
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
  int thr;
  thr_parms * parms;
  rt_thread_t * threads;
  rt_barrier_t * bar;
#if defined(MPI) && defined(THR)
  int row, numrowbars;
  rt_atomic_int_t * rowbars;
  rt_atomic_int_t * rowsdone;
#endif
#if defined(THR)
  rt_atomic_int_t * pixelsched;
  int sched_dynamic = 0; /* leave dynamic pixel scheduling off by default */

#if 1
  /* determine whether to enable dynamic pixel scheduling based   */
  /* on whether the scene uses any particularly costly rendering  */
  /* features such as ambient occlusion lighting, or greater than */
  /* 4-samples per-pixel antialiasing...                          */
  if (scene->ambocc.numsamples > 0 || scene->antialiasing > 4) {
    sched_dynamic = 1;
  } 
#else
  sched_dynamic = (getenv("SCHED_DYNAMIC") != NULL);
#endif
#endif

  /* allocate and initialize thread parameter buffers */
  threads = (rt_thread_t *) malloc(scene->numthreads * sizeof(rt_thread_t));
  parms = (thr_parms *) malloc(scene->numthreads * sizeof(thr_parms));

  bar = rt_thread_barrier_init(scene->numthreads);

#if defined(THR)
  /* initialize atomic pixel scheduler used for dynamic load balancing */
  pixelsched = (rt_atomic_int_t *) calloc(1, sizeof(rt_atomic_int_t));
  rt_atomic_int_init(pixelsched, 0);
#endif

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

#if defined(THR)
    parms[thr].sched_dynamic = sched_dynamic;
    parms[thr].pixelsched = pixelsched;
#endif

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

#if defined(THR)
    /* destroy the atomic pixel scheduler counter */
    rt_atomic_int_destroy(parms[0].pixelsched);
    free(parms[0].pixelsched);
#endif

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
static void rendercheck(scenedef * scene) {
  flt runtime;
  rt_timerhandle stth; /* setup time timer handle */

  if (scene->verbosemode && scene->mynode == 0) {
    char msgtxt[1024];
    int i, totalcpus;
    flt totalspeed;

    rt_ui_message(MSG_0, "CPU Information:");
    memset(msgtxt, 0, sizeof(msgtxt));
    if ((scene->nodes == 1) && (scene->cpuinfo[0].cpucaps != NULL)) {
      rt_cpu_caps_t *cpucaps = (rt_cpu_caps_t *) scene->cpuinfo[0].cpucaps;

      strcpy(msgtxt, "  CPU features: ");

#if (defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_AMD64))
      if (cpucaps->flags & CPU_SSE2)
        strcat(msgtxt, "SSE2 ");
      if (cpucaps->flags & CPU_SSE4_1)
        strcat(msgtxt, "SSE4.1 ");
      if (cpucaps->flags & CPU_AVX)
        strcat(msgtxt, "AVX ");
      if (cpucaps->flags & CPU_AVX2)
        strcat(msgtxt, "AVX2 ");
      if (cpucaps->flags & CPU_FMA)
        strcat(msgtxt, "FMA ");
      if (cpucaps->flags & CPU_F16C)
        strcat(msgtxt, "F16 ");

      if ((cpucaps->flags & CPU_KNL) == CPU_KNL) {
        strcat(msgtxt, "KNL:AVX-512F+CD+ER+PF ");
      } else {
        if (cpucaps->flags & CPU_AVX512F)
          strcat(msgtxt, "AVX512F ");
        if (cpucaps->flags & CPU_AVX512CD)
          strcat(msgtxt, "AVX512CD ");
        if (cpucaps->flags & CPU_AVX512ER)
          strcat(msgtxt, "AVX512ER ");
        if (cpucaps->flags & CPU_AVX512PF)
          strcat(msgtxt, "AVX512PF ");
      }

      if (cpucaps->flags & CPU_HT)
        strcat(msgtxt, "HT ");

      if (cpucaps->flags & CPU_HYPERVISOR) {
        rt_ui_message(MSG_0, msgtxt);
        rt_ui_message(MSG_0, "  Detected VM or hypervisor execution environment");
      }
#endif

#if (defined(__ARM_ARCH_ISA_A64) || defined(__ARM_NEON))
      if (cpucaps->flags & CPU_ARM64_FP)
        strcat(msgtxt, "FP ");
      if (cpucaps->flags & CPU_ARM64_SVE)
        strcat(msgtxt, "SVE ");

      if (cpucaps->flags & CPU_ARM64_ASIMD)
        strcat(msgtxt, "ASIMD ");
      if (cpucaps->flags & CPU_ARM64_ASIMDHP)
        strcat(msgtxt, "ASIMDHP ");
      if (cpucaps->flags & CPU_ARM64_ASIMDRDM)
        strcat(msgtxt, "ASIMDRDM ");
      if (cpucaps->flags & CPU_ARM64_ASIMDDP)
        strcat(msgtxt, "ASIMDDP ");
      if (cpucaps->flags & CPU_ARM64_ASIMDFHM)
        strcat(msgtxt, "ASIMDFHM ");

      if (cpucaps->flags & CPU_ARM64_AES)
        strcat(msgtxt, "AES ");
      if (cpucaps->flags & CPU_ARM64_CRC32)
        strcat(msgtxt, "CRC32 ");
      if (cpucaps->flags & CPU_ARM64_SHA1)
        strcat(msgtxt, "SHA1 ");
      if (cpucaps->flags & CPU_ARM64_SHA2)
        strcat(msgtxt, "SHA2 ");
      if (cpucaps->flags & CPU_ARM64_SHA3)
        strcat(msgtxt, "SHA3 ");
      if (cpucaps->flags & CPU_ARM64_SHA512)
        strcat(msgtxt, "SHA512 ");

#if defined(VMDCPUDISPATCH) && defined(__ARM_FEATURE_SVE)
      if (cpucaps->flags & CPU_ARM64_SVE) {
        rt_ui_message(MSG_0, msgtxt);
        sprintf(msgtxt, "  ARM64 SVE vector lengths  32-bit: %d,  64-bit: %d",
                arm_sve_vecsize_32bits(), arm_sve_vecsize_64bits());
      }
#endif
#endif

      rt_ui_message(MSG_0, msgtxt); 
    }

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

  rt_par_barrier_sync(scene->parhnd); /* synchronize all nodes at this point */
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

#if defined(RT_ACCUMULATE_ON)
  /* Allocate the accumulation buffer if necessary */
  if ((scene->accum_mode == RT_ACCUMULATE_ON) ||
      (scene->accum_mode == RT_ACCUMULATE_CLEAR)) {
    int bufsz = sizeof(float) * scene->hres * scene->vres * 3;

    /* handle resize events */
    if (scene->accum_buf != NULL) {
      free(scene->accum_buf);
      scene->accum_buf = NULL;
    }

    if (scene->accum_buf == NULL) {
      scene->accum_buf = calloc(1, bufsz);  /* allocate and clear buffer */
      scene->accum_mode = RT_ACCUMULATE_ON; /* reset to on from clear    */
      scene->accum_count = 0;               /* reset accumulation count  */
    } 

    if (scene->accum_mode == RT_ACCUMULATE_CLEAR) {
      int bufsz = sizeof(float) * scene->hres * scene->vres * 3;
      memset(scene->accum_buf, 0, bufsz);   /* clear accumulation buffer */
      scene->accum_count = 0;               /* reset accumulation count  */
      scene->accum_mode = RT_ACCUMULATE_ON; /* reset to on from clear    */
    }
  }
#endif

  /* if any threads are leftover from a previous scene, and the  */
  /* scene has changed significantly, we have to collect, and    */
  /* respawn the worker threads, since lots of things may have   */
  /* changed which would affect them.                            */
  destroy_render_threads(scene);
  create_render_threads(scene);

  /* allocate and initialize persistent scanline receive buffers */
  /* which are used by the parallel message passing code.        */
  scene->parbuf = rt_par_init_scanlinereceives(scene->parhnd, scene);

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

#if defined(RT_ACCUMULATE_ON)
  /* update accumulation buffer state on every frame */ 
  if (scene->accum_mode == RT_ACCUMULATE_CLEAR) {
    int bufsz = sizeof(float) * scene->hres * scene->vres * 3;
    memset(scene->accum_buf, 0, bufsz);   /* clear accumulation buffer */
    scene->accum_count = 0;               /* reset accumulation count  */
    scene->accum_mode = RT_ACCUMULATE_ON; /* reset to on from clear    */
  }
  scene->accum_count++;               /* increment accumulation count */
#endif

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

#if defined(THR)
  /* reset the pixel counter for this frame */
  rt_atomic_int_set(((thr_parms *) scene->threadparms)[0].pixelsched, 0);
#endif

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
  rt_par_start_scanlinereceives(scene->parhnd, scene->parbuf);
#endif

  /* Actually Ray Trace The Image */
  thread_trace(&((thr_parms *) scene->threadparms)[0]);

#ifdef MPI
  /* wait for all scanlines to recv/send  */
  rt_par_waitscanlines(scene->parhnd, scene->parbuf);
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

