/* 
 * trace.c - This file contains the functions for firing primary rays
 *           and handling subsequent calculations
 *
 *   $Id: trace.c,v 1.127 2013/04/21 08:28:14 johns Exp $
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TACHYON_INTERNAL 1
#include "tachyon.h"
#include "macros.h"
#include "vector.h"
#include "shade.h"
#include "camera.h"
#include "util.h"
#include "threads.h"
#include "parallel.h"
#include "intersect.h"
#include "ui.h"
#include "trace.h"
#if defined(_OPENMP)
#include <omp.h>
#endif

colora trace(ray * primary) {
  if (primary->depth > 0) {
    intersect_objects(primary);
    return primary->scene->shader(primary);
  }

  /* if the ray is truncated, return the background texture as its color */
  return primary->scene->bgtexfunc(primary);
}


#if defined(MPI)
int node_row_sendrecv(int my_tid, thr_parms * t, scenedef *scene, 
                      int *sentrows, int y) {
  /* If running with MPI and we have multiple nodes, we must exchange */
  /* pixel data for each row of the output image as we run.           */
  if (scene->nodes > 1) {

#if defined(THR)
    /* When mixing threads+MPI, we have to ensure that all threads in   */
    /* a given node have completed a row of pixels before we try to     */
    /* send it, which requires a barrier synchronization.               */

#if defined(USEATOMICBARRIERS)
    /* 
     * Use fast atomic integer ops for per-row MPI sendrecv barriers
     */
    int rowidx = y - 1;
    int rowbarcnt;
    int rowsdone=-1;

    rowbarcnt = rt_atomic_int_add_and_fetch(&t->rowbars[rowidx], 1);

    /* if we were the last thread to read the barrier, we increment the */
    /* rowsdone counter and continue on...                              */
    if (rowbarcnt == t->nthr) {
/* printf("node[%d] thr[%d] rowidx: %d\n", scene->mynode, my_tid, rowidx); */
      rowsdone = rt_atomic_int_add_and_fetch(t->rowsdone, 1);

      /* clear the row barrier so it is ready to be used again... */
      rt_atomic_int_set(&t->rowbars[rowidx], 0);
    }
 
    /* Since only thread 0 can make MPI calls, it checks how many rows */
    /* are done and sends any completed rows that weren't already sent */ 
    if (my_tid == 0) {
      int row;

      /* if we've already got rowsdone from a previous fetch-and-add, */
      /* we use it, otherwise we have to actually query it...         */
      if (rowsdone < 0)
        rowsdone = rt_atomic_int_get(t->rowsdone);
     
      /* send any rows that are completed but not already sent */ 
      for (row=(*sentrows); row<rowsdone; row++) { 
/* printf("node[%d] sending row: %d  (sentrows %d)\n", scene->mynode, row, *sentrows); */
        rt_sendrecvscanline(scene->parbuf); /* only thread 0 can use MPI */ 
/* printf("node[%d] row: %d sent!\n", scene->mynode, row); */
      }
      *sentrows = row;
    }
#else
    /*
     * Use the threadpool barriers to synchronize all worker threads
     * prior to invoking the MPI sendrecv operations.  This kind of
     * barrier is very costly for real-time renderings, so it has been
     * replaced by faster atomic counters.
     */
    rt_thread_barrier(t->runbar, 1);

    /* after all worker threads have completed the row, we can send it */
    if (my_tid == 0) {
      rt_sendrecvscanline(scene->parbuf); /* only thread 0 can use MPI */ 
    }
#endif  
#else
    /* For OpenMP, we must also check that we are thread ID 0 */
    if (my_tid == 0) {
      rt_sendrecvscanline(scene->parbuf); /* only thread 0 can use MPI */ 
    }
#endif

    /* Since all rows are stored in different memory locations     */
    /* there's no need to protect against race conditions between  */
    /* thread 0 MPI calls and ongoing work by peer threads running */
    /* farther ahead on subsequent rows.                           */
  }
  
  return 0;
}


int node_finish_row_sendrecvs(int my_tid, thr_parms * t, scenedef *scene, int *sentrows) {

  if (scene->nodes > 1) {
#if defined(THR)
    /* When mixing threads+MPI, we have to ensure that in the case that  */
    /* thread 0 of node 0 finishes early, we force it to finish handling */
    /* all oustanding row transfers before it returns.                   */
#if defined(USEATOMICBARRIERS)
#if 1
    /* XXX this barrier is very costly for real-time renderings, so it */
    /* is a candidate for replacement by a busy-wait..                 */
    rt_thread_barrier(t->runbar, 1);
#else
    /* wait for all peer threads to complete */
    if (my_tid == 0) {
      int rowsdone, totalrows;
      totalrows = rt_sendrecvscanline_get_totalrows(scene->parbuf);

/* printf("node[%d]: spinning waiting for totalrows: %d\n", scene->mynode, totalrows); */

      /* spin on the 'rowsdone' integer atomic counter */
      while ((rowsdone = rt_atomic_int_get(t->rowsdone)) < totalrows) {
/* printf("node[%d]: spinning waiting, rowsdone: %d totalrows: %d\n", scene->mynode, rowsdone, totalrows); */
      }
    }
#endif

    /* Since only thread 0 can make MPI calls, it checks how many rows */
    /* are done and sends any completed rows that weren't already sent */ 
    if (my_tid == 0) {
      int row; 
      int rowsdone = rt_atomic_int_get(t->rowsdone);
/* printf("node[%d] finish sendrecvs, rowsdone: %d  sentrows: %d\n", scene->mynode, rowsdone, *sentrows); */
      /* send any rows that are completed but not already sent */ 
      for (row=(*sentrows); row<rowsdone; row++) { 
/* printf("node[%d] sending row: %d (finishing)\n", scene->mynode, row); */
        rt_sendrecvscanline(scene->parbuf); /* only thread 0 can use MPI */ 
/* printf("node[%d] row: %d sent! (finishing)\n", scene->mynode, row); */
      }
      *sentrows = row;
    }
#else
    /* nothing to do for the old variant of the code since it kept all */
    /* worker threads in lockstep...                                   */
#endif  
#else
    /* nothing to do for OpenMP or other scenarios                     */
#endif
  }

  return 0;
}
#endif /* MPI */


void * thread_trace(thr_parms * t) {
#if defined(_OPENMP)
#pragma omp parallel default( none ) firstprivate(t)
{
#endif
  unsigned long * local_mbox = NULL;
  scenedef * scene;
  colora col;
  ray primary;
  int x, y, do_ui, hskip;
  int startx, stopx, xinc, starty, stopy, yinc, hsize, vres;
  rng_frand_handle cachefrng; /* Hold cached FP RNG state */
#if defined(MPI)
  int sentrows = 0;  /* no rows sent yet */
#endif

#if defined(_OPENMP)
  int my_tid = omp_get_thread_num(); /* get OpenMP thread ID */
  unsigned long my_serialno = 1; /* XXX should restore previous serialno */
#else
  int my_tid = t->tid;
  unsigned long my_serialno = t->serialno;
#endif

  /*
   * Copy all of the frequently used parameters into local variables.
   * This seems to improve performance, especially on NUMA systems.
   */
  startx = t->startx;
  stopx  = t->stopx;
  xinc   = t->xinc;
 
  starty = t->starty;
  stopy  = t->stopy;
  yinc   = t->yinc;
 
  scene  = t->scene;
  if (scene->imgbufformat == RT_IMAGE_BUFFER_RGBA32) {
	  hsize  = scene->hres*4;
	  hskip  = xinc * 4;
  }
  else {
	  hsize  = scene->hres*3;
	  hskip  = xinc * 3;
  }
  vres   = scene->vres;
  do_ui = (scene->mynode == 0 && my_tid == 0);

#if !defined(DISABLEMBOX)
   /* allocate mailbox array per thread... */
#if defined(_OPENMP)
  local_mbox = (unsigned long *)calloc(sizeof(unsigned long)*scene->objgroup.numobjects, 1);
#else
  if (t->local_mbox == NULL)  
    local_mbox = (unsigned long *)calloc(sizeof(unsigned long)*scene->objgroup.numobjects, 1);
  else 
    local_mbox = t->local_mbox;
#endif
#else
  local_mbox = NULL; /* mailboxes are disabled */
#endif

  /*
   * When compiled on platforms with a 64-bit long, ray serial numbers won't 
   * wraparound in _anyone's_ lifetime, so there's no need to even check....
   * On lesser-bit platforms, we're not quite so lucky, so we have to check.
   * We use a sizeof() check so that we can eliminate the LP64 macro tests
   * and eventually simplify the Makefiles.
   */
  if (sizeof(unsigned long) < 8) {
    /* 
     * If we are getting close to integer wraparound on the    
     * ray serial numbers, we need to re-clear the mailbox     
     * array(s).  Each thread maintains its own serial numbers 
     * so only those threads that are getting hit hard will    
     * need to re-clear their mailbox arrays.  In all likelihood,
     * the threads will tend to hit their counter limits at about
     * the same time though.
     */
    if (local_mbox != NULL) {
      /* reset counters if serial exceeds 1/8th largest possible ulong */
      if (my_serialno > (((unsigned long) 1) << ((sizeof(unsigned long) * 8) - 3))) {
        memset(local_mbox, 0, sizeof(unsigned long)*scene->objgroup.numobjects);
        my_serialno = 1;
      }
    }
  }

  /* setup the thread-specific properties of the primary ray(s) */
  camray_init(scene, &primary, my_serialno, local_mbox, 
              rng_seed_from_tid_nodeid(my_tid, scene->mynode));

  /* copy the RNG state to cause increased coherence among */
  /* AO sample rays, significantly reducing granulation    */
  cachefrng = primary.frng;

  /* 
   * Render the image in either RGB24 or RGB96F format
   */
  if (scene->imgbufformat == RT_IMAGE_BUFFER_RGB24 || scene->imgbufformat == RT_IMAGE_BUFFER_RGBA32) {
    /* 24-bit unsigned char RGB, RT_IMAGE_BUFFER_RGB24 */
    int addr, R,G,B,A;
    unsigned char *img = (unsigned char *) scene->img;

#if defined(_OPENMP)
#pragma omp for schedule(runtime)
#endif
    for (y=starty; y<=stopy; y+=yinc) {
      if(scene->imgbufformat == RT_IMAGE_BUFFER_RGB24) {
        addr = hsize * (y - 1) + (3 * (startx - 1));    /* row address */
      }
      else {
        addr = hsize * (y - 1) + (4 * (startx - 1));    /* row address */
      }
      for (x=startx; x<=stopx; x+=xinc,addr+=hskip) {
        primary.frng = cachefrng; /* each pixel uses the same AO RNG seed */
        col=scene->camera.cam_ray(&primary, x, y);    /* generate ray */ 

        R = (int) (col.r * 255.0f); /* quantize float to integer */
        G = (int) (col.g * 255.0f); /* quantize float to integer */
        B = (int) (col.b * 255.0f); /* quantize float to integer */

        if (R > 255) R = 255;       /* clamp pixel value to range 0-255      */
        if (R < 0) R = 0;
        img[addr    ] = (byte) R;   /* Store final pixel to the image buffer */

        if (G > 255) G = 255;       /* clamp pixel value to range 0-255      */
        if (G < 0) G = 0;
        img[addr + 1] = (byte) G;   /* Store final pixel to the image buffer */

        if (B > 255) B = 255;       /* clamp pixel value to range 0-255      */
        if (B < 0) B = 0;
        img[addr + 2] = (byte) B;   /* Store final pixel to the image buffer */

        if(scene->imgbufformat == RT_IMAGE_BUFFER_RGBA32) {
            A = (int) (col.a * 255.0f); /* quantize float to integer */
			if (A > 255) A = 255;       /* clamp pixel value to range 0-255      */
			if (A < 0) A = 0;
			img[addr + 3] = (byte) A;   /* Store final pixel to the image buffer */
        }
      } /* end of x-loop */

      if (do_ui && !((y-1) % 16)) {
        rt_ui_progress((100 * y) / vres);  /* call progress meter callback */
      } 

#if defined(MPI)
      /* Ensure all threads have completed this row, then send it */
      node_row_sendrecv(my_tid, t, scene, &sentrows, y);
#endif
    }        /* end y-loop */
  } else {   /* end of RGB24 loop */
    /* 96-bit float RGB, RT_IMAGE_BUFFER_RGB96F */
    int addr;
    float *img = (float *) scene->img;

#if defined(_OPENMP)
#pragma omp for schedule(runtime)
#endif
    for (y=starty; y<=stopy; y+=yinc) {
      addr = hsize * (y - 1) + (3 * (startx - 1));    /* row address */
      for (x=startx; x<=stopx; x+=xinc,addr+=hskip) {
        primary.frng = cachefrng; /* each pixel uses the same AO RNG seed */
        col=scene->camera.cam_ray(&primary, x, y);    /* generate ray */ 
        img[addr    ] = col.r;   /* Store final pixel to the image buffer */
        img[addr + 1] = col.g;   /* Store final pixel to the image buffer */
        img[addr + 2] = col.b;   /* Store final pixel to the image buffer */
      } /* end of x-loop */

      if (do_ui && !((y-1) % 16)) {
        rt_ui_progress((100 * y) / vres);  /* call progress meter callback */
      } 

#if defined(MPI)
      /* Ensure all threads have completed this row, then send it */
      node_row_sendrecv(my_tid, t, scene, &sentrows, y);
#endif
    }        /* end y-loop */
  }          /* end of RGB96F loop */

  /* 
   * Image has been rendered into the buffer in the appropriate pixel format
   */
  my_serialno = primary.serial + 1;

#if defined(_OPENMP)
  /* XXX The OpenMP code needs to find a way to save serialno for next */
  /* rendering pass, otherwise we need to force-clear the mailbox */
  /* t->serialno = my_serialno; */ /* save our serialno for next launch */

  /* XXX until we save/restore serial numbers, we have to clear the */
  /* mailbox before the next rendering pass */
  if (sizeof(unsigned long) < 8) {
    memset(local_mbox, 0, sizeof(unsigned long)*scene->objgroup.numobjects);
  }

  if (local_mbox != NULL)
    free(local_mbox);
#else
  t->serialno = my_serialno; /* save our serialno for next launch */

  if (t->local_mbox == NULL) {
    if (local_mbox != NULL)
      free(local_mbox);
  }
#endif

  /* ensure all threads have completed their pixels before return */
  if (scene->nodes == 1)
    rt_thread_barrier(t->runbar, 1);
#if defined(MPI)
  else 
    node_finish_row_sendrecvs(my_tid, t, scene, &sentrows);
#endif

/* printf("node[%d] thr[%d] done! *****************************\n", scene->mynode, my_tid); */

#if defined(_OPENMP)
  }
#endif

  return(NULL);  
}

