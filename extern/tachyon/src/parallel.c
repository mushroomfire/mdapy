/*
 *  parallel.c - This file contains all of the code for doing parallel
 *               message passing and such.
 *
 *  $Id: parallel.c,v 1.47 2013/04/21 08:11:09 johns Exp $
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TACHYON_INTERNAL 1
#include "tachyon.h"
#include "macros.h"
#include "parallel.h"
#include "tgafile.h"
#include "util.h"
#include "threads.h"

#if !defined(_MSC_VER)
#include <unistd.h>
#endif

#ifdef MPI
#include <mpi.h>
#endif

int rt_par_init(int * argc, char ***argv) {
  int a=0;  /* if sequential, do nothing */

#ifdef MPI
  MPI_Init(argc, argv);
#endif

  a = rt_mynode();

  return a;
}

int rt_par_finish(void) {
  int a=0; /* if sequential, do nothing */ 

#ifdef MPI
  MPI_Finalize();
#endif  

  return a;
}

int rt_mynode(void) {
  int a=0;   /* for non-parallel machines */

#ifdef MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &a);
#endif

  return a;
}

int rt_numnodes(void) {
  int a=1;  /* for non-parallel machines */

#ifdef MPI
  MPI_Comm_size(MPI_COMM_WORLD, &a);
#endif

  return a;
}

void rt_barrier_sync(void) {
  /* if sequential, do nothing */
#ifdef MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
}

int rt_getcpuinfo(nodeinfo **nodes) {
  int numnodes = rt_numnodes();
  int mynode = rt_mynode();
#ifdef MPI
  int namelen; 
  char namebuf[MPI_MAX_PROCESSOR_NAME];
#endif
 
  *nodes = (nodeinfo *) malloc(numnodes * sizeof(nodeinfo));
  (*nodes)[mynode].numcpus = rt_thread_numprocessors(); 
  (*nodes)[mynode].cpuspeed = 1.0; 
  (*nodes)[mynode].nodespeed = (*nodes)[mynode].numcpus * 
                                  (*nodes)[mynode].cpuspeed; 


#ifdef MPI
  MPI_Get_processor_name((char *) &namebuf, &namelen);
  strncpy((char *) &(*nodes)[mynode].machname, namebuf,
          (((namelen + 1) < 511) ? (namelen+1) : 511));
#if defined(USE_MPI_IN_PLACE)
  MPI_Allgather(MPI_IN_PLACE, sizeof(nodeinfo), MPI_BYTE, 
                &(*nodes)[     0], sizeof(nodeinfo), MPI_BYTE, 
                MPI_COMM_WORLD);
#else
  MPI_Allgather(&(*nodes)[mynode], sizeof(nodeinfo), MPI_BYTE, 
                &(*nodes)[     0], sizeof(nodeinfo), MPI_BYTE, 
                MPI_COMM_WORLD);
#endif
#else
#if defined(_MSC_VER)
  strcpy((*nodes)[mynode].machname, "Windows");
#elif defined(MCOS)
  strcpy((*nodes)[mynode].machname, "Mercury");
#else
  gethostname((*nodes)[mynode].machname, 511);
#endif
#endif

  return numnodes;
}



/* 
 * Communications implementation based on MPI persistent send/recv operations.
 *
 * MPI Request buffers are allocated
 *
 * Persistent Send/Recv channels are initialized for the scanlines in 
 *   the image(s) to be rendered.
 *
 * For each frame, the persistent communications are used once.
 *
 * After all frames are rendered, the persistent channels are closed down
 *   and the MPI Request buffers are freed.
 */

#ifdef MPI

typedef struct {
  int mynode;
  int nodes;
  int totalrows;
  int count;
  int curmsg;
  int haveinited;
  int havestarted;
  MPI_Request * requests;
  MPI_Status * statuses;
  int * indices;
} pardata;  

#endif

void * rt_allocate_reqbuf(int count) {
#ifdef MPI
  pardata * p;
  p = malloc(sizeof(pardata));
  p->mynode = rt_mynode();
  p->nodes = rt_numnodes();
  p->totalrows = 0;
  p->count = 0;
  p->curmsg = 0;
  p->haveinited = 0;
  p->havestarted = 0;
  p->requests = malloc(sizeof(MPI_Request)*count);
  p->statuses = malloc(sizeof(MPI_Status)*count);
  p->indices  = malloc(sizeof(int)*count);
  return p;
#else 
  return NULL;
#endif
}

void rt_free_reqbuf(void * voidhandle) {
#ifdef MPI
  pardata * p = (pardata *) voidhandle;

  if (p->requests != NULL)
    free(p->requests);

  if (p->statuses != NULL)
    free(p->statuses);

  if (p->indices != NULL)
    free(p->indices);

  if (p != NULL)
    free(p);
#endif
}


void * rt_init_scanlinereceives(scenedef * scene) {
#ifdef MPI
  int i, addr;
  pardata * p;

  p = (pardata *) rt_allocate_reqbuf(scene->vres);

  p->curmsg = 0;
  p->totalrows = 0;
  p->count = 0;
  p->haveinited = 1;

  if (scene->imgbufformat == RT_IMAGE_BUFFER_RGB24) {
    /* 24-bit RGB packed pixel format */
    unsigned char *imgbuf = (unsigned char *) scene->img;

    if (p->mynode == 0) {
      for (i=0; i<scene->vres; i++) {
        if (i % p->nodes != p->mynode) {
          addr = i * scene->hres * 3;
          MPI_Recv_init(&imgbuf[addr], scene->hres * 3, MPI_BYTE, 
                    i % p->nodes, i+1, MPI_COMM_WORLD, &p->requests[p->count]);
          p->count++; /* count of received rows */
        } else {
          p->totalrows++; /* count of our own rows */
        }
      }
    } else {
      for (i=0; i<scene->vres; i++) {
        if (i % p->nodes == p->mynode) {
          addr = i * scene->hres * 3;
          MPI_Send_init(&imgbuf[addr], scene->hres * 3, MPI_BYTE, 
                    0, i+1, MPI_COMM_WORLD, &p->requests[p->count]);
          p->count++; /* count of sent rows */
          p->totalrows++; /* count of sent rows */
        }
      }
    }
  } else if (scene->imgbufformat == RT_IMAGE_BUFFER_RGB96F) {
    /* 96-bit float RGB packed pixel format */
    float *imgbuf = (float *) scene->img;

    if (p->mynode == 0) {
      for (i=0; i<scene->vres; i++) {
        if (i % p->nodes != p->mynode) {
          addr = i * scene->hres * 3;
          MPI_Recv_init(&imgbuf[addr], scene->hres * 3, MPI_FLOAT, 
                    i % p->nodes, i+1, MPI_COMM_WORLD, &p->requests[p->count]);
          p->count++; /* count of received rows */
        } else {
          p->totalrows++; /* count of our own rows */
        }
      }
    } else {
      for (i=0; i<scene->vres; i++) {
        if (i % p->nodes == p->mynode) {
          addr = i * scene->hres * 3;
          MPI_Send_init(&imgbuf[addr], scene->hres * 3, MPI_FLOAT,
                    0, i+1, MPI_COMM_WORLD, &p->requests[p->count]);
          p->count++; /* count of sent rows */
          p->totalrows++; /* count of sent rows */
        }
      }
    }
  }
 
  return p;
#else
  return NULL;
#endif
}


void rt_start_scanlinereceives(void * voidhandle) {
#ifdef MPI
  pardata * p = (pardata *) voidhandle;

  p->havestarted = 1;
  if (p->mynode == 0)
    MPI_Startall(p->count, p->requests);

  p->curmsg = 0;
#endif
}

void rt_waitscanlines(void * voidhandle) {
#ifdef MPI
  pardata * p = (pardata *) voidhandle;
  
  MPI_Waitall(p->count, p->requests, p->statuses);

  p->havestarted=0;
#endif
}  

void rt_delete_scanlinereceives(void * voidhandle) {
#ifdef MPI
  int i;
  pardata * p = (pardata *) voidhandle;

  if (p == NULL)
    return; /* don't bomb if no valid handle */

  if (p->haveinited != 0 || p->havestarted != 0) {
    for (i=0; i<p->count; i++) {
      MPI_Request_free(&p->requests[i]);  
    }
  }

  rt_free_reqbuf(voidhandle);
#endif
}

int rt_sendrecvscanline_get_totalrows(void *voidhandle) {
#ifdef MPI
  pardata * p = (pardata *) voidhandle;
  return p->totalrows;
#else
  return 0;
#endif
}

void rt_sendrecvscanline(void * voidhandle) {
#ifdef MPI
  pardata * p = (pardata *) voidhandle;

  if (p->mynode == 0) {
#if   MPI_TUNE == 0  || !defined(MPI_TUNE)
    /* 
     * Default Technique 
     */
    int outcount;
    int numtorecv = p->nodes - 1; /* all nodes but node 0 */
 
    int numtotest = (numtorecv < (p->count - p->curmsg)) ?
                     numtorecv : (p->count - p->curmsg);

    if (numtotest < 1) {
      printf("Internal Tachyon MPI error, tried to recv zero/negative count!\n");
      return;
    }

    MPI_Testsome(numtotest, &p->requests[p->curmsg], &outcount,
                 &p->indices[p->curmsg], &p->statuses[p->curmsg]);
    p->curmsg += numtorecv;
#elif MPI_TUNE == 1
    /* 
     * Technique number 1
     */
    int index, flag;
    MPI_Testany(p->count, p->requests, &index, &flag, p->statuses);
#elif MPI_TUNE == 2
    /* 
     * Technique number 2 
     */
    int flag;
    MPI_Testall(p->count, p->requests, &flag, p->statuses);
#elif MPI_TUNE == 3
    /* 
     * Technique number 3 
     */
    int i, index, flag;
    for (i=1; i<p->nodes; i++)
      MPI_Testany(p->count, p->requests, &index, &flag, p->statuses);
#endif
  } else {
    if (p->curmsg >= p->count) {
      printf("Internal Tachyon MPI error, tried to send oob count!\n");
      return;
    }
    MPI_Start(&p->requests[p->curmsg]);
    p->curmsg++;
  }
#endif
}


