/*
 * parallel.c - This file contains all of the code for doing parallel
 *              message passing and such.
 *
 * (C) Copyright 1994-2022 John E. Stone
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * $Id: parallel.c,v 1.63 2022/02/18 17:55:28 johns Exp $
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

typedef struct {
  int totalrows;
  int count;
  int curmsg;
  int haveinited;
  int havestarted;
  MPI_Request * requests;
  MPI_Status * statuses;
  int * indices;
} pardata;  

/* 
 * Check to see if we have to pass the MPI_IN_PLACE flag 
 * for our allgather reductions during startup.
 */
#if !defined(USE_MPI_IN_PLACE)
#if (MPI_VERSION >= 2) || defined(MPI_IN_PLACE)
#define USE_MPI_IN_PLACE 1
#endif
#endif 

#endif /* MPI */


typedef struct {
  int mpienabled;   /* Whether MPI is enabled or not at runtime          */
  int mpi_client;   /* Whether or not Tachyon initialized MPI for itself */
  int owns_comm;    /* Whether we created the communicator or not        */
#ifdef MPI
  MPI_Comm comm;    /* Our MPI communicator se                           */
  int color;        /* Param given to MPI_Comm_split()                   */
  int key;          /* Param given to MPI_Comm_split()                   */
#endif
  int worldrank;    /* our rank within the MPI_COMM_WORLD                */ 
  int worldsize;    /* size of MPI_COMM_WORLD                            */
  int callrank;     /* our rank within the calling code's communicator   */ 
  int callsize;     /* size of calling communicator                      */
  int commrank;     /* our rank within our own sub-communicator          */ 
  int commsize;     /* size of our communicator                          */ 
} parhandle;


static void rt_par_comm_default(parhandle *ph) {
  if (ph != NULL) {
    ph->mpienabled=0;
#ifdef MPI
    ph->mpienabled=1;
    ph->comm = MPI_COMM_WORLD; /* Use global communicator by default */
#endif
    ph->mpi_client = 0; /* Tachyon initialized MPI for itself  */
    ph->owns_comm = 1;  /* we own the communicator we're using */
    ph->worldrank = 0;  /* we are rank 0 unless MPI is used    */
    ph->worldsize = 1;  /* group has size 1 unless MPI is used */
    ph->callrank = 0;   /* we are rank 0 unless MPI is used    */
    ph->callsize = 1;   /* group has size 1 unless MPI is used */
    ph->commrank = 0;   /* we are rank 0 unless MPI is used    */
    ph->commsize = 1;   /* group has size 1 unless MPI is used */
  }
}

#ifdef MPI
static void rt_par_comm_info(parhandle *ph, MPI_Comm *caller_comm) {
  if (ph != NULL) {
    /* record this node's rank among various communicators */
    MPI_Comm_rank(MPI_COMM_WORLD, &ph->worldrank);
    MPI_Comm_size(MPI_COMM_WORLD, &ph->worldsize);
    MPI_Comm_rank(*caller_comm, &ph->callrank);
    MPI_Comm_size(*caller_comm, &ph->callsize);
    MPI_Comm_rank(ph->comm, &ph->commrank);
    MPI_Comm_size(ph->comm, &ph->commsize);
  }
}
#endif


rt_parhandle rt_par_init_nompi(void) {
  parhandle *ph = (parhandle *) calloc(1, sizeof(parhandle));
  rt_par_comm_default(ph);
  ph->mpienabled=0; /* disable MPI for this run */
  return ph;
}


rt_parhandle rt_par_init(int * argc, char ***argv) {
  parhandle *ph = (parhandle *) calloc(1, sizeof(parhandle));
  rt_par_comm_default(ph);

#ifdef MPI
  MPI_Init(argc, argv);
  rt_par_comm_default(ph);   /* Reset to known default state                */
  ph->mpi_client = 0;        /* Tachyon initialized MPI itself              */
  ph->owns_comm  = 0;        /* We're using MPI_COMM_WORLD, don't delete it */
  ph->comm = MPI_COMM_WORLD; /* Use global communicator by default          */
  rt_par_comm_info(ph, &ph->comm);
#endif

  return ph;
}


rt_parhandle rt_par_init_mpi_comm(void * mpicomm) {
#ifdef MPI
  MPI_Comm *caller_comm = (MPI_Comm *) mpicomm;
  if (caller_comm != NULL) {
    parhandle *ph=(parhandle *) calloc(1, sizeof(parhandle));
    rt_par_comm_default(ph); /* Reset to known default state                */
    ph->mpi_client = 1;      /* Tachyon is a client of a calling MPI code   */
    ph->owns_comm  = 0;      /* Caller created the communicator we're using */
    ph->comm = *caller_comm; /* Use caller-provided communicator            */
    rt_par_comm_info(ph, &ph->comm);

    return ph;
  }
#endif

  return NULL; /* not supported for non-MPI builds */
}


rt_parhandle rt_par_init_mpi_comm_world(void) {
#ifdef MPI
  MPI_Comm comm = MPI_COMM_WORLD; 
  return rt_par_init_mpi_comm(&comm);
#endif
  return NULL; /* not supported for non-MPI builds */
}


rt_parhandle rt_par_init_mpi_comm_split(void * mpicomm, int color, int key) {
#ifdef MPI
  MPI_Comm *caller_comm = (MPI_Comm *) mpicomm;
  if (caller_comm != NULL) {
    parhandle *ph=(parhandle *) calloc(1, sizeof(parhandle));
    rt_par_comm_default(ph); /* Reset to known default state */
    ph->mpi_client = 1;      /* Tachyon is a client of a calling MPI code   */
    ph->owns_comm  = 1;      /* Tachyon created the communicator it's using */
    MPI_Comm_split(*caller_comm, color, key, &ph->comm);
    rt_par_comm_info(ph, caller_comm);

    return ph;
  }
#endif

  return NULL; /* not supported for non-MPI builds */
}


int rt_par_set_mpi_comm(rt_parhandle voidhandle, void * mpicomm) {
#ifdef MPI
  parhandle *ph=(parhandle *) voidhandle;
  if (ph->mpienabled) {
    MPI_Comm *caller_comm = (MPI_Comm *) mpicomm;
    if (caller_comm != NULL) {
      /* If Tachyon is a client library within a larger app          */
      /* then we clean up any sub-communicators we may have created. */
      if (ph->mpi_client && ph->owns_comm) {
        MPI_Comm_free(&ph->comm);
      }

      rt_par_comm_default(ph); /* Reset to known default state                */
      ph->mpi_client = 1;      /* Tachyon is a client of a calling MPI code   */
      ph->owns_comm  = 0;      /* Caller created the communicator we're using */
      ph->comm = *caller_comm; /* Use caller-provided communicator            */
      rt_par_comm_info(ph, &ph->comm);

      return 0;
    }
  }
#endif

  return -1; /* not supported for non-MPI builds */
}


int rt_par_set_mpi_comm_world(rt_parhandle voidhandle) {
#ifdef MPI
  parhandle *ph=(parhandle *) voidhandle;
  if (!ph->mpienabled) {
    MPI_Comm comm = MPI_COMM_WORLD;
    return rt_par_set_mpi_comm(voidhandle, &comm);
  }
#endif
  
  return -1; /* not supported for non-MPI builds */
}


int rt_par_set_mpi_comm_split(rt_parhandle voidhandle, void * mpicomm,
                              int color, int key) {
#ifdef MPI
  parhandle *ph=(parhandle *) voidhandle;
  if (ph->mpienabled) {
    MPI_Comm *caller_comm = (MPI_Comm *) mpicomm;
    if (caller_comm != NULL) {
      /* If Tachyon is a client library within a larger app          */
      /* then we clean up any sub-communicators we may have created. */
      if (ph->mpi_client && ph->owns_comm) {
        MPI_Comm_free(&ph->comm);
      }

      rt_par_comm_default(ph); /* Reset to known default state                */
      ph->mpi_client = 1;      /* Tachyon is a client of a calling MPI code   */
      ph->owns_comm  = 1;      /* Tachyon created the communicator it's using */
      MPI_Comm_split(*caller_comm, color, key, &ph->comm);
      rt_par_comm_info(ph, caller_comm);

      return 0;
    }
  }
#endif

  return -1; /* not supported for non-MPI builds */
}


int rt_par_set_mpi_comm_world_split(rt_parhandle voidhandle,
                                    int color, int key) {
#ifdef MPI
  parhandle *ph=(parhandle *) voidhandle;
  if (ph->mpienabled) {
    MPI_Comm comm = MPI_COMM_WORLD;
    return rt_par_set_mpi_comm_split(voidhandle, &comm, color, key);
  }
#endif
  
  return -1; /* not supported for non-MPI builds */
}


int rt_par_set_mpi_comm_world_split_all(rt_parhandle voidhandle) {
#ifdef MPI
  parhandle *ph=(parhandle *) voidhandle;
  if (ph->mpienabled) {
    int myrank;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &myrank); 
    return rt_par_set_mpi_comm_split(voidhandle, &comm, myrank, 0);
  }
#endif
  
  return -1; /* not supported for non-MPI builds */
}


int rt_par_finish(rt_parhandle voidhandle) {
  parhandle *ph = (parhandle *) voidhandle;
  int a=0; /* if sequential, do nothing */ 

  if (ph == NULL)
    return -1;

#ifdef MPI
  if (ph->mpienabled) {
    /* If Tachyon is a client library within a larger app */
    /* then we only clean up any sub-communicators we may */
    /* have created and free up data.  If Tachyon is not  */
    /* a client, but is in charge of MPI, then we have to */
    /* teardown everything at this point.                 */
    if (ph->mpi_client && ph->owns_comm) {
      MPI_Comm_free(&ph->comm);
    } else {
      free(ph); /* free the handle before calling MPI_Finalize() */
      ph=NULL;

      /* If Tachyon initialized MPI for itself, */
      /* then it must also shutdown MPI itself. */
      MPI_Finalize();
    }
  }
#endif  

  if (ph != NULL) 
    free(ph);

  return a;
}

int rt_par_rank(rt_parhandle voidhandle) {
  parhandle *ph = (parhandle *) voidhandle;
  return ph->commrank;
}

int rt_par_size(rt_parhandle voidhandle) {
  parhandle *ph = (parhandle *) voidhandle;
  return ph->commsize;
}

void rt_par_barrier_sync(rt_parhandle voidhandle) {
  /* if sequential, do nothing */
#ifdef MPI
  parhandle *ph = (parhandle *) voidhandle;
  if (ph->mpienabled)
    MPI_Barrier(ph->comm);
#endif
}

int rt_par_getcpuinfo(rt_parhandle voidhandle, nodeinfo **nodes) {
  parhandle *ph = (parhandle *) voidhandle;
  int numnodes = ph->commsize;
  int mynode = ph->commrank; 
#ifdef MPI
  int namelen; 
  char namebuf[MPI_MAX_PROCESSOR_NAME];
#endif
 
  *nodes = (nodeinfo *) malloc(numnodes * sizeof(nodeinfo));
  (*nodes)[mynode].numcpus = rt_thread_numprocessors(); 
  (*nodes)[mynode].cpuspeed = 1.0; 
  (*nodes)[mynode].nodespeed = (*nodes)[mynode].numcpus * 
                                  (*nodes)[mynode].cpuspeed; 
  (*nodes)[mynode].cpucaps = NULL;

#ifdef MPI
  if (ph->mpienabled) {
    MPI_Get_processor_name((char *) &namebuf, &namelen);
    strncpy((char *) &(*nodes)[mynode].machname, namebuf,
            (((namelen + 1) < 511) ? (namelen+1) : 511));
#if defined(USE_MPI_IN_PLACE)
    MPI_Allgather(MPI_IN_PLACE, sizeof(nodeinfo), MPI_BYTE, 
                  &(*nodes)[     0], sizeof(nodeinfo), MPI_BYTE, 
                  ph->comm);
#else
    MPI_Allgather(&(*nodes)[mynode], sizeof(nodeinfo), MPI_BYTE, 
                  &(*nodes)[     0], sizeof(nodeinfo), MPI_BYTE, 
                  ph->comm);
#endif
  } else
#endif
  {
#if defined(_MSC_VER)
    strcpy((*nodes)[mynode].machname, "Windows");
#elif defined(MCOS)
    strcpy((*nodes)[mynode].machname, "Mercury");
#else
    gethostname((*nodes)[mynode].machname, 511);
#endif
    (*nodes)[mynode].cpucaps = calloc(1, sizeof(rt_cpu_caps_t));
    if (rt_cpu_capability_flags((rt_cpu_caps_t *) (*nodes)[mynode].cpucaps) != 0) {
      free((*nodes)[mynode].cpucaps);
    }
  }

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


void * rt_par_allocate_reqbuf(rt_parhandle voidhandle, int count) {
#ifdef MPI
  parhandle * ph = (parhandle *) voidhandle;
  if (ph->mpienabled) {
    pardata * p;
    p = malloc(sizeof(pardata));
    p->totalrows = 0;
    p->count = 0;
    p->curmsg = 0;
    p->haveinited = 0;
    p->havestarted = 0;
    p->requests = malloc(sizeof(MPI_Request)*count);
    p->statuses = malloc(sizeof(MPI_Status)*count);
    p->indices  = malloc(sizeof(int)*count);
    return p;
  }
#endif
  return NULL;
}

void rt_par_free_reqbuf(rt_parhandle voidparhandle, rt_parbuf voidhandle) {
#ifdef MPI
  parhandle * ph = (parhandle *) voidparhandle;
  if (ph->mpienabled) {
    pardata *p = (pardata *) voidhandle;

    if (p->requests != NULL)
      free(p->requests);

    if (p->statuses != NULL)
      free(p->statuses);

    if (p->indices != NULL)
      free(p->indices);

    if (p != NULL)
      free(p);
  }
#endif
}


void * rt_par_init_scanlinereceives(rt_parhandle voidhandle, scenedef * scene) {
#ifdef MPI
  parhandle *ph = (parhandle *) voidhandle;
  if (ph->mpienabled) {
    int i, addr;
    pardata *p = (pardata *) rt_par_allocate_reqbuf(voidhandle, scene->vres);

    p->curmsg = 0;
    p->totalrows = 0;
    p->count = 0;
    p->haveinited = 1;

    if (scene->imgbufformat == RT_IMAGE_BUFFER_RGB24) {
      /* 24-bit RGB packed pixel format */
      unsigned char *imgbuf = (unsigned char *) scene->img;

      if (ph->commrank == 0) {
        for (i=0; i<scene->vres; i++) {
          if (i % ph->commsize != ph->commrank) {
            addr = i * scene->hres * 3;
            MPI_Recv_init(&imgbuf[addr], scene->hres * 3, MPI_BYTE, 
                      i % ph->commsize, i+1, ph->comm, &p->requests[p->count]);
            p->count++; /* count of received rows */
          } else {
            p->totalrows++; /* count of our own rows */
          }
        }
      } else {
        for (i=0; i<scene->vres; i++) {
          if (i % ph->commsize == ph->commrank) {
            addr = i * scene->hres * 3;
            MPI_Send_init(&imgbuf[addr], scene->hres * 3, MPI_BYTE, 
                      0, i+1, ph->comm, &p->requests[p->count]);
            p->count++; /* count of sent rows */
            p->totalrows++; /* count of sent rows */
          }
        }
      }
    } else if (scene->imgbufformat == RT_IMAGE_BUFFER_RGB96F) {
      /* 96-bit float RGB packed pixel format */
      float *imgbuf = (float *) scene->img;

      if (ph->commrank == 0) {
        for (i=0; i<scene->vres; i++) {
          if (i % ph->commsize != ph->commrank) {
            addr = i * scene->hres * 3;
            MPI_Recv_init(&imgbuf[addr], scene->hres * 3, MPI_FLOAT, 
                      i % ph->commsize, i+1, ph->comm, &p->requests[p->count]);
            p->count++; /* count of received rows */
          } else {
            p->totalrows++; /* count of our own rows */
          }
        }
      } else {
        for (i=0; i<scene->vres; i++) {
          if (i % ph->commsize == ph->commrank) {
            addr = i * scene->hres * 3;
            MPI_Send_init(&imgbuf[addr], scene->hres * 3, MPI_FLOAT,
                      0, i+1, ph->comm, &p->requests[p->count]);
            p->count++; /* count of sent rows */
            p->totalrows++; /* count of sent rows */
          }
        }
      }
    }
 
    return p;
  }
#endif

  return NULL;
}


void rt_par_start_scanlinereceives(rt_parhandle voidparhandle, rt_parbuf voidhandle) {
#ifdef MPI
  parhandle *ph = (parhandle *) voidparhandle;
  if (ph->mpienabled) {
    pardata *p = (pardata *) voidhandle;

    p->havestarted = 1;
    if (ph->commrank == 0)
      MPI_Startall(p->count, p->requests);

    p->curmsg = 0;
  }
#endif
}

void rt_par_waitscanlines(rt_parhandle voidparhandle, rt_parbuf voidhandle) {
#ifdef MPI
  parhandle *ph = (parhandle *) voidparhandle;
  if (ph->mpienabled) {
    pardata *p = (pardata *) voidhandle;
  
    MPI_Waitall(p->count, p->requests, p->statuses);

    p->havestarted=0;
  }
#endif
}  

void rt_par_delete_scanlinereceives(rt_parhandle voidparhandle, rt_parbuf voidhandle) {
#ifdef MPI
  parhandle *ph = (parhandle *) voidparhandle;
  if (ph->mpienabled) {
    int i;
    pardata *p = (pardata *) voidhandle;

    if (p == NULL)
      return; /* don't bomb if no valid handle */

    if (p->haveinited != 0 || p->havestarted != 0) {
      for (i=0; i<p->count; i++) {
        MPI_Request_free(&p->requests[i]);  
      }
    }

    rt_par_free_reqbuf(voidparhandle, voidhandle);
  }
#endif
}

int rt_par_sendrecvscanline_get_totalrows(rt_parhandle voidparhandle, rt_parbuf voidhandle) {
#ifdef MPI
  parhandle *ph = (parhandle *) voidparhandle;
  if (ph->mpienabled) {
    pardata *p = (pardata *) voidhandle;
    return p->totalrows;
  }
#endif

  return 0;
}

void rt_par_sendrecvscanline(rt_parhandle voidparhandle, rt_parbuf voidhandle) {
#ifdef MPI
  parhandle *ph = (parhandle *) voidparhandle;
  if (ph->mpienabled) {
    pardata *p = (pardata *) voidhandle;

    if (ph->commrank == 0) {
#if MPI_TUNE == 0  || !defined(MPI_TUNE)
      /* 
       * Default Technique 
       */
      int outcount;
      int numtorecv = ph->commsize - 1; /* all nodes but node 0 */
 
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
  }
#endif
}


