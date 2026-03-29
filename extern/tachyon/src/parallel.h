/*
 *  parallel.h - This file contains all of the defines for doing parallel
 *               message passing and such.
 *
 *  $Id: parallel.h,v 1.15 2013/04/21 08:11:09 johns Exp $
 *
 */

int rt_par_init(int *, char ***);
int rt_par_finish(void);
int rt_mynode(void);
int rt_numnodes(void);
int rt_getcpuinfo(nodeinfo **);
void rt_barrier_sync(void);

void * rt_allocate_reqbuf(int count);
void rt_free_reqbuf(void * voidhandle);

void * rt_init_scanlinereceives(scenedef * scene);
void rt_start_scanlinereceives(void * voidhandle);
void rt_waitscanlines(void * voidhandle);
void rt_delete_scanlinereceives(void * voidhandle);
int rt_sendrecvscanline_get_totalrows(void *voidhandle);
void rt_sendrecvscanline(void * voidhandle);
