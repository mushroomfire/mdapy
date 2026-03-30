/*
 * parallel.h - This file contains all of the defines for doing parallel
 *              message passing and such.
 *
 * (C) Copyright 1994-2022 John E. Stone
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * $Id: parallel.h,v 1.24 2022/02/18 17:55:28 johns Exp $
 *
 */

int rt_par_rank(rt_parhandle);
int rt_par_size(rt_parhandle);

rt_parhandle rt_par_init_nompi(void);
rt_parhandle rt_par_init(int *, char ***);
rt_parhandle rt_par_init_mpi_comm_world(void);
rt_parhandle rt_par_init_mpi_comm(void * mpicomm);
rt_parhandle rt_par_init_mpi_comm_split(void * mpicomm, int color, int key);

int rt_par_set_mpi_comm_world(rt_parhandle);
int rt_par_set_mpi_comm_world_split(rt_parhandle, int color, int key);
int rt_par_set_mpi_comm_world_split_all(rt_parhandle);
int rt_par_set_mpi_comm(rt_parhandle, void * mpicomm);
int rt_par_set_mpi_comm_split(rt_parhandle, void * mpicomm, int color, int key);

int rt_par_finish(rt_parhandle);

int rt_par_getcpuinfo(rt_parhandle, nodeinfo **);
void rt_par_barrier_sync(rt_parhandle);

void * rt_par_allocate_reqbuf(rt_parhandle, int count);
void rt_par_free_reqbuf(rt_parhandle, rt_parbuf);

void * rt_par_init_scanlinereceives(rt_parhandle, scenedef * scene);
void rt_par_start_scanlinereceives(rt_parhandle, rt_parbuf);
void rt_par_waitscanlines(rt_parhandle, rt_parbuf);
void rt_par_delete_scanlinereceives(rt_parhandle, rt_parbuf);
int rt_par_sendrecvscanline_get_totalrows(rt_parhandle, rt_parbuf);
void rt_par_sendrecvscanline(rt_parhandle, rt_parbuf);
