/*
 * global.h - any/all global data items etc should be in this file
 *
 * (C) Copyright 1994-2022 John E. Stone
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * $Id: global.h,v 1.21 2022/02/18 17:55:28 johns Exp $
 *
 */

extern rt_parhandle global_parhnd;  /**< parallel message passing data structures */
extern rawimage * global_imagelist[MAXIMGS]; /**< texture map cache */
extern int global_numimages;

extern void (* global_rt_ui_message) (int, char *);
extern void (* global_rt_ui_progress) (int);
extern int (* global_rt_ui_checkaction) (void);


