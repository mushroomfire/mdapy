/*
 * render.h - This file contains the defines for the top level functions 
 *
 * (C) Copyright 1994-2022 John E. Stone
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * $Id: render.h,v 1.5 2022/02/18 17:55:28 johns Exp $
 *
 */

void create_render_threads(scenedef * scene);
void destroy_render_threads(scenedef * scene);
void renderscene(scenedef *); 

