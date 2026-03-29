/*
 * render.h - This file contains the defines for the top level functions 
 *
 *  $Id: render.h,v 1.4 1998/07/26 06:43:09 johns Exp $
 */

void create_render_threads(scenedef * scene);
void destroy_render_threads(scenedef * scene);
void renderscene(scenedef *); 
void rendercheck(scenedef *);
