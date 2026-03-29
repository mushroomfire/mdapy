/*
 * ui.c - Contains functions for dealing with user interfaces
 *
 *  $Id: ui.c,v 1.9 2011/02/07 07:41:51 johns Exp $
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TACHYON_INTERNAL 1
#include "tachyon.h"
#include "macros.h"
#include "util.h"
#include "ui.h"

static void (* rt_static_ui_message) (int, char *) = NULL;
static void (* rt_static_ui_progress) (int) = NULL;
static int (* rt_static_ui_checkaction) (void) = NULL;

void rt_set_ui_message(void (* func) (int, char *)) {
  rt_static_ui_message = func;
}

void rt_set_ui_progress(void (* func) (int)) {
  rt_static_ui_progress = func;
}

void rt_ui_message(int level, char * msg) {
  if (rt_static_ui_message != NULL) 
    rt_static_ui_message(level, msg);
}

void rt_ui_progress(int percent) {
  if (rt_static_ui_progress != NULL)
    rt_static_ui_progress(percent);
}

int rt_ui_checkaction(void) {
  if (rt_static_ui_checkaction != NULL) 
    return rt_static_ui_checkaction();
  else
    return 0;
}














