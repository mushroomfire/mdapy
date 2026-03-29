/*
 * vol.h - Volume rendering definitions etc.
 *
 *  $Id: vol.h,v 1.7 2011/02/02 06:06:30 johns Exp $
 */

void * newscalarvol(void * intex, vector min, vector max, 
                    int xs, int ys, int zs, 
                    const char * fname, scalarvol * invol);

void  LoadVol(scalarvol *);
color scalar_volume_texture(const vector *, const texture *, ray *);

