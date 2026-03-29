/* 
 * vector.c - This file contains all of the vector arithmetic functions.
 *
 *  $Id: vector.c,v 1.8 2012/10/17 04:25:57 johns Exp $
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TACHYON_INTERNAL 1
#include "tachyon.h"
#include "macros.h"

flt VDot(const vector *a, const vector *b) {
  return (a->x*b->x + a->y*b->y + a->z*b->z);
}

void VCross(const vector * a, const vector * b, vector * c) {
  c->x = (a->y * b->z) - (a->z * b->y);
  c->y = (a->z * b->x) - (a->x * b->z);
  c->z = (a->x * b->y) - (a->y * b->x);
}

flt VLength(const vector * a) {
  return (flt) SQRT((a->x * a->x) + (a->y * a->y) + (a->z * a->z));
}

void VNorm(vector * a) {
  flt len;

  len=SQRT((a->x * a->x) + (a->y * a->y) + (a->z * a->z));
  if (len != 0.0) {
    a->x /= len;
    a->y /= len;
    a->z /= len;
  }
}

void VAdd(const vector * a, const vector * b, vector * c) {
  c->x = (a->x + b->x);
  c->y = (a->y + b->y);
  c->z = (a->z + b->z);
}
    
void VSub(const vector * a, const vector * b, vector * c) {
  c->x = (a->x - b->x);
  c->y = (a->y - b->y);
  c->z = (a->z - b->z);
}

void VAddS(flt a, const vector * A, const vector * B, vector * C) {
  C->x = (a * A->x) + B->x;
  C->y = (a * A->y) + B->y;
  C->z = (a * A->z) + B->z;
}

vector Raypnt(const ray * a, flt t) {
  vector temp;

  temp.x=a->o.x + (a->d.x * t);
  temp.y=a->o.y + (a->d.y * t);
  temp.z=a->o.z + (a->d.z * t);

  return temp;
}

void VScale(vector * a, flt s) {
  a->x *= s;
  a->y *= s;
  a->z *= s;
}

void ColorAddS(color * a, const color * b, flt s) {
  a->r += b->r * s;
  a->g += b->g * s;
  a->b += b->b * s;
}

void ColorAccum(color * a, const color * b) {
  a->r += b->r;
  a->g += b->g;
  a->b += b->b;
}

void ColorScale(color * a, flt s) {
  a->r *= s;
  a->g *= s;
  a->b *= s;
}

