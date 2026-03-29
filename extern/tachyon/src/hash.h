/*
 * hash.h - This file contains prototypes for hash table code etc.
 *
 * $Id: hash.h,v 1.4 2011/02/05 08:10:11 johns Exp $
 */

#ifndef HASH_H
#define HASH_H


#ifdef __cplusplus
extern "C" {
#endif

typedef struct rt_hash_t {
  struct hash_node_t **bucket;        /**< array of hash nodes */
  int size;                           /**< size of the array */
  int entries;                        /**< number of entries in table */
  int downshift;                      /**< shift cound, used in hash function */
  int mask;                           /**< used to select bits for hashing */
} rt_hash_t;

/** 
 * Return code when a hash key is not find, or there's no collision 
 * upon insertion.
 */
#define HASH_FAIL -1

void rt_hash_init(rt_hash_t *, int);
int rt_hash_lookup (rt_hash_t *, const char *);
int rt_hash_insert (rt_hash_t *, const char *, int);
int rt_hash_delete (rt_hash_t *, const char *);
void rt_hash_destroy(rt_hash_t *);
char *rt_hash_stats (rt_hash_t *);

#ifdef __cplusplus
}
#endif

#endif
