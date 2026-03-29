/*
 * util.c - Contains all of the timing functions for various platforms.
 *
 *  $Id: util.c,v 1.63 2011/02/07 15:20:39 johns Exp $ 
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#define TACHYON_INTERNAL 1
#include "tachyon.h"
#include "macros.h"
#include "util.h"
#include "parallel.h"
#include "ui.h"

#if defined(__PARAGON__) || defined(__IPSC__)
#if defined(__IPSC__)
#include <cube.h>
#endif     /* iPSC/860 specific */

#if defined(__PARAGON__)
#include <nx.h>
#endif     /* Paragon XP/S specific */

#include <estat.h>
#endif /* iPSC/860 and Paragon specific items */

/* most platforms will use the regular time function gettimeofday() */
#if !defined(__IPSC__) && !defined(__PARAGON__) && !defined(NEXT)
#define STDTIME
#endif

#if defined(NEXT) 
#include <time.h>
#undef STDTIME
#define OLDUNIXTIME
#endif

#if defined(_MSC_VER) || defined(WIN32)
#include <windows.h>
#undef STDTIME
#define WIN32GETTICKCOUNT
#endif

void rt_finalize(void); /* UGLY! tachyon.h needs more cleanup before it can */
                        /* be properly included without risk of bogosity    */

#if defined(__linux) || defined(Bsd) || defined(AIX) || defined(__APPLE__) || defined(__sun) || defined(__hpux) || defined(_CRAYT3E) || defined(_CRAY) || defined(_CRAYC) || defined(__osf__) || defined(__BEOS__) || defined(__CYGWIN__)
#include <sys/time.h>
#endif

#if defined(MCOS) || defined(VXWORKS)
#define POSIXTIME
#endif


#if defined(WIN32GETTICKCOUNT)
typedef struct {
  DWORD starttime;
  DWORD endtime;
} rt_timer;

void rt_timer_start(rt_timerhandle v) {
  rt_timer * t = (rt_timer *) v;
  t->starttime = GetTickCount();
}

void rt_timer_stop(rt_timerhandle v) {
  rt_timer * t = (rt_timer *) v;
  t->endtime = GetTickCount();
}

double rt_timer_time(rt_timerhandle v) {
  rt_timer * t = (rt_timer *) v;
  double ttime;

  ttime = ((double) (t->endtime - t->starttime)) / 1000.0;

  return ttime;
}
#endif


#if defined(POSIXTIME)
#undef STDTIME
#include <time.h>

typedef struct {
  struct timespec starttime;
  struct timespec endtime;
} rt_timer;

void rt_timer_start(rt_timerhandle v) {
  rt_timer * t = (rt_timer *) v;
  clock_gettime(CLOCK_REALTIME, &t->starttime);
}

void rt_timer_stop(rt_timerhandle v) {
  rt_timer * t = (rt_timer *) v;
  clock_gettime(CLOCK_REALTIME, &t->endtime);
}

double rt_timer_time(rt_timerhandle v) {
  rt_timer * t = (rt_timer *) v;
  double ttime;
  ttime = ((double) (t->endtime.tv_sec - t->starttime.tv_sec)) +
          ((double) (t->endtime.tv_nsec - t->starttime.tv_nsec)) / 1000000000.0;
  return ttime;
}
#endif



/* if we're running on a Paragon or iPSC/860, use mclock() hi res timers */
#if defined(__IPSC__) || defined(__PARAGON__)

typedef struct {
  long starttime;
  long stoptime;
} rt_timer;

void rt_timer_start(rt_timerhandle v) {
  rt_timer * t = (rt_timer *) v;
  t->starttime=mclock(); 
}

void rt_timer_stop(rt_timerhandle v) {
  rt_timer * t = (rt_timer *) v;
  t->stoptime=mclock();
}

double rt_timer_time(rt_timerhandle v) {
  rt_timer * t = (rt_timer *) v;
  double ttime; 
  ttime = ((double) t->stoptime - t->starttime) / 1000.0;
  return ttime;
}
#endif



/* if we're on a Unix with gettimeofday() we'll use newer timers */
#ifdef STDTIME 
typedef struct {
  struct timeval starttime, endtime;
#ifndef VMS
  struct timezone tz;
#endif
} rt_timer;

void rt_timer_start(rt_timerhandle v) {
  rt_timer * t = (rt_timer *) v;
#ifdef VMS
  gettimeofday(&t->starttime, NULL);
#else
  gettimeofday(&t->starttime, &t->tz);
#endif
} 
  
void rt_timer_stop(rt_timerhandle v) {
  rt_timer * t = (rt_timer *) v;
#ifdef VMS
  gettimeofday(&t->endtime, NULL);
#else
  gettimeofday(&t->endtime, &t->tz);
#endif
} 
  
double rt_timer_time(rt_timerhandle v) {
  rt_timer * t = (rt_timer *) v;
  double ttime;
  ttime = ((double) (t->endtime.tv_sec - t->starttime.tv_sec)) +
          ((double) (t->endtime.tv_usec - t->starttime.tv_usec)) / 1000000.0;
  return ttime;
}  
#endif



/* use the old fashioned Unix time functions */
#ifdef OLDUNIXTIME
typedef struct {
  time_t starttime;
  time_t stoptime;
} rt_timer;

void rt_timer_start(rt_timerhandle v) {
  rt_timer * t = (rt_timer *) v;
  time(&t->starttime);
}

void rt_timer_stop(rt_timerhandle v) {
  rt_timer * t = (rt_timer *) v;
  time(&t->stoptime);
}

double rt_timer_time(rt_timerhandle v) {
  rt_timer * t = (rt_timer *) v;
  double ttime;
  ttime = difftime(t->stoptime, t->starttime);
  return ttime;
}
#endif


/* 
 * system independent routines to create and destroy timers 
 */
rt_timerhandle rt_timer_create(void) {
  rt_timer * t;  
  t = (rt_timer *) malloc(sizeof(rt_timer));
  memset(t, 0, sizeof(rt_timer));
  return t;
}

void rt_timer_destroy(rt_timerhandle v) {
  free(v);
}

double rt_timer_timenow(rt_timerhandle v) {
  rt_timer_stop(v);
  return rt_timer_time(v);
}



/*
 * Code for machines with deficient libc's etc.
 */

#if defined(__IPSC__) && !defined(__PARAGON__) 

/* the iPSC/860 libc is *missing* strstr(), so here it is.. */
char * strstr(const char *s, const char *find) {
  register char c, sc;
  register size_t len;

  if ((c = *find++) != 0) {
    len = strlen(find);
    do {
      do {
        if ((sc = *s++) == 0)
          return (NULL);
      } while (sc != c);
    } while (strncmp(s, find, len) != 0);
    s--;
  }
  return ((char *)s);
}
#endif

/* the Mercury libc is *missing* isascii(), so here it is.. */
#if defined(MCOS)
   int isascii(int c) {
     return (!((c) & ~0177));
   }
#endif

/*
 * Thread Safe Random Number Generators
 * (no internal static data storage)
 * 
 *
 * Various useful RNG related pages:
 *  http://www.boost.org/libs/random/index.html
 *  http://www.agner.org/random/
 *  http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html
 *  http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2003/n1452.html
 *  http://www.gnu.org/software/gsl/manual/
 *  http://www.qbrundage.com/michaelb/pubs/essays/random_number_generation.html
 */


/*
 * VERY fast LCG random number generators for lousy but fast results
 *   Randval[j+1] = (A*V[j] + B) % M
 *   Generators where M=2^32 or 2^64 are fast since modulo is free, but
 *   they produce a bad distribution. 
 * DO NOT USE FOR MONTE CARLO SAMPLING
 * The rt_rand() API is similar to the reentrant "rand_r" version
 * found in some libc implementations.
 *
 * Note: Do not use LCG-generated random numbers in any way similar to 
 *       rand() % number. Only the high bits are very random [L'Ecuyer 1990].
 *
 * Unix drand48 uses: A=25214903917 B=11 M=2^48  [L'Ecuyer Testu01]
 *
 * L'Ecuyer suggests that all LCG's with moduli up to 2^61 fail
 * too many tests and should not be used.
 *
 */

#if 1

/*
 * Quick and dirty 32-bit LCG random number generator: 
 *   A=1099087573 B=0 M=2^32
 *   Period: 10^9
 * Fastest gun in the west, but fails many tests after 10^6 samples, 
 * and fails all statistics tests after 10^7 samples.
 * It fares better than the Numerical Recipes LCG.  This is the fastest
 * power of two rand, and has the best multiplier for 2^32, found by 
 * brute force[Fishman 1990].  Test results:
 *   http://www.iro.umontreal.ca/~lecuyer/myftp/papers/testu01.pdf 
 *   http://www.shadlen.org/ichbin/random/
 */
unsigned int rt_rand(unsigned int * idum) {
#if defined(_CRAYT3E)
  /* mask the lower 32 bits on machines where int is a 64-bit quantity */
  *idum = ((1099087573  * (*idum))) & ((unsigned int) 0xffffffff); 
#else
  /* on machines where int is 32-bits, no need to mask */
  *idum = (1099087573  * (*idum)); 
#endif
  return *idum;
}

#else

/*
 * Simple 32-bit LCG random number generator: 
     A=1664525 B=1013904223 M=2^32
 *   Period: 10^9
 *   Numerical Recipes suggests using: A=1664525 B=1013904223 M=2^32
 * Fails all of the same tests as the simpler one above, and returns
 * alternately even and odd results.
 */
unsigned int rt_rand(unsigned int * idum) {
#if defined(_CRAYT3E)
  /* mask the lower 32 bits on machines where int is a 64-bit quantity */
  *idum = ((1664525 * (*idum)) + 1013904223) & ((unsigned int) 0xffffffff); 
#else
  /* on machines where int is 32-bits, no need to mask */
  *idum = ((1664525 * (*idum)) + 1013904223);
#endif
  return *idum;
}

#endif


/* 
 * Higher quality random number generators which are safer for
 * use in monte carlo sampling etc.
 *
 */

#if defined(RT_RNG_USE_QUICK_AND_DIRTY)

unsigned int rng_urand(rng_urand_handle *rngh) {
#if defined(_CRAYT3E)
  /* mask the lower 32 bits on machines where int is a 64-bit quantity */
  rngh->randval = ((1099087573  * (rngh->randval))) & ((unsigned int) 0xffffffff); 
#else
  /* on machines where int is 32-bits, no need to mask */
  rngh->randval = (1099087573  * rngh->randval); 
#endif
  return rngh->randval;
}

void rng_urand_init(rng_urand_handle *rngh) {
  rng_urand_seed(rngh, 31337);
}

void rng_urand_seed(rng_urand_handle *rngh, unsigned int s) {
  rngh->randval = s;  
}

#elif defined(RT_RNG_USE_MERSENNE_TWISTER)

/* 
 * Mersenne Twister
 */

/* Period parameters */
#define N 624
#define M 397
#define MATRIX_A 0x9908b0dfUL   /* constant vector a */
#define UPPER_MASK 0x80000000UL /* most significant w-r bits */
#define LOWER_MASK 0x7fffffffUL /* least significant r bits */

void rng_urand_init(rng_urand_handle *rngh) {
  rngh->mti=N+1;            /* mti==N+1 means mt[N] is not initialized */
  rngh->mag01[0]=0x0UL;     /* mag01[x] = x * MATRIX_A  for x=0,1 */
  rngh->mag01[1]=MATRIX_A;
}

/* initializes mt[N] with a seed */
void rng_urand_seed(rng_urand_handle *rngh, unsigned int s) {
  unsigned int * mt = rngh->mt;
  int mti=rngh->mti;

  mt[0]= s & 0xffffffffUL;
  for (mti=1; mti<N; mti++) {
    mt[mti] = (1812433253UL * (mt[mti-1] ^ (mt[mti-1] >> 30)) + mti);
    /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
    /* In the previous versions, MSBs of the seed affect   */
    /* only MSBs of the array mt[].                        */
    /* 2002/01/09 modified by Makoto Matsumoto             */
  }
  rngh->mti=mti; /* update mti in handle */
}

/* generates a random number on [0,0xffffffff]-interval */
unsigned int rng_urand(rng_urand_handle *rngh) {
  unsigned int y;
  unsigned int * mt    = rngh->mt;
  unsigned int * mag01 = rngh->mag01;
  int mti = rngh->mti;

  if (mti >= N) { /* generate N words at one time */
    int kk;

    if (mti == N+1)   /* if init_genrand() has not been called, */
      rng_urand_seed(rngh, 5489); /* a default initial seed is used */

    for (kk=0;kk<N-M;kk++) {
      y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
      mt[kk] = mt[kk+M] ^ (y >> 1) ^ mag01[y & 0x1UL];
    }
    for (;kk<N-1;kk++) {
      y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
      mt[kk] = mt[kk+(M-N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
    }
    y = (mt[N-1]&UPPER_MASK)|(mt[0]&LOWER_MASK);
    mt[N-1] = mt[M-1] ^ (y >> 1) ^ mag01[y & 0x1UL];

    mti = 0;
  }

  y = mt[mti++];

  /* Tempering */
  y ^= (y >> 11);
  y ^= (y << 7) & 0x9d2c5680UL;
  y ^= (y << 15) & 0xefc60000UL;
  y ^= (y >> 18);

  rngh->mti = mti;

  return y;
}

#elif defined(RT_RNG_USE_KISS93)

/*
 * KISS93 random number generator by George Marsaglia
 * combines congruential with lag-1 multiply-with-carry
 *   Period: 2^127
 * Fails higher order tests. 
 *
 * The idea is to use simple, fast, individually promising
 * generators to get a composite that will be fast, easy to code
 * have a very long period and pass all the tests put to it.
 * The three components of KISS are
 *        x(n)=a*x(n-1)+1 mod 2^32
 *        y(n)=y(n-1)(I+L^13)(I+R^17)(I+L^5),
 *        z(n)=2*z(n-1)+z(n-2) +carry mod 2^32
 * The y's are a shift register sequence on 32bit binary vectors
 * period 2^32-1;
 * The z's are a simple multiply-with-carry sequence with period
 * 2^63+2^32-1.  The period of KISS is thus
 *      2^32*(2^32-1)*(2^63+2^32-1) > 2^127
 */

void rng_urand_init(rng_urand_handle *rngh) {
  rngh->x = 1;
  rngh->y = 2;
  rngh->z = 4;
  rngh->w = 8;
  rngh->c = 0;
  rngh->k = 0;
  rngh->m = 0;
}

void rng_urand_seed(rng_urand_handle *rngh, unsigned int seed) {
  rngh->x = seed | 1;
  rngh->y = seed | 2;
  rngh->z = seed | 4;
  rngh->w = seed | 8;
  rngh->c = seed | 0;
}

unsigned int rng_urand(rng_urand_handle *rngh) {
  rngh->x = rngh->x * 69069 + 1;
  rngh->y ^= rngh->y << 13;
  rngh->y ^= rngh->y >> 17;
  rngh->y ^= rngh->y << 5;
  rngh->k = (rngh->z >> 2) + (rngh->w >> 3) + (rngh->c >> 2);
  rngh->m = rngh->w + rngh->w + rngh->z + rngh->c;
  rngh->z = rngh->w;
  rngh->w = rngh->m;
  rngh->c = rngh->k >> 30;
  return rngh->x + rngh->y + rngh->w;
}

#else

/*
 * KISS99 random number generator by George Marsaglia
 * combines congruential with lag-1 multiply-with-carry
 *   Period: 2^123 
 */

void rng_urand_init(rng_urand_handle *rngh) {
  rngh->x = 123456789;
  rngh->y = 362436000;
  rngh->z = 521288629;
  rngh->c = 7654321;
}

void rng_urand_seed(rng_urand_handle *rngh, unsigned int seed) {
  rngh->x = seed | 1;
  rngh->y = seed | 2;
  rngh->z = seed | 4;
  rngh->c = seed | 0;
}

unsigned int rng_urand(rng_urand_handle *rngh) {
  /* yes, the below are 64-bit quantities, wonder if this compiles in */
  /* a portable manner... */
  unsigned long long t, a=698769069LL;
  rngh->x = 69069 * rngh->x + 12345;
  rngh->y ^= (rngh->y<<13);
  rngh->y ^= (rngh->y>>17);
  rngh->y ^= (rngh->y<<5);
  t=a*rngh->z+rngh->c;
  rngh->c=(t>>32);
  return rngh->x+rngh->y+(rngh->z=t);
}

#endif


/*
 * single precision random number generators returning range [0-1)
 * (uses unsigned int random generators above)
 */
void rng_frand_init(rng_frand_handle *rngh) {
  rng_urand_init(rngh);
}

float rng_frand(rng_frand_handle *rngh) {
  return rng_urand(rngh) / RT_RNG_MAX; 
}

void rng_frand_seed(rng_frand_handle *rngh, unsigned int seed) {
  rng_urand_seed(rngh, seed);
}

void rng_drand_init(rng_drand_handle *rngh) {
  rng_urand_init(rngh);
}

double rng_drand(rng_frand_handle *rngh) {
  return rng_urand(rngh) / RT_RNG_MAX; 
}

void rng_drand_seed(rng_frand_handle *rngh, unsigned int seed) {
  rng_urand_seed(rngh, seed);
}

/*
 * routine to help create seeds for parallel runs
 */
unsigned int rng_seed_from_tid_nodeid(int tid, int node) {
  unsigned int seedbuf[11] = {
    12345678,
     3498711,
    19872134,
     1004141,
     1275987,
    23904273,
     2091097, 
    19872727,
       31337,
    20872837,
     1020733
  };

  return seedbuf[tid % 11] + node * 31337;
}

/* calculate a pair of pixel jitter offset values */
/* that range from -0.5 to 0.5                    */
void jitter_offset2f(unsigned int *pval, float *xy) {
  xy[0] = (rt_rand(pval) / RT_RAND_MAX) - 0.5f;
  xy[1] = (rt_rand(pval) / RT_RAND_MAX) - 0.5f;
}

/* calculate a pair of pixel jitter offset values */
/* that range from -0.5 to 0.5                    */
void jitter_disc2f(unsigned int *pval, float *dir) {
  float dx, dy;
  do {
    dx = (rt_rand(pval) / RT_RAND_MAX) - 0.5f;
    dy = (rt_rand(pval) / RT_RAND_MAX) - 0.5f;
  } while ((dx*dx + dy*dy) > 0.250f);

  dir[0] = dx;
  dir[1] = dy;
}

/* Generate a randomly oriented ray */
void jitter_sphere3f(rng_frand_handle *rngh, float *dir) {
  float dx, dy, dz, len, invlen;
  /* In order to correctly sample a sphere, using rays    */
  /* generated randomly within a cube we must throw out   */
  /* direction vectors longer than 1.0, otherwise we'll   */
  /* oversample the corners of the cube relative to       */
  /* a true sphere.                                       */
  do {
    dx = rng_frand(rngh) - 0.5f;
    dy = rng_frand(rngh) - 0.5f;
    dz = rng_frand(rngh) - 0.5f;
    len = dx*dx + dy*dy + dz*dz;
  } while (len > 0.250f);
  invlen = 1.0f / sqrt(len);

  /* finish normalizing the direction vector */
  dir[0] = dx * invlen;
  dir[1] = dy * invlen;
  dir[2] = dz * invlen;
}


