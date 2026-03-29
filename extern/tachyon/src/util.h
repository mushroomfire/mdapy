/* 
 * util.h - This file contains defines for the timer functions...
 *
 *  $Id: util.h,v 1.23 2012/10/17 04:25:57 johns Exp $
 */

#if !defined(RT_UTIL_H) 
#define RT_UTIL_H 1

#if !defined(USESINGLEFLT)
#define ACOS(x)    acos(x)
#define COS(x)     cos(x)
#define EXP(x)     exp(x)
#define FABS(x)    fabs(x)
#define POW(x, y)  pow(x, y)
#define SIN(x)     sin(x)
#define SQRT(x)    sqrt(x)
#else
#define ACOS(x)    acosf(x)
#define COS(x)     cosf(x)
#define EXP(x)     expf(x)
#define FABS(x)    fabsf(x)
#define POW(x, y)  powf(x, y)
#define SIN(x)     sinf(x)
#define SQRT(x)    sqrtf(x)
#endif

typedef void * rt_timerhandle;           /* a timer handle */
rt_timerhandle rt_timer_create(void);    /* create a timer (clears timer)  */
void rt_timer_destroy(rt_timerhandle);   /* create a timer (clears timer)  */
void rt_timer_start(rt_timerhandle);     /* start a timer  (clears timer)  */
void rt_timer_stop(rt_timerhandle);      /* stop a timer                   */
double rt_timer_time(rt_timerhandle);    /* report elapsed time in seconds */
double rt_timer_timenow(rt_timerhandle); /* report elapsed time in seconds */

#define RT_RAND_MAX 4294967296.0         /* Max random value from rt_rand  */
unsigned int rt_rand(unsigned int *);    /* thread-safe 32-bit RNG         */

/* select the RNG to use as the basis for all of the floating point work */
#define RT_RNG_USE_KISS93               1

#if defined(RT_RNG_USE_QUICK_AND_DIRTY)

/* Quick and Dirty RNG */
typedef struct {
  unsigned int randval;
} rng_urand_handle;
#define RT_RNG_MAX 4294967296.0       /* max urand value: 2^32 */

#elif defined(RT_RNG_USE_MERSENNE_TWISTER)

/* Mersenne Twister */
typedef struct {
  int mti;                /* mti==N+1 means mt[N] is not initialized */
  unsigned int mt[624];   /* N: the array for the state vector  */
  unsigned int mag01[2];
} rng_urand_handle;
#define RT_RNG_MAX 4294967296.0       /* max urand value: 2^32 */

#elif defined(RT_RNG_USE_KISS93)

/* KISS93 */
typedef struct {
  unsigned int x;
  unsigned int y;
  unsigned int z;
  unsigned int w;
  unsigned int c;
  unsigned int k;
  unsigned int m;
} rng_urand_handle;
#define RT_RNG_MAX 4294967296.0       /* max urand value: 2^32 */

#else

/* KISS99 */
typedef struct {
  unsigned int x;
  unsigned int y;
  unsigned int z;
  unsigned int c;
} rng_urand_handle;
#define RT_RNG_MAX 4294967296.0       /* max urand value: 2^32 */

#endif

void rng_urand_init(rng_urand_handle *rngh);
void rng_urand_seed(rng_urand_handle *rngh, unsigned int seed);
unsigned int rng_urand(rng_urand_handle *rngh);

typedef rng_urand_handle rng_frand_handle;
typedef rng_urand_handle rng_drand_handle;

void rng_frand_init(rng_frand_handle *rngh);
/* generates a random number on [0,1)-real-interval */
float rng_frand(rng_frand_handle *rngh);
void rng_frand_seed(rng_frand_handle *rngh, unsigned int seed);

void rng_drand_init(rng_drand_handle *rngh);
/* generates a random number on [0,1)-real-interval */
double rng_drand(rng_frand_handle *rngh);
void rng_drand_seed(rng_frand_handle *rngh, unsigned int seed);

/* routine to help create seeds for parallel runs */
unsigned int rng_seed_from_tid_nodeid(int tid, int node);

void jitter_offset2f(unsigned int *pval, float *xy);
void jitter_disc2f(unsigned int *pval, float *xy);
void jitter_sphere3f(rng_frand_handle *rngh, float *dir);

#endif
