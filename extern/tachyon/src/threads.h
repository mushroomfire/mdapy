/*
 * threads.h - code for spawning threads on various platforms.
 *
 *  $Id: threads.h,v 1.50 2013/04/20 19:59:46 johns Exp $
 */ 

#ifndef RT_THREADS_INC
#define RT_THREADS_INC 1

#ifdef __cplusplus
extern "C" {
#endif

/* define which thread calls to use */
#if defined(USEPOSIXTHREADS) && defined(USEUITHREADS)
#error You may only define USEPOSIXTHREADS or USEUITHREADS, but not both
#endif

/* POSIX Threads */
#if defined(_AIX) || defined(__APPLE__) || defined(_CRAY) || defined(__hpux) || defined(__irix) || defined(__linux) || defined(__osf__) ||  defined(__PARAGON__) || defined(__CYGWIN__)
#if !defined(USEUITHREADS) && !defined(USEPOSIXTHREADS)
#define USEPOSIXTHREADS
#endif
#endif

/* Unix International Threads */
#if defined(SunOS)
#if !defined(USEPOSIXTHREADS) && !defined(USEUITHREADS)
#define USEUITHREADS
#endif
#endif


#ifdef THR
#ifdef USEPOSIXTHREADS
#include <pthread.h>

typedef pthread_t        rt_thread_t;
typedef pthread_mutex_t   rt_mutex_t;
typedef pthread_cond_t     rt_cond_t;

typedef struct rwlock_struct {
  pthread_mutex_t lock;          /**< read/write monitor lock */
  int rwlock;                    /**< if >0 = #rdrs, if <0 = wrtr, 0=none */
  pthread_cond_t  rdrs_ok;       /**< start waiting readers */
  unsigned int waiting_writers;  /**< # of waiting writers  */
  pthread_cond_t  wrtr_ok;       /**< start waiting writers */ 
} rt_rwlock_t;

#endif

#ifdef USEUITHREADS
#include <thread.h>

typedef thread_t  rt_thread_t;
typedef mutex_t   rt_mutex_t;
typedef cond_t    rt_cond_t;
typedef rwlock_t  rt_rwlock_t;
#endif


#ifdef WIN32
#include <windows.h>
typedef HANDLE rt_thread_t;
typedef CRITICAL_SECTION rt_mutex_t;

#if 0 && (NTDDI_VERSION >= NTDDI_WS08 || _WIN32_WINNT > 0x0600)
/* Use native condition variables only with Windows Server 2008 and newer... */
#define RTUSEWIN2008CONDVARS 1
typedef CONDITION_VARIABLE rt_cond_t;
#else
/* Every version of Windows prior to Vista/WS2008 must emulate */
/* variables using manually resettable events or other schemes */

/* For higher performance, use interlocked memory operations   */
/* rather than locking/unlocking mutexes when manipulating     */
/* internal state.                                             */
#if 1
#define RTUSEINTERLOCKEDATOMICOPS 1
#endif
#define RT_COND_SIGNAL    0
#define RT_COND_BROADCAST 1
typedef struct {
  LONG waiters;                  /**< MUST be 32-bit aligned for correct   */
                                 /**< operation with InterlockedXXX() APIs */
  CRITICAL_SECTION waiters_lock; /**< lock itself                          */
  HANDLE events[2];              /**< Signal and broadcast event HANDLEs.  */
} rt_cond_t;
#endif

typedef struct rwlock_struct {
  rt_mutex_t lock;               /**< read/write monitor lock */
  int rwlock;                    /**< if >0 = #rdrs, if <0 = wrtr, 0=none */
  rt_cond_t  rdrs_ok;            /**< start waiting readers */
  unsigned int waiting_writers;  /**< # of waiting writers  */
  rt_cond_t  wrtr_ok;            /**< start waiting writers */ 
} rt_rwlock_t;

#endif
#endif /* WIN32 */


#ifndef THR
typedef int rt_thread_t;
typedef int rt_mutex_t;
typedef int rt_cond_t;
typedef int rt_rwlock_t;
#endif


typedef struct atomic_int_struct {
  int padding1[8];        /**< Pad to avoid false sharing, cache aliasing */
  rt_mutex_t lock;        /**< Mutex lock for the structure */
  int val;                /**< Integer value to be atomically manipulated */
  int padding2[8];        /**< Pad to avoid false sharing, cache aliasing */
} rt_atomic_int_t;


typedef struct barrier_struct {
  int padding1[8];        /**< Pad to avoid false sharing, cache aliasing */
  rt_mutex_t lock;        /**< Mutex lock for the structure */
  int n_clients;          /**< Number of threads to wait for at barrier */
  int n_waiting;          /**< Number of currently waiting threads */
  int phase;              /**< Flag to separate waiters from fast workers */
  int sum;                /**< Sum of arguments passed to barrier_wait */
  int result;             /**< Answer to be returned by barrier_wait */
  rt_cond_t wait_cv;      /**< Clients wait on condition variable to proceed */
  int padding2[8];        /**< Pad to avoid false sharing, cache aliasing */
} rt_barrier_t;

typedef struct rt_run_barrier_struct {
  int padding1[8];        /**< Pad to avoid false sharing, cache aliasing */
  rt_mutex_t lock;        /**< Mutex lock for the structure */
  int n_clients;          /**< Number of threads to wait for at barrier */
  int n_waiting;          /**< Number of currently waiting threads */
  int phase;              /**< Flag to separate waiters from fast workers */
  void * (*fctn)(void *); /**< Fctn ptr to call, or NULL if done */
  void * parms;           /**< parms for fctn pointer */
  void * (*rslt)(void *); /**< Fctn ptr to return to barrier wait callers */
  void * rsltparms;       /**< parms to return to barrier wait callers */
  rt_cond_t wait_cv;      /**< Clients wait on condition variable to proceed */
  int padding2[8];        /**< Pad to avoid false sharing, cache aliasing */
} rt_run_barrier_t;


/*
 * Routines for querying processor counts, and managing CPU affinity
 */
/** number of physical processors available */
int rt_thread_numphysprocessors(void);

/** number of processors available, subject to user override */
int rt_thread_numprocessors(void);

/** query CPU affinity of the calling process (if allowed by host system) */
int * rt_cpu_affinitylist(int *cpuaffinitycount);

/** set the CPU affinity of the current thread (if allowed by host system) */
int rt_thread_set_self_cpuaffinity(int cpu);

/** set the concurrency level and scheduling scope for threads */
int rt_thread_setconcurrency(int);


/*
 * Thread management
 */
/** create a new child thread */
int rt_thread_create(rt_thread_t *, void * fctn(void *), void *);

/** join (wait for completion of, and merge with) a thread */
int rt_thread_join(rt_thread_t, void **);


/*
 * Mutex management
 */
/** initialize a mutex */
int rt_mutex_init(rt_mutex_t *);

/** lock a mutex */
int rt_mutex_lock(rt_mutex_t *);

/** try to lock a mutex */
int rt_mutex_trylock(rt_mutex_t *);

/** lock a mutex by spinning only */
int rt_mutex_spin_lock(rt_mutex_t *);

/** unlock a mutex */
int rt_mutex_unlock(rt_mutex_t *);

/** destroy a mutex */
int rt_mutex_destroy(rt_mutex_t *);


/*
 * Condition variable management
 */
/** initialize a condition variable */
int rt_cond_init(rt_cond_t *);

/** destroy a condition variable */
int rt_cond_destroy(rt_cond_t *);

/** wait on a condition variable */
int rt_cond_wait(rt_cond_t *, rt_mutex_t *);

/** signal a condition variable, waking at least one thread */
int rt_cond_signal(rt_cond_t *);

/** signal a condition variable, waking all threads */
int rt_cond_broadcast(rt_cond_t *);


/*
 * Atomic operations on integers
 */
/** initialize an atomic int variable */
int rt_atomic_int_init(rt_atomic_int_t * atomp, int val);

/** destroy an atomic int variable */
int rt_atomic_int_destroy(rt_atomic_int_t * atomp);

/** set an atomic int variable */
int rt_atomic_int_set(rt_atomic_int_t * atomp, int val);

/** get an atomic int variable */
int rt_atomic_int_get(rt_atomic_int_t * atomp);

/** fetch an atomic int and add inc to it, returning original value */
int rt_atomic_int_fetch_and_add(rt_atomic_int_t * atomp, int inc);

/** fetch an atomic int  and add inc to it, returning new value */
int rt_atomic_int_add_and_fetch(rt_atomic_int_t * atomp, int inc);


/*
 * Reader/writer lock management
 */
/** initialize a reader/writer lock */
int rt_rwlock_init(rt_rwlock_t *);

/** set reader lock */
int rt_rwlock_readlock(rt_rwlock_t *);

/** set writer lock */
int rt_rwlock_writelock(rt_rwlock_t *);

/** unlock reader/writer lock */
int rt_rwlock_unlock(rt_rwlock_t *);


/*
 * counting barrier
 */
/** initialize counting barrier primitive */
rt_barrier_t * rt_thread_barrier_init(int n_clients);

/** destroy counting barrier primitive */
void rt_thread_barrier_destroy(rt_barrier_t *barrier);

/** synchronize on counting barrier primitive */
int rt_thread_barrier(rt_barrier_t *barrier, int increment);


/*
 * This is a symmetric barrier routine designed to be used
 * in implementing a sleepable thread pool.
 */
/** initialize thread pool barrier */
int rt_thread_run_barrier_init(rt_run_barrier_t *barrier, int n_clients);

/** destroy thread pool barrier */
void rt_thread_run_barrier_destroy(rt_run_barrier_t *barrier);

/** sleeping barrier synchronization for thread pool */
void * (*rt_thread_run_barrier(rt_run_barrier_t *barrier,
                                void * fctn(void*),
                                void * parms,
                                void **rsltparms))(void *);

/** non-blocking poll to see if peers are already at the barrier */
int rt_thread_run_barrier_poll(rt_run_barrier_t *barrier);


/**
 * Task tile struct for stack, iterator, and scheduler routines;
 * 'start' is inclusive, 'end' is exclusive.  This yields a
 * half-open interval that corresponds to a typical 'for' loop.
 */
typedef struct rt_tasktile_struct {
  int start;         /**< starting task ID (inclusive) */
  int end;           /**< ending task ID (exclusive) */
} rt_tasktile_t;


/*
 * tile stack
 */
#define RT_TILESTACK_EMPTY -1

/**
 * stack of work tiles, for error handling
 */
typedef struct {
  rt_mutex_t mtx;    /**< Mutex lock for the structure */
  int growthrate;    /**< stack growth chunk size */
  int size;          /**< current allocated stack size */
  int top;           /**< index of top stack element */
  rt_tasktile_t *s;  /**< stack of task tiles */
} rt_tilestack_t;

/** initialize task tile stack (to empty) */
int rt_tilestack_init(rt_tilestack_t *s, int size);

/** destroy task tile stack */
void rt_tilestack_destroy(rt_tilestack_t *);

/** shrink memory buffers associated with task tile stack if possible */
int rt_tilestack_compact(rt_tilestack_t *);

/** push a task tile onto the stack */
int rt_tilestack_push(rt_tilestack_t *, const rt_tasktile_t *);

/** pop a task tile off of the stack */
int rt_tilestack_pop(rt_tilestack_t *, rt_tasktile_t *);

/** pop all of the task tiles off of the stack */
int rt_tilestack_popall(rt_tilestack_t *);

/** query if the task tile stack is empty or not */
int rt_tilestack_empty(rt_tilestack_t *);


/**
 * Shared iterators intended for trivial CPU/GPU load balancing with no
 * exception handling capability (all work units must complete with
 * no errors, or else the whole thing is canceled).
 */
#define RT_SCHED_DONE     -1   /**< no work left to process        */
#define RT_SCHED_CONTINUE  0   /**< some work remains in the queue */

/** iterator used for dynamic load balancing */
typedef struct rt_shared_iterator_struct {
  rt_mutex_t mtx;      /**< mutex lock */
  int start;           /**< starting value (inclusive) */
  int end;             /**< ending value (exlusive) */
  int current;         /**< current value */
  int fatalerror;      /**< cancel processing immediately for all threads */
} rt_shared_iterator_t;

/** initialize a shared iterator */
int rt_shared_iterator_init(rt_shared_iterator_t *it);

/** destroy a shared iterator */
int rt_shared_iterator_destroy(rt_shared_iterator_t *it);

/** Set shared iterator state to half-open interval defined by tile */
int rt_shared_iterator_set(rt_shared_iterator_t *it, rt_tasktile_t *tile);

/**
 * iterate the shared iterator with a requested tile size,
 * returns the tile received, and a return code of -1 if no
 * iterations left or a fatal error has occured during processing,
 * canceling all worker threads.
 */
int rt_shared_iterator_next_tile(rt_shared_iterator_t *it, int reqsize,
                                 rt_tasktile_t *tile);

/** worker thread calls this to indicate a fatal error */
int rt_shared_iterator_setfatalerror(rt_shared_iterator_t *it);

/** master thread calls this to query for fatal errors */
int rt_shared_iterator_getfatalerror(rt_shared_iterator_t *it);


/*
 * Thread pool.
 */
/** shortcut macro to tell the create routine we only want CPU cores */
#define RT_THREADPOOL_DEVLIST_CPUSONLY NULL

/** symbolic constant macro to test if we have a GPU or not */
#define RT_THREADPOOL_DEVID_CPU -1

/** thread-specific handle data for workers */
typedef struct rt_threadpool_workerdata_struct {
  int padding1[8];                        /**< avoid false sharing */
  rt_shared_iterator_t *iter;             /**< dynamic work scheduler */
  rt_tilestack_t *errorstack;             /**< stack of tiles that failed */
  int threadid;                           /**< worker thread's id */
  int threadcount;                        /**< total number of worker threads */
  int devid;                              /**< worker CPU/GPU device ID */
  float devspeed;                         /**< speed scaling for this device */
  void *parms;                            /**< fctn parms for this worker */
  void *thrpool;                          /**< void ptr to thread pool struct */
  int padding2[8];                        /**< avoid false sharing */
} rt_threadpool_workerdata_t;

typedef struct rt_threadpool_struct {
  int workercount;                        /**< number of worker threads */
  int *devlist;                           /**< per-worker CPU/GPU device IDs */
  rt_shared_iterator_t iter;              /**< dynamic work scheduler */
  rt_tilestack_t errorstack;              /**< stack of tiles that failed */
  rt_thread_t *threads;                   /**< worker threads */
  rt_threadpool_workerdata_t *workerdata; /**< per-worker data */
  rt_run_barrier_t runbar;                /**< master/worker run barrier */
} rt_threadpool_t;

/** create a thread pool with a specified number of worker threads */
rt_threadpool_t * rt_threadpool_create(int workercount, int *devlist);

/** launch threads onto a new function, with associated parms */
int rt_threadpool_launch(rt_threadpool_t *thrpool,
                         void *fctn(void *), void *parms, int blocking);

/** wait for all worker threads to complete their work */
int rt_threadpool_wait(rt_threadpool_t *thrpool);

/** join all worker threads and free resources */
int rt_threadpool_destroy(rt_threadpool_t *thrpool);

/** query number of worker threads in the pool */
int rt_threadpool_get_workercount(rt_threadpool_t *thrpool);

/** worker thread can call this to get its ID and number of peers */
int rt_threadpool_worker_getid(void *voiddata, int *threadid, int *threadcount);

/** worker thread can call this to get its CPU/GPU device ID */
int rt_threadpool_worker_getdevid(void *voiddata, int *devid);

/**
 * Worker thread calls this to set relative speed of this device
 * as determined by the SM/core count and clock rate
 * Note: this should only be called once, during the worker's
 * device initialization process
 */
int rt_threadpool_worker_setdevspeed(void *voiddata, float speed);

/**
 * Worker thread calls this to get relative speed of this device
 * as determined by the SM/core count and clock rate
 */
int rt_threadpool_worker_getdevspeed(void *voiddata, float *speed);

/**
 * worker thread calls this to scale max tile size by worker speed
 * as determined by the SM/core count and clock rate
 */
int rt_threadpool_worker_devscaletile(void *voiddata, int *tilesize);

/** worker thread can call this to get its client data pointer */
int rt_threadpool_worker_getdata(void *voiddata, void **clientdata);

/** Set dynamic scheduler state to half-open interval defined by tile */
int rt_threadpool_sched_dynamic(rt_threadpool_t *thrpool, rt_tasktile_t *tile);

/**
 * worker thread calls this to get its next work unit
 * iterate the shared iterator, returns -1 if no iterations left
 */
int rt_threadpool_next_tile(void *thrpool, int reqsize, rt_tasktile_t *tile);

/**
 * worker thread calls this when it fails computing a tile after
 * it has already taken it from the scheduler
 */
int rt_threadpool_tile_failed(void *thrpool, rt_tasktile_t *tile);

/** worker thread calls this to indicate that an unrecoverable error occured */
int rt_threadpool_setfatalerror(void *thrparms);

/** master thread calls this to query for fatal errors */
int rt_threadpool_getfatalerror(void *thrparms);


/**
 * Routines to generate a pool of threads which then grind through
 * a dynamically load balanced work queue implemented as a shared iterator.
 * No exception handling is possible, just a simple all-or-nothing attept.
 * Useful for simple calculations that take very little time.
 * An array of threads is generated, launched, and joined all with one call.
 */
typedef struct rt_threadlaunch_struct {
  int padding1[8];              /**< avoid false sharing, cache aliasing */
  rt_shared_iterator_t *iter;   /**< dynamic scheduler iterator */
  int threadid;                 /**< ID of worker thread */
  int threadcount;              /**< number of workers */
  void * clientdata;            /**< worker parameters */
  int padding2[8];              /**< avoid false sharing, cache aliasing */
} rt_threadlaunch_t;

/** launch up to numprocs threads using shared iterator as a load balancer */
int rt_threadlaunch(int numprocs, void *clientdata, void * fctn(void *),
                    rt_tasktile_t *tile);

/** worker thread can call this to get its ID and number of peers */
int rt_threadlaunch_getid(void *thrparms, int *threadid, int *threadcount);

/** worker thread can call this to get its client data pointer */
int rt_threadlaunch_getdata(void *thrparms, void **clientdata);

/**
 * worker thread calls this to get its next work unit
 * iterate the shared iterator, returns -1 if no iterations left
 */
int rt_threadlaunch_next_tile(void *voidparms, int reqsize,
                              rt_tasktile_t *tile);

/** worker thread calls this to indicate that an unrecoverable error occured */
int rt_threadlaunch_setfatalerror(void *thrparms);


#ifdef __cplusplus
}
#endif

#endif
