/*
 * TachyonOptiX.cu - OptiX host-side RT engine implementation
 *
 * (C) Copyright 2013-2022 John E. Stone
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * $Id: ProfileHooks.h,v 1.3 2022/03/11 00:45:37 johns Exp $
 *
 */

/**
 *  \file ProfileHooks.h
 *  \brief CPU and GPU profiling utility macros/routines 
 * 
 *  Exemplary use of NVTX is shown here:
 *    https://devblogs.nvidia.com/cuda-pro-tip-generate-custom-application-profile-timelines-nvtx/
 *
 *  NVTX3 eliminates the need for linkage w/ the nvtools library:
 *    https://github.com/NVIDIA/NVTX 
 *    https://nvidia.github.io/NVTX/doxygen/index.html
 * 
 *  NVTX3 C++ RAII-based tag documentation:
 *    https://github.com/NVIDIA/NVTX/tree/dev/cpp
 *    https://jrhemstad.github.io/nvtx_wrappers/html/index.html
 *    https://jrhemstad.github.io/nvtx_wrappers/html/nvtx3_8hpp.html
 *
 */

#ifndef PROFILEHOOKS_H
#define PROFILEHOOKS_H

#if defined(WKFNVTX)

#if 1
/// use gettid() to obtain thread IDs
#define WKFUSEGETTID 1
#include <unistd.h>
#include <sys/types.h>
#include <sys/syscall.h>

#ifndef gettid
/// equivalent to:  pid_t  gettid(void)
#define gettid() syscall(SYS_gettid)
#endif
#else 
/// use pthread_threadid_np() on MacOS X, other non-Linux platforms
#include <pthread.h>
#endif

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// We really only want to use CUDA >= 10.0 w/ NVTX V3+,
// which bypasses the need to include/ship/link against 
// any additional libs.
#if CUDART_VERSION >= 10000
#include <nvtx3/nvToolsExt.h>  // CUDA >= 10 has NVTX V3+
#else
#error NVTXv3 requires CUDA 10.0 or greater
#include <nvToolsExt.h>        // CUDA < 10 has NVTX V2
#endif


/// C++ note: declaring const variables implies static (internal) linkage,
/// and you have to explicitly specify "extern" to get external linkage.
/// Colors are specified in ARGB hexadecimal ordering below:
const uint32_t WKF_nvtx_colors[] = {
  0xff629f57, ///< 0 green
  0xffee8c40, ///< 1 orange
  0xff507ba4, ///< 2 blue
  0xffecc65c, ///< 3 yellow
  0xffac7c9f, ///< 4 purple
  0xff7cb7b2, ///< 5 teal
  0xffdb565c, ///< 6 red
  0xffb9b0ac, ///< 7 gray
  0xffffffff, ///< 8 white
};

/// Map an arbitrary caller-provided color index to our fixed size color table
const int WKF_nvtx_colors_len = sizeof(WKF_nvtx_colors)/sizeof(uint32_t);

#define PROFILE_GREEN  0
#define PROFILE_ORANGE 1
#define PROFILE_BLUE   2
#define PROFILE_YELLOW 3
#define PROFILE_PURPLE 4
#define PROFILE_TEAL   5
#define PROFILE_RED    6
#define PROFILE_GRAY   7
#define PROFILE_WHITE  8

/// Initialize the profiling system.
#define PROFILE_INITIALIZE() do { nvtxInitialize(NULL); } while(0) // terminate with semicolon

/// Trigger the beginning of profiler trace capture, for those that support it.
#define PROFILE_START() \
  do { \
    cudaProfilerStart(); \
  } while (0) // terminate with semicolon

/// Trigger the stop of profiler trace capture, for those that support it.
#define PROFILE_STOP() \
  do { \
    cudaDeviceSynchronize(); \
    cudaProfilerStop(); \
  } while (0) // terminate with semicolon


///
/// An alternative to using NVTX to name threads is to use OS- or 
/// runtime-specific threading APIs to assign string names independent
/// of the profiling tools being used.  On Linux we can do this using
/// pthread_setname_np() in combination with _GNU_SOURCE if we like.
/// It is noteworthy that the pthread_setname_np() APIs are resitricted
/// to 15 chars of thread name and 1 char for the terminating NUL char.
///
#if defined(WKFUSEGETTID)

/// Assign this as the "main" thread for the benefit of profile trace display.
#define PROFILE_MAIN_THREAD() \
  do { \
    /* On Linux use gettid() to get current thread ID */ \
    nvtxNameOsThread(gettid(), "Main thread"); \
  } while (0) // terminate with semicolon

/// Assign a caller-provided string as the name of the current thread for
/// use in profiler trace display
#define PROFILE_NAME_THREAD(name) \
  do { \
    nvtxNameOsThread(gettid(), name); \
  } while (0) // terminate with semicolon

#else

/// Assign this as the "main" thread for the benefit of profile trace display.
#define PROFILE_MAIN_THREAD() \
  do { \
    /* On MacOS X or other platforms use pthread_threadid_np() */ \
    __uint64_t tid;
    pthread_threadid_np(pthread_self(), &tid);
    nvtxNameOsThread(tid, "Main thread"); \
  } while (0) // terminate with semicolon

/// Assign a caller-provided string as the name of the current thread for
/// use in profiler trace display
#define PROFILE_NAME_THREAD(name) \
  do { \
    __uint64_t tid;
    pthread_threadid_np(pthread_self(), &tid);
    nvtxNameOsThread(gettid(), name); \
  } while (0) // terminate with semicolon

#endif


/// Insert an "event" into the profiler trace display, with the 
/// caller-provided string and color index.  The inserted event is
/// just a mark or timestamp, and has no defined time length associated 
/// with it.
#define PROFILE_MARK(name,cid) \
  do { \
    /* create an ASCII event marker */ \
    /* nvtxMarkA(name); */ \
    int color_id = cid; \
    color_id = color_id % WKF_nvtx_colors_len; \
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = WKF_nvtx_colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxMarkEx(&eventAttrib); \
  } while(0) // terminate with semicolon


/// Pushes a time range annotation onto the profiler's trace stack, beginning
/// at the time of submission, and ending when a matching PROFILE_POP_RANGE
/// is encountered.  The new annotation is named with a caller-provided string
/// and is colored according to the provided color index.
#define PROFILE_PUSH_RANGE(name,cid) \
  do { \
    int color_id = cid; \
    color_id = color_id % WKF_nvtx_colors_len; \
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = WKF_nvtx_colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
  } while(0)  // must terminate with semi-colon


/// Pops the innermost time range off of the profiler's trace stack, 
/// at the time of execution.
#define PROFILE_POP_RANGE(empty) \
  do { \
    nvtxRangePop(); \
  } while(0) // terminate with semicolon 


// embed event recording in class to automatically pop when destroyed
class WKF_NVTX_Tracer {
  public:
    WKF_NVTX_Tracer(const char *name, int cid = 0) { PROFILE_PUSH_RANGE(name, cid); }
    ~WKF_NVTX_Tracer() { PROFILE_POP_RANGE(); }
};

/// Embeds event recording into a class to automatically pop when destroyed.
#define PROFILE_RANGE(name,cid) \
  /* include cid as part of the name */ \
  /* call RANGE at beginning of function to push event recording */ \
  /* destructor is automatically called on return to pop event recording */ \
  WKF_NVTX_Tracer wkf_nvtx_tracer##cid(name,cid)
  // must terminate with semi-colon

#if defined(WKFNVTX_SYNCPRETTY)
/// Helper macro that can conditionally insert extra calls to 
/// cudaStreamSynchronize() into an application, which are not required
/// for correctness, but are sometimes beneficial to ensure that long 
/// sequences of asynchronous API calls don't result in illegible 
/// profile traces.
#define PROFILE_STREAM_SYNC_PRETTY(stream) \
  do { \
    /* Add a CUDA stream sync call, but only for the benefit of */ \
    /* profile trace clarity, so that it can be disabled on demand */ \
    cudaStreamSynchronize(stream); \
  } while(0) // terminate with semicolon

#else
/// Helper macro that can conditionally insert extra calls to 
/// cudaStreamSynchronize() into an application, which are not required
/// for correctness, but are sometimes beneficial to ensure that long 
/// sequences of asynchronous API calls don't result in illegible 
/// profile traces.
#define PROFILE_STREAM_SYNC_PRETTY(stream) do { } while(0) // term w/ semicolon 
#endif


#else

//
// If NVTX isn't enabled, then the profiling macros become no-ops.
// We add inline documentation here since Doxygen sees this branch by default.
//

/// Initialize the profiling system.
#define PROFILE_INITIALIZE()         do { } while(0) // terminate with semicolon

/// Trigger the beginning of profiler trace capture, for those that support it.
#define PROFILE_START()              do { } while(0) // terminate with semicolon

/// Trigger the stop of profiler trace capture, for those that support it.
#define PROFILE_STOP()               do { } while(0) // terminate with semicolon

/// Assign this as the "main" thread for the benefit of profile trace display.
#define PROFILE_MAIN_THREAD()        do { } while(0) // terminate with semicolon

/// Assign a caller-provided string as the name of the current thread for
/// use in profiler trace display
#define PROFILE_NAME_THREAD(name)    do { } while(0) // terminate with semicolon

/// Insert an "event" into the profiler trace display, with the 
/// caller-provided string and color index.  The inserted event is
/// just a mark or timestamp, and has no defined time length associated 
/// with it.
#define PROFILE_MARK(name,cid)       do { } while(0) // terminate with semicolon

/// Pushes a time range annotation onto the profiler's trace stack, beginning
/// at the time of submission, and ending when a matching PROFILE_POP_RANGE
/// is encountered.  The new annotation is named with a caller-provided string
/// and is colored according to the provided color index.
#define PROFILE_PUSH_RANGE(name,cid) do { } while(0) // terminate with semicolon

/// Pops the innermost time range off of the profiler's trace stack, 
/// at the time of execution.
#define PROFILE_POP_RANGE()          do { } while(0) // terminate with semicolon

/// Embeds event recording into a class to automatically pop when destroyed.
#define PROFILE_RANGE(name,cid)      do { } while(0) // terminate with semicolon

/// Helper macro that can conditionally insert extra calls to 
/// cudaStreamSynchronize() into an application, which are not required
/// for correctness, but are sometimes beneficial to ensure that long 
/// sequences of asynchronous API calls don't result in illegible 
/// profile traces.
#define PROFILE_STREAM_SYNC_PRETTY(stream) do { } while(0) // term w/ semicolon 
#endif

#endif
