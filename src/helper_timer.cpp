/////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////

// includes, project
#include "helper_timer.h"

//////////////////////////////////////////////////////////////////
// Begin Stopwatch timer class definitions for all OS platforms //
//////////////////////////////////////////////////////////////////
#ifdef WIN32
StopWatchWin::StopWatchWin() :
start_time(), end_time(),
diff_time(0.0f), total_time(0.0f),
running(false), clock_sessions(0), freq(0), freq_set(false)
{
  if (!freq_set) {
    // helper variable
    LARGE_INTEGER temp;

    // get the tick frequency from the OS
    QueryPerformanceFrequency((LARGE_INTEGER*)&temp);

    // convert to type in which it is needed
    freq = ((double)temp.QuadPart) / 1000.0;

    // rememeber query
    freq_set = true;
  }
}

StopWatchWin::~StopWatchWin() { }
// functions, d

////////////////////////////////////////////////////////////////////////////////
//! Start time measurement
////////////////////////////////////////////////////////////////////////////////
void
StopWatchWin::start()
{
  QueryPerformanceCounter((LARGE_INTEGER*)&start_time);
  running = true;
}

////////////////////////////////////////////////////////////////////////////////
//! Stop time measurement and increment add to the current diff_time summation
//! variable. Also increment the number of times this clock has been run.
////////////////////////////////////////////////////////////////////////////////
void
StopWatchWin::stop()
{
  QueryPerformanceCounter((LARGE_INTEGER*)&end_time);
  diff_time = (float)
    (((double)end_time.QuadPart - (double)start_time.QuadPart) / freq);

  total_time += diff_time;
  clock_sessions++;
  running = false;
}

////////////////////////////////////////////////////////////////////////////////
//! Reset the timer to 0. Does not change the timer running state but does 
//! recapture this point in time as the current start time if it is running.
////////////////////////////////////////////////////////////////////////////////
void
StopWatchWin::reset()
{
  diff_time = 0;
  total_time = 0;
  clock_sessions = 0;
  if (running)
    QueryPerformanceCounter((LARGE_INTEGER*)&start_time);
}


////////////////////////////////////////////////////////////////////////////////
//! Time in msec. after start. If the stop watch is still running (i.e. there
//! was no call to stop()) then the elapsed time is returned added to the 
//! current diff_time sum, otherwise the current summed time difference alone
//! is returned.
////////////////////////////////////////////////////////////////////////////////
float
StopWatchWin::getTime()
{
  // Return the TOTAL time to date
  float retval = total_time;
  if (running)
  {
    LARGE_INTEGER temp;
    QueryPerformanceCounter((LARGE_INTEGER*)&temp);
    retval += (float)
      (((double)(temp.QuadPart - start_time.QuadPart)) / freq);
  }

  return retval;
}

////////////////////////////////////////////////////////////////////////////////
//! Time in msec. for a single run based on the total number of COMPLETED runs
//! and the total time.
////////////////////////////////////////////////////////////////////////////////
float
StopWatchWin::getAverageTime()
{
  return (clock_sessions > 0) ? (total_time / clock_sessions) : 0.0f;
}
#else
StopWatchLinux::StopWatchLinux() :
start_time(), diff_time(0.0), total_time(0.0),
running(false), clock_sessions(0)
{ }

// Destructor
StopWatchLinux~StopWatchLinux() { }

// functions, d

////////////////////////////////////////////////////////////////////////////////
//! Start time measurement
////////////////////////////////////////////////////////////////////////////////
void
StopWatchLinux::start() {
  gettimeofday(&start_time, 0);
  running = true;
}

////////////////////////////////////////////////////////////////////////////////
//! Stop time measurement and increment add to the current diff_time summation
//! variable. Also increment the number of times this clock has been run.
////////////////////////////////////////////////////////////////////////////////
void
StopWatchLinux::stop() {
  diff_time = getDiffTime();
  total_time += diff_time;
  running = false;
  clock_sessions++;
}

////////////////////////////////////////////////////////////////////////////////
//! Reset the timer to 0. Does not change the timer running state but does 
//! recapture this point in time as the current start time if it is running.
////////////////////////////////////////////////////////////////////////////////
void
StopWatchLinux::reset()
{
  diff_time = 0;
  total_time = 0;
  clock_sessions = 0;
  if (running)
    gettimeofday(&start_time, 0);
}

////////////////////////////////////////////////////////////////////////////////
//! Time in msec. after start. If the stop watch is still running (i.e. there
//! was no call to stop()) then the elapsed time is returned added to the 
//! current diff_time sum, otherwise the current summed time difference alone
//! is returned.
////////////////////////////////////////////////////////////////////////////////
float
StopWatchLinux::getTime()
{
  // Return the TOTAL time to date
  float retval = total_time;
  if (running) {
    retval += getDiffTime();
  }

  return retval;
}

////////////////////////////////////////////////////////////////////////////////
//! Time in msec. for a single run based on the total number of COMPLETED runs
//! and the total time.
////////////////////////////////////////////////////////////////////////////////
float
StopWatchLinux::getAverageTime()
{
  return (clock_sessions > 0) ? (total_time / clock_sessions) : 0.0f;
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
float
StopWatchLinux::getDiffTime()
{
  struct timeval t_time;
  gettimeofday(&t_time, 0);

  // time difference in milli-seconds
  return  (float)(1000.0 * (t_time.tv_sec - start_time.tv_sec)
    + (0.001 * (t_time.tv_usec - start_time.tv_usec)));
}
#endif // _WIN32

////////////////////////////////////////////////////////////////////////////////
//! Timer functionality exported

////////////////////////////////////////////////////////////////////////////////
//! Create a new timer
//! @return true if a time has been created, otherwise false
//! @param  name of the new timer, 0 if the creation failed
////////////////////////////////////////////////////////////////////////////////
bool
sdkCreateTimer(StopWatchInterface **timer_interface)
{
  //printf("sdkCreateTimer called object %08x\n", (void *)*timer_interface);
#ifdef _WIN32
  *timer_interface = (StopWatchInterface *)new StopWatchWin();
#else
  *timer_interface = (StopWatchInterface *)new StopWatchLinux();
#endif
  return (*timer_interface != NULL) ? true : false;
}


////////////////////////////////////////////////////////////////////////////////
//! Delete a timer
//! @return true if a time has been deleted, otherwise false
//! @param  name of the timer to delete
////////////////////////////////////////////////////////////////////////////////
bool
sdkDeleteTimer(StopWatchInterface **timer_interface)
{
  //printf("sdkDeleteTimer called object %08x\n", (void *)*timer_interface);
  if (*timer_interface) delete *timer_interface;
  return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Start the time with name \a name
//! @param name  name of the timer to start
////////////////////////////////////////////////////////////////////////////////
bool
sdkStartTimer(StopWatchInterface **timer_interface)
{
  //printf("sdkStartTimer called object %08x\n", (void *)*timer_interface);
  if (*timer_interface) (*timer_interface)->start();
  return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Stop the time with name \a name. Does not reset.
//! @param name  name of the timer to stop
////////////////////////////////////////////////////////////////////////////////
bool
sdkStopTimer(StopWatchInterface **timer_interface)
{
  // printf("sdkStopTimer called object %08x\n", (void *)*timer_interface);
  if (*timer_interface) (*timer_interface)->stop();
  return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Resets the timer's counter.
//! @param name  name of the timer to reset.
////////////////////////////////////////////////////////////////////////////////
bool
sdkResetTimer(StopWatchInterface **timer_interface)
{
  // printf("sdkResetTimer called object %08x\n", (void *)*timer_interface);
  if (*timer_interface) (*timer_interface)->reset();
  return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Return the average time for timer execution as the total time
//! for the timer dividied by the number of completed (stopped) runs the timer 
//! has made.
//! Excludes the current running time if the timer is currently running.
//! @param name  name of the timer to return the time of
////////////////////////////////////////////////////////////////////////////////
float
sdkGetAverageTimerValue(StopWatchInterface **timer_interface)
{
  //  printf("sdkGetAverageTimerValue called object %08x\n", (void *)*timer_interface);
  if (*timer_interface)
    return (*timer_interface)->getAverageTime();
  else
    return 0.0f;
}

////////////////////////////////////////////////////////////////////////////////
//! Total execution time for the timer over all runs since the last reset
//! or timer creation.
//! @param name  name of the timer to obtain the value of.
////////////////////////////////////////////////////////////////////////////////
float
sdkGetTimerValue(StopWatchInterface **timer_interface)
{
  // printf("sdkGetTimerValue called object %08x\n", (void *)*timer_interface);
  if (*timer_interface)
    return (*timer_interface)->getTime();
  else
    return 0.0f;
}