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

// Helper Timer Functions (this is the inlined version)

#ifndef HELPER_TIMER_H
#define HELPER_TIMER_H

// includes, system
#include <vector>

// includes, project
#include "my_exception.h"

// Definition of the StopWatch Interface, this is used if we don't want to use the CUT functions
// But rather in a self contained class interface
class StopWatchInterface
{
public:
  StopWatchInterface() {};
  virtual ~StopWatchInterface() {};

public:
  //! Start time measurement
  virtual void start() = 0;

  //! Stop time measurement
  virtual void stop() = 0;

  //! Reset time counters to zero
  virtual void reset() = 0;

  //! Time in msec. after start. If the stop watch is still running (i.e. there
  //! was no call to stop()) then the elapsed time is returned, otherwise the
  //! time between the last start() and stop call is returned
  virtual float getTime() = 0;

  //! Mean time to date based on the number of times the stopwatch has been 
  //! _stopped_ (ie finished sessions) and the current total time
  virtual float getAverageTime() = 0;
};


//////////////////////////////////////////////////////////////////
// Begin Stopwatch timer class definitions for all OS platforms //
//////////////////////////////////////////////////////////////////
#ifdef WIN32
// includes, system
#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>
#undef min
#undef max

//! Windows specific implementation of StopWatch
class StopWatchWin : public StopWatchInterface
{
public:
  //! Constructor, default
  StopWatchWin();

  // Destructor
  ~StopWatchWin();

public:
  //! Start time measurement
  void start();

  //! Stop time measurement
  void stop();

  //! Reset time counters to zero
  void reset();

  //! Time in msec. after start. If the stop watch is still running (i.e. there
  //! was no call to stop()) then the elapsed time is returned, otherwise the
  //! time between the last start() and stop call is returned
  float getTime();

  //! Mean time to date based on the number of times the stopwatch has been 
  //! _stopped_ (ie finished sessions) and the current total time
  float getAverageTime();

private:
  // member variables

  //! Start of measurement
  LARGE_INTEGER  start_time;
  //! End of measurement
  LARGE_INTEGER  end_time;

  //! Time difference between the last start and stop
  float  diff_time;

  //! TOTAL time difference between starts and stops
  float  total_time;

  //! flag if the stop watch is running
  bool running;

  //! Number of times clock has been started
  //! and stopped to allow averaging
  int clock_sessions;

  //! tick frequency
  double  freq;

  //! flag if the frequency has been set
  bool  freq_set;
};
#else
// Declarations for Stopwatch on Linux and Mac OSX
// includes, system
#include <ctime>
#include <sys/time.h>

//! Windows specific implementation of StopWatch
class StopWatchLinux : public StopWatchInterface
{
public:
  //! Constructor, default
  StopWatchLinux();

  // Destructor
  virtual ~StopWatchLinux();
public:
  //! Start time measurement
  void start();

  //! Stop time measurement
  void stop();

  //! Reset time counters to zero
  void reset();

  //! Time in msec. after start. If the stop watch is still running (i.e. there
  //! was no call to stop()) then the elapsed time is returned, otherwise the
  //! time between the last start() and stop call is returned
  float getTime();

  //! Mean time to date based on the number of times the stopwatch has been 
  //! _stopped_ (ie finished sessions) and the current total time
  float getAverageTime();

private:

  // helper functions

  //! Get difference between start time and current time
  float getDiffTime();

private:

  // member variables

  //! Start of measurement
  struct timeval  start_time;

  //! Time difference between the last start and stop
  float  diff_time;

  //! TOTAL time difference between starts and stops
  float  total_time;

  //! flag if the stop watch is running
  bool running;

  //! Number of times clock has been started
  //! and stopped to allow averaging
  int clock_sessions;
};

#endif // _WIN32

////////////////////////////////////////////////////////////////////////////////
//! Timer functionality exported

////////////////////////////////////////////////////////////////////////////////
//! Create a new timer
//! @return true if a time has been created, otherwise false
//! @param  name of the new timer, 0 if the creation failed
////////////////////////////////////////////////////////////////////////////////
bool sdkCreateTimer(StopWatchInterface **timer_interface);
////////////////////////////////////////////////////////////////////////////////
//! Delete a timer
//! @return true if a time has been deleted, otherwise false
//! @param  name of the timer to delete
////////////////////////////////////////////////////////////////////////////////
bool sdkDeleteTimer(StopWatchInterface **timer_interface);
////////////////////////////////////////////////////////////////////////////////
//! Start the time with name \a name
//! @param name  name of the timer to start
////////////////////////////////////////////////////////////////////////////////
bool sdkStartTimer(StopWatchInterface **timer_interface);
////////////////////////////////////////////////////////////////////////////////
//! Stop the time with name \a name. Does not reset.
//! @param name  name of the timer to stop
////////////////////////////////////////////////////////////////////////////////
bool sdkStopTimer(StopWatchInterface **timer_interface);
////////////////////////////////////////////////////////////////////////////////
//! Resets the timer's counter.
//! @param name  name of the timer to reset.
////////////////////////////////////////////////////////////////////////////////
bool sdkResetTimer(StopWatchInterface **timer_interface);
////////////////////////////////////////////////////////////////////////////////
//! Return the average time for timer execution as the total time
//! for the timer dividied by the number of completed (stopped) runs the timer 
//! has made.
//! Excludes the current running time if the timer is currently running.
//! @param name  name of the timer to return the time of
////////////////////////////////////////////////////////////////////////////////
float sdkGetAverageTimerValue(StopWatchInterface **timer_interface);
////////////////////////////////////////////////////////////////////////////////
//! Total execution time for the timer over all runs since the last reset
//! or timer creation.
//! @param name  name of the timer to obtain the value of.
////////////////////////////////////////////////////////////////////////////////
float sdkGetTimerValue(StopWatchInterface **timer_interface);

#endif // HELPER_TIMER_H