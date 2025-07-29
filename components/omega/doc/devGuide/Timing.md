(omega-dev-timing)=

# Timing

The timing module provides free functions for timing selected regions of code.
It builds upon the E3SM Pacer library for wall clock timing and integrates,
whenever possible, platform-specific marker APIs.

# Initialization and reading config options

To initialize the timing module call
```c++
initTiming();
```

After initialization, Omega timers can be used, but only timers with the default level 0 will
be active. To get configuration options, including the timing level, from the config file call
```c++
readTimingConfig();
```

# Finalization

To finalize the timing module and write out timing data call
```c++
finalizeTiming(FileName);
```

# Basic timer use

To time a region of code enclose it with calls to `timerStart` and `timerStop` functions, like so
```c++
timerStart(Name, Level);
// region of code to be timed
timerStop(Name, Level);
```
These functions take a string `Name` and a non-negative integer `Level`.
The added timer will be active only if the timing level set in the config file is greater or equal to `Level`.

# Advanced timer use

The `timerStart` and `timerStop` functions can accept another argument, which is a flag or a collection of flags
that modify their behavior. There are three optional flags:
- `AddMpiBarrier`
- `PrefixParent`
- `DisableChildTimers`
that can be aribtrarily combined, for example
```c++
 timerStart(Name, Level, AddMpiBarrier | PrefixParent);
```

The `AddMpiBarrier` flag adds an MPI barrier before starting/stopping the timer,
but after the automatic Kokkos fence.
This ensures that all GPU work is done before any process arrives at the barrier.
Its main use is for timing MPI communication.
Since MPI barriers are costly, this flag should be used only with high timing levels
or for coarse-grained timing of large parts of the code.
This is the only timer flag that can be set independently in `timerStart` and `timerStop`.

The `PrefixParent` flag prefixes the timer name with the name of its enclosing parent timer,
if such a parent timer exits.
This is useful when timers are added inside general purpose routines
that are called from many places in the code, such as halo exchange.

The `DisableChildTimers` flag disables all child timers within the timed region.
This is useful mainly when the first call to some function takes much longer than subsequent calls.
In that case, it might be desirable to have a separate timer for the first call with its child timers disabled.
