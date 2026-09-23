(omega-dev-float-except)=

## Floating-point Exceptions

Omega provides functions that enable or disable selected
floating-point exceptions. Floating-point exception handling depends
on the target architecture, and is not always supported.

To enable floating-point exceptions `Exceptions` call
```c++
Error Err = enableFloatExceptions(Exceptions);
```
The returned `Err` value indicates whether this succeeded.
`Exceptions` is an integer mask, which can be composed of exception
types defined in `<cfenv>`. They are implementation defined, but
typically include
- `FE_DIVBYZERO`
- `FE_INEXACT`
- `FE_INVALID`
- `FE_OVERFLOW`
- `FE_UNDERFLOW`

For example, to enable divide-by-zero and underflow exceptions do
```c++
Error Err = enableFloatExceptions(FE_DIVBYZERO | FE_OVERFLOW);
```

You can call `enableFloatExceptions` without an argument, which
enables
- `FE_DIVBYZERO`
- `FE_INVALID`
- `FE_OVERFLOW`

To disable floating-point exceptions `Exceptions` call
```c++
Error Err = disableFloatExceptions(Exceptions);
```
Calling this function without an argument disables all exceptions.

Omega also provides a helper function `enableFloatExceptionsInTests(Exceptions)`.
This function is used in unit tests to conditionally call
`enableFloatExceptions(Exceptions)` based on the compiler and specified CMake options.
See [User's Guide](#omega-user-float-except) for details.
