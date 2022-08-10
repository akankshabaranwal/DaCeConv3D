#include <dace/dace.h>
typedef void * dace_simpleHandle_t;
extern "C" dace_simpleHandle_t __dace_init_dace_simple();
extern "C" void __dace_exit_dace_simple(dace_simpleHandle_t handle);
extern "C" void __program_dace_simple(dace_simpleHandle_t handle, float * __restrict__ Input, float * __restrict__ Output, float * __restrict__ kernel);
