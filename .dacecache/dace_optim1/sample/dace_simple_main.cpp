#include <cstdlib>
#include "../include/dace_simple.h"

int main(int argc, char **argv) {
    dace_simpleHandle_t handle;
    float * __restrict__ Input = (float*) calloc(131072, sizeof(float));
    float * __restrict__ Output = (float*) calloc(262144, sizeof(float));
    float * __restrict__ kernel = (float*) calloc(864, sizeof(float));


    handle = __dace_init_dace_simple();
    __program_dace_simple(handle, Input, Output, kernel);
    __dace_exit_dace_simple(handle);

    free(Input);
    free(Output);
    free(kernel);


    return 0;
}
