#include <cstdlib>
#include "../include/dace_simple.h"

int main(int argc, char **argv) {
    dace_simpleHandle_t handle;
    int d_inchannels = 42;
    int d_indepth = 42;
    int d_inheight = 42;
    int d_inwidth = 42;
    int d_outchannels = 42;
    int d_outdepth = 42;
    int d_outheight = 42;
    int d_outwidth = 42;
    float * __restrict__ Input = (float*) calloc((((d_inchannels * d_indepth) * d_inheight) * d_inwidth), sizeof(float));
    float * __restrict__ Output = (float*) calloc((((d_outchannels * d_outdepth) * d_outheight) * d_outwidth), sizeof(float));
    float * __restrict__ kernel = (float*) calloc(((27 * d_inchannels) * d_outchannels), sizeof(float));


    handle = __dace_init_dace_simple(d_inchannels, d_indepth, d_inheight, d_inwidth, d_outchannels, d_outdepth, d_outheight, d_outwidth);
    __program_dace_simple(handle, Input, Output, kernel, d_inchannels, d_indepth, d_inheight, d_inwidth, d_outchannels, d_outdepth, d_outheight, d_outwidth);
    __dace_exit_dace_simple(handle);

    free(Input);
    free(Output);
    free(kernel);


    return 0;
}
