#include "basic_transform.h"
#include <npp.h>

extern "C" __global__ void DLinearResize(uint8_t *dImage, uint8_t *dImageOut,
                                             uint32_t swidth, uint32_t sheight, uint32_t twidth, uint32_t theight,
                                             float wratio, float hratio) {
    uint32_t xIdx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t yIdx = threadIdx.y + blockIdx.y * blockDim.y;

    short short_temp, cbufx_1, cbufx_2, cbufy_1, cbufy_2;
    float x_float, y_float;
    int x_int, y_int;

    uint32_t top, left, ti, si_1, si_2, si_3, si_4;
    uint32_t sstep = 3 * swidth;

    if (twidth > theight) {
        top = (uint32_t) ((float) (twidth - theight) / 2.) * twidth * 3;
        ti = top + yIdx * 3 * twidth + xIdx * 3;
        if (xIdx < twidth && yIdx < theight) {
            // x direction calculation
            x_float = (float) ((xIdx + 0.5) * wratio - 0.5);
            x_int = (int) (x_float);
            x_float = x_float - (float) (x_int);
            //x_int = (x_int < swidth - 1) ? x_int : swidth - 2;
            x_int = (0 > x_int) ? 0 : x_int;
            short_temp = (short) ((1. - x_float) * 2048.);
            cbufx_1 = short_temp > 2048 ? 2048 : short_temp;
            cbufx_2 = 2048 - cbufx_1;

            // y direction calculation
            y_float = (float) ((yIdx + 0.5) * hratio - 0.5);
            y_int = (int) (y_float);
            y_float = y_float - (float) (y_int);
            //y_int = (y_int < sheight - 1) ? y_int : sheight - 2;
            y_int = (0 > y_int) ? 0 : y_int;
            short_temp = (short) ((1. - y_float) * 2048.);
            cbufy_1 = short_temp > 2048 ? 2048 : short_temp;
            cbufy_2 = 2048 - cbufy_1;

            // data to data
            if (x_int < swidth - 1 && y_int < sheight - 1) {
                si_1 = y_int * sstep + x_int * 3;
                si_2 = (y_int + 1) * sstep + x_int * 3;
                si_3 = y_int * sstep + (x_int + 1) * 3;
                si_4 = (y_int + 1) * sstep + (x_int + 1) * 3;

                dImageOut[ti] = ((short) dImage[si_1] * cbufx_1 * cbufy_1 + (short) dImage[si_2] * cbufx_1 * cbufy_2 \
 + (short) dImage[si_3] * cbufx_2 * cbufy_1 + (short) dImage[si_4] * cbufx_2 * cbufy_2) >> 22;
                dImageOut[ti + 1] =
                        ((short) dImage[si_1 + 1] * cbufx_1 * cbufy_1 + (short) dImage[si_2 + 1] * cbufx_1 * cbufy_2 \
 + (short) dImage[si_3 + 1] * cbufx_2 * cbufy_1 + (short) dImage[si_4 + 1] * cbufx_2 * cbufy_2) >> 22;
                dImageOut[ti + 2] =
                        ((short) dImage[si_1 + 2] * cbufx_1 * cbufy_1 + (short) dImage[si_2 + 2] * cbufx_1 * cbufy_2 \
 + (short) dImage[si_3 + 2] * cbufx_2 * cbufy_1 + (short) dImage[si_4 + 2] * cbufx_2 * cbufy_2) >> 22;
            } else if (x_int >= swidth - 1 && y_int < sheight - 1) { // right edge
                si_1 = y_int * sstep + x_int * 3;
                si_2 = (y_int + 1) * sstep + x_int * 3;

                dImageOut[ti] = ((short) dImage[si_1] * cbufx_1 * cbufy_1 + (short) dImage[si_2] * cbufx_1 * cbufy_2 \
 + (short) dImage[si_1] * cbufx_2 * cbufy_1 + (short) dImage[si_2] * cbufx_2 * cbufy_2) >> 22;
                dImageOut[ti + 1] =
                        ((short) dImage[si_1 + 1] * cbufx_1 * cbufy_1 + (short) dImage[si_2 + 1] * cbufx_1 * cbufy_2 \
 + (short) dImage[si_1 + 1] * cbufx_2 * cbufy_1 + (short) dImage[si_2 + 1] * cbufx_2 * cbufy_2) >> 22;
                dImageOut[ti + 2] =
                        ((short) dImage[si_1 + 2] * cbufx_1 * cbufy_1 + (short) dImage[si_2 + 2] * cbufx_1 * cbufy_2 \
 + (short) dImage[si_1 + 2] * cbufx_2 * cbufy_1 + (short) dImage[si_2 + 2] * cbufx_2 * cbufy_2) >> 22;
            } else if (x_int < swidth - 1 && y_int >= sheight - 1) { // bottom edge
                si_1 = y_int * sstep + x_int * 3;
                si_3 = y_int * sstep + (x_int + 1) * 3;

                dImageOut[ti] = ((short) dImage[si_1] * cbufx_1 * cbufy_1 + (short) dImage[si_1] * cbufx_1 * cbufy_2 \
 + (short) dImage[si_3] * cbufx_2 * cbufy_1 + (short) dImage[si_3] * cbufx_2 * cbufy_2) >> 22;
                dImageOut[ti + 1] =
                        ((short) dImage[si_1 + 1] * cbufx_1 * cbufy_1 + (short) dImage[si_1 + 1] * cbufx_1 * cbufy_2 \
 + (short) dImage[si_3 + 1] * cbufx_2 * cbufy_1 + (short) dImage[si_3 + 1] * cbufx_2 * cbufy_2) >> 22;
                dImageOut[ti + 2] =
                        ((short) dImage[si_1 + 2] * cbufx_1 * cbufy_1 + (short) dImage[si_1 + 2] * cbufx_1 * cbufy_2 \
 + (short) dImage[si_3 + 2] * cbufx_2 * cbufy_1 + (short) dImage[si_3 + 2] * cbufx_2 * cbufy_2) >> 22;
            } else {
                si_1 = y_int * sstep + x_int * 3;

                dImageOut[ti] = ((short) dImage[si_1] * cbufx_1 * cbufy_1 + (short) dImage[si_1] * cbufx_1 * cbufy_2 \
 + (short) dImage[si_1] * cbufx_2 * cbufy_1 + (short) dImage[si_1] * cbufx_2 * cbufy_2) >> 22;
                dImageOut[ti + 1] =
                        ((short) dImage[si_1 + 1] * cbufx_1 * cbufy_1 + (short) dImage[si_2 + 1] * cbufx_1 * cbufy_2 \
 + (short) dImage[si_1 + 1] * cbufx_2 * cbufy_1 + (short) dImage[si_1 + 1] * cbufx_2 * cbufy_2) >> 22;
                dImageOut[ti + 2] =
                        ((short) dImage[si_1 + 2] * cbufx_1 * cbufy_1 + (short) dImage[si_2 + 2] * cbufx_1 * cbufy_2 \
 + (short) dImage[si_1 + 2] * cbufx_2 * cbufy_1 + (short) dImage[si_1 + 2] * cbufx_2 * cbufy_2) >> 22;
            }
        }
    } else {
        left = (uint32_t) ((theight - twidth) / 2.);
        ti = yIdx * 3 * theight + (xIdx + left) * 3;
        if (xIdx < twidth && yIdx < theight) {
            // x direction calculation
            x_float = (float) ((xIdx + 0.5) * wratio - 0.5);
            x_int = (int) (x_float);
            x_float = x_float - (float) (x_int);
            //x_int = (x_int < swidth - 1) ? x_int : swidth - 2;
            x_int = (0 > x_int) ? 0 : x_int;
            short_temp = (short) ((1. - x_float) * 2048.);
            cbufx_1 = short_temp > 2048 ? 2048 : short_temp;
            cbufx_2 = 2048 - cbufx_1;

            // y direction calculation
            y_float = (float) ((yIdx + 0.5) * hratio - 0.5);
            y_int = (int) (y_float);
            y_float = y_float - (float) (y_int);
            //y_int = (y_int < sheight - 1) ? y_int : sheight - 2;
            y_int = (0 > y_int) ? 0 : y_int;
            short_temp = (short) ((1. - y_float) * 2048.);
            cbufy_1 = short_temp > 2048 ? 2048 : short_temp;
            cbufy_2 = 2048 - cbufy_1;

            // data to data
            if (x_int < swidth - 1 && y_int < sheight - 1) {
                si_1 = y_int * sstep + x_int * 3;
                si_2 = (y_int + 1) * sstep + x_int * 3;
                si_3 = y_int * sstep + (x_int + 1) * 3;
                si_4 = (y_int + 1) * sstep + (x_int + 1) * 3;

                dImageOut[ti] = ((short) dImage[si_1] * cbufx_1 * cbufy_1 + (short) dImage[si_2] * cbufx_1 * cbufy_2 \
 + (short) dImage[si_3] * cbufx_2 * cbufy_1 + (short) dImage[si_4] * cbufx_2 * cbufy_2) >> 22;
                dImageOut[ti + 1] =
                        ((short) dImage[si_1 + 1] * cbufx_1 * cbufy_1 + (short) dImage[si_2 + 1] * cbufx_1 * cbufy_2 \
 + (short) dImage[si_3 + 1] * cbufx_2 * cbufy_1 + (short) dImage[si_4 + 1] * cbufx_2 * cbufy_2) >> 22;
                dImageOut[ti + 2] =
                        ((short) dImage[si_1 + 2] * cbufx_1 * cbufy_1 + (short) dImage[si_2 + 2] * cbufx_1 * cbufy_2 \
 + (short) dImage[si_3 + 2] * cbufx_2 * cbufy_1 + (short) dImage[si_4 + 2] * cbufx_2 * cbufy_2) >> 22;
            } else if (x_int >= swidth - 1 && y_int < sheight - 1) { // right edge
                si_1 = y_int * sstep + x_int * 3;
                si_2 = (y_int + 1) * sstep + x_int * 3;

                dImageOut[ti] = ((short) dImage[si_1] * cbufx_1 * cbufy_1 + (short) dImage[si_2] * cbufx_1 * cbufy_2 \
 + (short) dImage[si_1] * cbufx_2 * cbufy_1 + (short) dImage[si_2] * cbufx_2 * cbufy_2) >> 22;
                dImageOut[ti + 1] =
                        ((short) dImage[si_1 + 1] * cbufx_1 * cbufy_1 + (short) dImage[si_2 + 1] * cbufx_1 * cbufy_2 \
 + (short) dImage[si_1 + 1] * cbufx_2 * cbufy_1 + (short) dImage[si_2 + 1] * cbufx_2 * cbufy_2) >> 22;
                dImageOut[ti + 2] =
                        ((short) dImage[si_1 + 2] * cbufx_1 * cbufy_1 + (short) dImage[si_2 + 2] * cbufx_1 * cbufy_2 \
 + (short) dImage[si_1 + 2] * cbufx_2 * cbufy_1 + (short) dImage[si_2 + 2] * cbufx_2 * cbufy_2) >> 22;
            } else if (x_int < swidth - 1 && y_int >= sheight - 1) { // bottom edge
                si_1 = y_int * sstep + x_int * 3;
                si_3 = y_int * sstep + (x_int + 1) * 3;

                dImageOut[ti] = ((short) dImage[si_1] * cbufx_1 * cbufy_1 + (short) dImage[si_1] * cbufx_1 * cbufy_2 \
 + (short) dImage[si_3] * cbufx_2 * cbufy_1 + (short) dImage[si_3] * cbufx_2 * cbufy_2) >> 22;
                dImageOut[ti + 1] =
                        ((short) dImage[si_1 + 1] * cbufx_1 * cbufy_1 + (short) dImage[si_1 + 1] * cbufx_1 * cbufy_2 \
 + (short) dImage[si_3 + 1] * cbufx_2 * cbufy_1 + (short) dImage[si_3 + 1] * cbufx_2 * cbufy_2) >> 22;
                dImageOut[ti + 2] =
                        ((short) dImage[si_1 + 2] * cbufx_1 * cbufy_1 + (short) dImage[si_1 + 2] * cbufx_1 * cbufy_2 \
 + (short) dImage[si_3 + 2] * cbufx_2 * cbufy_1 + (short) dImage[si_3 + 2] * cbufx_2 * cbufy_2) >> 22;
            } else {
                si_1 = y_int * sstep + x_int * 3;

                dImageOut[ti] = ((short) dImage[si_1] * cbufx_1 * cbufy_1 + (short) dImage[si_1] * cbufx_1 * cbufy_2 \
 + (short) dImage[si_1] * cbufx_2 * cbufy_1 + (short) dImage[si_1] * cbufx_2 * cbufy_2) >> 22;
                dImageOut[ti + 1] =
                        ((short) dImage[si_1 + 1] * cbufx_1 * cbufy_1 + (short) dImage[si_2 + 1] * cbufx_1 * cbufy_2 \
 + (short) dImage[si_1 + 1] * cbufx_2 * cbufy_1 + (short) dImage[si_1 + 1] * cbufx_2 * cbufy_2) >> 22;
                dImageOut[ti + 2] =
                        ((short) dImage[si_1 + 2] * cbufx_1 * cbufy_1 + (short) dImage[si_2 + 2] * cbufx_1 * cbufy_2 \
 + (short) dImage[si_1 + 2] * cbufx_2 * cbufy_1 + (short) dImage[si_1 + 2] * cbufx_2 * cbufy_2) >> 22;
            }
        }

    }
}

void PicResize(uint8_t *dImage, uint8_t *dImageOut, uint32_t swidth, uint32_t sheight, uint32_t twidth,
                uint32_t theight) {

    dim3 block(32, 16, 1);
    dim3 grid(0, 0, 1);

    grid.x = (int) ((twidth + block.x - 1) / block.x);
    grid.y = (int) ((theight + block.y - 1) / block.y);

    float wratio = (float) swidth / (float) twidth;
    float hratio = (float) sheight / (float) theight;

    DLinearResize << < grid, block >> > (dImage, dImageOut, swidth, sheight, twidth, theight, wratio, hratio);
    cudaDeviceSynchronize();
}



extern "C" __global__ void
DNormalize(uint8_t *dImage, float *dImageOut, uint32_t swidth, uint32_t sheight, uint32_t sstep) {
    uint32_t xIdx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t yIdx = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t idx = yIdx * sstep + xIdx * 3;
    dImageOut[yIdx * swidth + xIdx] = (float) dImage[idx + 2] / 255.;
    dImageOut[swidth * sheight + yIdx * swidth + xIdx] = (float) dImage[idx + 1] / 255.;
    dImageOut[2 * swidth * sheight + yIdx * swidth + xIdx] = (float) dImage[idx] / 255.;
}

void PicNormalize(uint8_t *dImage, float *dImageOut, uint32_t swidth, uint32_t sheight) {
    dim3 block(32, 16, 1);
    dim3 grid(0, 0, 1);
    grid.x = (int) ((swidth + block.x - 1) / block.x);
    grid.y = (int) ((sheight + block.y - 1) / block.y);
    uint32_t sstep = 3 * swidth;
    DNormalize << < grid, block >> > (dImage, dImageOut, swidth, sheight, sstep);
    cudaDeviceSynchronize();
}