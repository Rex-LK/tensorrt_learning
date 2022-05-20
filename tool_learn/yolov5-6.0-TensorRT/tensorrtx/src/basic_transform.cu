#include "basic_transform.h"
#include <npp.h>

extern "C" __global__ void dRGB2LinearResize(uint8_t *dImage, uint8_t *dImageOut,
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

void RGB2Resize(uint8_t *dImage, uint8_t *dImageOut, uint32_t swidth, uint32_t sheight, uint32_t twidth,
                uint32_t theight) {
    /*
     * @brief: resize and border, resize and add border on the (left and right) or (top and bottom) using equal scaling
     *          however, swidth is chaning to twidth using equal scaling
     *          such as swidth: 1920 * sheight: 1080 ---> twidth: 416 * sheight: 234, and output is 416 * 416
     * @swidth: such as 1920, orginal image width
     * @twidth: such as 416 or ratio_width, target image width
     */
    dim3 block(32, 16, 1);
    dim3 grid(0, 0, 1);

    grid.x = (int) ((twidth + block.x - 1) / block.x);
    grid.y = (int) ((theight + block.y - 1) / block.y);

    float wratio = (float) swidth / (float) twidth;
    float hratio = (float) sheight / (float) theight;

    dRGB2LinearResize << < grid, block >> > (dImage, dImageOut, swidth, sheight, twidth, theight, wratio, hratio);
    cudaDeviceSynchronize();
}

extern "C" __global__ void
dRGB2Gray3(uint8_t *dImage, uint8_t *dImageOut, uint32_t width, uint32_t height, uint32_t step){
    uint32_t xIdx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t yIdx = threadIdx.y + blockIdx.y * blockDim.y;
    if(xIdx >= width || yIdx >= height){
        return;
    }
    uint32_t idx = yIdx * step + xIdx * 3;
    uint8_t temp = round(dImage[idx] * 0.114 + dImage[idx + 1] * 0.587 + dImage[idx + 2] * 0.299);

    dImageOut[idx] = temp;
    dImageOut[idx + 1] = temp;
    dImageOut[idx + 2] = temp;
}

void RGB2Gray3(uint8_t *dImage, uint8_t *dImageOut, uint32_t width, uint32_t height){
    /*
     * @brief: BGR to Gray and then copy to 3 channels
     */
    dim3 block(32, 16, 1);
    dim3 grid(0, 0, 1);
    grid.x = (int) ((width + block.x - 1) / block.x);
    grid.y = (int) ((height + block.y - 1) / block.y);
    uint32_t step = 3 * width;
    dRGB2Gray3 << < grid, block >> > (dImage, dImageOut, width, height, step);
    cudaDeviceSynchronize();
}

extern "C" __global__ void
dBigRGB2Resize(uint8_t *dImage, uint8_t *dImageOut, uint32_t swidth, uint32_t sheight, uint32_t twidth, uint32_t left){
     uint32_t xIdx = threadIdx.x + blockIdx.x * blockDim.x;
     uint32_t yIdx = threadIdx.y + blockIdx.y * blockDim.y;
     uint32_t sIdx = yIdx * swidth * 3 + xIdx * 3;
     uint32_t tIdx = yIdx * twidth * 3 + (xIdx + left) * 3;
     if(xIdx < swidth && yIdx < sheight){
         dImageOut[tIdx] = dImage[sIdx];
         dImageOut[tIdx + 1] = dImage[sIdx + 1];
         dImageOut[tIdx + 2] = dImage[sIdx + 2];
     }
}



extern "C" __global__ void
dRGB2Normalize(uint8_t *dImage, float *dImageOut, uint32_t swidth, uint32_t sheight, uint32_t sstep) {
    uint32_t xIdx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t yIdx = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t idx = yIdx * sstep + xIdx * 3;
    dImageOut[yIdx * swidth + xIdx] = (float) dImage[idx + 2] / 255.;
    dImageOut[swidth * sheight + yIdx * swidth + xIdx] = (float) dImage[idx + 1] / 255.;
    dImageOut[2 * swidth * sheight + yIdx * swidth + xIdx] = (float) dImage[idx] / 255.;
}

void RGB2Normalize(uint8_t *dImage, float *dImageOut, uint32_t swidth, uint32_t sheight) {
    /*
     * @brief: normalize the image, pixel = pixel / 255.
     * @type: RGBRGBRGBRGBRGBRGBRGB convert to RRRRRRRRRRRRGGGGGGGGGGGGGBBBBBBBBBBBBBB
     */
    dim3 block(32, 16, 1);
    dim3 grid(0, 0, 1);
    grid.x = (int) ((swidth + block.x - 1) / block.x);
    grid.y = (int) ((sheight + block.y - 1) / block.y);
    uint32_t sstep = 3 * swidth;
    dRGB2Normalize << < grid, block >> > (dImage, dImageOut, swidth, sheight, sstep);
    cudaDeviceSynchronize();
}

extern "C" __global__ void
undistort_kernel(uint8_t *dImage, uint8_t *dImageout, int width, int height, float k1, float k2, float p1,
                 float p2, float k3, float fx, float fy, float cx, float cy) {
    float u_distorted = 0, v_distorted = 0;
    float x1, y1, x2, y2, r2;
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int undistort_index = 0;
    int u = index / (height * 3);
    int v = index % (height * 3) / 3;
    x1 = (u - cx) / fx;
    y1 = (v - cy) / fy;
    r2 = pow(x1, 2) + pow(y1, 2);
    x2 = x1 * (1 + k1 * r2 + k2 * pow(r2, 2) + k3 * pow(r2, 3)) + 2 * p1 * x1 * y1 + p2 * (r2 + 2 * x1 * x1);
    y2 = y1 * (1 + k1 * r2 + k2 * pow(r2, 2) + k3 * pow(r2, 3)) + p1 * (r2 + 2 * y1 * y1) + 2 * p2 * x1 * y1;
    u_distorted = fx * x2 + cx;
    v_distorted = fy * y2 + cy;
    if (u_distorted >= 0 && v_distorted >= 0 && u_distorted < width && v_distorted < height) {
        undistort_index = round(v_distorted) * width * 3 + round(u_distorted) * 3;
        dImageout[v * width * 3 + u * 3] = dImage[undistort_index];
        dImageout[v * width * 3 + u * 3 + 1] = dImage[undistort_index + 1];
        dImageout[v * width * 3 + u * 3 + 2] = dImage[undistort_index + 2];
    } else {
        dImageout[v * width * 3 + u * 3] = 0;
        dImageout[v * width * 3 + u * 3 + 1] = 0;
        dImageout[v * width * 3 + u * 3 + 2] = 0;
    }

}

void undistort(float *param, uint8_t *input_image_gpu, uint8_t *output_image_gpu, int height, int width) {
    float k1 = param[0];
    float k2 = param[1];
    float p1 = param[2];
    float p2 = param[3];
    float k3 = param[4];
    float fx = param[5];
    float fy = param[6];
    float cx = param[7];
    float cy = param[8];
    undistort_kernel << < width * height, 3 >> >
                                          (input_image_gpu, output_image_gpu, width, height, k1, k2, p1, p2, k3, fx, fy, cx, cy);
    cudaDeviceSynchronize();
}

extern "C" __global__ void
affine_transform_kernel(uint8_t *dImage, uint8_t *dImageout, uint32_t height, uint32_t width,
                        uint32_t channel, float k1, float k2, float k3, float k4, float k5, float k6) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int step = width * channel;
    if (idx < height * width * channel) {
        int i = (idx % step) / channel;
        int j = idx / step;
        int k = idx % channel;
        float x = k1 * i + k2 * j + k3;
        int x1 = floor(x);
        int x2 = x1 + 1;
        float y = k4 * i + k5 * j + k6;
        int y1 = floor(y);
        int y2 = y1 + 1;
        if (y1 > 0 && y2 < height && x1 > 0 && x2 < width) {
            float a1 = (x2 - x) * (y2 - y) / (x2 - x1) / (y2 - y1);
            float a2 = (x - x1) * (y2 - y) / (x2 - x1) / (y2 - y1);
            float a3 = (x2 - x) * (y - y1) / (x2 - x1) / (y2 - y1);
            float a4 = (x - x1) * (y - y1) / (x2 - x1) / (y2 - y1);
            dImageout[idx] = a1 * dImage[y1 * step + x1 * channel + k] + a2 * dImage[y1 * step + x2 * channel + k] +
                             a3 * dImage[y2 * step + x1 * channel + k] + a4 * dImage[y2 * step + x2 * channel + k];
        }
    }
}

void
affine_transform(uint8_t *dImage, uint8_t *dImageout, uint32_t height, uint32_t width, uint32_t channel,
                 float *param) {
    dim3 block(512);
    dim3 grid((height * width * channel - 1) / 512 + 1);
    float k1 = param[0];
    float k2 = param[1];
    float k3 = param[2];
    float k4 = param[3];
    float k5 = param[4];
    float k6 = param[5];
    affine_transform_kernel << < grid, block >> > (dImage, dImageout, height, width, channel, k1, k2, k3, k4, k5, k6);
    cudaDeviceSynchronize();
}


extern "C" __global__ void
perspective_transform_kernel(uint8_t *dImage, uint8_t *dImageout, uint32_t height, uint32_t width,
                             uint32_t channel, float k1, float k2, float k3, float k4, float k5, float k6, float k7,
                             float k8, float k9) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int step = width * channel;
    if (idx < height * width * channel) {

        int i = (idx % step) / channel;
        int j = idx / step;
        int k = idx % channel;
        float x = (k1 * i + k2 * j + k3) / (k7 * i + k8 * j + k9);
        int x1 = floor(x);
        int x2 = x1 + 1;
        float y = (k4 * i + k5 * j + k6) / (k7 * i + k8 * j + k9);
        int y1 = floor(y);
        int y2 = y1 + 1;
        if (y1 > 0 && y2 < height && x1 > 0 && x2 < width) {
            float a1 = (x2 - x) * (y2 - y) / (x2 - x1) / (y2 - y1);
            float a2 = (x - x1) * (y2 - y) / (x2 - x1) / (y2 - y1);
            float a3 = (x2 - x) * (y - y1) / (x2 - x1) / (y2 - y1);
            float a4 = (x - x1) * (y - y1) / (x2 - x1) / (y2 - y1);
            dImageout[idx] = a1 * dImage[y1 * step + x1 * channel + k] + a2 * dImage[y1 * step + x2 * channel + k] +
                             a3 * dImage[y2 * step + x1 * channel + k] + a4 * dImage[y2 * step + x2 * channel + k];
        }
    }
}

void perspective_transform(uint8_t *dImage, uint8_t *dImageout, uint32_t height, uint32_t width,
                           uint32_t channel, float *param) {
    float k1 = param[0];
    float k2 = param[1];
    float k3 = param[2];
    float k4 = param[3];
    float k5 = param[4];
    float k6 = param[5];
    float k7 = param[6];
    float k8 = param[7];
    float k9 = param[8];

    dim3 block(512);
    dim3 grid((height * width * channel - 1) / 512 + 1);

    perspective_transform_kernel << < grid, block >> >
                                            (dImage, dImageout, height, width, channel, k1, k2, k3, k4, k5, k6, k7, k8, k9);
    cudaDeviceSynchronize();

}

extern "C" __global__ void groi_rect(uint8_t *dImage, uint8_t *dImageout, uint32_t swith,
                                     uint32_t sheight, uint32_t x, uint32_t y, uint32_t w, uint32_t h) {
    uint32_t xIdx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t yIdx = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t s_idx = yIdx * w * 3 + xIdx * 3;
    uint32_t t_idx = (yIdx + y) * swith * 3 + (xIdx + x) * 3;

    if (xIdx < w && yIdx < h) {
        dImageout[s_idx] = dImage[t_idx];
        dImageout[s_idx + 1] = dImage[t_idx + 1];
        dImageout[s_idx + 2] = dImage[t_idx + 2];
    }
}

void roi_rect(uint8_t *dImage, uint8_t *dImageout, uint32_t swith,
              uint32_t sheight, uint32_t x, uint32_t y, uint32_t w, uint32_t h) {
    dim3 block(32, 16, 1);
    dim3 grid(0, 0, 1);
    grid.x = (int) ((w + block.x - 1) / block.x);
    grid.y = (int) ((h + block.y - 1) / block.y);

    groi_rect << < grid, block >> > (dImage, dImageout, swith, sheight, x, y, w, h);
    cudaDeviceSynchronize();
}

extern "C" __global__ void get_timestamp_kernel(uint8_t *dImage, uint8_t *dTimestamp, uint32_t image_width,
                                                uint32_t image_height, uint32_t timestamp_len) {
    uint8_t index = 0;
    for (uint32_t i = 0; i < image_width * 3 * timestamp_len; i += image_width * 3) {
        dTimestamp[index] = dImage[i];
        index += 1;
    }
}

void get_timestamp(uint8_t *dImage, uint8_t *dTimestamp, uint32_t image_width,
                   uint32_t image_height, uint32_t timestamp_len) {
    get_timestamp_kernel << < 1, 1 >> > (dImage, dTimestamp, image_width, image_height, timestamp_len);
    cudaDeviceSynchronize();
}
extern "C" __global__ void gseal_rect(unsigned char *dImage, unsigned char *dImageout, uint32_t width, uint32_t height, uint32_t maxlen)
{
    uint32_t xIdx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t yIdx = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t s_idx = yIdx * width * 3 + xIdx * 3;
    uint32_t t_idx = yIdx * maxlen * 3 + xIdx * 3;
    if (xIdx < width && yIdx < height) {
        dImageout[t_idx] = dImage[s_idx];
        dImageout[t_idx + 1] =  dImage[s_idx+1];
        dImageout[t_idx + 2] =  dImage[s_idx+2];
    }

}

void seal_rect(unsigned char *dImage, unsigned char *dImageout, uint32_t width, uint32_t height, uint32_t maxlen) {
    dim3 block(32, 16, 1);
    dim3 grid(0, 0, 1);
    grid.x = (int) ((width + block.x - 1) / block.x);
    grid.y = (int) ((height + block.y - 1) / block.y);
    gseal_rect << < grid, block >> > (dImage, dImageout, width, height, maxlen);

}

extern "C" __global__ void gmerge_image(unsigned char* dImage, unsigned char* dImageout, uint32_t height, uint32_t width,uint32_t sheight, uint32_t swidth, uint32_t xstart,uint32_t ystart)
{
    uint32_t xIdx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t yIdx = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t s_idx = yIdx * width * 3 + xIdx * 3;
    uint32_t t_idx = (yIdx +ystart) * swidth  * 3 + (xIdx + xstart) * 3;
    if (xIdx < width && yIdx < height) {
        dImageout[t_idx] = dImage[s_idx];
        dImageout[t_idx + 1] =  dImage[s_idx+1];
        dImageout[t_idx + 2] =  dImage[s_idx+2];
    }
}

void merge_image(unsigned char* dImage, unsigned char* dImageout, uint32_t height, uint32_t width,uint32_t sheight, uint32_t swidth, uint32_t xstart,uint32_t ystart)
{
    dim3 block(32,16,1);
    dim3 grid(0,0,1);
    grid.x = (int) ((width + block.x - 1) / block.x);
    grid.y = (int) ((height + block.y - 1) / block.y);
    gmerge_image << < grid, block >> > (dImage, dImageout,height,width,sheight,swidth,xstart,ystart);

}

extern "C" __global__ void gdraw_rect(unsigned char* dImage, uint32_t swith,
                                     uint32_t sheight, uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2,uint32_t r,uint32_t g, uint32_t b) {
    uint32_t xIdx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t yIdx = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t  s_Idx = xIdx * 3 + yIdx * swith * 3;

    if ((xIdx >= x1 - 1 & xIdx <= x1 + 1) || (xIdx >= x2 - 1 & xIdx <= x2 + 1)) {
        if (yIdx >= y1 - 1 && yIdx <= y2 + 1) {
            dImage[s_Idx] = r;
            dImage[s_Idx + 1] = g;
            dImage[s_Idx + 2] = b;
        }
    }
    if ((yIdx >= y1 - 1 & yIdx <= y1 + 1) || (yIdx >= y2 - 1 & yIdx <= y2 + 1)) {
        if (xIdx >= x1 - 1 && xIdx <= x2 + 1) {
            dImage[s_Idx] = r;
            dImage[s_Idx + 1] = g;
            dImage[s_Idx + 2] = b;
        }
    }

}

void draw_rect(unsigned char* dImage, uint32_t w, uint32_t h, uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2,uint32_t r,uint32_t g, uint32_t b) {
    dim3 block(32, 16, 1);
    dim3 grid(0, 0, 1);
    grid.x = (int) ((w + block.x - 1) / block.x);
    grid.y = (int) ((h + block.y - 1) / block.y);

    gdraw_rect << < grid, block >> > (dImage, w, h, x1, y1, x2, y2, r, g, b);
    cudaDeviceSynchronize();
}

