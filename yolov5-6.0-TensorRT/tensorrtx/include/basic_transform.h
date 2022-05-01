#ifndef BASIC_TRANSFORM_H
#define BASIC_TRANSFORM_H
#include <stdint.h>

#ifdef __cplusplus
  extern "C"
#endif


void RGB2Resize(uint8_t*, uint8_t*, uint32_t, uint32_t, uint32_t, uint32_t);
void RGB2Normalize(uint8_t*, float*, uint32_t, uint32_t);
void RGB2Gray3(uint8_t*, uint8_t*, uint32_t, uint32_t);
void roi_rect(uint8_t*, uint8_t*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
void get_timestamp(uint8_t*, uint8_t*, uint32_t, uint32_t, uint32_t timestamp_len = 13);
void undistort(float* param, uint8_t* input_image_gpu, uint8_t* output_image_gpu, int height, int width);
void affine_transform(uint8_t* dImage, uint8_t* dImageout,uint32_t height,uint32_t width, uint32_t channel,float* param);
void perspective_transform(uint8_t* dImage, uint8_t* dImageout,uint32_t height,uint32_t width, uint32_t channel,float* param);
void seal_rect(unsigned char* dImage, unsigned char* dImageout,uint32_t height,uint32_t width,uint32_t maxlen);
void merge_image(unsigned char* dImage, unsigned char* dImageout, uint32_t height, uint32_t width, uint32_t sheight, uint32_t swidth, uint32_t xstart,uint32_t ystart);
void draw_rect(unsigned char*, uint32_t w, uint32_t h, uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2, uint32_t r,uint32_t g, uint32_t b);
//
#endif
