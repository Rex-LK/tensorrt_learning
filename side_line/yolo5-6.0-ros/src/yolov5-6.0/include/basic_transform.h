#ifndef BASIC_TRANSFORM_H
#define BASIC_TRANSFORM_H
#include <stdint.h>
void PicResize(uint8_t*, uint8_t*, uint32_t, uint32_t, uint32_t, uint32_t);
void PicNormalize(uint8_t*, float*, uint32_t, uint32_t);
#endif
