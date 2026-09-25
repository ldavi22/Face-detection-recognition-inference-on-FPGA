#ifndef PREPROCESS_H
#define PREPROCESS_H

#include "types.h"

#include "types.h"

typedef struct {
  u32 new_w, new_h;
  u32 dx, dy;
} letterbox;

void embed_image(float *src, u16 w, u16 h, float *dst, u16 dst_w, u16 dst_h,
                 u16 dx, u16 dy);
void fill_image(float *dst, u16 w, u16 h, float value);
void resize_image(float *src, u16 old_w, u16 old_h, u16 new_w, u16 new_h,
                  float *out);
letterbox letterbox_calc(u16 w_i, u16 h_i, u16 w_f, u16 h_f);
void from_rgb(const unsigned char *src, u16 width, u16 height, float *out);
void to_rgb(const float *src, u16 width, u16 height, unsigned char *out);
void yuyv_to_rgb(unsigned char *src, u32 stride, u16 width, u16 height,
                 float *dst);

#endif
