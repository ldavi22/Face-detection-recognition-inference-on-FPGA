#include "preprocess.h"
#include "types.h"
#include <stdint.h>

void from_rgb(const unsigned char *src, u16 width, u16 height, float *out) {
  u32 plane = width * height;

  for (u32 pix = 0; pix < plane; pix++) {
    out[pix] = src[pix * 3 + 0] / 255.0f;
    out[pix + plane] = src[pix * 3 + 1] / 255.0f;
    out[pix + plane * 2] = src[pix * 3 + 2] / 255.0f;
  }
}

void to_rgb(const float *src, u16 width, u16 height, unsigned char *out) {
  u32 plane = width * height;

  for (u32 i = 0; i < plane; i++) {
    out[i * 3 + 0] = (unsigned char)(src[i] * 255.0f + 0.5f);
    out[i * 3 + 1] = (unsigned char)(src[i + plane] * 255.0f + 0.5f);
    out[i * 3 + 2] = (unsigned char)(src[i + 2 * plane] * 255.0f + 0.5f);
  }
}

static float clamp(double v) {
  if (v < 0.0)
    return 0.0f;
  if (v > 255.0)
    return 255.0f;
  return (float)v;
}

void yuyv_to_rgb(unsigned char *src, u32 stride, u16 width, u16 height,
                 float *dst) {

  u32 plane = width * height;

  for (u32 row = 0; row < height; row++) {
    unsigned char *line = src + row * stride;

    for (u32 col = 0; col < width; col += 2) {

      i32 y0 = line[col * 2 + 0];
      i32 u = line[col * 2 + 1];
      i32 y1 = line[col * 2 + 2];
      i32 v = line[col * 2 + 3];

      double cb = (u - 128) * (255.0 / 224.0);
      double cr = (v - 128) * (255.0 / 224.0);

      double r_term = 1.402 * cr;
      double g_term = -0.344136 * cb - 0.714136 * cr;
      double b_term = 1.772 * cb;

      u32 o = row * width + col;

      double br0 = (y0 - 16) * (255.0 / 219.0);
      double br1 = (y1 - 16) * (255.0 / 219.0);

      dst[o] = clamp(r_term + br0) / 255.0f;     // R0
      dst[o + 1] = clamp(r_term + br1) / 255.0f; // R1

      dst[o + plane] = clamp(g_term + br0) / 255.0f;     // G0
      dst[o + plane + 1] = clamp(g_term + br1) / 255.0f; // G1

      dst[o + 2 * plane] = clamp(b_term + br0) / 255.0f;     // B0
      dst[o + 2 * plane + 1] = clamp(b_term + br1) / 255.0f; // B1
    }
  }
}

void resize_image(float *src, u16 old_w, u16 old_h, u16 new_w, u16 new_h,
                  float *out) {

  float ratio_w = (float)(old_w - 1) / (new_w - 1);
  float ratio_h = (float)(old_h - 1) / (new_h - 1);

  u32 old_plane = (u32)old_w * old_h;
  u32 new_plane = (u32)new_w * new_h;

  for (u16 k = 0; k < 3; ++k) {
    for (u16 j = 0; j < new_h; ++j) {
      for (u16 i = 0; i < new_w; ++i) {
        float x_query = ratio_w * i;
        float y_query = ratio_h * j;

        u16 x_low = (u16)x_query;
        u16 y_low = (u16)y_query;
        u16 x_high = (x_low + 1 < old_w) ? x_low + 1 : x_low;
        u16 y_high = (y_low + 1 < old_h) ? y_low + 1 : y_low;

        float w_x = x_query - x_low;
        float w_y = y_query - y_low;

        float a = src[x_low + y_low * old_w + k * old_plane];
        float b = src[x_high + y_low * old_w + k * old_plane];
        float c = src[x_low + y_high * old_w + k * old_plane];
        float d = src[x_high + y_high * old_w + k * old_plane];

        float pixel_val = a * (1 - w_x) * (1 - w_y) + b * w_x * (1 - w_y) +
                          c * (1 - w_x) * w_y + d * w_x * w_y;

        out[i + j * new_w + k * new_plane] = pixel_val;
      }
    }
  }
}

void fill_image(float *dst, u16 w, u16 h, float value) {
  for (u32 i = 0; i < (u32)w * h * 3; ++i)
    dst[i] = value;
}

void embed_image(float *src, u16 w, u16 h, float *dst, u16 dst_w, u16 dst_h,
                 u16 dx, u16 dy) {
  u32 old_plane = w * h;
  u32 new_plane = dst_w * dst_h;
  for (u16 k = 0; k < 3; ++k) {
    for (u16 j = 0; j < h; ++j) {
      for (u16 i = 0; i < w; ++i) {
        dst[(i + dx) + (j + dy) * dst_w + k * new_plane] =
            src[i + j * w + k * old_plane];
      }
    }
  }
}

letterbox letterbox_calc(u16 w_i, u16 h_i, u16 w_f, u16 h_f) {
  letterbox smt;

  if ((float)w_f / w_i < (float)h_f / h_i) {
    smt.new_w = w_f;
    smt.new_h = (h_i * w_f) / w_i;
  } else {
    smt.new_h = h_f;
    smt.new_w = (w_i * h_f) / h_i;
  }

  smt.dx = (w_f - smt.new_w) / 2;
  smt.dy = (h_f - smt.new_h) / 2;

  return smt;
}
