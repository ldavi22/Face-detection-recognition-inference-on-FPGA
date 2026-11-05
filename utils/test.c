#include "preprocess.h"
#include "types.h"
#include <stdio.h>
#include <stdlib.h>

#define W 1280
#define H 960

int main() {

  FILE *f = fopen("darknet_im.bin", "rb");
  FILE *f_new = fopen("my_input.bin", "wb");
  float *original = malloc(W * H * 3 * sizeof(float));

  size_t tmp = fread(original, sizeof(float), W * H * 3, f);
  (void)tmp;

  float *altered = malloc(640 * 480 * 3 * sizeof(float));

  //  float *resized = malloc(416 * 312 * 3 * sizeof(float));

  unsigned char *orig_cp = malloc(416 * 416 * 3 * sizeof(unsigned char));

  float *embedded_image = malloc(416 * 416 * 3 * sizeof(float));
  letterbox im = letterbox_calc(W, H, 416, 416);

  float *resized = malloc((size_t)im.new_w * im.new_h * 3 * sizeof(float));
  //  from_rgb(original, 640, 480, altered);

  resize_image(original, W, H, im.new_w, im.new_h, resized);

  fill_image(embedded_image, 416, 416, 0.5);

  embed_image(resized, im.new_w, im.new_h, embedded_image, 416, 416, im.dx,
              im.dy);
  // to_rgb(embedded_image, 416, 416, orig_cp);

  fwrite(embedded_image, sizeof(float), 416 * 416 * 3, f_new);

  fclose(f);
  fclose(f_new);
  free(embedded_image);
  free(original);
  free(altered);
  free(resized);
  free(orig_cp);
  return 0;
}
