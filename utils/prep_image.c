#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "preprocess.h"
#include "types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NET_W 416
#define NET_H 416

static int write_floats(const char *path, const float *buf, size_t n) {
  FILE *f = fopen(path, "wb");
  if (!f) {
    perror(path);
    return 1;
  }
  size_t put = fwrite(buf, sizeof(float), n, f);
  if (fclose(f) != 0 || put != n) {
    fprintf(stderr, "write failed: %s\n", path);
    return 1;
  }
  printf("wrote %s  (%zu floats, %zu bytes)\n", path, n, n * sizeof(float));
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s <image.jpg> <out_prefix>\n", argv[0]);
    fprintf(stderr, "  e.g. %s face1.jpg testdata/temo_a\n", argv[0]);
    return 1;
  }

  const char *jpg = argv[1];
  const char *prefix = argv[2];

  int w, h, c;
  unsigned char *pixels = stbi_load(jpg, &w, &h, &c, 3);
  if (!pixels) {
    fprintf(stderr, "%s: %s\n", jpg, stbi_failure_reason());
    return 1;
  }
  printf("%s: %dx%d (%d channels in file)\n", jpg, w, h, c);

  size_t src_n = (size_t)w * h * 3;
  float *src = malloc(src_n * sizeof(float));
  if (!src) {
    fprintf(stderr, "malloc failed\n");
    return 1;
  }

  /* interleaved bytes 0-255  ->  planar float 0-1, same as darknet's
   * load_image_stb */
  from_rgb(pixels, (u16)w, (u16)h, src);
  stbi_image_free(pixels);

  /* letterbox into the 416x416 network input */
  letterbox lb = letterbox_calc((u16)w, (u16)h, NET_W, NET_H);
  printf("letterbox: %ux%u at (%u,%u)\n", lb.new_w, lb.new_h, lb.dx, lb.dy);

  float *resized = malloc((size_t)lb.new_w * lb.new_h * 3 * sizeof(float));
  float *net_in = malloc((size_t)NET_W * NET_H * 3 * sizeof(float));
  if (!resized || !net_in) {
    fprintf(stderr, "malloc failed\n");
    return 1;
  }

  my_resize_image(src, (u16)w, (u16)h, lb.new_w, lb.new_h, resized);
  my_fill_image(net_in, NET_W, NET_H, 0.5f);
  my_embed_image(resized, lb.new_w, lb.new_h, net_in, NET_W, NET_H, lb.dx,
                 lb.dy);

  char path[512];

  snprintf(path, sizeof path, "%s_im_%dx%dx3.bin", prefix, w, h);
  if (write_floats(path, src, src_n))
    return 1;

  snprintf(path, sizeof path, "%s_input_%dx%dx3.bin", prefix, NET_W, NET_H);
  if (write_floats(path, net_in, (size_t)NET_W * NET_H * 3))
    return 1;

  snprintf(path, sizeof path, "%s_meta.txt", prefix);
  FILE *m = fopen(path, "w");
  if (m) {
    fprintf(m, "source %s\n", jpg);
    fprintf(m, "src_w %d\n", w);
    fprintf(m, "src_h %d\n", h);
    fprintf(m, "net_w %d\n", NET_W);
    fprintf(m, "net_h %d\n", NET_H);
    fprintf(m, "new_w %u\n", lb.new_w);
    fprintf(m, "new_h %u\n", lb.new_h);
    fprintf(m, "dx %u\n", lb.dx);
    fprintf(m, "dy %u\n", lb.dy);
    fclose(m);
    printf("wrote %s\n", path);
  }

  free(src);
  free(resized);
  free(net_in);
  return 0;
}
