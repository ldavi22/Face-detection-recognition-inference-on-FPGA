#include "decode.h"
#include "detect.h"
#include "preprocess.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#define SRC_W 1280
#define SRC_H 960

int save_raw_images(const unsigned char *buf, int n, size_t img_size,
                    const char *prefix) {
  char path[256];

  for (int i = 0; i < n; i++) {
    snprintf(path, sizeof path, "%s_%03d_112x112x3.raw", prefix, i);

    FILE *f = fopen(path, "wb");
    if (!f) {
      perror(path);
      return -1;
    }

    size_t written = fwrite(buf + img_size * i, 1, img_size, f);

    if (fclose(f) != 0 || written != img_size) {
      fprintf(stderr, "write failed: %s\n", path);
      return -1;
    }
  }
  return 0;
}

int main(void) {
  int r = detector_darknet.init("face-yolov3-tiny.cfg",
                                "face-yolov3-tiny_41000.weights");
  if (r != 0) {
    fprintf(stderr, "detector init failed: %d\n", r);
    return 1;
  }

  /* 416x416 letterboxed tensor -- what the network consumes */
  FILE *input_image = fopen("darknet_inpuuut.bin", "rb");
  if (!input_image) {
    perror("darknet_inpuuut.bin");
    return 1;
  }
  float *buff = malloc(sizeof(float) * 416 * 416 * 3);
  assert(buff != NULL);
  size_t got = fread(buff, sizeof(float), 416 * 416 * 3, input_image);
  assert(got == (size_t)416 * 416 * 3);
  fclose(input_image);

  /* full-size source image -- what the crops come from */
  FILE *src_file = fopen("darknet_im.bin", "rb");
  if (!src_file) {
    perror("darknet_im.bin");
    return 1;
  }
  float *src = malloc(sizeof(float) * SRC_W * SRC_H * 3);
  assert(src != NULL);
  size_t got_src =
      fread(src, sizeof(float), (size_t)SRC_W * SRC_H * 3, src_file);
  assert(got_src == (size_t)SRC_W * SRC_H * 3);
  fclose(src_file);

  yolo_output final_image;
  detector_darknet.run(buff, &final_image);

  detection dets[256];
  int n = decode(&final_image, 0.25f, dets, 256);

  printf("found %d\n", n);
  for (int k = 0; k < n; ++k) {
    printf("%.3f %.3f %.3f %.3f  conf %.3f\n", dets[k].x, dets[k].y, dets[k].w,
           dets[k].h, dets[k].confidence);
  }

  int top_k = nms(dets, n, 0.45f);
  letterbox lb = letterbox_calc(SRC_W, SRC_H, 416, 416);
  unletterbox(dets, top_k, lb, SRC_W, SRC_H);

  printf("after nms %d\n", top_k);
  for (int k = 0; k < top_k; k++) {
    printf("%.0f %.0f %.0f %.0f  conf %.3f\n", dets[k].x, dets[k].y, dets[k].w,
           dets[k].h, dets[k].confidence);
  }

  unsigned char *dst_image =
      malloc(sizeof(unsigned char) * 112 * 112 * 3 * MAX_DETECTION);
  float *to_crop = malloc(sizeof(float) * MAX_SIDE * MAX_SIDE * 3);
  float *to_resize = malloc(sizeof(float) * 112 * 112 * 3);
  assert(dst_image && to_crop && to_resize);

  int n_cropped = pp_faces(dets, src, top_k, MAX_DETECTION, SRC_W, SRC_H,
                           dst_image, to_crop, to_resize);

  printf("cropped %d\n", n_cropped);

  if (save_raw_images(dst_image, n_cropped, 112 * 112 * 3, "face") != 0)
    fprintf(stderr, "saving faces failed\n");

  free(dst_image);
  free(to_crop);
  free(to_resize);
  free(buff);
  free(src);

  detector_darknet.cleanup();
  return 0;
}
