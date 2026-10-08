#include "decode.h"
#include "detect.h"
#include "preprocess.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
int main(void) {
  int r = detector_darknet.init("face-yolov3-tiny.cfg",
                                "face-yolov3-tiny_41000.weights");
  FILE *input_image = fopen("darknet_inpuuut.bin", "rb");
  float *buff = malloc(sizeof(float) * 416 * 416 * 3);
  assert(buff != NULL);

  size_t got = fread(buff, sizeof(float), 416 * 416 * 3, input_image);
  assert(got == 416 * 416 * 3);
  yolo_output final_image;
  detector_darknet.run(buff, &final_image);
  detection dets[256];
  int n = decode(&final_image, 0.25f, dets, 256);

  printf("found %d\n", n);

  for (u16 k = 0; k < n; ++k) {
    printf("%.3f %.3f %.3f %.3f  conf %.3f\n", dets[k].x, dets[k].y, dets[k].w,
           dets[k].h, dets[k].confidence);
  }
  int top_k = nms(dets, n, 0.65);
  letterbox lb = letterbox_calc(1280, 960, 416, 416);
  unletterbox(dets, top_k, lb, 1280, 960);
  for (u16 k = 0; k < top_k; k++) {
    printf("%.0f %.0f %.0f %.0f  conf %.3f\n", dets[k].x, dets[k].y, dets[k].w,
           dets[k].h, dets[k].confidence);
  }

  FILE *f1 = fopen("small_out.bin", "wb");
  assert(f1);
  FILE *f2 = fopen("large_out.bin", "wb");
  assert(f2);
  fwrite(final_image.small, sizeof(float), 3042, f1);
  fwrite(final_image.large, sizeof(float), 12168, f2);

  fclose(input_image);
  fclose(f1);
  fclose(f2);

  detector_darknet.cleanup();

  printf("init returned %d\n", r);
  return 0;
}
