#include "decode.h"
#include "detect.h"
#include "preprocess.h"
#include "types.h"
#include <math.h>
// #include <stdio.h>
#include <stdlib.h>

static const float anchors_small[3][2] = {{81, 82}, {135, 169}, {344, 319}};
static const float anchors_large[3][2] = {{10, 14}, {23, 27}, {37, 58}};

static int decode_head(const float *t, u16 grid, const float anchors[3][2],
                       float thresh, detection *det, int n, int max_dets) {

  u32 plane = (u32)grid * grid;
  u32 block = plane * 6;

  for (u16 anchor = 0; anchor < 3; anchor++) {
    for (u16 j = 0; j < grid; j++) {
      for (u16 i = 0; i < grid; i++) {

        if (n >= max_dets)
          return n;

        u32 base = anchor * block + j * grid + i;

        float obj = t[base + 4 * plane];
        float cls = t[base + 5 * plane];
        float conf = obj * cls;

        if (conf < thresh)
          continue;

        float tx = t[base + 0 * plane];
        float ty = t[base + 1 * plane];
        float tw = t[base + 2 * plane];
        float th = t[base + 3 * plane];

        det[n].x = (i + tx) / (float)grid;
        det[n].y = (j + ty) / (float)grid;
        det[n].w = anchors[anchor][0] * expf(tw) / 416.0f;
        det[n].h = anchors[anchor][1] * expf(th) / 416.0f;
        det[n].confidence = conf;
        det[n].crop_idx = -1;
        det[n].identity = -1;
        n++;
      }
    }
  }

  return n;
}

int decode(const yolo_output *out, float thresh, detection *det, int max_dets) {
  int n = 0;
  n = decode_head(out->small, 13, anchors_small, thresh, det, n, max_dets);
  n = decode_head(out->large, 26, anchors_large, thresh, det, n, max_dets);
  return n;
}

float my_box_iou(const detection *a, const detection *b) {
  float l1 = a->x - a->w / 2;
  float r1 = a->x + a->w / 2;
  float t1 = a->y - a->h / 2;
  float b1 = a->y + a->h / 2;
  float l2 = b->x - b->w / 2;
  float r2 = b->x + b->w / 2;
  float t2 = b->y - b->h / 2;
  float b2 = b->y + b->h / 2;

  float iw = fminf(r1, r2) - fmaxf(l1, l2);
  float ih = fminf(b1, b2) - fmaxf(t1, t2);

  if (iw < 0)
    iw = 0;
  if (ih < 0)
    ih = 0;

  float __intersection__ = iw * ih;
  float __union__ = a->w * a->h + b->w * b->h - __intersection__;

  return __intersection__ / __union__;
}
void unletterbox(detection *dets, int n, letterbox lb, u16 src_w, u16 src_h) {
  float scale = fminf(416.0f / src_w, 416.0f / src_h);
  for (int i = 0; i < n; ++i) {
    dets[i].x = (dets[i].x * 416.0f - lb.dx) / scale;
    dets[i].y = (dets[i].y * 416.0f - lb.dy) / scale;
    dets[i].w = (dets[i].w * 416.0f) / scale;
    dets[i].h = (dets[i].h * 416.0f) / scale;
  }
  (void)src_h;
}

static int cmp_conf(const void *a, const void *b) {
  const detection *d0 = a;
  const detection *d1 = b;

  if (d0->confidence < d1->confidence)
    return 1;
  if (d0->confidence > d1->confidence)
    return -1;
  return 0;
}

int nms(detection *dets, int n, float iou_thresh) {
  qsort(dets, n, sizeof(detection), cmp_conf);
  int curr = 0;
  for (int i = 0; i < n; i++) {
    if (dets[i].confidence == -1.0f)
      continue;
    for (int j = i + 1; j < n; j++) {
      if (dets[j].confidence == -1.0f)
        continue;
      if (my_box_iou(&dets[i], &dets[j]) > iou_thresh)
        dets[j].confidence = -1.0f;
    }
  }

  for (int i = 0; i < n; i++) {
    if (dets[i].confidence != -1.0f) {
      dets[curr] = dets[i];
      curr++;
    }
  }

  return curr;
}

// Memory is preallocated for faces and cropped images

int pp_faces(detection *dets, float *src_image, int n, int max_dets, u16 src_w,
             u16 src_h, unsigned char *dst_image, float *to_crop,
             float *to_resize) {
  int crops = 0;
  const size_t slot = 112 * 112 * 3;

  for (int i = 0; i < n && crops < max_dets; i++) {
    dets[i].crop_idx = -1;

    float side = fmaxf(dets[i].w, dets[i].h) * 1.2f;
    int iside = (int)(side + 0.5f);
    int left = (int)(dets[i].x - side / 2.0f + 0.5f);
    int top = (int)(dets[i].y - side / 2.0f + 0.5f);

    if (left < 0 || top < 0 || left + iside > src_w || top + iside > src_h)
      continue;

    my_crop_image(src_image, src_w, src_h, left, top, iside, to_crop);

    my_resize_image(to_crop, iside, iside, 112, 112, to_resize);
    unsigned char *dst = dst_image + slot * crops;
    to_rgb(to_resize, 112, 112, dst);
    dets[i].crop_idx = crops;
    crops++;
  }
  return crops;
}
