#include "detect.h"
#include "preprocess.h"

#define MAX_SIDE 1024
#define MAX_DETECTION 5

typedef struct {
  float x, y, w, h;
  float confidence;
  i16 crop_idx;
  i16 identity;
} detection;

int decode(const yolo_output *out, float thresh, detection *det, int max_dets);
float my_box_iou(const detection *a, const detection *b);
void unletterbox(detection *dets, int n, letterbox lb, u16 src_w, u16 src_h);
int nms(detection *dets, int n, float iou_thresh);
int pp_faces(detection *dets, float *src_image, int n, int max_crops, u16 src_w,
             u16 src_h, unsigned char *dst_image, float *to_crop,
             float *to_resize);
