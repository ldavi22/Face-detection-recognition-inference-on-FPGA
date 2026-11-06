#ifndef DETECT_H
#define DETECT_H

typedef struct {
  float *small; /* 13x13x18 */
  float *large; /* 26x26x18 */
} yolo_output;

typedef struct {
  const char *name;
  int (*init)(const char *input, const char *weights);
  void (*run)(const float *input, yolo_output *out);
  void (*cleanup)(void);
} detector_api;

extern const detector_api detector_darknet;

#endif
