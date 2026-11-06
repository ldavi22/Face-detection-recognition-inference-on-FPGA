#ifndef QNET_H
#define QNET_H

#include <stdint.h>

/* ---------------------------------------------------------------------- *
 * Network description
 *
 * yolov3-tiny, classes=1, 416x416 input. 24 layers, of which 13 are
 * convolutions. Shapes are fixed, so they live here rather than being
 * parsed from a cfg at runtime.
 * ---------------------------------------------------------------------- */

#define NET_W 416
#define NET_H 416
#define N_LAYERS 24
#define N_CONV 13

typedef enum { L_CONV, L_MAXPOOL, L_UPSAMPLE, L_ROUTE, L_YOLO } layer_type;

typedef struct {
  layer_type type;

  int in_w, in_h, in_c;
  int out_w, out_h, out_c;

  /* conv */
  int ksize;    /* 1 or 3            */
  int stride;   /* always 1 here     */
  int pad;      /* ksize/2           */
  int leaky;    /* 1 = leaky relu, 0 = linear (layers 15, 22) */
  int conv_idx; /* index into the weight arrays, -1 if not a conv */

  /* maxpool */
  int pool_size;
  int pool_stride;

  /* route: -1 when unused */
  int route_a;
  int route_b;
} layer_desc;

extern const layer_desc NET[N_LAYERS];

/* ---------------------------------------------------------------------- *
 * Quantization
 *
 *   weights     int8, symmetric, per output channel
 *   activations int8, symmetric, per tensor
 *   accumulator int32
 *
 * A real value x is represented by the integer q with  x ~= q * scale.
 * Convolution accumulates at scale  (scale_in * scale_w[f]); requantizing
 * to the output tensor's scale is a multiply by
 *
 *     M = scale_in * scale_w[f] / scale_out
 *
 * done in integer as  (acc * mult) >> shift,  which is exactly what the
 * RTL will do. Keeping the software model bit-identical to that is the
 * whole point -- the goldens it produces are what the fabric is checked
 * against.
 * ---------------------------------------------------------------------- */

typedef struct {
  int8_t *w;      /* n * c * k * k, output-channel major */
  int32_t *bias;  /* n, at scale (scale_in * scale_w[f]) */
  int32_t *mult;  /* n, requant multiplier               */
  int32_t *shift; /* n, requant right shift              */
  float *scale_w; /* n, kept for reporting               */
  int n, c, k;
} conv_weights;

typedef struct {
  conv_weights conv[N_CONV];
  float act_scale[N_LAYERS]; /* output scale of every layer */
} qmodel;

/* loading ------------------------------------------------------------- */

int qmodel_load(qmodel *m, const char *path);
void qmodel_free(qmodel *m);

/* inference ------------------------------------------------------------ */

/* in: NET_W*NET_H*3 floats, planar, 0-1 (the letterboxed image).
 * out_small / out_large: 3042 and 12168 floats, dequantized, matching
 * what detect.h's yolo_output expects.
 * dump_dir: if non-NULL, every layer's int8 output is written there as
 * qlayer_XX.bin -- those are the RTL goldens. */
int qnet_run(const qmodel *m, const float *in, float *out_small,
             float *out_large, const char *dump_dir);

/* leaky relu in fixed point: x<0  ->  (x * LEAKY_MUL) >> LEAKY_SHIFT
 * 13/128 = 0.1016, within 1.6% of darknet's 0.1, one multiply and a
 * shift in hardware. The cheaper 1/8 = 0.125 is a 25% error; measure
 * both against the float goldens before choosing. */
#define LEAKY_MUL 13
#define LEAKY_SHIFT 7

#endif /* QNET_H */
