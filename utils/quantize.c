/* Turn folded float weights into the int8 model the fabric will run.
 *
 *   ./quantize folded.bin ranges.txt qmodel.bin
 *
 * Per convolutional layer:
 *   - one scale per output channel, from that channel's largest |weight|
 *   - weights rounded to int8 in [-127, 127]
 *   - bias rounded to int32 at the accumulator's scale
 *   - a requantization multiplier and shift per channel
 *
 * The requantization is the part that has to match the RTL exactly.
 * After accumulating, the int32 sum is at scale (scale_in * scale_w[f]).
 * Converting to the output tensor's int8 scale is a multiply by
 *
 *     M = scale_in * scale_w[f] / scale_out
 *
 * M is a real number below 1. Hardware cannot multiply by a real number,
 * so it is written as an integer times a power of two:
 *
 *     M ~= mult * 2^-shift,   with mult in [2^30, 2^31)
 *
 * and the hardware computes (acc * mult) >> shift with a 64-bit
 * intermediate. Choosing mult as large as it fits keeps the relative error
 * near 2^-31, which is far below the int8 quantization error and therefore
 * invisible.
 */

#include "qnet.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static float act_scale[N_LAYERS];

static int read_ranges(const char *path) {
  FILE *f = fopen(path, "r");
  if (!f) {
    perror(path);
    return 1;
  }
  char line[256];
  int got = 0;
  while (fgets(line, sizeof line, f)) {
    int idx;
    float ma, sc;
    if (sscanf(line, "%d %f %f", &idx, &ma, &sc) == 3) {
      if (idx >= 0 && idx < N_LAYERS) {
        act_scale[idx] = sc;
        got++;
      }
    }
  }
  fclose(f);
  if (got != N_LAYERS) {
    fprintf(stderr, "%s: got %d of %d layer scales\n", path, got, N_LAYERS);
    return 1;
  }
  return 0;
}

/* M -> mult * 2^-shift, with mult normalized into [2^30, 2^31) */
static void choose_requant(double M, int32_t *mult, int32_t *shift) {
  if (M <= 0.0) {
    *mult = 0;
    *shift = 0;
    return;
  }
  int s = 0;
  while (M < 0.5 && s < 62) {
    M *= 2.0;
    s++;
  }
  while (M >= 1.0 && s > -62) {
    M /= 2.0;
    s--;
  }
  /* M now in [0.5, 1); scale it to 31 bits */
  int64_t m = (int64_t)llround(M * (double)(1LL << 31));
  if (m == (1LL << 31)) { /* rounded up out of range */
    m /= 2;
    s--;
  }
  *mult = (int32_t)m;
  *shift = s + 31;
}

/* Which layer feeds each conv -- needed for scale_in. Convs are at these
 * network indices, in the same order fold_bn wrote them. */
static const int CONV_AT[N_CONV] = {0,  2,  4,  6,  8,  10, 12,
                                    13, 14, 15, 18, 21, 22};

/* The layer whose output is the conv's input. For all of these it is just
 * the previous layer -- the two routes (17 and 20) sit directly before
 * convs 18 and 21 and act as their inputs. */
static float input_scale_of(int net_idx) {
  if (net_idx == 0)
    return 1.0f / 127.0f; /* the image: 0-1 floats, quantized at 1/127 */
  return act_scale[net_idx - 1];
}

int main(int argc, char **argv) {
  if (argc != 4) {
    fprintf(stderr, "usage: %s <folded.bin> <ranges.txt> <qmodel.bin>\n",
            argv[0]);
    return 1;
  }

  if (read_ranges(argv[2]))
    return 1;

  FILE *in = fopen(argv[1], "rb");
  if (!in) {
    perror(argv[1]);
    return 1;
  }
  FILE *out = fopen(argv[3], "wb");
  if (!out) {
    perror(argv[3]);
    return 1;
  }

  /* file header: the per-layer activation scales, so qnet can dequantize */
  if (fwrite(act_scale, sizeof(float), N_LAYERS, out) != N_LAYERS) {
    fprintf(stderr, "header write failed\n");
    return 1;
  }

  double worst_clip = 0.0;

  for (int ci = 0; ci < N_CONV; ci++) {
    int net_idx = CONV_AT[ci];
    const layer_desc *L = &NET[net_idx];

    int n = L->out_c, c = L->in_c, k = L->ksize;
    size_t per_filter = (size_t)c * k * k;
    size_t nw = (size_t)n * per_filter;

    float *bias_f = malloc((size_t)n * sizeof(float));
    float *w_f = malloc(nw * sizeof(float));
    int8_t *w_q = malloc(nw);
    int32_t *bias_q = malloc((size_t)n * sizeof(int32_t));
    int32_t *mult = malloc((size_t)n * sizeof(int32_t));
    int32_t *shift = malloc((size_t)n * sizeof(int32_t));
    float *scale_w = malloc((size_t)n * sizeof(float));

    if (!bias_f || !w_f || !w_q || !bias_q || !mult || !shift || !scale_w) {
      fprintf(stderr, "malloc failed\n");
      return 1;
    }

    if (fread(bias_f, sizeof(float), (size_t)n, in) != (size_t)n ||
        fread(w_f, sizeof(float), nw, in) != nw) {
      fprintf(stderr, "short read on conv %d\n", ci);
      return 1;
    }

    float s_in = input_scale_of(net_idx);
    float s_out = act_scale[net_idx];

    for (int f = 0; f < n; f++) {
      /* per-channel weight scale */
      float mx = 0.0f;
      for (size_t j = 0; j < per_filter; j++) {
        float a = fabsf(w_f[f * per_filter + j]);
        if (a > mx)
          mx = a;
      }
      float sw = (mx > 0.0f) ? mx / 127.0f : 1.0f / 127.0f;
      scale_w[f] = sw;

      for (size_t j = 0; j < per_filter; j++) {
        long q = lrintf(w_f[f * per_filter + j] / sw);
        if (q > 127)
          q = 127;
        if (q < -127)
          q = -127;
        w_q[f * per_filter + j] = (int8_t)q;
      }

      /* bias lives at the accumulator's scale */
      double acc_scale = (double)s_in * sw;
      double bq = (double)bias_f[f] / acc_scale;
      if (bq > 2147483647.0)
        bq = 2147483647.0;
      if (bq < -2147483648.0)
        bq = -2147483648.0;
      bias_q[f] = (int32_t)llround(bq);

      choose_requant(acc_scale / (double)s_out, &mult[f], &shift[f]);
    }

    /* how much of this layer's dynamic range gets clipped -- a rough
     * health check on the calibration */
    double clip = 0.0;
    for (size_t j = 0; j < nw; j++) {
      int f = (int)(j / per_filter);
      float back = (float)w_q[j] * scale_w[f];
      double e = fabs((double)back - w_f[j]);
      if (e > clip)
        clip = e;
    }
    if (clip > worst_clip)
      worst_clip = clip;

    fwrite(w_q, 1, nw, out);
    fwrite(bias_q, sizeof(int32_t), (size_t)n, out);
    fwrite(mult, sizeof(int32_t), (size_t)n, out);
    fwrite(shift, sizeof(int32_t), (size_t)n, out);
    fwrite(scale_w, sizeof(float), (size_t)n, out);

    printf("conv %2d (net %2d): %4d->%4d %dx%d  s_in %.3e s_out %.3e  "
           "max weight err %.3e\n",
           ci, net_idx, c, n, k, k, s_in, s_out, clip);

    free(bias_f);
    free(w_f);
    free(w_q);
    free(bias_q);
    free(mult);
    free(shift);
    free(scale_w);
  }

  long pos = ftell(in);
  fseek(in, 0, SEEK_END);
  if (pos != ftell(in))
    fprintf(stderr, "WARNING: folded.bin not fully consumed\n");

  printf("\nworst per-weight quantization error: %.3e\n", worst_clip);

  fclose(in);
  fclose(out);
  return 0;
}
