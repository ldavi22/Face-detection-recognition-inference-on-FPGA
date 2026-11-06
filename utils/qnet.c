#include "qnet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const int CONV_AT[N_CONV] = {0,  2,  4,  6,  8,  10, 12,
                                    13, 14, 15, 18, 21, 22};

int qmodel_load(qmodel *m, const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    perror(path);
    return 1;
  }

  memset(m, 0, sizeof(*m));

  if (fread(m->act_scale, sizeof(float), N_LAYERS, f) != N_LAYERS) {
    fprintf(stderr, "%s: truncated header\n", path);
    fclose(f);
    return 1;
  }

  for (int ci = 0; ci < N_CONV; ci++) {
    const layer_desc *L = &NET[CONV_AT[ci]];
    conv_weights *cw = &m->conv[ci];

    cw->n = L->out_c;
    cw->c = L->in_c;
    cw->k = L->ksize;

    size_t nw = (size_t)cw->n * cw->c * cw->k * cw->k;

    cw->w = malloc(nw);
    cw->bias = malloc((size_t)cw->n * sizeof(int32_t));
    cw->mult = malloc((size_t)cw->n * sizeof(int32_t));
    cw->shift = malloc((size_t)cw->n * sizeof(int32_t));
    cw->scale_w = malloc((size_t)cw->n * sizeof(float));

    if (!cw->w || !cw->bias || !cw->mult || !cw->shift || !cw->scale_w) {
      fprintf(stderr, "malloc failed on conv %d\n", ci);
      fclose(f);
      return 1;
    }

    if (fread(cw->w, 1, nw, f) != nw ||
        fread(cw->bias, sizeof(int32_t), (size_t)cw->n, f) != (size_t)cw->n ||
        fread(cw->mult, sizeof(int32_t), (size_t)cw->n, f) != (size_t)cw->n ||
        fread(cw->shift, sizeof(int32_t), (size_t)cw->n, f) != (size_t)cw->n ||
        fread(cw->scale_w, sizeof(float), (size_t)cw->n, f) != (size_t)cw->n) {
      fprintf(stderr, "%s: truncated at conv %d\n", path, ci);
      fclose(f);
      return 1;
    }
  }

  long pos = ftell(f);
  fseek(f, 0, SEEK_END);
  if (pos != ftell(f))
    fprintf(stderr, "WARNING: %s not fully consumed\n", path);

  fclose(f);
  return 0;
}

void qmodel_free(qmodel *m) {
  for (int i = 0; i < N_CONV; i++) {
    free(m->conv[i].w);
    free(m->conv[i].bias);
    free(m->conv[i].mult);
    free(m->conv[i].shift);
    free(m->conv[i].scale_w);
  }
  memset(m, 0, sizeof(*m));
}

static inline int32_t requantize(int32_t acc, int32_t mult, int32_t shift) {
  int64_t prod = (int64_t)acc * (int64_t)mult;
  if (shift > 0) {
    int64_t round = (int64_t)1 << (shift - 1);
    prod = (prod >= 0) ? (prod + round) : (prod - round);
    prod >>= shift;
  }
  return (int32_t)prod;
}

static inline int8_t clamp_int8(int32_t v) {
  if (v > 127)
    return 127;
  if (v < -127)
    return -127;
  return (int8_t)v;
}

static inline int32_t leaky(int32_t acc) {
  if (acc >= 0)
    return acc;
  return (int32_t)(((int64_t)acc * LEAKY_MUL) >> LEAKY_SHIFT);
}

static void conv_layer(const layer_desc *L, const conv_weights *cw,
                       const int8_t *in, int8_t *out) {
  const int W = L->in_w, H = L->in_h, C = L->in_c;
  const int K = L->ksize, P = L->pad;
  const int N = L->out_c;
  const long plane_in = (long)W * H;
  const long plane_out = (long)L->out_w * L->out_h;
  const long per_filter = (long)C * K * K;

  for (int f = 0; f < N; f++) {
    const int8_t *wf = cw->w + (long)f * per_filter;
    const int32_t bias = cw->bias[f];
    const int32_t mult = cw->mult[f];
    const int32_t shift = cw->shift[f];

    for (int oy = 0; oy < L->out_h; oy++) {
      for (int ox = 0; ox < L->out_w; ox++) {

        int32_t acc = bias;

        for (int c = 0; c < C; c++) {
          const int8_t *ic = in + (long)c * plane_in;
          const int8_t *wc = wf + (long)c * K * K;

          for (int ky = 0; ky < K; ky++) {
            int iy = oy + ky - P;
            if (iy < 0 || iy >= H)
              continue;

            for (int kx = 0; kx < K; kx++) {
              int ix = ox + kx - P;
              if (ix < 0 || ix >= W)
                continue;

              acc += (int32_t)ic[(long)iy * W + ix] * (int32_t)wc[ky * K + kx];
            }
          }
        }

        if (L->leaky)
          acc = leaky(acc);

        out[(long)f * plane_out + (long)oy * L->out_w + ox] =
            clamp_int8(requantize(acc, mult, shift));
      }
    }
  }
}

static void maxpool_layer(const layer_desc *L, const int8_t *in, int8_t *out) {
  const int W = L->in_w, H = L->in_h, C = L->in_c;
  const int sz = L->pool_size, st = L->pool_stride;
  const long plane_in = (long)W * H;
  const long plane_out = (long)L->out_w * L->out_h;

  for (int c = 0; c < C; c++) {
    const int8_t *ic = in + (long)c * plane_in;
    int8_t *oc = out + (long)c * plane_out;

    for (int oy = 0; oy < L->out_h; oy++) {
      for (int ox = 0; ox < L->out_w; ox++) {
        int32_t best = -128;

        for (int ky = 0; ky < sz; ky++) {
          int iy = oy * st + ky;
          if (iy >= H)
            continue;
          for (int kx = 0; kx < sz; kx++) {
            int ix = ox * st + kx;
            if (ix >= W)
              continue;
            int32_t v = ic[(long)iy * W + ix];
            if (v > best)
              best = v;
          }
        }
        oc[(long)oy * L->out_w + ox] = (int8_t)(best < -127 ? -127 : best);
      }
    }
  }
}

static void upsample_layer(const layer_desc *L, const int8_t *in, int8_t *out) {
  const int W = L->in_w, H = L->in_h, C = L->in_c;
  const long plane_in = (long)W * H;
  const long plane_out = (long)L->out_w * L->out_h;

  for (int c = 0; c < C; c++) {
    const int8_t *ic = in + (long)c * plane_in;
    int8_t *oc = out + (long)c * plane_out;
    for (int oy = 0; oy < L->out_h; oy++)
      for (int ox = 0; ox < L->out_w; ox++)
        oc[(long)oy * L->out_w + ox] = ic[(long)(oy / 2) * W + (ox / 2)];
  }
}

static void rescale(const int8_t *in, int8_t *out, long n, float s_from,
                    float s_to) {
  if (s_from == s_to) {
    memcpy(out, in, (size_t)n);
    return;
  }

  double M = (double)s_from / (double)s_to;
  int s = 0;
  while (M < 0.5 && s < 62) {
    M *= 2.0;
    s++;
  }
  while (M >= 1.0 && s > -62) {
    M /= 2.0;
    s--;
  }
  int64_t m = (int64_t)(M * (double)(1LL << 31) + 0.5);
  if (m == (1LL << 31)) {
    m /= 2;
    s--;
  }

  int32_t mult = (int32_t)m;
  int32_t shift = s + 31;

  for (long i = 0; i < n; i++)
    out[i] = clamp_int8(requantize(in[i], mult, shift));
}

static void yolo_layer(const layer_desc *L, const int8_t *in, int8_t *out,
                       float s_in, float s_out) {
  const long plane = (long)L->out_w * L->out_h;
  const long block = plane * 6;

  for (int n = 0; n < 3; n++) {
    for (int e = 0; e < 6; e++) {
      const long base = (long)n * block + (long)e * plane;
      const int logistic = (e == 0 || e == 1 || e == 4 || e == 5);

      for (long p = 0; p < plane; p++) {
        float v = (float)in[base + p] * s_in;
        if (logistic)
          v = 1.0f / (1.0f + expf(-v));
        long q = lrintf(v / s_out);
        out[base + p] = clamp_int8((int32_t)q);
      }
    }
  }
}

static long layer_elems(const layer_desc *L) {
  return (long)L->out_w * L->out_h * L->out_c;
}

static void dequantize(const int8_t *q, float *f, long n, float scale) {
  for (long i = 0; i < n; i++)
    f[i] = (float)q[i] * scale;
}

int qnet_run(const qmodel *m, const float *in, float *out_small,
             float *out_large, const char *dump_dir) {
  int8_t *buf[N_LAYERS];
  memset(buf, 0, sizeof(buf));

  long in_n = (long)NET_W * NET_H * 3;
  int8_t *img = malloc((size_t)in_n);
  if (!img)
    return 1;
  for (long i = 0; i < in_n; i++)
    img[i] = clamp_int8((int32_t)lrintf(in[i] * 127.0f));

  for (int i = 0; i < N_LAYERS; i++) {
    const layer_desc *L = &NET[i];
    long n = layer_elems(L);

    buf[i] = malloc((size_t)n);
    if (!buf[i]) {
      fprintf(stderr, "malloc failed at layer %d\n", i);
      return 1;
    }

    const int8_t *src = (i == 0) ? img : buf[i - 1];

    switch (L->type) {
    case L_CONV:
      conv_layer(L, &m->conv[L->conv_idx], src, buf[i]);
      break;

    case L_MAXPOOL:
      maxpool_layer(L, src, buf[i]);
      break;

    case L_UPSAMPLE:
      upsample_layer(L, src, buf[i]);
      break;

    case L_ROUTE:
      if (L->route_b < 0) {
        memcpy(buf[i], buf[L->route_a], (size_t)n);
      } else {
        long na = layer_elems(&NET[L->route_a]);
        long nb = layer_elems(&NET[L->route_b]);
        rescale(buf[L->route_a], buf[i], na, m->act_scale[L->route_a],
                m->act_scale[i]);
        rescale(buf[L->route_b], buf[i] + na, nb, m->act_scale[L->route_b],
                m->act_scale[i]);
      }
      break;

    case L_YOLO:
      yolo_layer(L, src, buf[i], m->act_scale[i - 1], m->act_scale[i]);
      break;
    }

    if (dump_dir) {
      char path[512];
      snprintf(path, sizeof path, "%s/qlayer_%02d.bin", dump_dir, i);
      FILE *f = fopen(path, "wb");
      if (f) {
        fwrite(buf[i], 1, (size_t)n, f);
        fclose(f);
      }
    }
  }

  dequantize(buf[16], out_small, layer_elems(&NET[16]), m->act_scale[16]);
  dequantize(buf[23], out_large, layer_elems(&NET[23]), m->act_scale[23]);

  free(img);
  for (int i = 0; i < N_LAYERS; i++)
    free(buf[i]);
  return 0;
}
