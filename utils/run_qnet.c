/* Run the fixed-point network and report how far each layer drifted from
 * darknet's float goldens.
 *
 *   ./run_qnet qmodel.bin golden/test1 [dumpdir]
 *
 * Reads golden/test1/darknet_input.bin as the input, runs the int8 model,
 * dequantizes each layer and compares against golden/test1/layer_XX.bin.
 *
 * The number to watch is the relative error per layer. It will grow with
 * depth -- each layer's quantization error feeds the next. If layer 0 is
 * already bad, the weight scales are wrong; if it climbs steeply after a
 * particular layer, that layer's activation range is badly calibrated.
 */

#include "qnet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static long layer_elems(int i) {
  return (long)NET[i].out_w * NET[i].out_h * NET[i].out_c;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <qmodel.bin> <golden_dir> [dump_dir]\n",
            argv[0]);
    return 1;
  }
  const char *dump_dir = (argc >= 4) ? argv[3] : NULL;

  qmodel m;
  if (qmodel_load(&m, argv[1]))
    return 1;

  char path[512];
  snprintf(path, sizeof path, "%s/darknet_input.bin", argv[2]);
  FILE *f = fopen(path, "rb");
  if (!f) {
    perror(path);
    return 1;
  }

  long in_n = (long)NET_W * NET_H * 3;
  float *in = malloc((size_t)in_n * sizeof(float));
  if (fread(in, sizeof(float), (size_t)in_n, f) != (size_t)in_n) {
    fprintf(stderr, "%s: short read\n", path);
    return 1;
  }
  fclose(f);

  float *out_small = malloc(3042 * sizeof(float));
  float *out_large = malloc(12168 * sizeof(float));

  if (qnet_run(&m, in, out_small, out_large, dump_dir))
    return 1;

  if (!dump_dir) {
    printf("no dump dir given -- skipping per-layer comparison\n");
    return 0;
  }

  /* --- per-layer comparison ------------------------------------------- */

  printf("\n%-5s %-10s %10s %12s %12s %8s\n", "layer", "type", "elems",
         "max abs err", "mean abs err", "rel %");

  static const char *TNAME[] = {"conv", "maxpool", "upsample", "route", "yolo"};

  for (int i = 0; i < N_LAYERS; i++) {
    long n = layer_elems(i);

    snprintf(path, sizeof path, "%s/layer_%02d.bin", argv[2], i);
    FILE *gf = fopen(path, "rb");
    if (!gf)
      continue;
    float *gold = malloc((size_t)n * sizeof(float));
    if (fread(gold, sizeof(float), (size_t)n, gf) != (size_t)n) {
      fclose(gf);
      free(gold);
      continue;
    }
    fclose(gf);

    snprintf(path, sizeof path, "%s/qlayer_%02d.bin", dump_dir, i);
    FILE *qf = fopen(path, "rb");
    if (!qf) {
      free(gold);
      continue;
    }
    signed char *q = malloc((size_t)n);
    if (fread(q, 1, (size_t)n, qf) != (size_t)n) {
      fclose(qf);
      free(gold);
      free(q);
      continue;
    }
    fclose(qf);

    double sum = 0.0, maxd = 0.0, gmax = 0.0;
    for (long j = 0; j < n; j++) {
      double dq = (double)q[j] * m.act_scale[i];
      double d = fabs(dq - gold[j]);
      sum += d;
      if (d > maxd)
        maxd = d;
      if (fabs(gold[j]) > gmax)
        gmax = fabs(gold[j]);
    }

    printf("%-5d %-10s %10ld %12.4e %12.4e %7.2f%%\n", i, TNAME[NET[i].type], n,
           maxd, sum / n, gmax > 0 ? 100.0 * (sum / n) / gmax : 0.0);

    free(gold);
    free(q);
  }

  free(in);
  free(out_small);
  free(out_large);
  qmodel_free(&m);
  return 0;
}
