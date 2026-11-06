/* Fold batch normalization into the convolution weights.
 *
 *   ./fold_bn face-yolov3-tiny_41000.weights folded.bin
 *
 * Reads darknet's .weights file and writes, per conv layer in network
 * order:  float bias[n], then float weights[n*c*k*k].
 *
 * The fold, per output channel f:
 *
 *     s           = scales[f] / (sqrt(variance[f]) + EPS)
 *     W_folded    = W * s
 *     bias_folded = biases[f] - mean[f] * s
 *
 * Note EPS is added AFTER the sqrt -- that is what darknet's normalize_cpu
 * does, and it differs from the sqrt(var + eps) most frameworks use.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EPS 0.000001f

typedef struct {
  int net_idx; /* index in the darknet layer table */
  int c;       /* input channels  */
  int n;       /* output channels */
  int k;       /* kernel size     */
  int bn;      /* 1 if batch_normalize */
} conv_spec;

/* yolov3-tiny with classes=1: 13 convolutional layers */
static const conv_spec LAYERS[] = {
    {0, 3, 16, 3, 1},      {2, 16, 32, 3, 1},     {4, 32, 64, 3, 1},
    {6, 64, 128, 3, 1},    {8, 128, 256, 3, 1},   {10, 256, 512, 3, 1},
    {12, 512, 1024, 3, 1}, {13, 1024, 256, 1, 1}, {14, 256, 512, 3, 1},
    {15, 512, 18, 1, 0},   {18, 256, 128, 1, 1},  {21, 384, 256, 3, 1},
    {22, 256, 18, 1, 0},
};
#define NLAYERS ((int)(sizeof(LAYERS) / sizeof(LAYERS[0])))

static float *xmalloc(size_t n) {
  float *p = malloc(n * sizeof(float));
  if (!p) {
    fprintf(stderr, "malloc of %zu floats failed\n", n);
    exit(1);
  }
  return p;
}

static void must_read(void *dst, size_t n, FILE *f, const char *what) {
  if (fread(dst, sizeof(float), n, f) != n) {
    fprintf(stderr, "short read on %s (%zu floats)\n", what, n);
    exit(1);
  }
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s <in.weights> <out.bin>\n", argv[0]);
    return 1;
  }

  FILE *in = fopen(argv[1], "rb");
  if (!in) {
    perror(argv[1]);
    return 1;
  }
  FILE *out = fopen(argv[2], "wb");
  if (!out) {
    perror(argv[2]);
    return 1;
  }

  /* --- header ---------------------------------------------------------- */

  int major, minor, revision;
  if (fread(&major, sizeof(int), 1, in) != 1 ||
      fread(&minor, sizeof(int), 1, in) != 1 ||
      fread(&revision, sizeof(int), 1, in) != 1) {
    fprintf(stderr, "truncated header\n");
    return 1;
  }

  /* darknet writes 'seen' as size_t when version >= 2.0, else int */
  if ((major * 10 + minor) >= 2) {
    size_t seen;
    if (fread(&seen, sizeof(size_t), 1, in) != 1) {
      fprintf(stderr, "truncated header (seen)\n");
      return 1;
    }
    printf("weights v%d.%d.%d, seen %zu\n", major, minor, revision, seen);
  } else {
    int seen;
    if (fread(&seen, sizeof(int), 1, in) != 1) {
      fprintf(stderr, "truncated header (seen)\n");
      return 1;
    }
    printf("weights v%d.%d.%d, seen %d\n", major, minor, revision, seen);
  }

  FILE *man = fopen("folded_manifest.txt", "w");
  if (man)
    fprintf(man, "# idx net_idx c n k bn n_weights\n");

  /* --- layers ---------------------------------------------------------- */

  long total_weights = 0;

  for (int li = 0; li < NLAYERS; li++) {
    const conv_spec *L = &LAYERS[li];
    size_t nw = (size_t)L->n * L->c * L->k * L->k;

    float *biases = xmalloc(L->n);
    float *scales = NULL, *mean = NULL, *var = NULL;
    float *w = xmalloc(nw);

    must_read(biases, L->n, in, "biases");

    if (L->bn) {
      scales = xmalloc(L->n);
      mean = xmalloc(L->n);
      var = xmalloc(L->n);
      must_read(scales, L->n, in, "scales");
      must_read(mean, L->n, in, "rolling_mean");
      must_read(var, L->n, in, "rolling_variance");
    }

    must_read(w, nw, in, "weights");

    /* --- the fold ------------------------------------------------------ */

    if (L->bn) {
      size_t per_filter = (size_t)L->c * L->k * L->k;

      for (int f = 0; f < L->n; f++) {
        /* darknet: (x - mean) / (sqrt(var) + eps), eps AFTER the sqrt */
        float s = scales[f] / (sqrtf(var[f]) + EPS);

        for (size_t j = 0; j < per_filter; j++)
          w[(size_t)f * per_filter + j] *= s;

        biases[f] = biases[f] - mean[f] * s;
      }
    }
    /* layers 15 and 22 have no batchnorm -- biases and weights pass
     * through untouched */

    /* --- write --------------------------------------------------------- */

    if (fwrite(biases, sizeof(float), L->n, out) != (size_t)L->n ||
        fwrite(w, sizeof(float), nw, out) != nw) {
      fprintf(stderr, "write failed on layer %d\n", L->net_idx);
      return 1;
    }

    printf("layer %2d: %4d->%4d %dx%d %s  %zu weights\n", L->net_idx, L->c,
           L->n, L->k, L->k, L->bn ? "bn " : "   ", nw);

    if (man)
      fprintf(man, "%d %d %d %d %d %d %zu\n", li, L->net_idx, L->c, L->n, L->k,
              L->bn, nw);

    total_weights += (long)nw + L->n;

    free(biases);
    free(w);
    free(scales);
    free(mean);
    free(var);
  }

  /* --- sanity ---------------------------------------------------------- */

  long pos = ftell(in);
  fseek(in, 0, SEEK_END);
  long end = ftell(in);

  if (pos != end)
    fprintf(stderr,
            "WARNING: %ld bytes left unread -- the layer table probably "
            "does not match this weights file\n",
            end - pos);
  else
    printf("consumed the whole file (%ld bytes)\n", end);

  printf("wrote %ld floats (%ld bytes) to %s\n", total_weights,
         total_weights * 4, argv[2]);

  if (man)
    fclose(man);
  fclose(in);
  fclose(out);
  return 0;
}
