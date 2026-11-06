/* Work out the activation range of every layer.
 *
 *   ./calibrate golden/test1 golden/test2 golden/test3 > ranges.txt
 *
 * Each directory holds darknet's float goldens (layer_00.bin .. layer_23.bin)
 * for one image. The widest absolute value seen across all of them sets that
 * layer's int8 scale:
 *
 *     scale = max_abs / 127
 *
 * Symmetric quantization -- no zero point. Activations after leaky relu are
 * not symmetric (the negative side is squashed by 10x), so a zero point
 * would be more accurate. Symmetric is chosen anyway because it makes the
 * fabric's accumulator a plain signed multiply-add with no offset
 * correction term, which is worth more than the accuracy.
 *
 * Use several images. One image gives you ranges tuned to one scene, and
 * the first unusual frame then clips.
 */

#include "qnet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static long layer_elems(int i) {
  const layer_desc *L = &NET[i];
  return (long)L->out_w * L->out_h * L->out_c;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <golden_dir> [golden_dir ...]\n", argv[0]);
    return 1;
  }

  float max_abs[N_LAYERS];
  for (int i = 0; i < N_LAYERS; i++)
    max_abs[i] = 0.0f;

  for (int d = 1; d < argc; d++) {
    for (int i = 0; i < N_LAYERS; i++) {
      char path[512];
      snprintf(path, sizeof path, "%s/layer_%02d.bin", argv[d], i);

      FILE *f = fopen(path, "rb");
      if (!f) {
        fprintf(stderr, "missing %s\n", path);
        return 1;
      }

      long n = layer_elems(i);
      float *buf = malloc((size_t)n * sizeof(float));
      if (!buf) {
        fprintf(stderr, "malloc failed\n");
        return 1;
      }

      if (fread(buf, sizeof(float), (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "%s: short read (expected %ld floats)\n", path, n);
        return 1;
      }
      fclose(f);

      for (long j = 0; j < n; j++) {
        float a = fabsf(buf[j]);
        if (a > max_abs[i])
          max_abs[i] = a;
      }
      free(buf);
    }
    fprintf(stderr, "scanned %s\n", argv[d]);
  }

  printf("# layer max_abs scale\n");
  printf("# scale = max_abs / 127\n");
  for (int i = 0; i < N_LAYERS; i++) {
    float s = max_abs[i] / 127.0f;
    if (s <= 0.0f)
      s = 1.0f / 127.0f; /* a dead layer would otherwise divide by zero */
    printf("%d %.8f %.10f\n", i, max_abs[i], s);
  }

  return 0;
}
