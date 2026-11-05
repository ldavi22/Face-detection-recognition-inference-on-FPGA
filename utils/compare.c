#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  if (argc != 4) {
    fprintf(stderr, "usage: compare a.bin b.bin count\n");
    return 1;
  }
  size_t n = strtoul(argv[3], NULL, 10);

  FILE *fa = fopen(argv[1], "rb");
  FILE *fb = fopen(argv[2], "rb");
  if (!fa || !fb) {
    perror("fopen");
    return 1;
  }

  float *a = malloc(n * sizeof(float));
  float *b = malloc(n * sizeof(float));
  if (fread(a, sizeof(float), n, fa) != n) {
    fprintf(stderr, "short read a\n");
    return 1;
  }
  if (fread(b, sizeof(float), n, fb) != n) {
    fprintf(stderr, "short read b\n");
    return 1;
  }

  double sum = 0.0, maxd = 0.0;
  size_t maxi = 0;
  for (size_t i = 0; i < n; i++) {
    double d = fabs((double)a[i] - b[i]);
    sum += d;
    if (d > maxd) {
      maxd = d;
      maxi = i;
    }
  }

  printf("n=%zu  mean=%.3e  max=%.3e at %zu (%.6f vs %.6f)\n", n, sum / n, maxd,
         maxi, a[maxi], b[maxi]);
  return 0;
}
