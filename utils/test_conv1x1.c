/* On-board bring-up test for the conv1x1 accelerator.
 *
 *   sudo ./test_conv1x1 qmodel.bin qdump [component_offset_hex]
 *
 * Loads layer 18 into the fabric, runs it, and compares the result
 * against qdump/qlayer_18.bin byte for byte. Also times each phase
 * separately, because the split between data movement and compute is
 * the result worth reporting.
 *
 * Needs root: /dev/mem is unrestricted access to physical memory.
 */

#include "../qnet.h"
#include "conv1x1_hps.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define L_IDX 18
#define L_IN 17
#define CONV_IDX 10

#define C_IN 256
#define C_OUT 128
#define POS 169

/* Default: whatever Platform Designer assigned the component inside the
 * lightweight bridge window. Override on the command line while you are
 * still moving things around in Qsys. */
#define DEFAULT_OFFSET 0x0UL

static double now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec / 1e9;
}

static int8_t *load_i8(const char *path, long n) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    perror(path);
    return NULL;
  }
  int8_t *b = malloc((size_t)n);
  if (!b) {
    fclose(f);
    return NULL;
  }
  if (fread(b, 1, (size_t)n, f) != (size_t)n) {
    fprintf(stderr, "%s: short read (wanted %ld bytes)\n", path, n);
    fclose(f);
    free(b);
    return NULL;
  }
  fclose(f);
  return b;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <qmodel.bin> <qdump_dir> [offset_hex]\n",
            argv[0]);
    return 1;
  }

  unsigned long offset = DEFAULT_OFFSET;
  if (argc >= 4)
    offset = strtoul(argv[3], NULL, 16);

  qmodel m;
  if (qmodel_load(&m, argv[1]))
    return 1;

  const conv_weights *cw = &m.conv[CONV_IDX];
  if (cw->n != C_OUT || cw->c != C_IN || cw->k != 1) {
    fprintf(stderr, "conv %d is %d->%d k=%d, expected %d->%d k=1\n", CONV_IDX,
            cw->c, cw->n, cw->k, C_IN, C_OUT);
    return 1;
  }

  char path[512];

  snprintf(path, sizeof path, "%s/qlayer_%02d.bin", argv[2], L_IN);
  int8_t *in = load_i8(path, (long)C_IN * POS);

  snprintf(path, sizeof path, "%s/qlayer_%02d.bin", argv[2], L_IDX);
  int8_t *expect = load_i8(path, (long)C_OUT * POS);

  if (!in || !expect)
    return 1;

  int8_t *got = malloc((size_t)C_OUT * POS);
  if (!got)
    return 1;

  conv1x1_hps h;
  if (c1_open(&h, offset, C_IN, C_OUT, POS)) {
    fprintf(stderr, "could not map the fabric\n");
    return 1;
  }
  printf("mapped lightweight bridge, component offset 0x%lx\n", offset);

  /* --- one-time loads -------------------------------------------------- */

  double t0 = now_sec();
  c1_load_weights(&h, cw->w);
  double t_w = now_sec() - t0;

  t0 = now_sec();
  c1_load_params(&h, cw->bias, cw->mult, cw->shift);
  double t_p = now_sec() - t0;

  /* --- per-frame path -------------------------------------------------- */

  t0 = now_sec();
  c1_load_activations(&h, in);
  double t_a = now_sec() - t0;

  t0 = now_sec();
  long cycles = c1_run(&h);
  double t_c = now_sec() - t0;

  if (cycles < 0) {
    fprintf(stderr, "run failed\n");
    return 1;
  }

  t0 = now_sec();
  c1_read_output(&h, got);
  double t_o = now_sec() - t0;

  /* --- compare --------------------------------------------------------- */

  long n = (long)C_OUT * POS;
  long errors = 0, first = -1;

  for (long i = 0; i < n; i++) {
    if (got[i] != expect[i]) {
      errors++;
      if (first < 0) {
        first = i;
        fprintf(stderr, "first mismatch at %ld (f=%ld p=%ld): got %d want %d\n",
                i, i / POS, i % POS, got[i], expect[i]);
      }
    }
  }

  printf("\n");
  printf("weights  load : %8.2f ms  (%d bytes)\n", 1000 * t_w, C_OUT * C_IN);
  printf("params   load : %8.2f ms\n", 1000 * t_p);
  printf("activation ld : %8.2f ms  (%d bytes)   <- per frame\n", 1000 * t_a,
         C_IN * POS);
  printf("compute       : %8.2f ms  (%ld cycles) <- per frame\n", 1000 * t_c,
         cycles);
  printf("output   read : %8.2f ms  (%ld bytes)  <- per frame\n", 1000 * t_o,
         n);
  printf("per-frame total: %7.2f ms\n", 1000 * (t_a + t_c + t_o));
  printf("\n");

  if (errors == 0)
    printf("PASS -- all %ld bytes match\n", n);
  else
    printf("FAIL -- %ld of %ld bytes differ\n", errors, n);

  c1_close(&h);
  free(in);
  free(expect);
  free(got);
  qmodel_free(&m);
  return errors ? 1 : 0;
}
