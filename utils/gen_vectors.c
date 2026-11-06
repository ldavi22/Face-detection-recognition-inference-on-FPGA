/* Produce testbench vectors for the layer 18 accelerator.
 *
 *   ./gen_vectors qmodel.bin qdump vectors
 *
 * Writes four hex files the testbench reads with $readmemh:
 *
 *   vectors/act.hex   activations, channel-interleaved, 4 int8 per word
 *   vectors/wgt.hex   weights, same packing
 *   vectors/prm.hex   bias, mult, shift per output channel
 *   vectors/exp.hex   expected output, one int8 byte per line
 *
 * The activation transpose from planar to channel-interleaved happens
 * here. On the board the HPS does exactly the same thing before writing
 * into the accelerator -- it is a software memcpy loop, cheap next to
 * everything else, and it keeps the fabric's addressing a plain counter.
 */

#include "qnet.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define L_IDX 18    /* the conv we are accelerating */
#define L_IN 17     /* its input: route layer 17     */
#define CONV_IDX 10 /* layer 18 is the 11th conv (0-based) */

#define C_IN 256
#define C_OUT 128
#define POS 169

static uint32_t pack4(const int8_t *p) {
  return ((uint32_t)(uint8_t)p[0]) | ((uint32_t)(uint8_t)p[1] << 8) |
         ((uint32_t)(uint8_t)p[2] << 16) | ((uint32_t)(uint8_t)p[3] << 24);
}

int main(int argc, char **argv) {
  if (argc != 4) {
    fprintf(stderr, "usage: %s <qmodel.bin> <qdump_dir> <out_dir>\n", argv[0]);
    return 1;
  }

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

  /* --- activations: planar in the dump, interleaved for the fabric --- */

  snprintf(path, sizeof path, "%s/qlayer_%02d.bin", argv[2], L_IN);
  FILE *f = fopen(path, "rb");
  if (!f) {
    perror(path);
    return 1;
  }
  int8_t *in = malloc((size_t)C_IN * POS);
  if (fread(in, 1, (size_t)C_IN * POS, f) != (size_t)C_IN * POS) {
    fprintf(stderr, "%s: short read\n", path);
    return 1;
  }
  fclose(f);

  snprintf(path, sizeof path, "%s/act.hex", argv[3]);
  FILE *fa = fopen(path, "w");
  if (!fa) {
    perror(path);
    return 1;
  }
  for (int p = 0; p < POS; p++)
    for (int ch = 0; ch < C_IN; ch += 4) {
      int8_t t[4];
      for (int j = 0; j < 4; j++)
        t[j] = in[(long)(ch + j) * POS + p]; /* planar -> interleaved */
      fprintf(fa, "%08x\n", pack4(t));
    }
  fclose(fa);

  /* --- weights: already output-channel major, just repack ------------ */

  snprintf(path, sizeof path, "%s/wgt.hex", argv[3]);
  FILE *fw = fopen(path, "w");
  if (!fw) {
    perror(path);
    return 1;
  }
  for (int o = 0; o < C_OUT; o++)
    for (int ch = 0; ch < C_IN; ch += 4)
      fprintf(fw, "%08x\n", pack4(&cw->w[(long)o * C_IN + ch]));
  fclose(fw);

  /* --- params -------------------------------------------------------- */

  snprintf(path, sizeof path, "%s/prm.hex", argv[3]);
  FILE *fp = fopen(path, "w");
  if (!fp) {
    perror(path);
    return 1;
  }
  for (int o = 0; o < C_OUT; o++) {
    fprintf(fp, "%08x\n", (uint32_t)cw->bias[o]);
    fprintf(fp, "%08x\n", (uint32_t)cw->mult[o]);
    fprintf(fp, "%08x\n", (uint32_t)cw->shift[o]);
    if (cw->shift[o] < 0 || cw->shift[o] > 63)
      fprintf(stderr, "WARNING: channel %d shift %d outside 0..63\n", o,
              cw->shift[o]);
  }
  fclose(fp);

  /* --- expected output ----------------------------------------------- */

  snprintf(path, sizeof path, "%s/qlayer_%02d.bin", argv[2], L_IDX);
  f = fopen(path, "rb");
  if (!f) {
    perror(path);
    return 1;
  }
  int8_t *exp = malloc((size_t)C_OUT * POS);
  if (fread(exp, 1, (size_t)C_OUT * POS, f) != (size_t)C_OUT * POS) {
    fprintf(stderr, "%s: short read\n", path);
    return 1;
  }
  fclose(f);

  snprintf(path, sizeof path, "%s/exp.hex", argv[3]);
  FILE *fe = fopen(path, "w");
  if (!fe) {
    perror(path);
    return 1;
  }
  for (long i = 0; i < (long)C_OUT * POS; i++)
    fprintf(fe, "%02x\n", (uint8_t)exp[i]);
  fclose(fe);

  printf("wrote %s/{act,wgt,prm,exp}.hex\n", argv[3]);
  printf("  act %d words, wgt %d words, prm %d words, exp %d bytes\n",
         POS * C_IN / 4, C_OUT * C_IN / 4, C_OUT * 3, C_OUT * POS);

  free(in);
  free(exp);
  qmodel_free(&m);
  return 0;
}
