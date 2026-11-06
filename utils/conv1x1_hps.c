#include "conv1x1_hps.h"
#include <stdio.h>
#include <string.h>

int c1_open(conv1x1_hps *h, unsigned long component_offset, int c_in, int c_out,
            int pos) {
  memset(h, 0, sizeof(*h));
  h->c_in = c_in;
  h->c_out = c_out;
  h->pos = pos;
  return fpga_open(&h->dev, component_offset);
}

void c1_close(conv1x1_hps *h) { fpga_close(&h->dev); }

/* pack four consecutive int8 into one 32-bit word, little-endian so the
 * byte order matches what the fabric unpacks */
static inline uint32_t pack4(const int8_t *p) {
  return ((uint32_t)(uint8_t)p[0]) | ((uint32_t)(uint8_t)p[1] << 8) |
         ((uint32_t)(uint8_t)p[2] << 16) | ((uint32_t)(uint8_t)p[3] << 24);
}

void c1_load_weights(conv1x1_hps *h, const int8_t *w) {
  volatile uint32_t *r = h->dev.regs;
  int chunks = h->c_in / 4;

  for (int f = 0; f < h->c_out; f++)
    for (int k = 0; k < chunks; k++)
      r[C1_RGN_WGT + (unsigned)(f * chunks + k)] =
          pack4(&w[(long)f * h->c_in + k * 4]);
}

void c1_load_params(conv1x1_hps *h, const int32_t *bias, const int32_t *mult,
                    const int32_t *shift) {
  volatile uint32_t *r = h->dev.regs;

  for (int f = 0; f < h->c_out; f++) {
    r[C1_RGN_PRM + (unsigned)(f * 3 + 0)] = (uint32_t)bias[f];
    r[C1_RGN_PRM + (unsigned)(f * 3 + 1)] = (uint32_t)mult[f];
    r[C1_RGN_PRM + (unsigned)(f * 3 + 2)] = (uint32_t)shift[f];
  }
}

void c1_load_activations(conv1x1_hps *h, const int8_t *in) {
  volatile uint32_t *r = h->dev.regs;
  int chunks = h->c_in / 4;

  /* planar (channel-major) -> channel-interleaved (position-major).
   * Doing the transpose in software keeps the fabric's address
   * generation a plain counter instead of a strided one. */
  for (int p = 0; p < h->pos; p++) {
    for (int k = 0; k < chunks; k++) {
      int8_t t[4];
      for (int j = 0; j < 4; j++)
        t[j] = in[(long)(k * 4 + j) * h->pos + p];
      r[C1_RGN_ACT + (unsigned)(p * chunks + k)] = pack4(t);
    }
  }
}

long c1_run(conv1x1_hps *h) {
  volatile uint32_t *r = h->dev.regs;

  r[C1_REG_CTRL] = 1u;

  /* Poll. The bound is generous: the design needs about 1.4M cycles at
   * P=4, and each read here crosses the bridge, so a few million
   * iterations is far more than enough. */
  for (long i = 0; i < 50000000L; i++) {
    uint32_t st = r[C1_REG_STATUS];
    if (st & 1u)
      return (long)r[C1_REG_CYCLES];
  }

  fprintf(stderr, "c1_run: timeout waiting for done\n");
  return -1;
}

void c1_read_output(conv1x1_hps *h, int8_t *out) {
  volatile uint32_t *r = h->dev.regs;
  long n = (long)h->c_out * h->pos;

  /* Output is already planar (f*pos + p), so this is a straight copy --
   * only the 4-bytes-per-word unpacking is needed. */
  for (long i = 0; i < n; i++)
    out[i] = (int8_t)(r[C1_RGN_OUT + (unsigned)i] & 0xFFu);
}
