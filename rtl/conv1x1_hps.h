#ifndef CONV1X1_HPS_H
#define CONV1X1_HPS_H

#include "fpga_dev.h"
#include <stdint.h>

/* Driver for the conv1x1 accelerator.
 *
 * Register map, in 32-bit WORDS. The component decodes address bits
 * [18:16] as the region, so byte offsets are these shifted left by 2.
 *
 *   region 0, word 0   CTRL    write bit0 = start
 *   region 0, word 1   STATUS  bit0 done, bit1 busy
 *   region 0, word 2   CYCLES  cycles taken by the last run
 *   region 1           activation RAM  (write)
 *   region 2           weight RAM      (write)
 *   region 3           param RAM       (write)
 *   region 4           output RAM      (read, 4 int8 per word)
 */

#define C1_REG_CTRL   0x00000u
#define C1_REG_STATUS 0x00001u
#define C1_REG_CYCLES 0x00002u
#define C1_RGN_ACT    0x08000u
#define C1_RGN_WGT    0x10000u
#define C1_RGN_PRM    0x18000u
#define C1_RGN_OUT    0x20000u

typedef struct {
  fpga_dev dev;
  int c_in, c_out, pos;
} conv1x1_hps;

int c1_open(conv1x1_hps *h, unsigned long component_offset, int c_in,
            int c_out, int pos);
void c1_close(conv1x1_hps *h);

/* Weights and params change once per layer; activations change per frame.
 * They are separate calls so the per-frame cost can be measured on its
 * own -- that split is the result worth reporting. */
void c1_load_weights(conv1x1_hps *h, const int8_t *w);
void c1_load_params(conv1x1_hps *h, const int32_t *bias, const int32_t *mult,
                    const int32_t *shift);

/* in: planar int8, c_in * pos. Transposed to channel-interleaved here --
 * the fabric wants position-major so its addressing stays a counter. */
void c1_load_activations(conv1x1_hps *h, const int8_t *in);

/* Returns the cycle count reported by the hardware, or -1 on timeout. */
long c1_run(conv1x1_hps *h);

/* out: planar int8, c_out * pos */
void c1_read_output(conv1x1_hps *h, int8_t *out);

#endif /* CONV1X1_HPS_H */
