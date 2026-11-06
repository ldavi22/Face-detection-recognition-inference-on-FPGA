#ifndef FPGA_DEV_H
#define FPGA_DEV_H

#include <stdint.h>

/* Access to the FPGA fabric from Linux on the HPS.
 *
 * The lightweight HPS-to-FPGA bridge is memory mapped: writing to a
 * physical address in its window sets a register inside your design.
 * /dev/mem exposes all of physical memory as a file, so mmap-ing the
 * bridge's base gives you a pointer you dereference like any other.
 *
 * This is the same mechanism V4L2 uses for camera buffers -- install a
 * translation, then use ordinary pointers. Only the thing being mapped
 * differs.
 *
 * Two things are mandatory and both fail silently if omitted:
 *
 *   O_SYNC on the open   -- maps the region uncached. Without it your
 *                           register writes can sit in cache and never
 *                           reach the fabric, and reads return stale
 *                           cached values.
 *
 *   volatile on the ptr  -- the compiler assumes memory does not change
 *                           on its own. A poll loop on a status register
 *                           gets read once and reused forever, giving an
 *                           infinite loop that disappears at -O0.
 */

/* Conventional Cyclone V lightweight bridge base and span.
 * VERIFY against the Cyclone V HPS Technical Reference Manual and your
 * own Platform Designer address assignment before trusting these. */
#define LWBRIDGE_BASE 0xFF200000UL
#define LWBRIDGE_SPAN 0x00200000UL /* 2 MB */

typedef struct {
  int fd;
  void *base;          /* what mmap returned          */
  size_t span;         /* needed by munmap            */
  volatile uint32_t *regs; /* base + the component offset */
} fpga_dev;

/* component_offset is the base address Platform Designer assigned to the
 * accelerator inside the bridge window, in BYTES. Returns 0 on success. */
int fpga_open(fpga_dev *d, unsigned long component_offset);
void fpga_close(fpga_dev *d);

#endif /* FPGA_DEV_H */
