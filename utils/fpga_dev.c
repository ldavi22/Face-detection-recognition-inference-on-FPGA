#include "fpga_dev.h"
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

int fpga_open(fpga_dev *d, unsigned long component_offset) {
  memset(d, 0, sizeof(*d));

  /* O_SYNC is what makes the mapping uncached. Dropping it is the
   * classic cause of "my writes never reach the FPGA". */
  d->fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (d->fd == -1) {
    perror("open /dev/mem (are you root?)");
    return 1;
  }

  d->span = LWBRIDGE_SPAN;
  d->base = mmap(NULL, d->span, PROT_READ | PROT_WRITE, MAP_SHARED, d->fd,
                 LWBRIDGE_BASE);

  /* mmap signals failure with MAP_FAILED, not NULL */
  if (d->base == MAP_FAILED) {
    perror("mmap lightweight bridge");
    close(d->fd);
    d->fd = -1;
    return 1;
  }

  d->regs = (volatile uint32_t *)((char *)d->base + component_offset);
  return 0;
}

void fpga_close(fpga_dev *d) {
  if (d->base && d->base != MAP_FAILED)
    munmap(d->base, d->span);
  if (d->fd != -1)
    close(d->fd);
  memset(d, 0, sizeof(*d));
  d->fd = -1;
}
