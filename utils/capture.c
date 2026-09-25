#include "preprocess.h"
#include "types.h"
#include <errno.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define DEVICE "/dev/video0"
#define NBUF 8
#define WIDTH 640
#define HEIGHT 480

static volatile sig_atomic_t stop = 0;

static double now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec / 1e9;
}

static void signalHandler(int sig) {
  (void)sig;
  stop = 1;
}

typedef struct {
  void *start;
  size_t length;
} buffer;

static buffer buffers[NBUF];
static int fd = -1;

static void die(const char *msg) {
  perror(msg);
  exit(EXIT_FAILURE);
}

static int xioctl(int fh, unsigned long request, void *arg) {
  int r;
  do {
    r = ioctl(fh, request, arg);
  } while (r == -1 && errno == EINTR);
  /* errno == EINTR means a signal occurred while the syscall was in
   * progress. This retry logic ensures that if that happens, ioctl is
   * retried. */
  return r;
}

static void enqueue(int index) {
  struct v4l2_buffer buf;
  memset(&buf, 0, sizeof(buf));
  buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  buf.index = index;
  buf.memory = V4L2_MEMORY_MMAP;
  if (xioctl(fd, VIDIOC_QBUF, &buf) == -1) {
    die("VIDIOC_QBUF ( couldn't enqueue )");
  }
}

int main(void) {

  fd = open(DEVICE, O_RDWR | O_NONBLOCK);
  if (fd == -1) {
    perror("open " DEVICE);
    exit(EXIT_FAILURE);
  }

  struct v4l2_capability cap;
  memset(&cap, 0, sizeof(cap));

  if (xioctl(fd, VIDIOC_QUERYCAP, &cap) == -1) {
    die("VIDIOC_QUERYCAP");
  }

  // printf("driver: %.*s\n", (int)sizeof(cap.driver), (char *)cap.driver);
  // printf("card:   %.*s\n", (int)sizeof(cap.card), (char *)cap.card);

  __u32 caps =
      (cap.capabilities & V4L2_CAP_DEVICE_CAPS)
          ? cap.device_caps   /* bit set   -> use the per-node value */
          : cap.capabilities; /* bit clear -> fall back to device-wide */

  // printf("caps:   0x%08X\n", caps);

  if (!(caps & V4L2_CAP_VIDEO_CAPTURE)) {
    fprintf(stderr, "%s: this device node is not a video capture device\n",
            DEVICE);
    exit(EXIT_FAILURE);
  }
  // printf("this device node is a capturing device\n");

  if (!(caps & V4L2_CAP_STREAMING)) {
    fprintf(stderr, "%s: this device node is not a streaming device\n", DEVICE);
    exit(EXIT_FAILURE);
  }
  // printf("this device node is a streaming device\n");

  struct v4l2_format fmt;
  memset(&fmt, 0, sizeof(fmt));

  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  fmt.fmt.pix.width = WIDTH;
  fmt.fmt.pix.height = HEIGHT;
  fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
  fmt.fmt.pix.field = V4L2_FIELD_NONE;

  if (xioctl(fd, VIDIOC_S_FMT, &fmt) == -1) {
    /* I filled the necessary parts; the kernel fills the remaining fields. */
    die("VIDIOC_S_FMT");
  }

  if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_YUYV ||
      fmt.fmt.pix.width != WIDTH || fmt.fmt.pix.height != HEIGHT) {
    fprintf(stderr, "driver gave %ux%u instead of %ux%u YUYV\n",
            fmt.fmt.pix.width, fmt.fmt.pix.height, WIDTH, HEIGHT);
    exit(EXIT_FAILURE);
  }

  /*
   printf("colorspace:   %u\n", fmt.fmt.pix.colorspace);
   printf("quantization: %u\n", fmt.fmt.pix.quantization);
   printf("format:       %ux%u YUYV, %u bytes/frame\n",
  fmt.fmt.pix.width,fmt.fmt.pix.height, fmt.fmt.pix.sizeimage);
  printf("bytesperline: %u\n", fmt.fmt.pix.bytesperline);
  */
  u32 stride = fmt.fmt.pix.bytesperline;

  /* Init of buffer */

  struct v4l2_requestbuffers req;
  memset(&req, 0, sizeof(req));

  req.count = NBUF;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;

  if (xioctl(fd, VIDIOC_REQBUFS, &req) == -1) {
    die("VIDIOC_REQBUFS");
  }
  // printf("buffer request: successful\n");

  if (req.count < 2) {
    fprintf(stderr, "not enough buffer memory\n");
    exit(EXIT_FAILURE);
  }
  if (req.count > NBUF) {
    fprintf(stderr, "driver granted %u buffers, array holds %u\n", req.count,
            (unsigned)NBUF);
    exit(EXIT_FAILURE);
  }
  // printf("buffer count: %u\n", req.count);

  /* querying buffers and mmap */

  for (unsigned int i = 0; i < req.count; i++) {
    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.index = i;
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    if (xioctl(fd, VIDIOC_QUERYBUF, &buf) == -1) {
      die("VIDIOC_QUERYBUF");
    }

    buffers[i].length = buf.length;
    buffers[i].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE,
                            MAP_SHARED, fd, buf.m.offset);

    if (buffers[i].start == MAP_FAILED) {
      die("mmap");
    }
    // printf("buffer[%u] mapping: successful\n", i);
  }

  /* enqueueing buffers */

  for (unsigned int i = 0; i < req.count; i++) {
    enqueue(i);
  }

  float *dst = malloc(sizeof(float) * WIDTH * HEIGHT * 3);
  if (dst == NULL) {
    fprintf(stderr, "malloc of RGB buffer failed\n");
    exit(EXIT_FAILURE);
  }

  /* start streaming */

  signal(SIGINT, signalHandler);

  enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (xioctl(fd, VIDIOC_STREAMON, &type) == -1) {
    die("VIDIOC_STREAMON");
  }

  printf("streaming started\n");

  unsigned long processed = 0, dropped = 0;
  unsigned long processed_t = 0;
  double t_initial = now_sec();
  double t_final = t_initial;

  while (!stop) {
    struct pollfd pfd = {.fd = fd, .events = POLLIN};
    int pr = poll(&pfd, 1, 2000);

    if (pr == -1) {
      if (errno == EINTR)
        continue;
      die("poll");
    }
    if (pr == 0) {
      printf("timeout: no frame in 2s\n");
      continue;
    }

    /* At least one buffer is full */

    int newest = -1;
    unsigned int newest_bytes = 0;
    for (;;) {
      struct v4l2_buffer buf;
      memset(&buf, 0, sizeof(buf));
      buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buf.memory = V4L2_MEMORY_MMAP;

      if (xioctl(fd, VIDIOC_DQBUF, &buf) == -1) {
        if (errno == EAGAIN)
          break; /* outgoing queue empty*/
        die("VIDIOC_DQBUF");
      }

      if (newest != -1) {
        enqueue(newest);
        dropped++;
      }

      newest = buf.index;
      newest_bytes = buf.bytesused;
    }

    if (newest == -1) {
      continue;
    }

    /* preprocess */
    processed++;
    processed_t++;

    yuyv_to_rgb(buffers[newest].start, stride, WIDTH, HEIGHT, dst);

    if (processed == 100) {
      FILE *f = fopen("rgb.raw", "wb");
      unsigned char *rgb = malloc((size_t)WIDTH * HEIGHT * 3);
      if (f != NULL && rgb != NULL) {
        to_rgb(dst, WIDTH, HEIGHT, rgb);
        fwrite(rgb, 1, (size_t)WIDTH * HEIGHT * 3, f);
        fclose(f);
        printf("wrote rgb.raw (%u source bytes -> %u RGB bytes)\n",
               newest_bytes, WIDTH * HEIGHT * 3);
      }
      free(rgb);
    }

    /* t_final = now_sec();
    if (t_final - t_initial >= 1.0) {
      printf("fps: %.1f | dropped %lu | total %lu\n",
             processed_t / (t_final - t_initial), dropped, processed);
      t_initial = t_final;
      processed_t = 0;
    }
    */

    /* Always give the buffer back. Holding one starves the driver. */
    enqueue(newest);
  }

  printf("shutting down\n");

  if (xioctl(fd, VIDIOC_STREAMOFF, &type) == -1) {
    die("VIDIOC_STREAMOFF");
  }

  for (unsigned int i = 0; i < req.count; i++) {
    munmap(buffers[i].start, buffers[i].length);
  }
  free(dst);
  close(fd);
  return 0;
}
