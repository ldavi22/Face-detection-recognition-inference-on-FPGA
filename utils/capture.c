#include "decode.h"
#include "detect.h"
#include "math.h"
#include "preprocess.h"
#include "recognize.h"
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

#define DEVICE "/dev/video2"
#define NBUF 8
#define WIDTH 640
#define HEIGHT 480
#define NET_W 416
#define NET_H 416

#define YOLO_CFG "face-yolov3-tiny.cfg"
#define YOLO_WEIGHTS "face-yolov3-tiny_41000.weights"
#define MFN_PARAM "mobilefacenet.param"
#define MFN_BIN "mobilefacenet.bin"
#define DB_PATH "faces.db"

#define CONF_THRESH 0.25f
#define NMS_THRESH 0.45f
#define MATCH_THRESH 0.45f

#define ENROLL_FRAMES 10

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
  return r;
}

static void enqueue(int index) {
  struct v4l2_buffer buf;
  memset(&buf, 0, sizeof(buf));
  buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  buf.index = index;
  buf.memory = V4L2_MEMORY_MMAP;
  if (xioctl(fd, VIDIOC_QBUF, &buf) == -1)
    die("VIDIOC_QBUF ( couldn't enqueue )");
}

static void draw_box(float *img, u16 w, u16 h, const detection *d, float r,
                     float g, float b) {
  u32 plane = (u32)w * h;
  int x0 = (int)(d->x - d->w / 2.0f + 0.5f);
  int y0 = (int)(d->y - d->h / 2.0f + 0.5f);
  int x1 = (int)(d->x + d->w / 2.0f + 0.5f);
  int y1 = (int)(d->y + d->h / 2.0f + 0.5f);

  if (x0 < 0)
    x0 = 0;
  if (y0 < 0)
    y0 = 0;
  if (x1 > w - 1)
    x1 = w - 1;
  if (y1 > h - 1)
    y1 = h - 1;
  if (x1 <= x0 || y1 <= y0)
    return;

  float col[3] = {r, g, b};

  for (int t = 0; t < 2; t++) { /* 2px thick */
    int yt = y0 + t, yb = y1 - t;
    int xl = x0 + t, xr = x1 - t;
    if (yt > y1 || yb < y0 || xl > x1 || xr < x0)
      break;

    for (int x = x0; x <= x1; x++)
      for (int k = 0; k < 3; k++) {
        img[x + yt * w + k * plane] = col[k];
        img[x + yb * w + k * plane] = col[k];
      }
    for (int y = y0; y <= y1; y++)
      for (int k = 0; k < 3; k++) {
        img[xl + y * w + k * plane] = col[k];
        img[xr + y * w + k * plane] = col[k];
      }
  }
}

int main(int argc, char **argv) {
  int enrolling = 0;
  const char *enroll_name = NULL;

  if (argc >= 3 && strcmp(argv[1], "enroll") == 0) {
    enrolling = 1;
    enroll_name = argv[2];
  } else if (argc >= 2 && strcmp(argv[1], "enroll") == 0) {
    fprintf(stderr, "usage: %s enroll <name>\n", argv[0]);
    return 1;
  }

  if (detector_darknet.init(YOLO_CFG, YOLO_WEIGHTS) != 0) {
    fprintf(stderr, "detector init failed\n");
    return 1;
  }
  if (rec_init(MFN_PARAM, MFN_BIN) != 0) {
    fprintf(stderr, "recognizer init failed\n");
    return 1;
  }

  enrolled_face table[MAX_ENROLLED];
  int n_enrolled = rec_load_enrolled(DB_PATH, table, MAX_ENROLLED);
  printf("enrolled rows: %d\n", n_enrolled);

  fd = open(DEVICE, O_RDWR | O_NONBLOCK);
  if (fd == -1) {
    perror("open " DEVICE);
    return 1;
  }

  struct v4l2_capability cap;
  memset(&cap, 0, sizeof(cap));
  if (xioctl(fd, VIDIOC_QUERYCAP, &cap) == -1)
    die("VIDIOC_QUERYCAP");

  __u32 caps = (cap.capabilities & V4L2_CAP_DEVICE_CAPS) ? cap.device_caps
                                                         : cap.capabilities;
  if (!(caps & V4L2_CAP_VIDEO_CAPTURE)) {
    fprintf(stderr, "%s: not a video capture device\n", DEVICE);
    return 1;
  }
  if (!(caps & V4L2_CAP_STREAMING)) {
    fprintf(stderr, "%s: not a streaming device\n", DEVICE);
    return 1;
  }

  struct v4l2_format fmt;
  memset(&fmt, 0, sizeof(fmt));
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  fmt.fmt.pix.width = WIDTH;
  fmt.fmt.pix.height = HEIGHT;
  fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
  fmt.fmt.pix.field = V4L2_FIELD_NONE;
  if (xioctl(fd, VIDIOC_S_FMT, &fmt) == -1)
    die("VIDIOC_S_FMT");

  if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_YUYV ||
      fmt.fmt.pix.width != WIDTH || fmt.fmt.pix.height != HEIGHT) {
    fprintf(stderr, "driver gave %ux%u instead of %ux%u YUYV\n",
            fmt.fmt.pix.width, fmt.fmt.pix.height, WIDTH, HEIGHT);
    return 1;
  }

  u32 stride = fmt.fmt.pix.bytesperline;

  struct v4l2_requestbuffers req;
  memset(&req, 0, sizeof(req));
  req.count = NBUF;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;
  if (xioctl(fd, VIDIOC_REQBUFS, &req) == -1)
    die("VIDIOC_REQBUFS");

  if (req.count < 2) {
    fprintf(stderr, "not enough buffer memory\n");
    return 1;
  }
  if (req.count > NBUF) {
    fprintf(stderr, "driver granted %u buffers, array holds %u\n", req.count,
            (unsigned)NBUF);
    return 1;
  }

  for (unsigned int i = 0; i < req.count; i++) {
    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.index = i;
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    if (xioctl(fd, VIDIOC_QUERYBUF, &buf) == -1)
      die("VIDIOC_QUERYBUF");

    buffers[i].length = buf.length;
    buffers[i].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE,
                            MAP_SHARED, fd, buf.m.offset);
    if (buffers[i].start == MAP_FAILED)
      die("mmap");
  }

  for (unsigned int i = 0; i < req.count; i++)
    enqueue(i);

  letterbox lb = letterbox_calc(WIDTH, HEIGHT, NET_W, NET_H);
  printf("letterbox: %ux%u at (%u,%u)\n", lb.new_w, lb.new_h, lb.dx, lb.dy);

  float *frame = malloc(sizeof(float) * WIDTH * HEIGHT * 3);
  float *resized = malloc(sizeof(float) * lb.new_w * lb.new_h * 3);
  float *net_in = malloc(sizeof(float) * NET_W * NET_H * 3);
  unsigned char *crops = malloc((size_t)FACE_BYTES * MAX_DETECTION);
  float *to_crop = malloc(sizeof(float) * MAX_SIDE * MAX_SIDE * 3);
  float *to_resize = malloc(sizeof(float) * FACE_SIDE * FACE_SIDE * 3);
  unsigned char *rgb_out = malloc((size_t)WIDTH * HEIGHT * 3);

  if (!frame || !resized || !net_in || !crops || !to_crop || !to_resize ||
      !rgb_out) {
    fprintf(stderr, "malloc failed\n");
    return 1;
  }

  my_fill_image(net_in, NET_W, NET_H, 0.5f);

  static float embs[MAX_DETECTION][EMB_DIM];
  static float enroll_acc[EMB_DIM];
  int enroll_count = 0;
  memset(enroll_acc, 0, sizeof(enroll_acc));

  signal(SIGINT, signalHandler);

  enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (xioctl(fd, VIDIOC_STREAMON, &type) == -1)
    die("VIDIOC_STREAMON");

  printf("streaming started\n");
  if (enrolling)
    printf("enrolling '%s' -- hold still, one face in frame\n", enroll_name);

  unsigned long processed = 0, dropped = 0, processed_t = 0;
  double prep_t = 0, det_t = 0, rec_t = 0;
  double t_initial = now_sec(), t_final = t_initial;

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

    int newest = -1;
    for (;;) {
      struct v4l2_buffer buf;
      memset(&buf, 0, sizeof(buf));
      buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buf.memory = V4L2_MEMORY_MMAP;

      if (xioctl(fd, VIDIOC_DQBUF, &buf) == -1) {
        if (errno == EAGAIN)
          break;
        die("VIDIOC_DQBUF");
      }
      if (newest != -1) {
        enqueue(newest);
        dropped++;
      }
      newest = buf.index;
    }
    if (newest == -1)
      continue;

    processed++;
    processed_t++;

    double t0 = now_sec();
    yuyv_to_rgb(buffers[newest].start, stride, WIDTH, HEIGHT, frame);
    my_resize_image(frame, WIDTH, HEIGHT, lb.new_w, lb.new_h, resized);
    my_embed_image(resized, lb.new_w, lb.new_h, net_in, NET_W, NET_H, lb.dx,
                   lb.dy);
    prep_t += now_sec() - t0;
    enqueue(newest);

    t0 = now_sec();
    yolo_output yo;
    detector_darknet.run(net_in, &yo);

    detection dets[256];
    int n = decode(&yo, CONF_THRESH, dets, 256);
    int top_k = nms(dets, n, NMS_THRESH);
    unletterbox(dets, top_k, lb, WIDTH, HEIGHT);
    det_t += now_sec() - t0;

    t0 = now_sec();
    int n_cropped = pp_faces(dets, frame, top_k, MAX_DETECTION, WIDTH, HEIGHT,
                             crops, to_crop, to_resize);

    for (int k = 0; k < n_cropped; k++)
      if (rec_embed(crops + (size_t)FACE_BYTES * k, embs[k]) != 0)
        embs[k][0] = 2.0f; /* sentinel: embedding failed, never matches */

    rec_t += now_sec() - t0;

    if (enrolling) {
      if (n_cropped == 1 && embs[0][0] <= 1.0f) {
        for (int i = 0; i < EMB_DIM; i++)
          enroll_acc[i] += embs[0][i];
        enroll_count++;
        printf("  captured %d/%d\n", enroll_count, ENROLL_FRAMES);
        if (enroll_count >= ENROLL_FRAMES)
          break;
      }
      continue;
    }

    for (int k = 0; k < n_cropped; k++) {
      float score = 0.0f;
      int id = (embs[k][0] > 1.0f) ? -1
                                   : rec_match(embs[k], table, n_enrolled,
                                               MATCH_THRESH, &score);
      for (int d = 0; d < top_k; d++)
        if (dets[d].crop_idx == k)
          dets[d].identity = (i16)id;
    }

    for (int d = 0; d < top_k; d++) {
      if (dets[d].identity >= 0)
        draw_box(frame, WIDTH, HEIGHT, &dets[d], 0.0f, 1.0f, 0.0f); /* known */
      else
        draw_box(frame, WIDTH, HEIGHT, &dets[d], 1.0f, 0.0f,
                 0.0f); /* unknown */
    }

    t_final = now_sec();
    if (t_final - t_initial >= 1.0) {
      printf("fps %.1f | prep %.1f det %.1f rec %.1f ms | faces %d | dropped "
             "%lu\n",
             processed_t / (t_final - t_initial), 1000.0 * prep_t / processed_t,
             1000.0 * det_t / processed_t, 1000.0 * rec_t / processed_t, top_k,
             dropped);

      for (int d = 0; d < top_k; d++)
        printf("    box %d: (%.0f,%.0f) %.0fx%.0f -> %s\n", d, dets[d].x,
               dets[d].y, dets[d].w, dets[d].h,
               dets[d].identity < 0 ? "unknown" : table[dets[d].identity].name);

      /* last annotated frame, for eyeballing:
       *   ffmpeg -y -f rawvideo -pix_fmt rgb24 -s 640x480 -i frame.raw out.png
       */
      to_rgb(frame, WIDTH, HEIGHT, rgb_out);
      FILE *ff = fopen("frame.raw", "wb");
      if (ff) {
        fwrite(rgb_out, 1, (size_t)WIDTH * HEIGHT * 3, ff);
        fclose(ff);
      }

      t_initial = t_final;
      processed_t = 0;
      prep_t = det_t = rec_t = 0;
    }
  }

  if (enrolling && enroll_count > 0) {
    float norm = 0.0f;
    for (int i = 0; i < EMB_DIM; i++)
      norm += enroll_acc[i] * enroll_acc[i];
    norm = sqrtf(norm);

    if (norm > 1e-8f && n_enrolled < MAX_ENROLLED) {
      snprintf(table[n_enrolled].name, NAME_LEN, "%s", enroll_name);
      for (int i = 0; i < EMB_DIM; i++)
        table[n_enrolled].emb[i] = enroll_acc[i] / norm;
      n_enrolled++;

      if (rec_save_enrolled(DB_PATH, table, n_enrolled) == 0)
        printf("enrolled '%s' from %d frames as row %d\n", enroll_name,
               enroll_count, n_enrolled - 1);
      else
        fprintf(stderr, "save failed\n");
    } else {
      fprintf(stderr, "enrollment table full or degenerate embedding\n");
    }
  } else if (enrolling) {
    fprintf(stderr, "no usable frames captured\n");
  }

  printf("shutting down\n");

  if (xioctl(fd, VIDIOC_STREAMOFF, &type) == -1)
    die("VIDIOC_STREAMOFF");

  for (unsigned int i = 0; i < req.count; i++)
    munmap(buffers[i].start, buffers[i].length);

  free(frame);
  free(resized);
  free(net_in);
  free(crops);
  free(to_crop);
  free(to_resize);
  free(rgb_out);
  close(fd);

  rec_cleanup();
  detector_darknet.cleanup();
  return 0;
}
