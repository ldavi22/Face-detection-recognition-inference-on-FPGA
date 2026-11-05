/* End-to-end test: tensors -> yolo -> decode -> nms -> crop -> mobilefacenet
 *
 * Usage:
 *   ./pipeline <prefix>                  embed every face, match against db
 *   ./pipeline <prefix> enroll <name>    embed face 0, append it to the db
 *
 * <prefix> is what you passed to prep_image, e.g. testdata/temo_a.
 * The source dimensions are read from <prefix>_meta.txt, so one binary
 * handles images of any size.
 */

#include "decode.h"
#include "detect.h"
#include "preprocess.h"
#include "recognize.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static float *load_floats(const char *path, size_t n) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    perror(path);
    return NULL;
  }
  float *buf = malloc(n * sizeof(float));
  if (!buf) {
    fclose(f);
    return NULL;
  }
  size_t got = fread(buf, sizeof(float), n, f);
  fclose(f);
  if (got != n) {
    fprintf(stderr, "%s: got %zu of %zu floats\n", path, got, n);
    free(buf);
    return NULL;
  }
  return buf;
}

/* Pulls src_w and src_h out of <prefix>_meta.txt. Returns 0 on success. */
static int read_meta(const char *prefix, int *src_w, int *src_h) {
  char path[512];
  snprintf(path, sizeof path, "%s_meta.txt", prefix);

  FILE *f = fopen(path, "r");
  if (!f) {
    perror(path);
    return 1;
  }

  char line[256];
  int val;
  int have_w = 0, have_h = 0;

  /* Line-based so non-numeric fields (e.g. "source face1.jpg") are skipped
   * instead of stalling the parse. */
  while (fgets(line, sizeof line, f)) {
    if (sscanf(line, "src_w %d", &val) == 1) {
      *src_w = val;
      have_w = 1;
    } else if (sscanf(line, "src_h %d", &val) == 1) {
      *src_h = val;
      have_h = 1;
    }
  }
  fclose(f);

  if (!have_w || !have_h) {
    fprintf(stderr, "%s: missing src_w/src_h\n", path);
    return 1;
  }
  return 0;
}

static void save_crop(const unsigned char *buf, int idx) {
  char path[64];
  snprintf(path, sizeof path, "face_%03d_112x112x3.raw", idx);
  FILE *f = fopen(path, "wb");
  if (!f)
    return;
  fwrite(buf, 1, FACE_BYTES, f);
  fclose(f);
  printf("  wrote %s\n", path);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <prefix> [enroll <name>]\n", argv[0]);
    fprintf(stderr, "  e.g. %s testdata/temo_a\n", argv[0]);
    fprintf(stderr, "       %s testdata/temo_a enroll Temo\n", argv[0]);
    return 1;
  }

  const char *prefix = argv[1];
  int enrolling = 0;
  const char *enroll_name = NULL;

  if (argc >= 4 && strcmp(argv[2], "enroll") == 0) {
    enrolling = 1;
    enroll_name = argv[3];
  }

  int src_w = 0, src_h = 0;
  if (read_meta(prefix, &src_w, &src_h) != 0)
    return 1;
  printf("source %dx%d\n", src_w, src_h);

  /* --- models ---------------------------------------------------------- */

  if (detector_darknet.init(YOLO_CFG, YOLO_WEIGHTS) != 0) {
    fprintf(stderr, "detector init failed\n");
    return 1;
  }
  if (rec_init(MFN_PARAM, MFN_BIN) != 0) {
    fprintf(stderr, "recognizer init failed\n");
    return 1;
  }

  /* --- input ----------------------------------------------------------- */

  char path[512];

  snprintf(path, sizeof path, "%s_input_%dx%dx3.bin", prefix, NET_W, NET_H);
  float *net_in = load_floats(path, (size_t)NET_W * NET_H * 3);

  snprintf(path, sizeof path, "%s_im_%dx%dx3.bin", prefix, src_w, src_h);
  float *src = load_floats(path, (size_t)src_w * src_h * 3);

  if (!net_in || !src)
    return 1;

  /* --- detect ---------------------------------------------------------- */

  yolo_output yo;
  detector_darknet.run(net_in, &yo);

  detection dets[256];
  int n = decode(&yo, CONF_THRESH, dets, 256);
  printf("decoded %d\n", n);

  int top_k = nms(dets, n, NMS_THRESH);
  printf("after nms %d\n", top_k);

  letterbox lb = letterbox_calc((u16)src_w, (u16)src_h, NET_W, NET_H);
  unletterbox(dets, top_k, lb, (u16)src_w, (u16)src_h);

  for (int k = 0; k < top_k; k++)
    printf("  box %d: centre (%.0f,%.0f) size %.0fx%.0f conf %.3f\n", k,
           dets[k].x, dets[k].y, dets[k].w, dets[k].h, dets[k].confidence);

  /* --- crop ------------------------------------------------------------ */

  unsigned char *crops = malloc((size_t)FACE_BYTES * MAX_DETECTION);
  float *to_crop = malloc(sizeof(float) * MAX_SIDE * MAX_SIDE * 3);
  float *to_resize = malloc(sizeof(float) * FACE_SIDE * FACE_SIDE * 3);
  assert(crops && to_crop && to_resize);

  int n_cropped = pp_faces(dets, src, top_k, MAX_DETECTION, (u16)src_w,
                           (u16)src_h, crops, to_crop, to_resize);
  printf("cropped %d\n", n_cropped);

  for (int k = 0; k < n_cropped; k++)
    save_crop(crops + (size_t)FACE_BYTES * k, k);

  /* --- embed ----------------------------------------------------------- */

  static float embs[MAX_DETECTION][EMB_DIM];
  for (int k = 0; k < n_cropped; k++) {
    int rc = rec_embed(crops + (size_t)FACE_BYTES * k, embs[k]);
    if (rc != 0) {
      fprintf(stderr, "rec_embed failed on crop %d: %d\n", k, rc);
      return 1;
    }
  }

  if (n_cropped > 1) {
    printf("pairwise cosine:\n");
    for (int a = 0; a < n_cropped; a++)
      for (int b = a + 1; b < n_cropped; b++)
        printf("  %d vs %d: %.4f\n", a, b, rec_similarity(embs[a], embs[b]));
  }

  /* --- enroll or match ------------------------------------------------- */

  enrolled_face table[MAX_ENROLLED];
  int n_enrolled = rec_load_enrolled(DB_PATH, table, MAX_ENROLLED);
  printf("enrolled: %d\n", n_enrolled);

  if (enrolling) {
    if (n_cropped != 1) {
      fprintf(stderr, "enroll needs exactly one face, got %d\n", n_cropped);
      return 1;
    }
    if (n_enrolled >= MAX_ENROLLED) {
      fprintf(stderr, "enrollment table full\n");
      return 1;
    }
    snprintf(table[n_enrolled].name, NAME_LEN, "%s", enroll_name);
    memcpy(table[n_enrolled].emb, embs[0], sizeof(float) * EMB_DIM);
    n_enrolled++;

    if (rec_save_enrolled(DB_PATH, table, n_enrolled) != 0) {
      fprintf(stderr, "save failed\n");
      return 1;
    }
    printf("enrolled '%s' as entry %d\n", enroll_name, n_enrolled - 1);
  } else {
    for (int k = 0; k < n_cropped; k++) {
      float score = 0.0f;
      int id = rec_match(embs[k], table, n_enrolled, MATCH_THRESH, &score);

      for (int d = 0; d < top_k; d++)
        if (dets[d].crop_idx == k)
          dets[d].identity = (i16)id;

      printf("  face %d: %s (best %.4f)\n", k,
             id < 0 ? "unknown" : table[id].name, score);
    }

    for (int d = 0; d < top_k; d++)
      printf("  det %d: crop %d identity %d -> %s\n", d, dets[d].crop_idx,
             dets[d].identity,
             dets[d].identity < 0 ? "unknown" : table[dets[d].identity].name);
  }

  free(crops);
  free(to_crop);
  free(to_resize);
  free(net_in);
  free(src);
  rec_cleanup();
  detector_darknet.cleanup();
  return 0;
}
