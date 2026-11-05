#include "recognize.h"
#include "net.h"
#include <cmath>
#include <cstdio>
#include <cstring>

static ncnn::Net g_net;
static int g_loaded = 0;

extern "C" int rec_init(const char *param_path, const char *bin_path) {
  if (g_net.load_param(param_path) != 0) {
    fprintf(stderr, "rec_init: load_param failed: %s\n", param_path);
    return 1;
  }
  if (g_net.load_model(bin_path) != 0) {
    fprintf(stderr, "rec_init: load_model failed: %s\n", bin_path);
    return 2;
  }
  g_loaded = 1;
  return 0;
}

extern "C" int rec_embed(const unsigned char *rgb112, float *out128) {
  if (!g_loaded)
    return 1;

  ncnn::Mat in = ncnn::Mat::from_pixels(rgb112, ncnn::Mat::PIXEL_RGB2BGR,
                                        FACE_SIDE, FACE_SIDE);

  /* A fresh Extractor per call: ncnn caches results inside one, so reusing
   * it returns the first face's embedding for every subsequent face. */
  ncnn::Extractor ex = g_net.create_extractor();
  if (ex.input("data", in) != 0)
    return 2;

  ncnn::Mat out;
  if (ex.extract("fc1", out) != 0)
    return 3;

  if (out.total() != EMB_DIM) {
    fprintf(stderr, "rec_embed: expected %d dims, got %d\n", EMB_DIM,
            (int)out.total());
    return 4;
  }

  float norm = 0.0f;
  for (int i = 0; i < EMB_DIM; i++)
    norm += out[i] * out[i];
  norm = sqrtf(norm);
  if (norm < 1e-8f)
    return 5;

  for (int i = 0; i < EMB_DIM; i++)
    out128[i] = out[i] / norm;

  return 0;
}

extern "C" void rec_cleanup(void) {
  g_net.clear();
  g_loaded = 0;
}

/* Enrollment table ------------------------------------------------------- */

extern "C" int rec_load_enrolled(const char *path, enrolled_face *table,
                                 int max_entries) {
  FILE *f = fopen(path, "rb");
  if (!f)
    return 0;

  int count = 0;
  if (fread(&count, sizeof(int), 1, f) != 1) {
    fclose(f);
    return 0;
  }
  if (count < 0 || count > max_entries)
    count = max_entries;

  size_t got = fread(table, sizeof(enrolled_face), (size_t)count, f);
  fclose(f);
  return (int)got;
}

extern "C" int rec_save_enrolled(const char *path, const enrolled_face *table,
                                 int n) {
  FILE *f = fopen(path, "wb");
  if (!f) {
    perror(path);
    return 1;
  }
  if (fwrite(&n, sizeof(int), 1, f) != 1) {
    fclose(f);
    return 2;
  }
  size_t put = fwrite(table, sizeof(enrolled_face), (size_t)n, f);
  if (fclose(f) != 0 || put != (size_t)n)
    return 3;
  return 0;
}

extern "C" float rec_similarity(const float *a, const float *b) {
  float dot = 0.0f;
  for (int i = 0; i < EMB_DIM; i++)
    dot += a[i] * b[i];
  return dot;
}

extern "C" int rec_match(const float *emb, const enrolled_face *table, int n,
                         float thresh, float *best_score) {
  int best = -1;
  float best_s = -2.0f;

  for (int i = 0; i < n; i++) {
    float s = rec_similarity(emb, table[i].emb);
    if (s > best_s) {
      best_s = s;
      best = i;
    }
  }

  if (best_score)
    *best_score = best_s;

  return (best_s >= thresh) ? best : -1;
}
