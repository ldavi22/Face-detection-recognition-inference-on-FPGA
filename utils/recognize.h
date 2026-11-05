#ifndef RECOGNIZE_H
#define RECOGNIZE_H

#define EMB_DIM 128
#define FACE_SIDE 112
#define FACE_BYTES (FACE_SIDE * FACE_SIDE * 3)
#define MAX_ENROLLED 16
#define NAME_LEN 32

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  char name[NAME_LEN];
  float emb[EMB_DIM];
} enrolled_face;

/* Loads the MobileFaceNet model. Returns 0 on success. */
int rec_init(const char *param_path, const char *bin_path);

/* rgb112: FACE_BYTES of interleaved RGB, raw 0-255 (the model does its own
 * normalization internally). out128: caller-supplied, filled with an
 * L2-normalized embedding. Returns 0 on success. */
int rec_embed(const unsigned char *rgb112, float *out128);

void rec_cleanup(void);

/* Enrollment table ------------------------------------------------------- */

/* Reads the table from disk. Returns the number loaded, or 0 if absent. */
int rec_load_enrolled(const char *path, enrolled_face *table, int max_entries);

/* Writes the whole table back. Returns 0 on success. */
int rec_save_enrolled(const char *path, const enrolled_face *table, int n);

/* Dot product of two L2-normalized embeddings == cosine similarity. */
float rec_similarity(const float *a, const float *b);

/* Best match for emb. Returns the index into table, or -1 if none scores
 * above thresh. best_score may be NULL. */
int rec_match(const float *emb, const enrolled_face *table, int n, float thresh,
              float *best_score);

#ifdef __cplusplus
}
#endif

#endif /* RECOGNIZE_H */
