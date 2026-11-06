#include "qnet.h"

/* conv_idx counts only convolutional layers, in network order, and must
 * match the order fold_bn wrote them to folded.bin. */

#define CONV(iw, ih, ic, oc, k, lk, ci)                                        \
  {L_CONV, iw, ih, ic, iw, ih, oc, k, 1, (k) / 2, lk, ci, 0, 0, -1, -1}

#define POOL(iw, ih, c, ow, oh, sz, st)                                        \
  {L_MAXPOOL, iw, ih, c, ow, oh, c, 0, 0, 0, 0, -1, sz, st, -1, -1}

#define UPSAMPLE(iw, ih, c)                                                    \
  {L_UPSAMPLE, iw, ih, c, (iw) * 2, (ih) * 2, c, 0, 0, 0, 0, -1, 0, 0, -1, -1}

#define ROUTE1(w, h, c, a)                                                     \
  {L_ROUTE, w, h, c, w, h, c, 0, 0, 0, 0, -1, 0, 0, a, -1}

#define ROUTE2(w, h, c, a, b)                                                  \
  {L_ROUTE, w, h, c, w, h, c, 0, 0, 0, 0, -1, 0, 0, a, b}

#define YOLO(w, h, c) {L_YOLO, w, h, c, w, h, c, 0, 0, 0, 0, -1, 0, 0, -1, -1}

const layer_desc NET[N_LAYERS] = {
    /*  0 */ CONV(416, 416, 3, 16, 3, 1, 0),
    /*  1 */ POOL(416, 416, 16, 208, 208, 2, 2),
    /*  2 */ CONV(208, 208, 16, 32, 3, 1, 1),
    /*  3 */ POOL(208, 208, 32, 104, 104, 2, 2),
    /*  4 */ CONV(104, 104, 32, 64, 3, 1, 2),
    /*  5 */ POOL(104, 104, 64, 52, 52, 2, 2),
    /*  6 */ CONV(52, 52, 64, 128, 3, 1, 3),
    /*  7 */ POOL(52, 52, 128, 26, 26, 2, 2),
    /*  8 */ CONV(26, 26, 128, 256, 3, 1, 4),
    /*  9 */ POOL(26, 26, 256, 13, 13, 2, 2),
    /* 10 */ CONV(13, 13, 256, 512, 3, 1, 5),
    /* 11 */ POOL(13, 13, 512, 13, 13, 2, 1), /* stride 1 -- asymmetric pad */
    /* 12 */ CONV(13, 13, 512, 1024, 3, 1, 6),
    /* 13 */ CONV(13, 13, 1024, 256, 1, 1, 7),
    /* 14 */ CONV(13, 13, 256, 512, 3, 1, 8),
    /* 15 */ CONV(13, 13, 512, 18, 1, 0, 9), /* linear, no batchnorm */
    /* 16 */ YOLO(13, 13, 18),
    /* 17 */ ROUTE1(13, 13, 256, 13),
    /* 18 */ CONV(13, 13, 256, 128, 1, 1, 10),
    /* 19 */ UPSAMPLE(13, 13, 128),
    /* 20 */ ROUTE2(26, 26, 384, 19, 8),
    /* 21 */ CONV(26, 26, 384, 256, 3, 1, 11),
    /* 22 */ CONV(26, 26, 256, 18, 1, 0, 12), /* linear, no batchnorm */
    /* 23 */ YOLO(26, 26, 18),
};
