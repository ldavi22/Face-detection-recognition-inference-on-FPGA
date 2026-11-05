// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_NORMALIZE_X86_AVX_H
#define LAYER_NORMALIZE_X86_AVX_H

#include "normalize.h"

namespace ncnn {

class Normalize_x86_avx : public Normalize
{
public:
    Normalize_x86_avx();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_NORMALIZE_X86_AVX_H
