// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_REDUCTION_X86_AVX_H
#define LAYER_REDUCTION_X86_AVX_H

#include "reduction.h"

namespace ncnn {

class Reduction_x86_avx : public Reduction
{
public:
    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_REDUCTION_X86_AVX_H
