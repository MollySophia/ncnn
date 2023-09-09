// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "gemv_arm.h"

#include <assert.h>
#include <arm_neon.h>
#include <iostream>

namespace ncnn {

int Gemv_arm::create_pipeline(const Option& opt)
{
    assert(K % KT == 0);
    assert(N % 4 == 0);
    BT_data.create(K * N, B_data.elemsize);
    if (BT_data.empty())
        return -100;

    const float* ptr0 = B_data;
    float* ptr = BT_data;

    // (K, N)
    // (K / 64, N / 4, 64, 4)
    for (int a = 0; a < K / KT; a++)
    {
        for (int b = 0; b < N / 4; b++)
        {
            for (int c = 0; c < KT; c++)
            {
                for (int d = 0; d < 4; d++)
                {
                    int k = a * KT + c;
                    int n = b * 4 + d;
                    *ptr++ = ptr0[k * N + n];
                }
            }
        }
    }

    if (opt.lightmode)
    {
        B_data.release();
    }

    return 0;
}

int Gemv_arm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& A = bottom_blobs[0];

    size_t elemsize = A.elemsize;

    Mat& top_blob = top_blobs[0];
    top_blob.create(N, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // A and B_data will both only be read once
    const float* a_ptr = A;
    const float* b_ptr = BT_data;
    for (int k = 0; k < K; k += KT, a_ptr += KT)
    {
        float32x4_t _a0 = vld1q_f32(a_ptr);
        float32x4_t _a1 = vld1q_f32(a_ptr + 4);
        float32x4_t _a2 = vld1q_f32(a_ptr + 8);
        float32x4_t _a3 = vld1q_f32(a_ptr + 12);
        float32x4_t _a4 = vld1q_f32(a_ptr + 16);
        float32x4_t _a5 = vld1q_f32(a_ptr + 20);
        float32x4_t _a6 = vld1q_f32(a_ptr + 24);
        float32x4_t _a7 = vld1q_f32(a_ptr + 28);
        float32x4_t _a8 = vld1q_f32(a_ptr + 32);
        float32x4_t _a9 = vld1q_f32(a_ptr + 36);
        float32x4_t _a10 = vld1q_f32(a_ptr + 40);
        float32x4_t _a11 = vld1q_f32(a_ptr + 44);
        float32x4_t _a12 = vld1q_f32(a_ptr + 48);
        float32x4_t _a13 = vld1q_f32(a_ptr + 52);
        float32x4_t _a14 = vld1q_f32(a_ptr + 56);
        float32x4_t _a15 = vld1q_f32(a_ptr + 60);

        for (int i = 0; i < N; i += 4)
        {
            float* output_ptr = (float*)top_blob + i;
            float32x4_t output = vld1q_f32(output_ptr);
            if (k == 0)
            {
                output = vdupq_n_f32(0.f);
            }

            float32x4_t _b0;
            float32x4_t _b1;
            float32x4_t _b2;
            float32x4_t _b3;

#define GEMV_KERNEL4x4(a_register_idx)                            \
    _b0 = vld1q_f32(b_ptr);                                       \
    b_ptr += 4;                                                   \
    _b1 = vld1q_f32(b_ptr);                                       \
    b_ptr += 4;                                                   \
    _b2 = vld1q_f32(b_ptr);                                       \
    b_ptr += 4;                                                   \
    _b3 = vld1q_f32(b_ptr);                                       \
    b_ptr += 4;                                                   \
    output = vfmaq_laneq_f32(output, _b0, _a##a_register_idx, 0); \
    output = vfmaq_laneq_f32(output, _b1, _a##a_register_idx, 1); \
    output = vfmaq_laneq_f32(output, _b2, _a##a_register_idx, 2); \
    output = vfmaq_laneq_f32(output, _b3, _a##a_register_idx, 3);

            // 64x4
            GEMV_KERNEL4x4(0);
            GEMV_KERNEL4x4(1);
            GEMV_KERNEL4x4(2);
            GEMV_KERNEL4x4(3);
            GEMV_KERNEL4x4(4);
            GEMV_KERNEL4x4(5);
            GEMV_KERNEL4x4(6);
            GEMV_KERNEL4x4(7);
            GEMV_KERNEL4x4(8);
            GEMV_KERNEL4x4(9);
            GEMV_KERNEL4x4(10);
            GEMV_KERNEL4x4(11);
            GEMV_KERNEL4x4(12);
            GEMV_KERNEL4x4(13);
            GEMV_KERNEL4x4(14);
            GEMV_KERNEL4x4(15);

#undef GEMV_KERNEL4x4

            vst1q_f32(output_ptr, output);
        }
    }

    return 0;
}

} // namespace ncnn
