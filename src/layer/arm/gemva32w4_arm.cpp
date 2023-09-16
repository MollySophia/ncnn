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

#include "gemva32w4_arm.h"

#include <array>
#include <assert.h>
#include <arm_neon.h>
#include <cmath>
#include <iostream>

namespace ncnn {

std::string float32x4_to_string(float32x4_t a)
{
    float* ptr = (float*)&a;
    std::string str = "[";
    for (int i = 0; i < 4; i++)
    {
        str += std::to_string(ptr[i]);
        if (i != 3)
        {
            str += ", ";
        }
    }
    str += "]";
    return str;
}

int GemvA32W4_arm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& A = bottom_blobs[0];

    size_t elemsize = A.elemsize;

    Mat& top_blob = top_blobs[0];
    top_blob.create(N, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // A and B_data will both only be read once
    for (int k = 0; k < K; k += KT)
    {
        const float* a_ptr = (const float*)A + k;
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

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < N; i += 4)
        {
            const uint8_t* b_ptr = (const uint8_t*)BT_data + k * N + i * 64;
            const int block_id = (k / KT) * (N / 4) + (i / 4);
            // 64x4

            const float32x4_t scale = vld1q_f32(&scales[block_id * 4]);
            const float32x4_t zero_point = vld1q_f32(&zero_points[block_id * 4]);

            float* output_ptr = (float*)top_blob + i;
            float32x4_t output = vld1q_f32(output_ptr);

            if (k == 0)
            {
                output = vdupq_n_f32(0.f);
            }

            uint8x16_t tmp;
            uint8x16_t tmp2;
            uint16x8_t tmp_low;
            uint16x8_t tmp_high;

            float32x4_t _b0;
            float32x4_t _b1;
            float32x4_t _b2;
            float32x4_t _b3;

#define GEMV_KERNEL8x4(a_register_idx1, a_register_idx2)                            \
    tmp = vld1q_u8(b_ptr);                                        \
    tmp2 = vshrq_n_u8(tmp, 4);                                    \
    tmp = vandq_u8(tmp, vdupq_n_u8(0xf));                         \
    tmp_low = vmovl_u8(vget_low_u8(tmp));                         \
    tmp_high = vmovl_u8(vget_high_u8(tmp));                       \
    _b0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(tmp_low)));        \
    _b0 = vmlaq_f32(zero_point, _b0, scale);                      \
    _b1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(tmp_low)));       \
    _b1 = vmlaq_f32(zero_point, _b1, scale);                      \
    _b2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(tmp_high)));       \
    _b2 = vmlaq_f32(zero_point, _b2, scale);                      \
    _b3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(tmp_high)));      \
    _b3 = vmlaq_f32(zero_point, _b3, scale);                      \
    output = vfmaq_laneq_f32(output, _b0, _a##a_register_idx1, 0); \
    output = vfmaq_laneq_f32(output, _b1, _a##a_register_idx1, 1); \
    output = vfmaq_laneq_f32(output, _b2, _a##a_register_idx1, 2); \
    output = vfmaq_laneq_f32(output, _b3, _a##a_register_idx1, 3); \
    tmp = tmp2; \
    tmp_low = vmovl_u8(vget_low_u8(tmp));                         \
    tmp_high = vmovl_u8(vget_high_u8(tmp));                       \
    _b0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(tmp_low)));        \
    _b0 = vmlaq_f32(zero_point, _b0, scale);                      \
    _b1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(tmp_low)));       \
    _b1 = vmlaq_f32(zero_point, _b1, scale);                      \
    _b2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(tmp_high)));       \
    _b2 = vmlaq_f32(zero_point, _b2, scale);                      \
    _b3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(tmp_high)));      \
    _b3 = vmlaq_f32(zero_point, _b3, scale);                      \
    output = vfmaq_laneq_f32(output, _b0, _a##a_register_idx2, 0); \
    output = vfmaq_laneq_f32(output, _b1, _a##a_register_idx2, 1); \
    output = vfmaq_laneq_f32(output, _b2, _a##a_register_idx2, 2); \
    output = vfmaq_laneq_f32(output, _b3, _a##a_register_idx2, 3); \
    b_ptr += 16;                                                  \

            GEMV_KERNEL8x4(0, 1);
            GEMV_KERNEL8x4(2, 3);
            GEMV_KERNEL8x4(4, 5);
            GEMV_KERNEL8x4(6, 7);
            GEMV_KERNEL8x4(8, 9);
            GEMV_KERNEL8x4(10, 11);
            GEMV_KERNEL8x4(12, 13);
            GEMV_KERNEL8x4(14, 15);

#undef GEMV_KERNEL4x4

            vst1q_f32(output_ptr, output);
        }
    }

    return 0;
}

} // namespace ncnn

