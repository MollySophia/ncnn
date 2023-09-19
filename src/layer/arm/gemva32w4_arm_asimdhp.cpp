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

static std::string float32x4_to_string(float32x4_t a)
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

int GemvA32W4_arm::forward_with_fp16(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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
        // read 64 a elements
        const float* a_ptr = (const float*)A + k;
        float32x4_t _t0 = vld1q_f32(a_ptr);
        float32x4_t _t1 = vld1q_f32(a_ptr + 4);
        float32x4_t _t2 = vld1q_f32(a_ptr + 8);
        float32x4_t _t3 = vld1q_f32(a_ptr + 12);
        float32x4_t _t4 = vld1q_f32(a_ptr + 16);
        float32x4_t _t5 = vld1q_f32(a_ptr + 20);
        float32x4_t _t6 = vld1q_f32(a_ptr + 24);
        float32x4_t _t7 = vld1q_f32(a_ptr + 28);
        float16x4_t _tfp16_0 = vcvt_f16_f32(_t0);
        float16x4_t _tfp16_2 = vcvt_f16_f32(_t2);
        float16x4_t _tfp16_4 = vcvt_f16_f32(_t4);
        float16x4_t _tfp16_6 = vcvt_f16_f32(_t6);
        float16x8_t _a0 = vcvt_high_f16_f32(_tfp16_0, _t1);
        float16x8_t _a1 = vcvt_high_f16_f32(_tfp16_2, _t3);
        float16x8_t _a2 = vcvt_high_f16_f32(_tfp16_4, _t5);
        float16x8_t _a3 = vcvt_high_f16_f32(_tfp16_6, _t7);
        float32x4_t _t8 = vld1q_f32(a_ptr + 32);
        float32x4_t _t9 = vld1q_f32(a_ptr + 36);
        float32x4_t _t10 = vld1q_f32(a_ptr + 40);
        float32x4_t _t11 = vld1q_f32(a_ptr + 44);
        float32x4_t _t12 = vld1q_f32(a_ptr + 48);
        float32x4_t _t13 = vld1q_f32(a_ptr + 52);
        float32x4_t _t14 = vld1q_f32(a_ptr + 56);
        float32x4_t _t15 = vld1q_f32(a_ptr + 60);
        float16x4_t _tfp16_8 = vcvt_f16_f32(_t8);
        float16x4_t _tfp16_10 = vcvt_f16_f32(_t10);
        float16x4_t _tfp16_12 = vcvt_f16_f32(_t12);
        float16x4_t _tfp16_14 = vcvt_f16_f32(_t14);
        float16x8_t _a4 = vcvt_high_f16_f32(_tfp16_8, _t9);
        float16x8_t _a5 = vcvt_high_f16_f32(_tfp16_10, _t11);
        float16x8_t _a6 = vcvt_high_f16_f32(_tfp16_12, _t13);
        float16x8_t _a7 = vcvt_high_f16_f32(_tfp16_14, _t15);

        int8x16_t nf4_table;
        nf4_table[0] = -127;
        nf4_table[1] = -88;
        nf4_table[2] = -67;
        nf4_table[3] = -50;
        nf4_table[4] = -36;
        nf4_table[5] = -23;
        nf4_table[6] = -12;
        nf4_table[7] = 0;
        nf4_table[8] = 10;
        nf4_table[9] = 20;
        nf4_table[10] = 31;
        nf4_table[11] = 43;
        nf4_table[12] = 56;
        nf4_table[13] = 71;
        nf4_table[14] = 92;
        nf4_table[15] = 127;

        const int kBlockCols = 8;

#pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < N; i += kBlockCols)
        {
            const int block_id = (k / KT) * (N / kBlockCols) + (i / kBlockCols);
            // the offset is half of a32w8
            const uint8_t* b_ptr = (const uint8_t*)BT_data + (block_id * KT * kBlockCols) / 2;

            // 64x8 (KT*8)

            const float16x8_t scale0 = vld1q_f16(static_cast<const float16_t*>(scales) + block_id * kBlockCols * 2);
            const float16x8_t zero_point0 = vld1q_f16(static_cast<const float16_t*>(zero_points) + block_id * kBlockCols * 2);
            const float16x8_t scale1 = vld1q_f16(static_cast<const float16_t*>(scales) + block_id * kBlockCols * 2 + kBlockCols);
            const float16x8_t zero_point1 = vld1q_f16(static_cast<const float16_t*>(zero_points) + block_id * kBlockCols * 2 + kBlockCols);

            float* output_ptr = (float*)top_blob + i;
            float32x4_t output_low = vld1q_f32(output_ptr);
            float32x4_t output_high = vld1q_f32(output_ptr + 4);
            float16x8_t fp16_acc = vdupq_n_f16(0.f);

            uint8x16_t tmp;
            uint8x16_t tmp2;
            uint8x16_t tmp3;

            float16x8_t _b0;
            float16x8_t _b1;
            float16x8_t _b2;
            float16x8_t _b3;

            uint8x8_t _u0;
            uint8x8_t _u1;
            uint8x8_t _u2;
            uint8x8_t _u3;
            // uint16x8_t tmp_low;
            // uint16x8_t tmp_high;
            // _b0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(tmp_low)));
            // _b0 = vmlaq_f32(zero_point, _b0, scale);
            // _b1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(tmp_low)));
            // _b1 = vmlaq_f32(zero_point, _b1, scale);
            // _b2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(tmp_high)));
            // _b2 = vmlaq_f32(zero_point, _b2, scale);
            // _b3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(tmp_high)));
            // _b3 = vmlaq_f32(zero_point, _b3, scale);

#define GEMV_KERNEL8x8(a_register_idx1, scale_idx)                     \
    tmp = vld1q_u8(b_ptr);                                             \
    tmp3 = vld1q_u8(b_ptr + 16);                                       \
    tmp2 = vshrq_n_u8(tmp, 4);                                         \
    tmp = vandq_u8(tmp, vdupq_n_u8(15));                               \
    _u0 = vget_low_u8(tmp);                                            \
    _u1 = vget_high_u8(tmp);                                           \
    _u0 = vqtbl1_u8(nf4_table, _u0);                                   \
    _u1 = vqtbl1_u8(nf4_table, _u1);                                   \
    _b0 = vcvtq_f16_s16(vmovl_s8(vreinterpret_s8_u8(_u0)));            \
    _b1 = vcvtq_f16_s16(vmovl_s8(vreinterpret_s8_u8(_u1)));            \
    _b0 = vmulq_f16(_b0, scale##scale_idx);                            \
    _b1 = vmulq_f16(_b1, scale##scale_idx);                            \
    _b0 = vaddq_f16(_b0, zero_point##scale_idx);                       \
    _b1 = vaddq_f16(_b1, zero_point##scale_idx);                       \
                                                                       \
    _u2 = vget_low_u8(tmp2);                                           \
    _u3 = vget_high_u8(tmp2);                                          \
    _u2 = vqtbl1_u8(nf4_table, _u2);                                   \
    _u3 = vqtbl1_u8(nf4_table, _u3);                                   \
    _b2 = vcvtq_f16_s16(vmovl_s8(vreinterpret_s8_u8(_u2)));            \
    _b3 = vcvtq_f16_s16(vmovl_s8(vreinterpret_s8_u8(_u3)));            \
    _b2 = vmulq_f16(_b2, scale##scale_idx);                            \
    _b3 = vmulq_f16(_b3, scale##scale_idx);                            \
    _b2 = vaddq_f16(_b2, zero_point##scale_idx);                       \
    _b3 = vaddq_f16(_b3, zero_point##scale_idx);                       \
    fp16_acc = vfmaq_laneq_f16(fp16_acc, _b0, _a##a_register_idx1, 0); \
    fp16_acc = vfmaq_laneq_f16(fp16_acc, _b1, _a##a_register_idx1, 1); \
    fp16_acc = vfmaq_laneq_f16(fp16_acc, _b2, _a##a_register_idx1, 2); \
    fp16_acc = vfmaq_laneq_f16(fp16_acc, _b3, _a##a_register_idx1, 3); \
                                                                       \
    tmp = tmp3;                                                        \
    tmp2 = vshrq_n_u8(tmp, 4);                                         \
    tmp = vandq_u8(tmp, vdupq_n_u8(15));                               \
    _u0 = vget_low_u8(tmp);                                            \
    _u1 = vget_high_u8(tmp);                                           \
    _u0 = vqtbl1_u8(nf4_table, _u0);                                   \
    _u1 = vqtbl1_u8(nf4_table, _u1);                                   \
    _b0 = vcvtq_f16_s16(vmovl_s8(vreinterpret_s8_u8(_u0)));            \
    _b1 = vcvtq_f16_s16(vmovl_s8(vreinterpret_s8_u8(_u1)));            \
    _b0 = vmulq_f16(_b0, scale##scale_idx);                            \
    _b1 = vmulq_f16(_b1, scale##scale_idx);                            \
    _b0 = vaddq_f16(_b0, zero_point##scale_idx);                       \
    _b1 = vaddq_f16(_b1, zero_point##scale_idx);                       \
                                                                       \
    _u2 = vget_low_u8(tmp2);                                           \
    _u3 = vget_high_u8(tmp2);                                          \
    _u2 = vqtbl1_u8(nf4_table, _u2);                                   \
    _u3 = vqtbl1_u8(nf4_table, _u3);                                   \
    _b2 = vcvtq_f16_s16(vmovl_s8(vreinterpret_s8_u8(_u2)));            \
    _b3 = vcvtq_f16_s16(vmovl_s8(vreinterpret_s8_u8(_u3)));            \
    _b2 = vmulq_f16(_b2, scale##scale_idx);                            \
    _b3 = vmulq_f16(_b3, scale##scale_idx);                            \
    _b2 = vaddq_f16(_b2, zero_point##scale_idx);                       \
    _b3 = vaddq_f16(_b3, zero_point##scale_idx);                       \
    fp16_acc = vfmaq_laneq_f16(fp16_acc, _b0, _a##a_register_idx1, 4); \
    fp16_acc = vfmaq_laneq_f16(fp16_acc, _b1, _a##a_register_idx1, 5); \
    fp16_acc = vfmaq_laneq_f16(fp16_acc, _b2, _a##a_register_idx1, 6); \
    fp16_acc = vfmaq_laneq_f16(fp16_acc, _b3, _a##a_register_idx1, 7); \
    b_ptr += 32;

            GEMV_KERNEL8x8(0, 0);
            GEMV_KERNEL8x8(1, 0);
            GEMV_KERNEL8x8(2, 0);
            GEMV_KERNEL8x8(3, 0);
            GEMV_KERNEL8x8(4, 1);
            GEMV_KERNEL8x8(5, 1);
            GEMV_KERNEL8x8(6, 1);
            GEMV_KERNEL8x8(7, 1);

#undef GEMV_KERNEL8x8

            if (k == 0)
            {
                output_low = vdupq_n_f32(0.f);
                output_high = vdupq_n_f32(0.f);
            }
            vst1q_f32(output_ptr, vaddq_f32(vcvt_f32_f16(vget_low_f16(fp16_acc)), output_low));
            vst1q_f32(output_ptr + 4, vaddq_f32(vcvt_f32_f16(vget_high_f16(fp16_acc)), output_high));
        }
    }

    return 0;
}

} // namespace ncnn
