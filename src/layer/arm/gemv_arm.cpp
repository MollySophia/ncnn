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

#include <array>
#include <assert.h>
#include <arm_neon.h>
#include <cmath>
#include <iostream>

namespace ncnn {

int Gemv_arm::create_pipeline(const Option& opt)
{
    assert(K % KT == 0);
    assert(N % 4 == 0);
    const int B_numel = B_data.total();
    // std::cout << "B_numel = " << B_numel << std::endl;
    BT_data.create(B_numel, 1u, opt.workspace_allocator);
    const int block_numel = KT * 4;
    scales.create(B_numel / KT, 4u, opt.workspace_allocator);
    zero_points.create(B_numel / KT, 4u, opt.workspace_allocator);

    if (BT_data.empty() || scales.empty() || zero_points.empty())
        return -100;

    const float* const ptr0 = B_data;

    // (K, N)
    // (K / 64, N / 4, 64, 4)
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int a = 0; a < K / KT; a++)
    {
        uint8_t* ptr = (uint8_t*)BT_data + a * KT * N;
        int block_id = a * (N / 4);
        for (int b = 0; b < N / 4; b++)
        {
            std::array<float, KT * 4> block_data;
            int index = 0;
            // every (64, 1) block in (64, 4) superblock has a scale and a zero_point
            for (int c = 0; c < KT; c++)
            {
                for (int d = 0; d < 4; d++)
                {
                    int k = a * KT + c;
                    int n = b * 4 + d;
                    block_data[index++] = ptr0[k * N + n];
                }
            }

            const auto [col_scales, col_zeropoints] = [&]() -> std::pair<std::array<float, 4>, std::array<float, 4> > {
                std::array<std::array<float, KT>, 4> col_datas;
                std::array<float, 4> col_scales;
                std::array<float, 4> col_zeropoints;

                for (int i = 0; i < KT * 4; i++)
                {
                    col_datas[i % 4][i / 4] = block_data[i];
                }

                // calculate scale and zero point
                // float[i] = int[i] * scale + zero_point
                // int[i] = (float[i] - zero_point) / scale
                // scale = (max - min) / 255
                for (int col = 0; col < 4; col++)
                {
                    const auto& col_data = col_datas[col];
                    float scale;
                    float zero_point;
                    float max = col_data[0];
                    float min = col_data[0];
                    for (int i = 1; i < static_cast<int>(col_data.size()); i++)
                    {
                        if (col_data[i] > max)
                        {
                            max = col_data[i];
                        }
                        if (col_data[i] < min)
                        {
                            min = col_data[i];
                        }
                    }
                    // std::cout << "max = " << max << std::endl;
                    // std::cout << "min = " << min << std::endl;
                    if (max == min)
                    {
                        scale = 1.f;
                    }
                    else
                    {
                        scale = (max - min) / 255.f;
                    }
                    zero_point = min;
                    col_scales[col] = scale;
                    col_zeropoints[col] = zero_point;

                    scales[block_id * 4 + col] = scale;
                    zero_points[block_id * 4 + col] = zero_point;
                }

                return {col_scales, col_zeropoints};
            }();

            // std::cout << "scales[" << block_id << "] = " << scale << std::endl;
            // std::cout << "zero_points[" << block_id << "] = " << zero_point << std::endl;
            block_id++;

            for (int i = 0; i < KT * 4; i++)
            {
                // std::cout << "pre quant, col_datas[" << i << "] = " << col_datas[i] << std::endl;
                block_data[i] = (block_data[i] - col_zeropoints[i % 4]) / col_scales[i % 4];
                assert(block_data[i] >= 0 && block_data[i] <= 255);
                *ptr++ = std::round(block_data[i]);
                // std::cout << "col_datas[" << i << "] = " << col_datas[i] << std::endl;
                // std::cout << "(int)col_datas[" << i << "] = " << std::round(col_datas[i]) << std::endl;
            }
        }
    }

    if (opt.lightmode)
    {
        B_data.release();
    }

    return 0;
}

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

int Gemv_arm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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
            uint16x8_t tmp_low;
            uint16x8_t tmp_high;

            float32x4_t _b0;
            float32x4_t _b1;
            float32x4_t _b2;
            float32x4_t _b3;

#define GEMV_KERNEL4x4(a_register_idx)                            \
    tmp = vld1q_u8(b_ptr);                                        \
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
    b_ptr += 16;                                                  \
    output = vfmaq_laneq_f32(output, _b0, _a##a_register_idx, 0); \
    output = vfmaq_laneq_f32(output, _b1, _a##a_register_idx, 1); \
    output = vfmaq_laneq_f32(output, _b2, _a##a_register_idx, 2); \
    output = vfmaq_laneq_f32(output, _b3, _a##a_register_idx, 3);

            // 64x4
            // GEMV_KERNEL4x4(0);
            tmp = vld1q_u8(b_ptr);
            tmp_low = vmovl_u8(vget_low_u8(tmp));
            tmp_high = vmovl_u8(vget_high_u8(tmp));
            _b0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(tmp_low)));
            // std::cout << "before dequant, _b0 = " << float32x4_to_string(_b0) << std::endl;
            _b0 = vmlaq_f32(zero_point, _b0, scale);
            // std::cout << "after dequant, _b0 = " << float32x4_to_string(_b0) << std::endl;
            _b1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(tmp_low)));
            _b1 = vmlaq_f32(zero_point, _b1, scale);
            _b2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(tmp_high)));
            _b2 = vmlaq_f32(zero_point, _b2, scale);
            _b3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(tmp_high)));
            _b3 = vmlaq_f32(zero_point, _b3, scale);
            b_ptr += 16;
            output = vfmaq_laneq_f32(output, _b0, _a0, 0);
            output = vfmaq_laneq_f32(output, _b1, _a0, 1);
            output = vfmaq_laneq_f32(output, _b2, _a0, 2);
            output = vfmaq_laneq_f32(output, _b3, _a0, 3);

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
