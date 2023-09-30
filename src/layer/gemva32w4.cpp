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

#include "gemva32w4.h"

#include <array>
#include <cassert>
#include <iostream>

namespace ncnn {

GemvA32W4::GemvA32W4()
{
    one_blob_only = true;
    support_inplace = false;
}

int GemvA32W4::load_param(const ParamDict& pd)
{
    N = pd.get(0, 0);
    K = pd.get(1, 0);
    group_size = pd.get(11, 32);
    assert(64 % group_size == 0);
    group_num = 64 / group_size;
    return 0;
}

int GemvA32W4::load_model(const ModelBin& mb)
{
    BT_data = mb.load(K / 2, N, 0);
    if (BT_data.empty())
        return -100;
    if (BT_data.elemsize != 1)
        return -99;

    // The frist 2 comes from a float32 contains two float16
    // The second 2 comes from a col contains two scales/zero_points
    scales = mb.load(K / 2 * N / 64 * group_num, 1);
    if (scales.empty())
        return -100;
    return 0;
}

int GemvA32W4::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    std::vector<Mat> bottom_blobs(1, bottom_blob);
    std::vector<Mat> top_blobs(1, top_blob);
    int ret = forward(bottom_blobs, top_blobs, opt);
    top_blob = top_blobs[0];
    return ret;
}

int GemvA32W4::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if 1
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
        std::array<float, 64> _a;
        for (int i = 0; i < 64; i++)
        {
            _a[i] = a_ptr[i];
        }

        const int kBlockCols = 8;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < N; i += kBlockCols)
        {
            const int block_id = (k / KT) * (N / kBlockCols) + (i / kBlockCols);
            // the offset is half of a32w8
            const uint8_t* b_ptr = (const uint8_t*)BT_data + (block_id * KT * kBlockCols) / 2;

            // 64x8

            std::vector<std::array<float, kBlockCols>> scales_vec;
            scales_vec.resize(group_num);
            for (int j = 0; j < kBlockCols; j++)
            {
                for (int k = 0; k < group_num; k++)
                {
                    scales_vec[k][j] = float16_to_float32(static_cast<const unsigned short*>(scales)[block_id * kBlockCols * group_num + kBlockCols * k + j]);
                }
            }

            float* output_ptr = (float*)top_blob + i;
            std::array<float, kBlockCols> output;
            for (int j = 0; j < kBlockCols; j++)
            {
                output[j] = output_ptr[j];
            }

            if (k == 0)
            {
                output.fill(0.f);
            }

            int8_t nf4_table[16];
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

            // from arm neon to naive impl, we "unroll" the simd
            // Example:
            // neon:
            // _b0 = vmulq_f16(_b0, scale##scale_idx);
            // _b1 = vmulq_f16(_b1, scale##scale_idx);
            // _b2 = vmulq_f16(_b2, scale##scale_idx);
            // _b3 = vmulq_f16(_b3, scale##scale_idx);
            //
            // naive (note the for loop i from 0 to 8):
            // for (int i = 0; i < 8; i++) {
            //    _b0[i] = _b0[i] * scale##scale_idx;
            //    _b1[i] = _b1[i] * scale##scale_idx;
            //    _b2[i] = _b2[i] * scale##scale_idx;
            //    _b3[i] = _b3[i] * scale##scale_idx;
            // }
            int tmp = 8 / group_num;
#define GEMV_KERNEL8x8(a_register_idx1)                                   \
    for (int j = 0; j < 8; j++)                                                      \
    {                                                                                \
        int row0 = b_ptr[j + 0] & 15;                                                \
        int row1 = b_ptr[j + 8] & 15;                                                \
        int row2 = b_ptr[j + 0] >> 4;                                                \
        int row3 = b_ptr[j + 8] >> 4;                                                \
        float dq_row0 = (static_cast<float>(nf4_table[row0]) * scales_vec[a_register_idx1 / tmp][j]); \
        float dq_row1 = (static_cast<float>(nf4_table[row1]) * scales_vec[a_register_idx1 / tmp][j]); \
        float dq_row2 = (static_cast<float>(nf4_table[row2]) * scales_vec[a_register_idx1 / tmp][j]); \
        float dq_row3 = (static_cast<float>(nf4_table[row3]) * scales_vec[a_register_idx1 / tmp][j]); \
        output[j] += _a[a_register_idx1 * 8 + 0] * dq_row0;                          \
        output[j] += _a[a_register_idx1 * 8 + 1] * dq_row1;                          \
        output[j] += _a[a_register_idx1 * 8 + 2] * dq_row2;                          \
        output[j] += _a[a_register_idx1 * 8 + 3] * dq_row3;                          \
    }                                                                                \
    for (int j = 0; j < 8; j++)                                                      \
    {                                                                                \
        int row4 = b_ptr[j + 16 + 0] & 15;                                           \
        int row5 = b_ptr[j + 16 + 8] & 15;                                           \
        int row6 = b_ptr[j + 16 + 0] >> 4;                                           \
        int row7 = b_ptr[j + 16 + 8] >> 4;                                           \
        float dq_row4 = (static_cast<float>(nf4_table[row4]) * scales_vec[a_register_idx1 / tmp][j]); \
        float dq_row5 = (static_cast<float>(nf4_table[row5]) * scales_vec[a_register_idx1 / tmp][j]); \
        float dq_row6 = (static_cast<float>(nf4_table[row6]) * scales_vec[a_register_idx1 / tmp][j]); \
        float dq_row7 = (static_cast<float>(nf4_table[row7]) * scales_vec[a_register_idx1 / tmp][j]); \
        output[j] += _a[a_register_idx1 * 8 + 4] * dq_row4;                          \
        output[j] += _a[a_register_idx1 * 8 + 5] * dq_row5;                          \
        output[j] += _a[a_register_idx1 * 8 + 6] * dq_row6;                          \
        output[j] += _a[a_register_idx1 * 8 + 7] * dq_row7;                          \
    }                                                                                \
    b_ptr += 32;

            GEMV_KERNEL8x8(0);
            assert((b_ptr - (const uint8_t*)BT_data) <= BT_data.total() * BT_data.elemsize);
            GEMV_KERNEL8x8(1);
            GEMV_KERNEL8x8(2);
            GEMV_KERNEL8x8(3);
            GEMV_KERNEL8x8(4);
            GEMV_KERNEL8x8(5);
            GEMV_KERNEL8x8(6);
            GEMV_KERNEL8x8(7);

#undef GEMV_KERNEL8x8

            for (int j = 0; j < 8; j++)
            {
                output_ptr[j] = output[j];
            }
        }
    }
#endif

    return 0;
}

} // namespace ncnn
