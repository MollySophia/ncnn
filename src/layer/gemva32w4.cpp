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
    return 0;
}

int GemvA32W4::load_model(const ModelBin& mb)
{
    BT_data = mb.load(K / 2, N, 0);
    if (BT_data.empty())
        return -100;
    if (BT_data.elemsize != 1)
        return -99;
    scales = mb.load(K * N / 64, 1);
    if (scales.empty())
        return -100;
    zero_points = mb.load(K * N / 64, 1);
    if (zero_points.empty())
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

#pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < N; i += 4)
        {
            // 32 instead of 64
            const uint8_t* b_ptr = (const uint8_t*)BT_data + k * N + i * 32;
            const int block_id = (k / KT) * (N / 4) + (i / 4);
            // 64x4

            std::array<float, 4> scale; // = vld1q_f32(&scales[block_id * 4]);
            for (int j = 0; j < 4; j++)
            {
                scale[j] = scales[block_id * 4 + j];
            }
            std::array<float, 4> zero_point; // = vld1q_f32(&zero_points[block_id * 4]);
            for (int j = 0; j < 4; j++)
            {
                zero_point[j] = zero_points[block_id * 4 + j];
            }

            float* output_ptr = (float*)top_blob + i;
            std::array<float, 4> output; // = vld1q_f32(output_ptr);
            for (int j = 0; j < 4; j++)
            {
                output[j] = output_ptr[j];
            }

            if (k == 0)
            {
                output.fill(0.f);
            }

#define GEMV_KERNEL8x4(a_register_idx1, a_register_idx2)                                                  \
    for (int j = 0; j < 4; j++)                                                                           \
    {                                                                                                     \
        int row0 = b_ptr[j + 0] & 15;                                                                     \
        int row1 = b_ptr[j + 4] & 15;                                                                     \
        int row2 = b_ptr[j + 8] & 15;                                                                     \
        int row3 = b_ptr[j + 12] & 15;                                                                    \
        int row4 = b_ptr[j + 0] >> 4;                                                                     \
        int row5 = b_ptr[j + 4] >> 4;                                                                     \
        int row6 = b_ptr[j + 8] >> 4;                                                                     \
        int row7 = b_ptr[j + 12] >> 4;                                                                    \
        output[j] += _a[a_register_idx1 * 4 + 0] * (static_cast<float>(row0) * scale[j] + zero_point[j]); \
        output[j] += _a[a_register_idx1 * 4 + 1] * (static_cast<float>(row1) * scale[j] + zero_point[j]); \
        output[j] += _a[a_register_idx1 * 4 + 2] * (static_cast<float>(row2) * scale[j] + zero_point[j]); \
        output[j] += _a[a_register_idx1 * 4 + 3] * (static_cast<float>(row3) * scale[j] + zero_point[j]); \
        output[j] += _a[a_register_idx2 * 4 + 0] * (static_cast<float>(row4) * scale[j] + zero_point[j]); \
        output[j] += _a[a_register_idx2 * 4 + 1] * (static_cast<float>(row5) * scale[j] + zero_point[j]); \
        output[j] += _a[a_register_idx2 * 4 + 2] * (static_cast<float>(row6) * scale[j] + zero_point[j]); \
        output[j] += _a[a_register_idx2 * 4 + 3] * (static_cast<float>(row7) * scale[j] + zero_point[j]); \
    }                                                                                                     \
    b_ptr += 16;

            GEMV_KERNEL8x4(0, 1);
            GEMV_KERNEL8x4(2, 3);
            GEMV_KERNEL8x4(4, 5);
            GEMV_KERNEL8x4(6, 7);
            GEMV_KERNEL8x4(8, 9);
            GEMV_KERNEL8x4(10, 11);
            GEMV_KERNEL8x4(12, 13);
            GEMV_KERNEL8x4(14, 15);

#undef GEMV_KERNEL8x4

            for (int j = 0; j < 4; j++)
            {
                output_ptr[j] = output[j];
            }
        }
    }
#endif

    return 0;
}

} // namespace ncnn
