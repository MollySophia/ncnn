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

#include "gemv.h"

namespace ncnn {

Gemv::Gemv()
{
    one_blob_only = false;
    support_inplace = false;
}

int Gemv::load_param(const ParamDict& pd)
{
    one_blob_only = true;
    N = pd.get(0, 0);
    K = pd.get(1, 0);
    return 0;
}

int Gemv::load_model(const ModelBin& mb)
{
    B_data = mb.load(K, N, 0);
    return 0;
}

int Gemv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    std::vector<Mat> bottom_blobs(1, bottom_blob);
    std::vector<Mat> top_blobs(1, top_blob);
    int ret = forward(bottom_blobs, top_blobs, opt);
    top_blob = top_blobs[0];
    return ret;
}

int Gemv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& A = bottom_blobs[0];

    size_t elemsize = A.elemsize;

    Mat& top_blob = top_blobs[0];
    top_blob.create(N, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    for (int i = 0; i < N; i++)
    {
        const float* ptrB = (const float*)B_data + i;
        float sum = 0.f;
        for (int k = 0; k < K; k++)
        {
            sum += ptrB[k * N] * A[k];
        }
        top_blob[i] = sum;
    }

    return 0;
}

} // namespace ncnn

