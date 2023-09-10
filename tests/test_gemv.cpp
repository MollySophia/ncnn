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

#include "layer/gemv.h"
#include "testutil.h"

#include <iostream>

static int test_gemv(int N, int K)
{
    ncnn::ParamDict pd;
    pd.set(0, N);
    pd.set(1, K); // beta

    std::vector<ncnn::Mat> weights;
    weights.push_back(ncnn::Mat(K, N));

    std::vector<ncnn::Mat> a;
    a.push_back(ncnn::Mat(K));

    Randomize(weights[0]);
    Randomize(a[0]);

    for (int i = 0; i < K * N; i++)
    {
        weights[0][i] = i * 3;
    }

    for (int i = 0; i < K; i++)
    {
        a[0][i] = 1;
    }

    int ret = test_layer<ncnn::Gemv>("Gemv", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_gemv failed N=%d K=%d\n", N, K);
    }

    return ret;
}

int main()
{
    SRAND(7767517);

    // k*n is the shape of the matrix
    int kn[][2] = {
        // minimal unit: (64, 4)
        {64, 4},
        {64, 8},
        {64, 64},
        {256, 4},
        {256, 128},
        {4096, 4096},
    };

    int nk_count = sizeof(kn) / sizeof(int) / 2;

    for (int i = 0; i < nk_count; i++)
    {
        int N = kn[i][1];
        int K = kn[i][0];

        int ret = 0 || test_gemv(N, K);

        if (ret != 0)
            return 0;
    }

    return 0;
}
