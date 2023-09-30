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

#ifndef LAYER_GEMV_A32W4_H
#define LAYER_GEMV_A32W4_H

#include "layer.h"

namespace ncnn {

// A * B, where B is a matrix and A is a vector. It is equivalent to B * A with tranB=1.
class GemvA32W4 : public Layer
{
public:
    GemvA32W4();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

public:
    int M = 1;
    int N;
    int K;
    Mat scales;
    Mat BT_data;
    static const int KT = 16 * 4;
    int group_size;
    int group_num;
    // static constexpr int kGroupSize = 16;
    // static constexpr int kGroupNum = 64 / kGroupSize;
};

} // namespace ncnn

#endif // LAYER_GEMV_A32W4_H
