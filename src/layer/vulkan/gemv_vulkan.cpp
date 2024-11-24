// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "gemv_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

Gemv_vulkan::Gemv_vulkan()
{
    support_vulkan = true;
    support_image_storage = false;

    pipeline_gemv = 0;
}

int Gemv_vulkan::create_pipeline(const Option& opt)
{
    std::vector<vk_specialization_type> specializations(3);
    specializations[0].i = M;
    specializations[1].i = N;
    specializations[2].i = K;

    Mat local_size_xyz;
    // if (shape_packed.dims == 2)
    // {
    //     local_size_xyz.w = std::min(8, shape_packed.w);
    //     local_size_xyz.h = std::min(8, shape_packed.h);
    //     local_size_xyz.c = 1;
    // }

    // pack1
    // if (shape.dims == 0 || elempack == 1)
    {
        pipeline_gemv = new Pipeline(vkdev);
        pipeline_gemv->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_gemv->set_local_size_xyz(1, 8, 1);
        pipeline_gemv->create(LayerShaderType::gemv, opt, specializations);
    }

    return 0;
}

int Gemv_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_gemv;
    pipeline_gemv = 0;

    return 0;
}

int Gemv_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    cmd.record_upload(B_data, B_data_gpu, opt);

    return 0;
}

int Gemv_vulkan::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    const VkMat& A0 = bottom_blobs[0];
    const VkMat& B0 = B_data_gpu;

    VkMat A;
    VkMat B;
    VkMat C;
    vkdev->convert_packing(A0, A, 1, cmd, opt);
    vkdev->convert_packing(B0, B, 1, cmd, opt);
    vkdev->convert_packing(C0, C, 1, cmd, opt);

    int elempack = A.elempack;
    size_t elemsize = A.elemsize;

    VkMat& top_blob = top_blobs[0];
    top_blob.create(N, M, elemsize, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(3);
    bindings[0] = top_blob;
    bindings[1] = A;
    bindings[2] = B;

    std::vector<vk_constant_type> constants(1);
    constants[0].i = A.dims;

    const Pipeline* pipeline = pipeline_gemv;

    VkMat dispatcher;
    dispatcher.w = N;
    dispatcher.h = (K + 7) / 8;
    dispatcher.c = 1;
    cmd.record_pipeline(pipeline, bindings, constants, dispatcher);

    return 0;
}

int Gemv_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    std::vector<VkMat> bottom_blobs(1);
    std::vector<VkMat> top_blobs(1);
    bottom_blobs[0] = bottom_blob;
    int ret = forward(bottom_blobs, top_blobs, cmd, opt);
    top_blob = top_blobs[0];
    return ret;
}

} // namespace ncnn
