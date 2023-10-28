#ifndef LAYER_GEMV_VULKAN_H
#define LAYER_GEMV_VULKAN_H

#include "gemv.h"

namespace ncnn {

class Gemv_vulkan : virtual public Gemv
{
public:
    Gemv_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int upload_model(VkTransfer& cmd, const Option& opt);

    using Gemv::forward;
    virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

public:
    VkMat B_data_gpu;

    Pipeline* pipeline_gemv;
};

} // namespace ncnn

#endif // LAYER_GEMV_VULKAN_H
