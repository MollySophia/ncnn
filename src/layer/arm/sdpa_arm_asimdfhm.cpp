// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "sdpa_arm.h"
#include "sdpa_arm_flash.h"

namespace ncnn {

int flash_kv_cache2_fp16_path_asimdfhm(
    Mat& top_blob, const Mat& query, const Mat& past_key, const Mat& past_value,
    bool has_attn_mask, const Mat& attn_mask_blob,
    int src_seqlen, int dst_seqlen, int embed_dim, int out_embed_dim,
    int num_heads, int num_heads_per_group, float scale, const Option& opt)
{
    return flash_kv_cache2_fp16_path(
        top_blob, query, past_key, past_value, has_attn_mask, attn_mask_blob,
        src_seqlen, dst_seqlen, embed_dim, out_embed_dim,
        num_heads, num_heads_per_group, scale, opt);
}

} // namespace ncnn
