// FlashAttention-2 kernels for ARM NEON
// 2D tiled (BrxBc) with online softmax - avoids O(MxN) attention matrix
// Reads K/V directly from preallocated buffer (zero-copy for kv_cache=2)

#pragma once

#include <float.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#include "mat.h"
#include "option.h"
#include "cpu.h"
#include "benchmark.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif

// Forward decls for runtime-CPU dispatch wrappers.
// Defined in sdpa_arm_asimdhp.cpp (compiled with +fp16) and sdpa_arm_asimdfhm.cpp
// (compiled with +fp16fml). Each wrapper just calls flash_kv_cache2_fp16_path
// from this header - but its TU has the right -march flag, so the proper
// kernel branches inside this header become live.
#if NCNN_RUNTIME_CPU && NCNN_ARM82FP16FML && __aarch64__ && !__ARM_FEATURE_FP16_FML
namespace ncnn {
int flash_kv_cache2_fp16_path_asimdfhm(
    Mat& top_blob, const Mat& query, const Mat& past_key, const Mat& past_value,
    bool has_attn_mask, const Mat& attn_mask_blob,
    int src_seqlen, int dst_seqlen, int embed_dim, int out_embed_dim,
    int num_heads, int num_heads_per_group, float scale, const Option& opt);
} // namespace ncnn
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82 && __aarch64__ && !__ARM_FEATURE_FP16_VECTOR_ARITHMETIC
namespace ncnn {
int flash_kv_cache2_fp16_path_asimdhp(
    Mat& top_blob, const Mat& query, const Mat& past_key, const Mat& past_value,
    bool has_attn_mask, const Mat& attn_mask_blob,
    int src_seqlen, int dst_seqlen, int embed_dim, int out_embed_dim,
    int num_heads, int num_heads_per_group, float scale, const Option& opt);
} // namespace ncnn
#endif

// ============================================================================
// Fast vectorized exp approximation for NEON
// Uses: exp(x) = 2^(x * log2e) = 2^n * 2^f, polynomial approx for 2^f
// Accuracy: ~0.01% relative error, sufficient for softmax
// Speed: ~5 cycles for 4 floats vs ~100 cycles for 4x scalar fast_exp_f32()
// ============================================================================

#if __ARM_NEON

static inline float32x4_t fast_exp_f32x4(float32x4_t x)
{
    // Clamp to prevent overflow/underflow
    x = vmaxq_f32(x, vdupq_n_f32(-88.f));
    x = vminq_f32(x, vdupq_n_f32(88.f));

    // t = x * log2(e)
    const float32x4_t log2e = vdupq_n_f32(1.4426950408889634f);
    float32x4_t t = vmulq_f32(x, log2e);

    // n = floor(t), f = t - n  (f in [0, 1))
    float32x4_t n = vrndmq_f32(t);
    float32x4_t f = vsubq_f32(t, n);

    // 2^n via exponent bit manipulation
    int32x4_t ni = vcvtq_s32_f32(n);
    float32x4_t pow2n = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(ni, vdupq_n_s32(127)), 23));

    // 2^f ~= degree-4 minimax polynomial on [0,1]
    // Coefficients: Taylor series of 2^x truncated (relative error < 2e-5)
    const float32x4_t c0 = vdupq_n_f32(1.0f);
    const float32x4_t c1 = vdupq_n_f32(0.6931472f);
    const float32x4_t c2 = vdupq_n_f32(0.2402265f);
    const float32x4_t c3 = vdupq_n_f32(0.0555049f);
    const float32x4_t c4 = vdupq_n_f32(0.0096813f);

    // Horner: ((c4*f + c3)*f + c2)*f + c1)*f + c0
    float32x4_t pow2f = vfmaq_f32(c3, c4, f);
    pow2f = vfmaq_f32(c2, pow2f, f);
    pow2f = vfmaq_f32(c1, pow2f, f);
    pow2f = vfmaq_f32(c0, pow2f, f);

    return vmulq_f32(pow2n, pow2f);
}

// Scalar fast exp (same algorithm, for tail elements)
static inline float fast_exp_f32(float x)
{
    if (x < -88.f) return 0.f;
    if (x > 88.f) return INFINITY;
    float t = x * 1.4426950408889634f;
    float n = floorf(t);
    float f = t - n;
    union { float f; int32_t i; } pow2n;
    pow2n.i = ((int32_t)n + 127) << 23;
    float pow2f = 1.0f + f * (0.6931472f + f * (0.2402265f + f * (0.0555049f + f * 0.0096813f)));
    return pow2n.f * pow2f;
}

#endif // __ARM_NEON

// ============================================================================
// fp32 kernels
// ============================================================================

#if __ARM_NEON

// Vectorized dot product for fp32, general d_k
static inline float dot_fp32_neon(const float* a, const float* b, int d)
{
    float32x4_t s0 = vdupq_n_f32(0.f);
    float32x4_t s1 = vdupq_n_f32(0.f);
    int k = 0;
    for (; k + 7 < d; k += 8)
    {
        s0 = vfmaq_f32(s0, vld1q_f32(a + k), vld1q_f32(b + k));
        s1 = vfmaq_f32(s1, vld1q_f32(a + k + 4), vld1q_f32(b + k + 4));
    }
    for (; k + 3 < d; k += 4)
        s0 = vfmaq_f32(s0, vld1q_f32(a + k), vld1q_f32(b + k));
    float s = vaddvq_f32(vaddq_f32(s0, s1));
    for (; k < d; k++)
        s += a[k] * b[k];
    return s;
}

// Decode kernel: M=1, Bc=32
static void flash_attn_fp32_decode(
    const float* Q,        // [d_k]
    const float* K,        // [N * d_k]
    const float* V,        // [N * d_v]
    float* O,              // [d_v]
    int N, int d_k, int d_v, float scale, const float* mask)
{
    const int Bc = 32;
    float m = -FLT_MAX;
    float l = 0.f;

    memset(O, 0, d_v * sizeof(float));

    for (int jb = 0; jb < N; jb += Bc)
    {
        const int bc = (jb + Bc <= N) ? Bc : (N - jb);

        // 1. Compute QK scores for this block
        float scores[32];
        for (int j = 0; j < bc; j++)
        {
            scores[j] = dot_fp32_neon(Q, K + (jb + j) * d_k, d_k) * scale;
            if (mask)
                scores[j] += mask[jb + j];
        }

        // 2. Find block max
        float m_block = scores[0];
        for (int j = 1; j < bc; j++)
            m_block = fmaxf(m_block, scores[j]);

        // 3. Online softmax: rescale O once per block
        float m_new = fmaxf(m, m_block);
        float alpha = (m_new == m) ? 1.f : fast_exp_f32(m - m_new);

        if (l != 0.f && alpha != 1.f)
        {
            float32x4_t valpha = vdupq_n_f32(alpha);
            int d = 0;
            for (; d + 7 < d_v; d += 8)
            {
                vst1q_f32(O + d, vmulq_f32(vld1q_f32(O + d), valpha));
                vst1q_f32(O + d + 4, vmulq_f32(vld1q_f32(O + d + 4), valpha));
            }
            for (; d + 3 < d_v; d += 4)
                vst1q_f32(O + d, vmulq_f32(vld1q_f32(O + d), valpha));
            for (; d < d_v; d++)
                O[d] *= alpha;
        }
        l *= alpha;

        // 4. Accumulate P[j] * V[j]
        for (int j = 0; j < bc; j++)
        {
            float p = fast_exp_f32(scores[j] - m_new);
            l += p;

            const float* vj = V + (jb + j) * d_v;
            float32x4_t vp = vdupq_n_f32(p);
            int d = 0;
            for (; d + 7 < d_v; d += 8)
            {
                vst1q_f32(O + d, vfmaq_f32(vld1q_f32(O + d), vp, vld1q_f32(vj + d)));
                vst1q_f32(O + d + 4, vfmaq_f32(vld1q_f32(O + d + 4), vp, vld1q_f32(vj + d + 4)));
            }
            for (; d + 3 < d_v; d += 4)
                vst1q_f32(O + d, vfmaq_f32(vld1q_f32(O + d), vp, vld1q_f32(vj + d)));
            for (; d < d_v; d++)
                O[d] += p * vj[d];
        }

        m = m_new;
    }

    // 5. Normalize
    if (l > 0.f)
    {
        float inv_l = 1.f / l;
        float32x4_t vinv = vdupq_n_f32(inv_l);
        int d = 0;
        for (; d + 3 < d_v; d += 4)
            vst1q_f32(O + d, vmulq_f32(vld1q_f32(O + d), vinv));
        for (; d < d_v; d++)
            O[d] *= inv_l;
    }
}

// Prefill kernel: Br=4, Bc=32, 2D tiled
static void flash_attn_fp32_prefill(
    const float* Q,        // [M * d_k]
    const float* K,        // [N * d_k]
    const float* V,        // [N * d_v]
    float* O,              // [M * d_v]
    int M, int N, int d_k, int d_v, float scale, const float* mask)
{
    const int Br = 4;
    const int Bc = 32;

    float* row_m = (float*)malloc(M * sizeof(float));
    float* row_l = (float*)malloc(M * sizeof(float));

    memset(O, 0, M * d_v * sizeof(float));
    for (int i = 0; i < M; i++)
    {
        row_m[i] = -FLT_MAX;
        row_l[i] = 0.f;
    }

    // Outer: KV blocks
    for (int jb = 0; jb < N; jb += Bc)
    {
        const int bc = (jb + Bc <= N) ? Bc : (N - jb);

        // Inner: Q blocks
        for (int ib = 0; ib < M; ib += Br)
        {
            const int br = (ib + Br <= M) ? Br : (M - ib);

            // Compute S[br][bc] = Q_tile x K_tile^T
            float S[4][32];
            for (int i = 0; i < br; i++)
            {
                const float* qi = Q + (ib + i) * d_k;
                for (int j = 0; j < bc; j++)
                {
                    S[i][j] = dot_fp32_neon(qi, K + (jb + j) * d_k, d_k) * scale;
                    if (mask)
                        S[i][j] += mask[(ib + i) * N + jb + j];
                }
            }

            // Per-row online softmax + PV accumulation
            for (int i = 0; i < br; i++)
            {
                const int row = ib + i;
                float m_old = row_m[row];
                float l_old = row_l[row];
                float* oi = O + row * d_v;

                // Block max
                float m_block = S[i][0];
                for (int j = 1; j < bc; j++)
                    m_block = fmaxf(m_block, S[i][j]);

                float m_new = fmaxf(m_old, m_block);
                float alpha = fast_exp_f32(m_old - m_new);

                // Rescale O
                {
                    float32x4_t valpha = vdupq_n_f32(alpha);
                    int d = 0;
                    for (; d + 7 < d_v; d += 8)
                    {
                        vst1q_f32(oi + d, vmulq_f32(vld1q_f32(oi + d), valpha));
                        vst1q_f32(oi + d + 4, vmulq_f32(vld1q_f32(oi + d + 4), valpha));
                    }
                    for (; d + 3 < d_v; d += 4)
                        vst1q_f32(oi + d, vmulq_f32(vld1q_f32(oi + d), valpha));
                    for (; d < d_v; d++)
                        oi[d] *= alpha;
                }

                // Accumulate P*V
                float l_block = 0.f;
                for (int j = 0; j < bc; j++)
                {
                    float p = fast_exp_f32(S[i][j] - m_new);
                    l_block += p;

                    const float* vj = V + (jb + j) * d_v;
                    float32x4_t vp = vdupq_n_f32(p);
                    int d = 0;
                    for (; d + 7 < d_v; d += 8)
                    {
                        vst1q_f32(oi + d, vfmaq_f32(vld1q_f32(oi + d), vp, vld1q_f32(vj + d)));
                        vst1q_f32(oi + d + 4, vfmaq_f32(vld1q_f32(oi + d + 4), vp, vld1q_f32(vj + d + 4)));
                    }
                    for (; d + 3 < d_v; d += 4)
                        vst1q_f32(oi + d, vfmaq_f32(vld1q_f32(oi + d), vp, vld1q_f32(vj + d)));
                    for (; d < d_v; d++)
                        oi[d] += p * vj[d];
                }

                row_l[row] = alpha * l_old + l_block;
                row_m[row] = m_new;
            }
        }
    }

    // Final normalization
    for (int i = 0; i < M; i++)
    {
        float* oi = O + i * d_v;
        float inv_l = (row_l[i] > 0.f) ? 1.f / row_l[i] : 0.f;
        float32x4_t vinv = vdupq_n_f32(inv_l);
        int d = 0;
        for (; d + 3 < d_v; d += 4)
            vst1q_f32(oi + d, vmulq_f32(vld1q_f32(oi + d), vinv));
        for (; d < d_v; d++)
            oi[d] *= inv_l;
    }

    free(row_m);
    free(row_l);
}

#endif // __ARM_NEON

// ============================================================================
// fp16 kernels (ARM NEON fp16 vector arithmetic)
// QK dot and softmax in fp32 for precision, V accumulation in fp32, output fp16
// ============================================================================

#if __ARM_NEON && __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#include <arm_fp16.h>
#include "arm_usability.h"  // transpose4x4_u16, transpose8x8_u16

static inline void pack_P_8x12_fp16(const float* P_all, int br, __fp16* P_T)
{
    if (br == 8)
    {
        for (int j = 0; j < 12; j += 4)
        {
            float32x4_t r0 = vld1q_f32(P_all + 0 * 12 + j);
            float32x4_t r1 = vld1q_f32(P_all + 1 * 12 + j);
            float32x4_t r2 = vld1q_f32(P_all + 2 * 12 + j);
            float32x4_t r3 = vld1q_f32(P_all + 3 * 12 + j);
            float32x4_t r4 = vld1q_f32(P_all + 4 * 12 + j);
            float32x4_t r5 = vld1q_f32(P_all + 5 * 12 + j);
            float32x4_t r6 = vld1q_f32(P_all + 6 * 12 + j);
            float32x4_t r7 = vld1q_f32(P_all + 7 * 12 + j);

            float16x4_t h0 = vcvt_f16_f32(r0);
            float16x4_t h1 = vcvt_f16_f32(r1);
            float16x4_t h2 = vcvt_f16_f32(r2);
            float16x4_t h3 = vcvt_f16_f32(r3);
            float16x4_t h4 = vcvt_f16_f32(r4);
            float16x4_t h5 = vcvt_f16_f32(r5);
            float16x4_t h6 = vcvt_f16_f32(r6);
            float16x4_t h7 = vcvt_f16_f32(r7);

            uint16x4_t u0 = vreinterpret_u16_f16(h0);
            uint16x4_t u1 = vreinterpret_u16_f16(h1);
            uint16x4_t u2 = vreinterpret_u16_f16(h2);
            uint16x4_t u3 = vreinterpret_u16_f16(h3);
            uint16x4_t u4 = vreinterpret_u16_f16(h4);
            uint16x4_t u5 = vreinterpret_u16_f16(h5);
            uint16x4_t u6 = vreinterpret_u16_f16(h6);
            uint16x4_t u7 = vreinterpret_u16_f16(h7);
            transpose4x4_u16(u0, u1, u2, u3);
            transpose4x4_u16(u4, u5, u6, u7);

            uint16_t* dst = (uint16_t*)(P_T + j * 8);
            vst1_u16(dst + 0 * 8 + 0, u0);
            vst1_u16(dst + 0 * 8 + 4, u4);
            vst1_u16(dst + 1 * 8 + 0, u1);
            vst1_u16(dst + 1 * 8 + 4, u5);
            vst1_u16(dst + 2 * 8 + 0, u2);
            vst1_u16(dst + 2 * 8 + 4, u6);
            vst1_u16(dst + 3 * 8 + 0, u3);
            vst1_u16(dst + 3 * 8 + 4, u7);
        }
        return;
    }

    for (int j = 0; j < 12; j++)
    {
        for (int i = 0; i < br; i++)
            P_T[j * 8 + i] = (__fp16)P_all[i * 12 + j];
        for (int i = br; i < 8; i++)
            P_T[j * 8 + i] = (__fp16)0.f;
    }
}

// D=192 specialized dot product: fully unrolled 12 iterations of 16 elements
static inline float dot_fp16_d192(const __fp16* Q, const __fp16* K)
{
    float32x4_t s0 = vdupq_n_f32(0.f);
    float32x4_t s1 = vdupq_n_f32(0.f);
    float32x4_t s2 = vdupq_n_f32(0.f);
    float32x4_t s3 = vdupq_n_f32(0.f);

    #define DOT16(off) do { \
        float16x8_t q0 = vld1q_f16(Q + (off)); \
        float16x8_t q1 = vld1q_f16(Q + (off) + 8); \
        float16x8_t k0 = vld1q_f16(K + (off)); \
        float16x8_t k1 = vld1q_f16(K + (off) + 8); \
        s0 = vfmaq_f32(s0, vcvt_f32_f16(vget_low_f16(q0)), vcvt_f32_f16(vget_low_f16(k0))); \
        s1 = vfmaq_f32(s1, vcvt_f32_f16(vget_high_f16(q0)), vcvt_f32_f16(vget_high_f16(k0))); \
        s2 = vfmaq_f32(s2, vcvt_f32_f16(vget_low_f16(q1)), vcvt_f32_f16(vget_low_f16(k1))); \
        s3 = vfmaq_f32(s3, vcvt_f32_f16(vget_high_f16(q1)), vcvt_f32_f16(vget_high_f16(k1))); \
    } while(0)

    DOT16(0); DOT16(16); DOT16(32); DOT16(48);
    DOT16(64); DOT16(80); DOT16(96); DOT16(112);
    DOT16(128); DOT16(144); DOT16(160); DOT16(176);

    #undef DOT16

    return vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
}

// General fp16 dot product
static inline float dot_fp16_neon(const __fp16* Q, const __fp16* K, int d)
{
    float32x4_t s0 = vdupq_n_f32(0.f);
    float32x4_t s1 = vdupq_n_f32(0.f);
    int k = 0;
    for (; k + 15 < d; k += 16)
    {
        float16x8_t q0 = vld1q_f16(Q + k);
        float16x8_t q1 = vld1q_f16(Q + k + 8);
        float16x8_t k0 = vld1q_f16(K + k);
        float16x8_t k1 = vld1q_f16(K + k + 8);
#if __ARM_FEATURE_FP16_FML
        s0 = vfmlalq_low_f16(s0, q0, k0);
        s1 = vfmlalq_high_f16(s1, q0, k0);
        s0 = vfmlalq_low_f16(s0, q1, k1);
        s1 = vfmlalq_high_f16(s1, q1, k1);
#else
        s0 = vfmaq_f32(s0, vcvt_f32_f16(vget_low_f16(q0)), vcvt_f32_f16(vget_low_f16(k0)));
        s1 = vfmaq_f32(s1, vcvt_f32_f16(vget_high_f16(q0)), vcvt_f32_f16(vget_high_f16(k0)));
        s0 = vfmaq_f32(s0, vcvt_f32_f16(vget_low_f16(q1)), vcvt_f32_f16(vget_low_f16(k1)));
        s1 = vfmaq_f32(s1, vcvt_f32_f16(vget_high_f16(q1)), vcvt_f32_f16(vget_high_f16(k1)));
#endif
    }
    for (; k + 7 < d; k += 8)
    {
        float16x8_t q0 = vld1q_f16(Q + k);
        float16x8_t k0 = vld1q_f16(K + k);
#if __ARM_FEATURE_FP16_FML
        s0 = vfmlalq_low_f16(s0, q0, k0);
        s1 = vfmlalq_high_f16(s1, q0, k0);
#else
        s0 = vfmaq_f32(s0, vcvt_f32_f16(vget_low_f16(q0)), vcvt_f32_f16(vget_low_f16(k0)));
        s1 = vfmaq_f32(s1, vcvt_f32_f16(vget_high_f16(q0)), vcvt_f32_f16(vget_high_f16(k0)));
#endif
    }
    float s = vaddvq_f32(vaddq_f32(s0, s1));
    for (; k < d; k++)
        s += (float)Q[k] * (float)K[k];
    return s;
}

// 4-way batched dot product: compute 4 dot products sharing Q loads
// Saves 75% of Q loads and Q fp16->fp32 conversions
static inline void dot_fp16_4x(const __fp16* Q,
    const __fp16* K0, const __fp16* K1, const __fp16* K2, const __fp16* K3,
    int d_k, float* out)
{
    float32x4_t a0 = vdupq_n_f32(0.f);
    float32x4_t a1 = vdupq_n_f32(0.f);
    float32x4_t a2 = vdupq_n_f32(0.f);
    float32x4_t a3 = vdupq_n_f32(0.f);

    int k = 0;
    for (; k + 15 < d_k; k += 16)
    {
        // Load Q once - shared across 4 K rows
        float16x8_t qh0 = vld1q_f16(Q + k);
        float16x8_t qh1 = vld1q_f16(Q + k + 8);
#if __ARM_FEATURE_FP16_FML
        float16x8_t kh0 = vld1q_f16(K0 + k);
        float16x8_t kh1 = vld1q_f16(K0 + k + 8);
        a0 = vfmlalq_low_f16(a0, qh0, kh0);
        a0 = vfmlalq_high_f16(a0, qh0, kh0);
        a0 = vfmlalq_low_f16(a0, qh1, kh1);
        a0 = vfmlalq_high_f16(a0, qh1, kh1);

        kh0 = vld1q_f16(K1 + k);
        kh1 = vld1q_f16(K1 + k + 8);
        a1 = vfmlalq_low_f16(a1, qh0, kh0);
        a1 = vfmlalq_high_f16(a1, qh0, kh0);
        a1 = vfmlalq_low_f16(a1, qh1, kh1);
        a1 = vfmlalq_high_f16(a1, qh1, kh1);

        kh0 = vld1q_f16(K2 + k);
        kh1 = vld1q_f16(K2 + k + 8);
        a2 = vfmlalq_low_f16(a2, qh0, kh0);
        a2 = vfmlalq_high_f16(a2, qh0, kh0);
        a2 = vfmlalq_low_f16(a2, qh1, kh1);
        a2 = vfmlalq_high_f16(a2, qh1, kh1);

        kh0 = vld1q_f16(K3 + k);
        kh1 = vld1q_f16(K3 + k + 8);
        a3 = vfmlalq_low_f16(a3, qh0, kh0);
        a3 = vfmlalq_high_f16(a3, qh0, kh0);
        a3 = vfmlalq_low_f16(a3, qh1, kh1);
        a3 = vfmlalq_high_f16(a3, qh1, kh1);
#else
        float32x4_t q0 = vcvt_f32_f16(vget_low_f16(qh0));
        float32x4_t q1 = vcvt_f32_f16(vget_high_f16(qh0));
        float32x4_t q2 = vcvt_f32_f16(vget_low_f16(qh1));
        float32x4_t q3 = vcvt_f32_f16(vget_high_f16(qh1));

        // K row 0
        {
            float16x8_t kh0 = vld1q_f16(K0 + k);
            float16x8_t kh1 = vld1q_f16(K0 + k + 8);
            a0 = vfmaq_f32(a0, q0, vcvt_f32_f16(vget_low_f16(kh0)));
            a0 = vfmaq_f32(a0, q1, vcvt_f32_f16(vget_high_f16(kh0)));
            a0 = vfmaq_f32(a0, q2, vcvt_f32_f16(vget_low_f16(kh1)));
            a0 = vfmaq_f32(a0, q3, vcvt_f32_f16(vget_high_f16(kh1)));
        }
        // K row 1
        {
            float16x8_t kh0 = vld1q_f16(K1 + k);
            float16x8_t kh1 = vld1q_f16(K1 + k + 8);
            a1 = vfmaq_f32(a1, q0, vcvt_f32_f16(vget_low_f16(kh0)));
            a1 = vfmaq_f32(a1, q1, vcvt_f32_f16(vget_high_f16(kh0)));
            a1 = vfmaq_f32(a1, q2, vcvt_f32_f16(vget_low_f16(kh1)));
            a1 = vfmaq_f32(a1, q3, vcvt_f32_f16(vget_high_f16(kh1)));
        }
        // K row 2
        {
            float16x8_t kh0 = vld1q_f16(K2 + k);
            float16x8_t kh1 = vld1q_f16(K2 + k + 8);
            a2 = vfmaq_f32(a2, q0, vcvt_f32_f16(vget_low_f16(kh0)));
            a2 = vfmaq_f32(a2, q1, vcvt_f32_f16(vget_high_f16(kh0)));
            a2 = vfmaq_f32(a2, q2, vcvt_f32_f16(vget_low_f16(kh1)));
            a2 = vfmaq_f32(a2, q3, vcvt_f32_f16(vget_high_f16(kh1)));
        }
        // K row 3
        {
            float16x8_t kh0 = vld1q_f16(K3 + k);
            float16x8_t kh1 = vld1q_f16(K3 + k + 8);
            a3 = vfmaq_f32(a3, q0, vcvt_f32_f16(vget_low_f16(kh0)));
            a3 = vfmaq_f32(a3, q1, vcvt_f32_f16(vget_high_f16(kh0)));
            a3 = vfmaq_f32(a3, q2, vcvt_f32_f16(vget_low_f16(kh1)));
            a3 = vfmaq_f32(a3, q3, vcvt_f32_f16(vget_high_f16(kh1)));
        }
#endif
    }

    out[0] = vaddvq_f32(a0);
    out[1] = vaddvq_f32(a1);
    out[2] = vaddvq_f32(a2);
    out[3] = vaddvq_f32(a3);

    // Handle remaining elements
    for (; k < d_k; k++)
    {
        float qv = (float)Q[k];
        out[0] += qv * (float)K0[k];
        out[1] += qv * (float)K1[k];
        out[2] += qv * (float)K2[k];
        out[3] += qv * (float)K3[k];
    }
}

#if __ARM_FEATURE_FP16_FML
static inline void dot_fp16_4x_d192_fml(const __fp16* Q,
    const __fp16* K0, const __fp16* K1, const __fp16* K2, const __fp16* K3,
    float* out)
{
    float32x4_t a0 = vdupq_n_f32(0.f);
    float32x4_t a1 = vdupq_n_f32(0.f);
    float32x4_t a2 = vdupq_n_f32(0.f);
    float32x4_t a3 = vdupq_n_f32(0.f);

    #define DOT4X16(off) do { \
        float16x8_t qh0 = vld1q_f16(Q + (off)); \
        float16x8_t qh1 = vld1q_f16(Q + (off) + 8); \
        float16x8_t kh0 = vld1q_f16(K0 + (off)); \
        float16x8_t kh1 = vld1q_f16(K0 + (off) + 8); \
        a0 = vfmlalq_low_f16(a0, qh0, kh0); \
        a0 = vfmlalq_high_f16(a0, qh0, kh0); \
        a0 = vfmlalq_low_f16(a0, qh1, kh1); \
        a0 = vfmlalq_high_f16(a0, qh1, kh1); \
        kh0 = vld1q_f16(K1 + (off)); \
        kh1 = vld1q_f16(K1 + (off) + 8); \
        a1 = vfmlalq_low_f16(a1, qh0, kh0); \
        a1 = vfmlalq_high_f16(a1, qh0, kh0); \
        a1 = vfmlalq_low_f16(a1, qh1, kh1); \
        a1 = vfmlalq_high_f16(a1, qh1, kh1); \
        kh0 = vld1q_f16(K2 + (off)); \
        kh1 = vld1q_f16(K2 + (off) + 8); \
        a2 = vfmlalq_low_f16(a2, qh0, kh0); \
        a2 = vfmlalq_high_f16(a2, qh0, kh0); \
        a2 = vfmlalq_low_f16(a2, qh1, kh1); \
        a2 = vfmlalq_high_f16(a2, qh1, kh1); \
        kh0 = vld1q_f16(K3 + (off)); \
        kh1 = vld1q_f16(K3 + (off) + 8); \
        a3 = vfmlalq_low_f16(a3, qh0, kh0); \
        a3 = vfmlalq_high_f16(a3, qh0, kh0); \
        a3 = vfmlalq_low_f16(a3, qh1, kh1); \
        a3 = vfmlalq_high_f16(a3, qh1, kh1); \
    } while (0)

    DOT4X16(0); DOT4X16(16); DOT4X16(32); DOT4X16(48);
    DOT4X16(64); DOT4X16(80); DOT4X16(96); DOT4X16(112);
    DOT4X16(128); DOT4X16(144); DOT4X16(160); DOT4X16(176);

    #undef DOT4X16

    out[0] = vaddvq_f32(a0);
    out[1] = vaddvq_f32(a1);
    out[2] = vaddvq_f32(a2);
    out[3] = vaddvq_f32(a3);
}
#endif

// Native fp16 FMA 4-way dot product - 2.5x fewer instructions than fp32 convert path
// Accumulates in fp16, converts to fp32 only for final horizontal reduction
static inline void dot_fp16_native_4x(const __fp16* Q,
    const __fp16* K0, const __fp16* K1, const __fp16* K2, const __fp16* K3,
    int d_k, float* out)
{
    float16x8_t a0_lo = vdupq_n_f16((__fp16)0.f), a0_hi = vdupq_n_f16((__fp16)0.f);
    float16x8_t a1_lo = vdupq_n_f16((__fp16)0.f), a1_hi = vdupq_n_f16((__fp16)0.f);
    float16x8_t a2_lo = vdupq_n_f16((__fp16)0.f), a2_hi = vdupq_n_f16((__fp16)0.f);
    float16x8_t a3_lo = vdupq_n_f16((__fp16)0.f), a3_hi = vdupq_n_f16((__fp16)0.f);

    int k = 0;
    for (; k + 15 < d_k; k += 16)
    {
        float16x8_t q0 = vld1q_f16(Q + k);
        float16x8_t q1 = vld1q_f16(Q + k + 8);

        a0_lo = vfmaq_f16(a0_lo, q0, vld1q_f16(K0 + k));
        a0_hi = vfmaq_f16(a0_hi, q1, vld1q_f16(K0 + k + 8));
        a1_lo = vfmaq_f16(a1_lo, q0, vld1q_f16(K1 + k));
        a1_hi = vfmaq_f16(a1_hi, q1, vld1q_f16(K1 + k + 8));
        a2_lo = vfmaq_f16(a2_lo, q0, vld1q_f16(K2 + k));
        a2_hi = vfmaq_f16(a2_hi, q1, vld1q_f16(K2 + k + 8));
        a3_lo = vfmaq_f16(a3_lo, q0, vld1q_f16(K3 + k));
        a3_hi = vfmaq_f16(a3_hi, q1, vld1q_f16(K3 + k + 8));
    }
    for (; k + 7 < d_k; k += 8)
    {
        float16x8_t q0 = vld1q_f16(Q + k);
        a0_lo = vfmaq_f16(a0_lo, q0, vld1q_f16(K0 + k));
        a1_lo = vfmaq_f16(a1_lo, q0, vld1q_f16(K1 + k));
        a2_lo = vfmaq_f16(a2_lo, q0, vld1q_f16(K2 + k));
        a3_lo = vfmaq_f16(a3_lo, q0, vld1q_f16(K3 + k));
    }

    // Reduce fp16 accumulators -> fp32 result
    float16x8_t s0 = vaddq_f16(a0_lo, a0_hi);
    float16x8_t s1 = vaddq_f16(a1_lo, a1_hi);
    float16x8_t s2 = vaddq_f16(a2_lo, a2_hi);
    float16x8_t s3 = vaddq_f16(a3_lo, a3_hi);

    out[0] = vaddvq_f32(vaddq_f32(vcvt_f32_f16(vget_low_f16(s0)), vcvt_f32_f16(vget_high_f16(s0))));
    out[1] = vaddvq_f32(vaddq_f32(vcvt_f32_f16(vget_low_f16(s1)), vcvt_f32_f16(vget_high_f16(s1))));
    out[2] = vaddvq_f32(vaddq_f32(vcvt_f32_f16(vget_low_f16(s2)), vcvt_f32_f16(vget_high_f16(s2))));
    out[3] = vaddvq_f32(vaddq_f32(vcvt_f32_f16(vget_low_f16(s3)), vcvt_f32_f16(vget_high_f16(s3))));

    // Handle tail
    for (; k < d_k; k++)
    {
        float qv = (float)Q[k];
        out[0] += qv * (float)K0[k];
        out[1] += qv * (float)K1[k];
        out[2] += qv * (float)K2[k];
        out[3] += qv * (float)K3[k];
    }
}

// Decode kernel: M=1, Bc=64, fp16 input, fp32 accumulation, fp16 output
static void flash_attn_fp16_decode(
    const __fp16* Q,       // [d_k]
    const __fp16* K,       // [N * d_k]
    const __fp16* V,       // [N * d_v]
    __fp16* O,             // [d_v]
    int N, int d_k, int d_v, float scale, const __fp16* mask)
{
    const int Bc = 64;

    float m = -FLT_MAX;
    float l = 0.f;

    // O accumulator in fp32
    float O_buf[256];
    float* O_fp32 = (d_v <= 256) ? O_buf : (float*)malloc(d_v * sizeof(float));
    memset(O_fp32, 0, d_v * sizeof(float));

    for (int jb = 0; jb < N; jb += Bc)
    {
        const int bc = (jb + Bc <= N) ? Bc : (N - jb);

        // 1. Compute QK scores using 4-way batched dot product (fp32 convert path)
        float scores[64];
        {
            int j = 0;
            for (; j + 3 < bc; j += 4)
            {
#if __ARM_FEATURE_FP16_FML
                if (d_k == 192)
                {
                    dot_fp16_4x_d192_fml(Q,
                                         K + (jb + j) * d_k, K + (jb + j + 1) * d_k,
                                         K + (jb + j + 2) * d_k, K + (jb + j + 3) * d_k,
                                         scores + j);
                }
                else
#endif
                {
                dot_fp16_4x(Q,
                            K + (jb + j) * d_k, K + (jb + j + 1) * d_k,
                            K + (jb + j + 2) * d_k, K + (jb + j + 3) * d_k,
                            d_k, scores + j);
                }
                scores[j] *= scale; scores[j+1] *= scale;
                scores[j+2] *= scale; scores[j+3] *= scale;
                if (mask)
                {
                    scores[j] += (float)mask[jb+j]; scores[j+1] += (float)mask[jb+j+1];
                    scores[j+2] += (float)mask[jb+j+2]; scores[j+3] += (float)mask[jb+j+3];
                }
            }
            for (; j < bc; j++)
            {
                const __fp16* kj = K + (jb + j) * d_k;
                scores[j] = dot_fp16_neon(Q, kj, d_k) * scale;
                if (mask)
                    scores[j] += (float)mask[jb + j];
            }
        }

        // 2. Find block max
        float m_block = scores[0];
        for (int j = 1; j < bc; j++)
            m_block = fmaxf(m_block, scores[j]);

        // 3. Online softmax: rescale O once per block
        float m_new = fmaxf(m, m_block);
        float alpha = (m_new == m) ? 1.f : fast_exp_f32(m - m_new);

        if (l != 0.f && alpha != 1.f)
        {
            float32x4_t valpha = vdupq_n_f32(alpha);
            int d = 0;
            for (; d + 7 < d_v; d += 8)
            {
                vst1q_f32(O_fp32 + d, vmulq_f32(vld1q_f32(O_fp32 + d), valpha));
                vst1q_f32(O_fp32 + d + 4, vmulq_f32(vld1q_f32(O_fp32 + d + 4), valpha));
            }
            for (; d + 3 < d_v; d += 4)
                vst1q_f32(O_fp32 + d, vmulq_f32(vld1q_f32(O_fp32 + d), valpha));
            for (; d < d_v; d++)
                O_fp32[d] *= alpha;
        }
        l *= alpha;

        // 4. Convert scores in-place to P[j] = exp(scores[j] - m_new).
        {
            float32x4_t vm = vdupq_n_f32(m_new);
            float32x4_t vl = vdupq_n_f32(0.f);
            int j = 0;
            for (; j + 3 < bc; j += 4)
            {
                float32x4_t s4 = vsubq_f32(vld1q_f32(scores + j), vm);
                float32x4_t p4 = fast_exp_f32x4(s4);
                vst1q_f32(scores + j, p4);
                vl = vaddq_f32(vl, p4);
            }
            l += vaddvq_f32(vl);
            for (; j < bc; j++)
            {
                scores[j] = fast_exp_f32(scores[j] - m_new);
                l += scores[j];
            }
        }

        // 5. Register-tiled PV accumulation: O stays in registers, iterate over all KV positions
        // Split d_v into tiles of 64 (16 float32x4_t regs), keep O in registers for the entire inner loop
        {
            const int TILE_DV = 64;
            int d_start = 0;
            for (; d_start + TILE_DV <= d_v; d_start += TILE_DV)
            {
                // Load O tile into registers
                float32x4_t o0  = vld1q_f32(O_fp32 + d_start + 0);
                float32x4_t o1  = vld1q_f32(O_fp32 + d_start + 4);
                float32x4_t o2  = vld1q_f32(O_fp32 + d_start + 8);
                float32x4_t o3  = vld1q_f32(O_fp32 + d_start + 12);
                float32x4_t o4  = vld1q_f32(O_fp32 + d_start + 16);
                float32x4_t o5  = vld1q_f32(O_fp32 + d_start + 20);
                float32x4_t o6  = vld1q_f32(O_fp32 + d_start + 24);
                float32x4_t o7  = vld1q_f32(O_fp32 + d_start + 28);
                float32x4_t o8  = vld1q_f32(O_fp32 + d_start + 32);
                float32x4_t o9  = vld1q_f32(O_fp32 + d_start + 36);
                float32x4_t o10 = vld1q_f32(O_fp32 + d_start + 40);
                float32x4_t o11 = vld1q_f32(O_fp32 + d_start + 44);
                float32x4_t o12 = vld1q_f32(O_fp32 + d_start + 48);
                float32x4_t o13 = vld1q_f32(O_fp32 + d_start + 52);
                float32x4_t o14 = vld1q_f32(O_fp32 + d_start + 56);
                float32x4_t o15 = vld1q_f32(O_fp32 + d_start + 60);

                // Iterate over all KV positions - O stays in registers!
                for (int j = 0; j < bc; j++)
                {
#if __ARM_FEATURE_FP16_FML
                    float16x4_t vp = vdup_n_f16((__fp16)scores[j]);
                    const __fp16* vj = V + (jb + j) * d_v + d_start;

                    float16x8_t v01 = vld1q_f16(vj + 0);
                    float16x8_t v23 = vld1q_f16(vj + 8);
                    float16x8_t v45 = vld1q_f16(vj + 16);
                    float16x8_t v67 = vld1q_f16(vj + 24);
                    float16x8_t v89 = vld1q_f16(vj + 32);
                    float16x8_t vab = vld1q_f16(vj + 40);
                    float16x8_t vcd = vld1q_f16(vj + 48);
                    float16x8_t vef = vld1q_f16(vj + 56);

                    o0  = vfmlalq_lane_low_f16 (o0,  v01, vp, 0);
                    o1  = vfmlalq_lane_high_f16(o1,  v01, vp, 0);
                    o2  = vfmlalq_lane_low_f16 (o2,  v23, vp, 0);
                    o3  = vfmlalq_lane_high_f16(o3,  v23, vp, 0);
                    o4  = vfmlalq_lane_low_f16 (o4,  v45, vp, 0);
                    o5  = vfmlalq_lane_high_f16(o5,  v45, vp, 0);
                    o6  = vfmlalq_lane_low_f16 (o6,  v67, vp, 0);
                    o7  = vfmlalq_lane_high_f16(o7,  v67, vp, 0);
                    o8  = vfmlalq_lane_low_f16 (o8,  v89, vp, 0);
                    o9  = vfmlalq_lane_high_f16(o9,  v89, vp, 0);
                    o10 = vfmlalq_lane_low_f16 (o10, vab, vp, 0);
                    o11 = vfmlalq_lane_high_f16(o11, vab, vp, 0);
                    o12 = vfmlalq_lane_low_f16 (o12, vcd, vp, 0);
                    o13 = vfmlalq_lane_high_f16(o13, vcd, vp, 0);
                    o14 = vfmlalq_lane_low_f16 (o14, vef, vp, 0);
                    o15 = vfmlalq_lane_high_f16(o15, vef, vp, 0);
#else
                    float32x4_t vp = vdupq_n_f32(scores[j]);
                    const __fp16* vj = V + (jb + j) * d_v + d_start;

                    float16x8_t v01 = vld1q_f16(vj + 0);
                    float16x8_t v23 = vld1q_f16(vj + 8);
                    float16x8_t v45 = vld1q_f16(vj + 16);
                    float16x8_t v67 = vld1q_f16(vj + 24);
                    float16x8_t v89 = vld1q_f16(vj + 32);
                    float16x8_t vab = vld1q_f16(vj + 40);
                    float16x8_t vcd = vld1q_f16(vj + 48);
                    float16x8_t vef = vld1q_f16(vj + 56);

                    o0  = vfmaq_f32(o0,  vp, vcvt_f32_f16(vget_low_f16(v01)));
                    o1  = vfmaq_f32(o1,  vp, vcvt_f32_f16(vget_high_f16(v01)));
                    o2  = vfmaq_f32(o2,  vp, vcvt_f32_f16(vget_low_f16(v23)));
                    o3  = vfmaq_f32(o3,  vp, vcvt_f32_f16(vget_high_f16(v23)));
                    o4  = vfmaq_f32(o4,  vp, vcvt_f32_f16(vget_low_f16(v45)));
                    o5  = vfmaq_f32(o5,  vp, vcvt_f32_f16(vget_high_f16(v45)));
                    o6  = vfmaq_f32(o6,  vp, vcvt_f32_f16(vget_low_f16(v67)));
                    o7  = vfmaq_f32(o7,  vp, vcvt_f32_f16(vget_high_f16(v67)));
                    o8  = vfmaq_f32(o8,  vp, vcvt_f32_f16(vget_low_f16(v89)));
                    o9  = vfmaq_f32(o9,  vp, vcvt_f32_f16(vget_high_f16(v89)));
                    o10 = vfmaq_f32(o10, vp, vcvt_f32_f16(vget_low_f16(vab)));
                    o11 = vfmaq_f32(o11, vp, vcvt_f32_f16(vget_high_f16(vab)));
                    o12 = vfmaq_f32(o12, vp, vcvt_f32_f16(vget_low_f16(vcd)));
                    o13 = vfmaq_f32(o13, vp, vcvt_f32_f16(vget_high_f16(vcd)));
                    o14 = vfmaq_f32(o14, vp, vcvt_f32_f16(vget_low_f16(vef)));
                    o15 = vfmaq_f32(o15, vp, vcvt_f32_f16(vget_high_f16(vef)));
#endif
                }

                // Store O tile back
                vst1q_f32(O_fp32 + d_start + 0, o0);
                vst1q_f32(O_fp32 + d_start + 4, o1);
                vst1q_f32(O_fp32 + d_start + 8, o2);
                vst1q_f32(O_fp32 + d_start + 12, o3);
                vst1q_f32(O_fp32 + d_start + 16, o4);
                vst1q_f32(O_fp32 + d_start + 20, o5);
                vst1q_f32(O_fp32 + d_start + 24, o6);
                vst1q_f32(O_fp32 + d_start + 28, o7);
                vst1q_f32(O_fp32 + d_start + 32, o8);
                vst1q_f32(O_fp32 + d_start + 36, o9);
                vst1q_f32(O_fp32 + d_start + 40, o10);
                vst1q_f32(O_fp32 + d_start + 44, o11);
                vst1q_f32(O_fp32 + d_start + 48, o12);
                vst1q_f32(O_fp32 + d_start + 52, o13);
                vst1q_f32(O_fp32 + d_start + 56, o14);
                vst1q_f32(O_fp32 + d_start + 60, o15);
            }

            for (; d_start + 7 < d_v; d_start += 8)
            {
                float32x4_t o0 = vld1q_f32(O_fp32 + d_start);
                float32x4_t o1 = vld1q_f32(O_fp32 + d_start + 4);

                for (int j = 0; j < bc; j++)
                {
                    float32x4_t vp = vdupq_n_f32(scores[j]);
                    const __fp16* vj = V + (jb + j) * d_v + d_start;
                    float16x8_t v = vld1q_f16(vj);
                    o0 = vfmaq_f32(o0, vp, vcvt_f32_f16(vget_low_f16(v)));
                    o1 = vfmaq_f32(o1, vp, vcvt_f32_f16(vget_high_f16(v)));
                }

                vst1q_f32(O_fp32 + d_start, o0);
                vst1q_f32(O_fp32 + d_start + 4, o1);
            }

            for (; d_start + 3 < d_v; d_start += 4)
            {
                float32x4_t o0 = vld1q_f32(O_fp32 + d_start);

                for (int j = 0; j < bc; j++)
                {
                    float32x4_t vp = vdupq_n_f32(scores[j]);
                    const __fp16* vj = V + (jb + j) * d_v + d_start;
                    o0 = vfmaq_f32(o0, vp, vcvt_f32_f16(vld1_f16(vj)));
                }

                vst1q_f32(O_fp32 + d_start, o0);
            }

            for (; d_start < d_v; d_start++)
            {
                float o0 = O_fp32[d_start];

                for (int j = 0; j < bc; j++)
                    o0 += scores[j] * (float)V[(jb + j) * d_v + d_start];

                O_fp32[d_start] = o0;
            }
        }

        m = m_new;
    }

    // 5. Normalize and convert to fp16
    if (l > 0.f)
    {
        float inv_l = 1.f / l;
        float32x4_t vinv = vdupq_n_f32(inv_l);
        int d = 0;
        for (; d + 7 < d_v; d += 8)
        {
            float32x4_t o0 = vmulq_f32(vld1q_f32(O_fp32 + d), vinv);
            float32x4_t o1 = vmulq_f32(vld1q_f32(O_fp32 + d + 4), vinv);
            vst1_f16(O + d, vcvt_f16_f32(o0));
            vst1_f16(O + d + 4, vcvt_f16_f32(o1));
        }
        for (; d + 3 < d_v; d += 4)
        {
            float32x4_t o0 = vmulq_f32(vld1q_f32(O_fp32 + d), vinv);
            vst1_f16(O + d, vcvt_f16_f32(o0));
        }
        for (; d < d_v; d++)
            O[d] = (__fp16)(O_fp32[d] * inv_l);
    }

    if (d_v > 256)
        free(O_fp32);
}

// =============================================================================
// FlashAttention-2 Prefill: Br=8, Bc=12, FP16FML 8x12 micro-kernel
// =============================================================================
// - Outer-Q / inner-KV loop ordering (FA-2 spec): O_local stays hot in stack/L1
// - K_packed prepared ONCE upfront for the entire N (workspace_allocator pool)
// - 8x12 micro-kernel uses vfmlalq_lane_low/high_f16 (FP16FML) - same pattern
//   as ncnn's gemm_fp16s.h to match its throughput
// - Tail handling: M%8 zero-padded Q, N%12 -inf-masked S
//
// Per-thread workspace layout (callers in sdpa_arm.cpp must size accordingly):
//   K_packed:  ceil(N/12) * 12 * d_k * sizeof(fp16)   <= ~1.5MB at N=4096
//   Q_packed:  d_k * 8 * sizeof(fp16)                 = 3KB
// =============================================================================

#if __ARM_FEATURE_FP16_FML

struct FlashPrefillProfile
{
    double qpack;
    double qk;
    double softmax;
    double pv;
    double norm;
};

static FlashPrefillProfile g_flash_prefill_profile;

static inline bool flash_prefill_profile_enabled()
{
    const char* s = getenv("NCNN_SDPA_FLASH_PROFILE");
    return s && s[0] && s[0] != '0';
}

static inline void flash_prefill_profile_reset()
{
    g_flash_prefill_profile.qpack = 0.0;
    g_flash_prefill_profile.qk = 0.0;
    g_flash_prefill_profile.softmax = 0.0;
    g_flash_prefill_profile.pv = 0.0;
    g_flash_prefill_profile.norm = 0.0;
}

static inline void flash_prefill_profile_add(const FlashPrefillProfile& p)
{
    #pragma omp critical
    {
        g_flash_prefill_profile.qpack += p.qpack;
        g_flash_prefill_profile.qk += p.qk;
        g_flash_prefill_profile.softmax += p.softmax;
        g_flash_prefill_profile.pv += p.pv;
        g_flash_prefill_profile.norm += p.norm;
    }
}

static inline void flash_prefill_profile_print()
{
    const double total = g_flash_prefill_profile.qpack + g_flash_prefill_profile.qk + g_flash_prefill_profile.softmax + g_flash_prefill_profile.pv + g_flash_prefill_profile.norm;
    fprintf(stderr, "flash_prefill_profile qpack=%.3f qk=%.3f softmax=%.3f pv=%.3f norm=%.3f total=%.3f ms\n",
            g_flash_prefill_profile.qpack, g_flash_prefill_profile.qk,
            g_flash_prefill_profile.softmax, g_flash_prefill_profile.pv,
            g_flash_prefill_profile.norm, total);
}

static inline bool allow_noncausal_flash_prefill()
{
    const char* s = getenv("NCNN_SDPA_FLASH_PREFILL_NONCAUSAL");
    return s && s[0] && s[0] != '0';
}

static inline bool is_causal_mask_fp16(const __fp16* mask, int M, int N)
{
    const int past = N - M;
    if (!mask || past < 0)
        return false;

    for (int i = 0; i < M; i++)
    {
        const __fp16* row = mask + i * N;
        const int valid = past + i + 1;
        for (int j = 0; j < N; j++)
        {
            const float v = (float)row[j];
            if (j < valid)
            {
                if (fabsf(v) > 1e-3f)
                    return false;
            }
            else
            {
                if (v > -10000.f)
                    return false;
            }
        }
    }

    return true;
}

// Pack 8xd_k of Q into [d_k][8] layout (k-major, 8 lanes per kk)
// If br < 8, pad with zeros so the micro-kernel can still run full Br=8.
static inline void pack_Q_8_fp16(const __fp16* Q, int br, int d_k, __fp16* Q_pack)
{
    // Q is row-major [br][d_k]; output Q_pack is [d_k][8]
    // We zero-pad up to 8 rows.
    if (br == 8)
    {
        // Fast path: 8x8 transposed blocks
        int k = 0;
        for (; k + 7 < d_k; k += 8)
        {
            uint16x8_t r0 = vld1q_u16((const uint16_t*)(Q + 0 * d_k + k));
            uint16x8_t r1 = vld1q_u16((const uint16_t*)(Q + 1 * d_k + k));
            uint16x8_t r2 = vld1q_u16((const uint16_t*)(Q + 2 * d_k + k));
            uint16x8_t r3 = vld1q_u16((const uint16_t*)(Q + 3 * d_k + k));
            uint16x8_t r4 = vld1q_u16((const uint16_t*)(Q + 4 * d_k + k));
            uint16x8_t r5 = vld1q_u16((const uint16_t*)(Q + 5 * d_k + k));
            uint16x8_t r6 = vld1q_u16((const uint16_t*)(Q + 6 * d_k + k));
            uint16x8_t r7 = vld1q_u16((const uint16_t*)(Q + 7 * d_k + k));
            transpose8x8_u16(r0, r1, r2, r3, r4, r5, r6, r7);
            uint16_t* dst = (uint16_t*)(Q_pack + k * 8);
            vst1q_u16(dst + 0 * 8, r0);
            vst1q_u16(dst + 1 * 8, r1);
            vst1q_u16(dst + 2 * 8, r2);
            vst1q_u16(dst + 3 * 8, r3);
            vst1q_u16(dst + 4 * 8, r4);
            vst1q_u16(dst + 5 * 8, r5);
            vst1q_u16(dst + 6 * 8, r6);
            vst1q_u16(dst + 7 * 8, r7);
        }
        for (; k < d_k; k++)
        {
            for (int i = 0; i < 8; i++)
                Q_pack[k * 8 + i] = Q[i * d_k + k];
        }
    }
    else
    {
        // Tail path: scalar with zero-pad
        for (int k = 0; k < d_k; k++)
        {
            for (int i = 0; i < br; i++)
                Q_pack[k * 8 + i] = Q[i * d_k + k];
            for (int i = br; i < 8; i++)
                Q_pack[k * 8 + i] = (__fp16)0.f;
        }
    }
}

// Pack 12xd_k of K into [d_k][12] layout (k-major, 12 lanes per kk)
// If bc < 12, pad with zeros (caller must mask resulting S to -inf for j>=bc)
static inline void pack_K_12_fp16(const __fp16* K, int bc, int d_k, __fp16* K_pack)
{
    // K is row-major [bc][d_k]; output K_pack is [d_k][12]
    if (bc == 12)
    {
        // Fast path: process 4 k-elements at a time using 4x4 transpose for each group of 4 rows
        int k = 0;
        for (; k + 3 < d_k; k += 4)
        {
            uint16x4_t r0 = vld1_u16((const uint16_t*)(K + 0 * d_k + k));
            uint16x4_t r1 = vld1_u16((const uint16_t*)(K + 1 * d_k + k));
            uint16x4_t r2 = vld1_u16((const uint16_t*)(K + 2 * d_k + k));
            uint16x4_t r3 = vld1_u16((const uint16_t*)(K + 3 * d_k + k));
            uint16x4_t r4 = vld1_u16((const uint16_t*)(K + 4 * d_k + k));
            uint16x4_t r5 = vld1_u16((const uint16_t*)(K + 5 * d_k + k));
            uint16x4_t r6 = vld1_u16((const uint16_t*)(K + 6 * d_k + k));
            uint16x4_t r7 = vld1_u16((const uint16_t*)(K + 7 * d_k + k));
            uint16x4_t r8 = vld1_u16((const uint16_t*)(K + 8 * d_k + k));
            uint16x4_t r9 = vld1_u16((const uint16_t*)(K + 9 * d_k + k));
            uint16x4_t ra = vld1_u16((const uint16_t*)(K + 10 * d_k + k));
            uint16x4_t rb = vld1_u16((const uint16_t*)(K + 11 * d_k + k));
            transpose4x4_u16(r0, r1, r2, r3);
            transpose4x4_u16(r4, r5, r6, r7);
            transpose4x4_u16(r8, r9, ra, rb);
            // After transpose: r0 = K[0..3][k+0], r1 = K[0..3][k+1], etc.
            // We want K_pack[(k+kk)*12 + j] for kk in 0..3, j in 0..11
            uint16_t* dst = (uint16_t*)(K_pack + k * 12);
            // K_pack[k+0][0..11]
            vst1_u16(dst + 0 * 12 + 0, r0);
            vst1_u16(dst + 0 * 12 + 4, r4);
            vst1_u16(dst + 0 * 12 + 8, r8);
            // K_pack[k+1][0..11]
            vst1_u16(dst + 1 * 12 + 0, r1);
            vst1_u16(dst + 1 * 12 + 4, r5);
            vst1_u16(dst + 1 * 12 + 8, r9);
            // K_pack[k+2][0..11]
            vst1_u16(dst + 2 * 12 + 0, r2);
            vst1_u16(dst + 2 * 12 + 4, r6);
            vst1_u16(dst + 2 * 12 + 8, ra);
            // K_pack[k+3][0..11]
            vst1_u16(dst + 3 * 12 + 0, r3);
            vst1_u16(dst + 3 * 12 + 4, r7);
            vst1_u16(dst + 3 * 12 + 8, rb);
        }
        for (; k < d_k; k++)
        {
            for (int j = 0; j < 12; j++)
                K_pack[k * 12 + j] = K[j * d_k + k];
        }
    }
    else
    {
        // Tail path: scalar with zero-pad
        for (int k = 0; k < d_k; k++)
        {
            for (int j = 0; j < bc; j++)
                K_pack[k * 12 + j] = K[j * d_k + k];
            for (int j = bc; j < 12; j++)
                K_pack[k * 12 + j] = (__fp16)0.f;
        }
    }
}

// 8x12 FP16FML micro-kernel: S[8][12] = Q_pack[d_k][8] x K_pack[d_k][12]
// S output layout: S_out[i*12 + j] (row-major, Br=8, Bc=12)
//
// D_STATIC > 0 hardcodes the kk loop bound at compile time so the compiler can
// fully unroll (e.g. D_STATIC=192 for MLA configs). D_STATIC == 0 falls back
// to the runtime d_k argument.
//
// The K_pack pointer for the NEXT block (or NULL if there is no next) is
// prefetched while we compute. K rows are 12*d_k*2 = 4.5KB each at d_k=192;
// fetching them into L1/L2 ahead of time hides DRAM latency on long N.
template <int D_STATIC>
static inline void qk_micro_8x12_fp16fml(
    const __fp16* __restrict Q_pack,       // [d_k][8] packed
    const __fp16* __restrict K_pack,       // [d_k][12] packed
    const __fp16* __restrict K_pack_next,  // [d_k][12] packed, NULL if no next block
    int d_k_runtime,              // ignored when D_STATIC > 0
    float scale,
    float* __restrict S_out)      // [8][12] row-major output
{
    (void)K_pack_next;

    const int d_k = (D_STATIC > 0) ? D_STATIC : d_k_runtime;
    // 24 fp32x4 accumulators: s_jJ_lo holds S[0..3][J], s_jJ_hi holds S[4..7][J]
    float32x4_t s0_lo = vdupq_n_f32(0.f), s0_hi = vdupq_n_f32(0.f);
    float32x4_t s1_lo = vdupq_n_f32(0.f), s1_hi = vdupq_n_f32(0.f);
    float32x4_t s2_lo = vdupq_n_f32(0.f), s2_hi = vdupq_n_f32(0.f);
    float32x4_t s3_lo = vdupq_n_f32(0.f), s3_hi = vdupq_n_f32(0.f);
    float32x4_t s4_lo = vdupq_n_f32(0.f), s4_hi = vdupq_n_f32(0.f);
    float32x4_t s5_lo = vdupq_n_f32(0.f), s5_hi = vdupq_n_f32(0.f);
    float32x4_t s6_lo = vdupq_n_f32(0.f), s6_hi = vdupq_n_f32(0.f);
    float32x4_t s7_lo = vdupq_n_f32(0.f), s7_hi = vdupq_n_f32(0.f);
    float32x4_t s8_lo = vdupq_n_f32(0.f), s8_hi = vdupq_n_f32(0.f);
    float32x4_t s9_lo = vdupq_n_f32(0.f), s9_hi = vdupq_n_f32(0.f);
    float32x4_t sa_lo = vdupq_n_f32(0.f), sa_hi = vdupq_n_f32(0.f);
    float32x4_t sb_lo = vdupq_n_f32(0.f), sb_hi = vdupq_n_f32(0.f);

    for (int kk = 0; kk < d_k; kk++)
    {
        float16x8_t pA = vld1q_f16(Q_pack + kk * 8);
        float16x4_t pB0 = vld1_f16(K_pack + kk * 12 + 0);
        float16x4_t pB1 = vld1_f16(K_pack + kk * 12 + 4);
        float16x4_t pB2 = vld1_f16(K_pack + kk * 12 + 8);

        // 24 FP16FML instructions: each accumulates 4 fp32 FMAs from fp16 inputs
        // s_jJ accumulates Q_pack[kk][:] * K_pack[kk][J] for that column J
        s0_lo = vfmlalq_lane_low_f16 (s0_lo, pA, pB0, 0);
        s0_hi = vfmlalq_lane_high_f16(s0_hi, pA, pB0, 0);
        s1_lo = vfmlalq_lane_low_f16 (s1_lo, pA, pB0, 1);
        s1_hi = vfmlalq_lane_high_f16(s1_hi, pA, pB0, 1);
        s2_lo = vfmlalq_lane_low_f16 (s2_lo, pA, pB0, 2);
        s2_hi = vfmlalq_lane_high_f16(s2_hi, pA, pB0, 2);
        s3_lo = vfmlalq_lane_low_f16 (s3_lo, pA, pB0, 3);
        s3_hi = vfmlalq_lane_high_f16(s3_hi, pA, pB0, 3);
        s4_lo = vfmlalq_lane_low_f16 (s4_lo, pA, pB1, 0);
        s4_hi = vfmlalq_lane_high_f16(s4_hi, pA, pB1, 0);
        s5_lo = vfmlalq_lane_low_f16 (s5_lo, pA, pB1, 1);
        s5_hi = vfmlalq_lane_high_f16(s5_hi, pA, pB1, 1);
        s6_lo = vfmlalq_lane_low_f16 (s6_lo, pA, pB1, 2);
        s6_hi = vfmlalq_lane_high_f16(s6_hi, pA, pB1, 2);
        s7_lo = vfmlalq_lane_low_f16 (s7_lo, pA, pB1, 3);
        s7_hi = vfmlalq_lane_high_f16(s7_hi, pA, pB1, 3);
        s8_lo = vfmlalq_lane_low_f16 (s8_lo, pA, pB2, 0);
        s8_hi = vfmlalq_lane_high_f16(s8_hi, pA, pB2, 0);
        s9_lo = vfmlalq_lane_low_f16 (s9_lo, pA, pB2, 1);
        s9_hi = vfmlalq_lane_high_f16(s9_hi, pA, pB2, 1);
        sa_lo = vfmlalq_lane_low_f16 (sa_lo, pA, pB2, 2);
        sa_hi = vfmlalq_lane_high_f16(sa_hi, pA, pB2, 2);
        sb_lo = vfmlalq_lane_low_f16 (sb_lo, pA, pB2, 3);
        sb_hi = vfmlalq_lane_high_f16(sb_hi, pA, pB2, 3);
    }

    // Apply scale and store transposed to row-major S[Br=8][Bc=12]
    float32x4_t vs = vdupq_n_f32(scale);
    s0_lo = vmulq_f32(s0_lo, vs); s0_hi = vmulq_f32(s0_hi, vs);
    s1_lo = vmulq_f32(s1_lo, vs); s1_hi = vmulq_f32(s1_hi, vs);
    s2_lo = vmulq_f32(s2_lo, vs); s2_hi = vmulq_f32(s2_hi, vs);
    s3_lo = vmulq_f32(s3_lo, vs); s3_hi = vmulq_f32(s3_hi, vs);
    s4_lo = vmulq_f32(s4_lo, vs); s4_hi = vmulq_f32(s4_hi, vs);
    s5_lo = vmulq_f32(s5_lo, vs); s5_hi = vmulq_f32(s5_hi, vs);
    s6_lo = vmulq_f32(s6_lo, vs); s6_hi = vmulq_f32(s6_hi, vs);
    s7_lo = vmulq_f32(s7_lo, vs); s7_hi = vmulq_f32(s7_hi, vs);
    s8_lo = vmulq_f32(s8_lo, vs); s8_hi = vmulq_f32(s8_hi, vs);
    s9_lo = vmulq_f32(s9_lo, vs); s9_hi = vmulq_f32(s9_hi, vs);
    sa_lo = vmulq_f32(sa_lo, vs); sa_hi = vmulq_f32(sa_hi, vs);
    sb_lo = vmulq_f32(sb_lo, vs); sb_hi = vmulq_f32(sb_hi, vs);

    transpose8x12_ps(s0_lo, s0_hi, s1_lo, s1_hi, s2_lo, s2_hi, s3_lo, s3_hi,
                     s4_lo, s4_hi, s5_lo, s5_hi, s6_lo, s6_hi, s7_lo, s7_hi,
                     s8_lo, s8_hi, s9_lo, s9_hi, sa_lo, sa_hi, sb_lo, sb_hi);

    vst1q_f32(S_out, s0_lo);
    vst1q_f32(S_out + 4, s0_hi);
    vst1q_f32(S_out + 8, s1_lo);
    vst1q_f32(S_out + 12, s1_hi);
    vst1q_f32(S_out + 16, s2_lo);
    vst1q_f32(S_out + 20, s2_hi);
    vst1q_f32(S_out + 24, s3_lo);
    vst1q_f32(S_out + 28, s3_hi);
    vst1q_f32(S_out + 32, s4_lo);
    vst1q_f32(S_out + 36, s4_hi);
    vst1q_f32(S_out + 40, s5_lo);
    vst1q_f32(S_out + 44, s5_hi);
    vst1q_f32(S_out + 48, s6_lo);
    vst1q_f32(S_out + 52, s6_hi);
    vst1q_f32(S_out + 56, s7_lo);
    vst1q_f32(S_out + 60, s7_hi);
    vst1q_f32(S_out + 64, s8_lo);
    vst1q_f32(S_out + 68, s8_hi);
    vst1q_f32(S_out + 72, s9_lo);
    vst1q_f32(S_out + 76, s9_hi);
    vst1q_f32(S_out + 80, sa_lo);
    vst1q_f32(S_out + 84, sa_hi);
    vst1q_f32(S_out + 88, sb_lo);
    vst1q_f32(S_out + 92, sb_hi);
}

static inline void qk_softmax_8x12_d192_fp16fml(
    const __fp16* __restrict Q_pack,
    const __fp16* __restrict K_pack,
    float scale,
    float* __restrict m_local,
    float* __restrict l_local,
    float* __restrict alpha_out,
    float* __restrict l_old_out,
    __fp16* __restrict P_T)
{
    float32x4_t s0_lo = vdupq_n_f32(0.f), s0_hi = vdupq_n_f32(0.f);
    float32x4_t s1_lo = vdupq_n_f32(0.f), s1_hi = vdupq_n_f32(0.f);
    float32x4_t s2_lo = vdupq_n_f32(0.f), s2_hi = vdupq_n_f32(0.f);
    float32x4_t s3_lo = vdupq_n_f32(0.f), s3_hi = vdupq_n_f32(0.f);
    float32x4_t s4_lo = vdupq_n_f32(0.f), s4_hi = vdupq_n_f32(0.f);
    float32x4_t s5_lo = vdupq_n_f32(0.f), s5_hi = vdupq_n_f32(0.f);
    float32x4_t s6_lo = vdupq_n_f32(0.f), s6_hi = vdupq_n_f32(0.f);
    float32x4_t s7_lo = vdupq_n_f32(0.f), s7_hi = vdupq_n_f32(0.f);
    float32x4_t s8_lo = vdupq_n_f32(0.f), s8_hi = vdupq_n_f32(0.f);
    float32x4_t s9_lo = vdupq_n_f32(0.f), s9_hi = vdupq_n_f32(0.f);
    float32x4_t sa_lo = vdupq_n_f32(0.f), sa_hi = vdupq_n_f32(0.f);
    float32x4_t sb_lo = vdupq_n_f32(0.f), sb_hi = vdupq_n_f32(0.f);

    for (int kk = 0; kk < 192; kk++)
    {
        float16x8_t pA = vld1q_f16(Q_pack + kk * 8);
        float16x4_t pB0 = vld1_f16(K_pack + kk * 12 + 0);
        float16x4_t pB1 = vld1_f16(K_pack + kk * 12 + 4);
        float16x4_t pB2 = vld1_f16(K_pack + kk * 12 + 8);

        s0_lo = vfmlalq_lane_low_f16 (s0_lo, pA, pB0, 0);
        s0_hi = vfmlalq_lane_high_f16(s0_hi, pA, pB0, 0);
        s1_lo = vfmlalq_lane_low_f16 (s1_lo, pA, pB0, 1);
        s1_hi = vfmlalq_lane_high_f16(s1_hi, pA, pB0, 1);
        s2_lo = vfmlalq_lane_low_f16 (s2_lo, pA, pB0, 2);
        s2_hi = vfmlalq_lane_high_f16(s2_hi, pA, pB0, 2);
        s3_lo = vfmlalq_lane_low_f16 (s3_lo, pA, pB0, 3);
        s3_hi = vfmlalq_lane_high_f16(s3_hi, pA, pB0, 3);
        s4_lo = vfmlalq_lane_low_f16 (s4_lo, pA, pB1, 0);
        s4_hi = vfmlalq_lane_high_f16(s4_hi, pA, pB1, 0);
        s5_lo = vfmlalq_lane_low_f16 (s5_lo, pA, pB1, 1);
        s5_hi = vfmlalq_lane_high_f16(s5_hi, pA, pB1, 1);
        s6_lo = vfmlalq_lane_low_f16 (s6_lo, pA, pB1, 2);
        s6_hi = vfmlalq_lane_high_f16(s6_hi, pA, pB1, 2);
        s7_lo = vfmlalq_lane_low_f16 (s7_lo, pA, pB1, 3);
        s7_hi = vfmlalq_lane_high_f16(s7_hi, pA, pB1, 3);
        s8_lo = vfmlalq_lane_low_f16 (s8_lo, pA, pB2, 0);
        s8_hi = vfmlalq_lane_high_f16(s8_hi, pA, pB2, 0);
        s9_lo = vfmlalq_lane_low_f16 (s9_lo, pA, pB2, 1);
        s9_hi = vfmlalq_lane_high_f16(s9_hi, pA, pB2, 1);
        sa_lo = vfmlalq_lane_low_f16 (sa_lo, pA, pB2, 2);
        sa_hi = vfmlalq_lane_high_f16(sa_hi, pA, pB2, 2);
        sb_lo = vfmlalq_lane_low_f16 (sb_lo, pA, pB2, 3);
        sb_hi = vfmlalq_lane_high_f16(sb_hi, pA, pB2, 3);
    }

    const float32x4_t vs = vdupq_n_f32(scale);
    s0_lo = vmulq_f32(s0_lo, vs); s0_hi = vmulq_f32(s0_hi, vs);
    s1_lo = vmulq_f32(s1_lo, vs); s1_hi = vmulq_f32(s1_hi, vs);
    s2_lo = vmulq_f32(s2_lo, vs); s2_hi = vmulq_f32(s2_hi, vs);
    s3_lo = vmulq_f32(s3_lo, vs); s3_hi = vmulq_f32(s3_hi, vs);
    s4_lo = vmulq_f32(s4_lo, vs); s4_hi = vmulq_f32(s4_hi, vs);
    s5_lo = vmulq_f32(s5_lo, vs); s5_hi = vmulq_f32(s5_hi, vs);
    s6_lo = vmulq_f32(s6_lo, vs); s6_hi = vmulq_f32(s6_hi, vs);
    s7_lo = vmulq_f32(s7_lo, vs); s7_hi = vmulq_f32(s7_hi, vs);
    s8_lo = vmulq_f32(s8_lo, vs); s8_hi = vmulq_f32(s8_hi, vs);
    s9_lo = vmulq_f32(s9_lo, vs); s9_hi = vmulq_f32(s9_hi, vs);
    sa_lo = vmulq_f32(sa_lo, vs); sa_hi = vmulq_f32(sa_hi, vs);
    sb_lo = vmulq_f32(sb_lo, vs); sb_hi = vmulq_f32(sb_hi, vs);

    float32x4_t m_block_lo = vmaxq_f32(vmaxq_f32(vmaxq_f32(s0_lo, s1_lo), vmaxq_f32(s2_lo, s3_lo)),
                                       vmaxq_f32(vmaxq_f32(s4_lo, s5_lo), vmaxq_f32(s6_lo, s7_lo)));
    m_block_lo = vmaxq_f32(m_block_lo, vmaxq_f32(vmaxq_f32(s8_lo, s9_lo), vmaxq_f32(sa_lo, sb_lo)));
    float32x4_t m_block_hi = vmaxq_f32(vmaxq_f32(vmaxq_f32(s0_hi, s1_hi), vmaxq_f32(s2_hi, s3_hi)),
                                       vmaxq_f32(vmaxq_f32(s4_hi, s5_hi), vmaxq_f32(s6_hi, s7_hi)));
    m_block_hi = vmaxq_f32(m_block_hi, vmaxq_f32(vmaxq_f32(s8_hi, s9_hi), vmaxq_f32(sa_hi, sb_hi)));

    float32x4_t m_old_lo = vld1q_f32(m_local);
    float32x4_t m_old_hi = vld1q_f32(m_local + 4);
    float32x4_t l_old_lo = vld1q_f32(l_local);
    float32x4_t l_old_hi = vld1q_f32(l_local + 4);
    float32x4_t m_new_lo = vmaxq_f32(m_old_lo, m_block_lo);
    float32x4_t m_new_hi = vmaxq_f32(m_old_hi, m_block_hi);
    float32x4_t alpha_lo = fast_exp_f32x4(vsubq_f32(m_old_lo, m_new_lo));
    float32x4_t alpha_hi = fast_exp_f32x4(vsubq_f32(m_old_hi, m_new_hi));

    float32x4_t l_block_lo = vdupq_n_f32(0.f);
    float32x4_t l_block_hi = vdupq_n_f32(0.f);

    #define STORE_P_COL(col, slo, shi) do { \
        float32x4_t p_lo = fast_exp_f32x4(vsubq_f32((slo), m_new_lo)); \
        float32x4_t p_hi = fast_exp_f32x4(vsubq_f32((shi), m_new_hi)); \
        l_block_lo = vaddq_f32(l_block_lo, p_lo); \
        l_block_hi = vaddq_f32(l_block_hi, p_hi); \
        vst1_f16(P_T + (col) * 8, vcvt_f16_f32(p_lo)); \
        vst1_f16(P_T + (col) * 8 + 4, vcvt_f16_f32(p_hi)); \
    } while (0)

    STORE_P_COL(0, s0_lo, s0_hi);
    STORE_P_COL(1, s1_lo, s1_hi);
    STORE_P_COL(2, s2_lo, s2_hi);
    STORE_P_COL(3, s3_lo, s3_hi);
    STORE_P_COL(4, s4_lo, s4_hi);
    STORE_P_COL(5, s5_lo, s5_hi);
    STORE_P_COL(6, s6_lo, s6_hi);
    STORE_P_COL(7, s7_lo, s7_hi);
    STORE_P_COL(8, s8_lo, s8_hi);
    STORE_P_COL(9, s9_lo, s9_hi);
    STORE_P_COL(10, sa_lo, sa_hi);
    STORE_P_COL(11, sb_lo, sb_hi);

    #undef STORE_P_COL

    float32x4_t l_new_lo = vaddq_f32(vmulq_f32(alpha_lo, l_old_lo), l_block_lo);
    float32x4_t l_new_hi = vaddq_f32(vmulq_f32(alpha_hi, l_old_hi), l_block_hi);
    vst1q_f32(m_local, m_new_lo);
    vst1q_f32(m_local + 4, m_new_hi);
    vst1q_f32(l_local, l_new_lo);
    vst1q_f32(l_local + 4, l_new_hi);
    vst1q_f32(alpha_out, alpha_lo);
    vst1q_f32(alpha_out + 4, alpha_hi);
    vst1q_f32(l_old_out, l_old_lo);
    vst1q_f32(l_old_out + 4, l_old_hi);
}

// Prefill kernel: Br=8, Bc=12, FA-2 outer-Q / inner-KV, FP16FML micro-kernel.
// Workspace pointers (must be sized by caller):
//   K_pack_buf:  ceil(N/12) * 12 * d_k fp16 elements
//   Q_pack_buf:  d_k * 8 fp16 elements (stack-allocated by caller via workspace)
static void flash_attn_fp16_prefill(
    const __fp16* __restrict Q,         // [M][d_k]
    const __fp16* __restrict K,         // [N][d_k]
    const __fp16* __restrict V,         // [N][d_v]
    __fp16* __restrict O,               // [M][d_v]
    int M, int N, int d_k, int d_v, float scale, const __fp16* mask,
    __fp16* __restrict K_pack_buf,      // [(N/12+1)*12*d_k]
    __fp16* __restrict Q_pack_buf,      // [d_k * 8]
    bool K_prepacked,        // true when K_pack_buf already contains packed K
    bool causal_mask)
{
    const int Br = 8;
    const int Bc = 12;
    const bool profile = flash_prefill_profile_enabled();
    FlashPrefillProfile prof;
    prof.qpack = 0.0;
    prof.qk = 0.0;
    prof.softmax = 0.0;
    prof.pv = 0.0;
    prof.norm = 0.0;

    const bool own_K_pack = (K_pack_buf == NULL);
    const bool own_Q_pack = (Q_pack_buf == NULL);

    const int Nblocks = (N + Bc - 1) / Bc;
    const int K_pack_size = Nblocks * Bc * d_k;
    if (!K_pack_buf) K_pack_buf = (__fp16*)malloc(K_pack_size * sizeof(__fp16));
    if (!Q_pack_buf) Q_pack_buf = (__fp16*)malloc(d_k * Br * sizeof(__fp16));

    // 1. Pre-pack ALL of K^T in Bc-blocks, once for entire N.
    if (!K_prepacked)
    {
        for (int jb_idx = 0; jb_idx < Nblocks; jb_idx++)
        {
            int jb = jb_idx * Bc;
            int bc = (jb + Bc <= N) ? Bc : (N - jb);
            pack_K_12_fp16(K + jb * d_k, bc, d_k, K_pack_buf + jb_idx * Bc * d_k);
        }
    }

    // 2. Outer Q loop
    for (int ib = 0; ib < M; ib += Br)
    {
        const int br = (ib + Br <= M) ? Br : (M - ib);

        // Stack-local accumulators (4KB for d_v=128)
        // Use VLA via fixed max size (d_v <= 256 covers MLA d_v=128 with margin)
        float O_local[8 * 256];
        float m_local[8];
        float l_local[8];
        for (int i = 0; i < br; i++)
        {
            m_local[i] = -FLT_MAX;
            l_local[i] = 0.f;
        }
        // Zero only the rows we'll touch
        for (int i = 0; i < br; i++)
            memset(O_local + i * d_v, 0, d_v * sizeof(float));

        // Pack Q for this Q block (per-ib; pre-packing entire M didn't help -
        // L2 pressure offset the savings)
        double t0 = profile ? ncnn::get_current_time() : 0.0;
        pack_Q_8_fp16(Q + ib * d_k, br, d_k, Q_pack_buf);
        if (profile) prof.qpack += ncnn::get_current_time() - t0;

        // 3. Inner KV loop
        const int past = N - M;
        int Nblocks_visible = Nblocks;
        if (causal_mask)
        {
            Nblocks_visible = (past + ib + br + Bc - 1) / Bc;
            if (Nblocks_visible > Nblocks)
                Nblocks_visible = Nblocks;
        }
        for (int jb_idx = 0; jb_idx < Nblocks_visible; jb_idx++)
        {
            int jb = jb_idx * Bc;
            int bc = (jb + Bc <= N) ? Bc : (N - jb);

            // 8x12 micro-kernel.
            const __fp16* K_pack_cur  = K_pack_buf + jb_idx * Bc * d_k;
            const __fp16* K_pack_next = (jb_idx + 1 < Nblocks_visible)
                ? K_pack_buf + (jb_idx + 1) * Bc * d_k
                : NULL;

            __fp16 P_T[12 * 8];
            float P_all[8 * 12];
            const bool full_visible_block = !causal_mask || (jb + bc <= past + ib + 1);
            const bool fused_qk_softmax = (br == 8 && bc == 12 && d_k == 192 && d_v == 128 && full_visible_block && (!mask || causal_mask));
            if (fused_qk_softmax)
            {
                float alpha_row[8];
                float l_old_row[8];
                t0 = profile ? ncnn::get_current_time() : 0.0;
                qk_softmax_8x12_d192_fp16fml(Q_pack_buf, K_pack_cur, scale, m_local, l_local, alpha_row, l_old_row, P_T);
                if (profile) prof.qk += ncnn::get_current_time() - t0;

                t0 = profile ? ncnn::get_current_time() : 0.0;
                #define RESCALE_O_ROW_128(row) do { \
                    const float alpha = alpha_row[(row)]; \
                    if (l_old_row[(row)] != 0.f && alpha != 1.f) \
                    { \
                        float* oi = O_local + (row) * d_v; \
                        float32x4_t valpha = vdupq_n_f32(alpha); \
                        for (int d = 0; d < 128; d += 8) \
                        { \
                            vst1q_f32(oi + d,     vmulq_f32(vld1q_f32(oi + d),     valpha)); \
                            vst1q_f32(oi + d + 4, vmulq_f32(vld1q_f32(oi + d + 4), valpha)); \
                        } \
                    } \
                } while (0)
                RESCALE_O_ROW_128(0);
                RESCALE_O_ROW_128(1);
                RESCALE_O_ROW_128(2);
                RESCALE_O_ROW_128(3);
                RESCALE_O_ROW_128(4);
                RESCALE_O_ROW_128(5);
                RESCALE_O_ROW_128(6);
                RESCALE_O_ROW_128(7);
                #undef RESCALE_O_ROW_128
                if (profile) prof.softmax += ncnn::get_current_time() - t0;
            }
            else
            {
            float S[8 * 12];
            t0 = profile ? ncnn::get_current_time() : 0.0;
            if (d_k == 192)
                qk_micro_8x12_fp16fml<192>(Q_pack_buf, K_pack_cur, K_pack_next,
                                           d_k, scale, S);
            else
                qk_micro_8x12_fp16fml<0>(Q_pack_buf, K_pack_cur, K_pack_next,
                                         d_k, scale, S);
            if (profile) prof.qk += ncnn::get_current_time() - t0;

            // Apply mask + suppress N-tail columns to -inf
            t0 = profile ? ncnn::get_current_time() : 0.0;
            if (causal_mask)
            {
                const bool full_visible = (jb + bc <= past + ib + 1);
                if (!full_visible)
                {
                    for (int i = 0; i < br; i++)
                    {
                        const int valid = past + ib + i + 1 - jb;
                        const int valid_clamped = valid < 0 ? 0 : valid > bc ? bc : valid;
                        for (int j = valid_clamped; j < Bc; j++)
                            S[i * Bc + j] = -FLT_MAX;
                    }
                }
            }
            else if (mask)
            {
                for (int i = 0; i < br; i++)
                {
                    const __fp16* mp = mask + (ib + i) * N + jb;
                    for (int j = 0; j < bc; j++)
                        S[i * Bc + j] += (float)mp[j];
                    for (int j = bc; j < Bc; j++)
                        S[i * Bc + j] = -FLT_MAX;
                }
            }
            else if (bc < Bc)
            {
                for (int i = 0; i < br; i++)
                    for (int j = bc; j < Bc; j++)
                        S[i * Bc + j] = -FLT_MAX;
            }

            // 4. Two-phase update:
            //    Phase A: per-row softmax (compute m_new, alpha, P, rescale O by alpha,
            //             update m_local & l_local)
            //    Phase B: PV - load V[jb+j] ONCE per j, reuse across all br rows
            //             (8x less V bandwidth than per-row PV)

            // Phase A: per-row softmax. P[Br][Bc] kept on stack across phases.
            #define PROCESS_SOFTMAX_ROW_BC(row, bc_len) do { \
                float* oi = O_local + (row) * d_v; \
                const float* Si = S + (row) * Bc; \
                float m_block = Si[0]; \
                for (int j = 1; j < (bc_len); j++) \
                    m_block = fmaxf(m_block, Si[j]); \
                float m_old = m_local[(row)]; \
                float m_new = fmaxf(m_old, m_block); \
                float alpha = (m_new == m_old) ? 1.f : fast_exp_f32(m_old - m_new); \
                if (l_local[(row)] != 0.f && alpha != 1.f) \
                { \
                    float32x4_t valpha = vdupq_n_f32(alpha); \
                    int d = 0; \
                    for (; d + 7 < d_v; d += 8) \
                    { \
                        vst1q_f32(oi + d,     vmulq_f32(vld1q_f32(oi + d),     valpha)); \
                        vst1q_f32(oi + d + 4, vmulq_f32(vld1q_f32(oi + d + 4), valpha)); \
                    } \
                    for (; d + 3 < d_v; d += 4) \
                        vst1q_f32(oi + d, vmulq_f32(vld1q_f32(oi + d), valpha)); \
                    for (; d < d_v; d++) \
                        oi[d] *= alpha; \
                } \
                float* Pi = P_all + (row) * Bc; \
                float l_block; \
                { \
                    float32x4_t vm = vdupq_n_f32(m_new); \
                    float32x4_t vlsum = vdupq_n_f32(0.f); \
                    int j = 0; \
                    for (; j + 3 < (bc_len); j += 4) \
                    { \
                        float32x4_t s4 = vsubq_f32(vld1q_f32(Si + j), vm); \
                        float32x4_t p4 = fast_exp_f32x4(s4); \
                        vst1q_f32(Pi + j, p4); \
                        vlsum = vaddq_f32(vlsum, p4); \
                    } \
                    l_block = vaddvq_f32(vlsum); \
                    for (; j < (bc_len); j++) \
                    { \
                        Pi[j] = fast_exp_f32(Si[j] - m_new); \
                        l_block += Pi[j]; \
                    } \
                } \
                for (int j = (bc_len); j < Bc; j++) \
                    Pi[j] = 0.f; \
                l_local[(row)] = alpha * l_local[(row)] + l_block; \
                m_local[(row)] = m_new; \
            } while (0)

            if (br == 8 && bc == 12)
            {
                PROCESS_SOFTMAX_ROW_BC(0, 12);
                PROCESS_SOFTMAX_ROW_BC(1, 12);
                PROCESS_SOFTMAX_ROW_BC(2, 12);
                PROCESS_SOFTMAX_ROW_BC(3, 12);
                PROCESS_SOFTMAX_ROW_BC(4, 12);
                PROCESS_SOFTMAX_ROW_BC(5, 12);
                PROCESS_SOFTMAX_ROW_BC(6, 12);
                PROCESS_SOFTMAX_ROW_BC(7, 12);
            }
            else if (br == 8)
            {
                PROCESS_SOFTMAX_ROW_BC(0, bc);
                PROCESS_SOFTMAX_ROW_BC(1, bc);
                PROCESS_SOFTMAX_ROW_BC(2, bc);
                PROCESS_SOFTMAX_ROW_BC(3, bc);
                PROCESS_SOFTMAX_ROW_BC(4, bc);
                PROCESS_SOFTMAX_ROW_BC(5, bc);
                PROCESS_SOFTMAX_ROW_BC(6, bc);
                PROCESS_SOFTMAX_ROW_BC(7, bc);
            }
            else
            {
                for (int i = 0; i < br; i++)
                {
                    PROCESS_SOFTMAX_ROW_BC(i, bc);
                }
            }
            #undef PROCESS_SOFTMAX_ROW_BC
            if (profile) prof.softmax += ncnn::get_current_time() - t0;

            // Phase B prep: pack P transposed and converted to fp16.
            // Layout P_T[j*8 + i] for j in 0..12, i in 0..8 (zero-padded for tail rows).
            // This lets the PV inner loop use FP16FML lane-broadcast on a single P load per j.
            pack_P_8x12_fp16(P_all, br, P_T);
            }

            // Phase B: V-shared PV using FP16FML. Tile d_v into chunks of 8 floats so we keep
            // 16 fp32x4 O accumulators (8 rows x 2 halves of an 8-element block) in regs.
            // Per d_block: load O[br][d_block:+8] once, sweep all bc j's loading V/P once,
            // FMA into all br O rows via vfmlalq_laneq, then store O once.
            t0 = profile ? ncnn::get_current_time() : 0.0;
            #define PROCESS_PV_D_BLOCK(d_block) do { \
                float32x4_t o0_lo, o0_hi, o1_lo, o1_hi, o2_lo, o2_hi, o3_lo, o3_hi; \
                float32x4_t o4_lo, o4_hi, o5_lo, o5_hi, o6_lo, o6_hi, o7_lo, o7_hi; \
                const float* p0 = O_local + 0 * d_v + (d_block); \
                const float* p1 = O_local + 1 * d_v + (d_block); \
                const float* p2 = O_local + 2 * d_v + (d_block); \
                const float* p3 = O_local + 3 * d_v + (d_block); \
                const float* p4 = O_local + 4 * d_v + (d_block); \
                const float* p5 = O_local + 5 * d_v + (d_block); \
                const float* p6 = O_local + 6 * d_v + (d_block); \
                const float* p7 = O_local + 7 * d_v + (d_block); \
                o0_lo = vld1q_f32(p0); o0_hi = vld1q_f32(p0 + 4); \
                o1_lo = vld1q_f32(p1); o1_hi = vld1q_f32(p1 + 4); \
                o2_lo = vld1q_f32(p2); o2_hi = vld1q_f32(p2 + 4); \
                o3_lo = vld1q_f32(p3); o3_hi = vld1q_f32(p3 + 4); \
                o4_lo = vld1q_f32(p4); o4_hi = vld1q_f32(p4 + 4); \
                o5_lo = vld1q_f32(p5); o5_hi = vld1q_f32(p5 + 4); \
                o6_lo = vld1q_f32(p6); o6_hi = vld1q_f32(p6 + 4); \
                o7_lo = vld1q_f32(p7); o7_hi = vld1q_f32(p7 + 4); \
                const int pv_bc = bc; \
                if (pv_bc == 12) \
                { \
                    _Pragma("GCC unroll 12") \
                    for (int j = 0; j < 12; j++) \
                    { \
                        float16x8_t vj_h = vld1q_f16(V + (jb + j) * d_v + (d_block)); \
                        float16x8_t p_j  = vld1q_f16(P_T + j * 8); \
                        o0_lo = vfmlalq_laneq_low_f16 (o0_lo, vj_h, p_j, 0); \
                        o0_hi = vfmlalq_laneq_high_f16(o0_hi, vj_h, p_j, 0); \
                        o1_lo = vfmlalq_laneq_low_f16 (o1_lo, vj_h, p_j, 1); \
                        o1_hi = vfmlalq_laneq_high_f16(o1_hi, vj_h, p_j, 1); \
                        o2_lo = vfmlalq_laneq_low_f16 (o2_lo, vj_h, p_j, 2); \
                        o2_hi = vfmlalq_laneq_high_f16(o2_hi, vj_h, p_j, 2); \
                        o3_lo = vfmlalq_laneq_low_f16 (o3_lo, vj_h, p_j, 3); \
                        o3_hi = vfmlalq_laneq_high_f16(o3_hi, vj_h, p_j, 3); \
                        o4_lo = vfmlalq_laneq_low_f16 (o4_lo, vj_h, p_j, 4); \
                        o4_hi = vfmlalq_laneq_high_f16(o4_hi, vj_h, p_j, 4); \
                        o5_lo = vfmlalq_laneq_low_f16 (o5_lo, vj_h, p_j, 5); \
                        o5_hi = vfmlalq_laneq_high_f16(o5_hi, vj_h, p_j, 5); \
                        o6_lo = vfmlalq_laneq_low_f16 (o6_lo, vj_h, p_j, 6); \
                        o6_hi = vfmlalq_laneq_high_f16(o6_hi, vj_h, p_j, 6); \
                        o7_lo = vfmlalq_laneq_low_f16 (o7_lo, vj_h, p_j, 7); \
                        o7_hi = vfmlalq_laneq_high_f16(o7_hi, vj_h, p_j, 7); \
                    } \
                } \
                else \
                { \
                    for (int j = 0; j < pv_bc; j++) \
                    { \
                        float16x8_t vj_h = vld1q_f16(V + (jb + j) * d_v + (d_block)); \
                        float16x8_t p_j  = vld1q_f16(P_T + j * 8); \
                        o0_lo = vfmlalq_laneq_low_f16 (o0_lo, vj_h, p_j, 0); \
                        o0_hi = vfmlalq_laneq_high_f16(o0_hi, vj_h, p_j, 0); \
                        o1_lo = vfmlalq_laneq_low_f16 (o1_lo, vj_h, p_j, 1); \
                        o1_hi = vfmlalq_laneq_high_f16(o1_hi, vj_h, p_j, 1); \
                        o2_lo = vfmlalq_laneq_low_f16 (o2_lo, vj_h, p_j, 2); \
                        o2_hi = vfmlalq_laneq_high_f16(o2_hi, vj_h, p_j, 2); \
                        o3_lo = vfmlalq_laneq_low_f16 (o3_lo, vj_h, p_j, 3); \
                        o3_hi = vfmlalq_laneq_high_f16(o3_hi, vj_h, p_j, 3); \
                        o4_lo = vfmlalq_laneq_low_f16 (o4_lo, vj_h, p_j, 4); \
                        o4_hi = vfmlalq_laneq_high_f16(o4_hi, vj_h, p_j, 4); \
                        o5_lo = vfmlalq_laneq_low_f16 (o5_lo, vj_h, p_j, 5); \
                        o5_hi = vfmlalq_laneq_high_f16(o5_hi, vj_h, p_j, 5); \
                        o6_lo = vfmlalq_laneq_low_f16 (o6_lo, vj_h, p_j, 6); \
                        o6_hi = vfmlalq_laneq_high_f16(o6_hi, vj_h, p_j, 6); \
                        o7_lo = vfmlalq_laneq_low_f16 (o7_lo, vj_h, p_j, 7); \
                        o7_hi = vfmlalq_laneq_high_f16(o7_hi, vj_h, p_j, 7); \
                    } \
                } \
                float* q0 = O_local + 0 * d_v + (d_block); \
                vst1q_f32(q0, o0_lo); vst1q_f32(q0 + 4, o0_hi); \
                if (br == 8) \
                { \
                    float* q1 = O_local + 1 * d_v + (d_block); vst1q_f32(q1, o1_lo); vst1q_f32(q1 + 4, o1_hi); \
                    float* q2 = O_local + 2 * d_v + (d_block); vst1q_f32(q2, o2_lo); vst1q_f32(q2 + 4, o2_hi); \
                    float* q3 = O_local + 3 * d_v + (d_block); vst1q_f32(q3, o3_lo); vst1q_f32(q3 + 4, o3_hi); \
                    float* q4 = O_local + 4 * d_v + (d_block); vst1q_f32(q4, o4_lo); vst1q_f32(q4 + 4, o4_hi); \
                    float* q5 = O_local + 5 * d_v + (d_block); vst1q_f32(q5, o5_lo); vst1q_f32(q5 + 4, o5_hi); \
                    float* q6 = O_local + 6 * d_v + (d_block); vst1q_f32(q6, o6_lo); vst1q_f32(q6 + 4, o6_hi); \
                    float* q7 = O_local + 7 * d_v + (d_block); vst1q_f32(q7, o7_lo); vst1q_f32(q7 + 4, o7_hi); \
                } \
                else \
                { \
                    if (br > 1) { float* q1 = O_local + 1 * d_v + (d_block); vst1q_f32(q1, o1_lo); vst1q_f32(q1 + 4, o1_hi); } \
                    if (br > 2) { float* q2 = O_local + 2 * d_v + (d_block); vst1q_f32(q2, o2_lo); vst1q_f32(q2 + 4, o2_hi); } \
                    if (br > 3) { float* q3 = O_local + 3 * d_v + (d_block); vst1q_f32(q3, o3_lo); vst1q_f32(q3 + 4, o3_hi); } \
                    if (br > 4) { float* q4 = O_local + 4 * d_v + (d_block); vst1q_f32(q4, o4_lo); vst1q_f32(q4 + 4, o4_hi); } \
                    if (br > 5) { float* q5 = O_local + 5 * d_v + (d_block); vst1q_f32(q5, o5_lo); vst1q_f32(q5 + 4, o5_hi); } \
                    if (br > 6) { float* q6 = O_local + 6 * d_v + (d_block); vst1q_f32(q6, o6_lo); vst1q_f32(q6 + 4, o6_hi); } \
                    if (br > 7) { float* q7 = O_local + 7 * d_v + (d_block); vst1q_f32(q7, o7_lo); vst1q_f32(q7 + 4, o7_hi); } \
                } \
            } while (0)

            if (d_v == 128)
            {
                #pragma GCC unroll 2
                for (int d_block = 0; d_block < 128; d_block += 8)
                {
                    PROCESS_PV_D_BLOCK(d_block);
                }
            }
            else
            {
                for (int d_block = 0; d_block < d_v; d_block += 8)
                {
                    const int dv_end = (d_block + 8 <= d_v) ? 8 : (d_v - d_block);
                    if (dv_end < 8)
                    {
                        // Tail (rare for d_v=128 multiple of 8). Scalar fallback.
                        for (int i = 0; i < br; i++)
                        {
                            float* oi = O_local + i * d_v + d_block;
                            const float* Pi = P_all + i * Bc;
                            for (int j = 0; j < bc; j++)
                            {
                                const __fp16* vj = V + (jb + j) * d_v + d_block;
                                for (int d = 0; d < dv_end; d++)
                                    oi[d] += Pi[j] * (float)vj[d];
                            }
                        }
                        continue;
                    }

                    PROCESS_PV_D_BLOCK(d_block);
                }
            }
            #undef PROCESS_PV_D_BLOCK
            if (profile) prof.pv += ncnn::get_current_time() - t0;
        }

        // 5. Final: normalize O_local by l and write fp16 to O[ib..ib+br]
        t0 = profile ? ncnn::get_current_time() : 0.0;
        #define PROCESS_NORM_ROW_128(row) do { \
            const float* oi_src = O_local + (row) * d_v; \
            __fp16* O_out = O + (ib + (row)) * d_v; \
            float inv_l = (l_local[(row)] > 0.f) ? 1.f / l_local[(row)] : 0.f; \
            float32x4_t vinv = vdupq_n_f32(inv_l); \
            for (int d = 0; d < 128; d += 8) \
            { \
                float32x4_t a = vmulq_f32(vld1q_f32(oi_src + d),     vinv); \
                float32x4_t b = vmulq_f32(vld1q_f32(oi_src + d + 4), vinv); \
                vst1_f16(O_out + d,     vcvt_f16_f32(a)); \
                vst1_f16(O_out + d + 4, vcvt_f16_f32(b)); \
            } \
        } while (0)

        if (br == 8 && d_v == 128)
        {
            PROCESS_NORM_ROW_128(0);
            PROCESS_NORM_ROW_128(1);
            PROCESS_NORM_ROW_128(2);
            PROCESS_NORM_ROW_128(3);
            PROCESS_NORM_ROW_128(4);
            PROCESS_NORM_ROW_128(5);
            PROCESS_NORM_ROW_128(6);
            PROCESS_NORM_ROW_128(7);
        }
        else
        {
            for (int i = 0; i < br; i++)
            {
                const float* oi_src = O_local + i * d_v;
                __fp16* O_out = O + (ib + i) * d_v;
                float inv_l = (l_local[i] > 0.f) ? 1.f / l_local[i] : 0.f;
                float32x4_t vinv = vdupq_n_f32(inv_l);
                int d = 0;
                for (; d + 7 < d_v; d += 8)
                {
                    float32x4_t a = vmulq_f32(vld1q_f32(oi_src + d),     vinv);
                    float32x4_t b = vmulq_f32(vld1q_f32(oi_src + d + 4), vinv);
                    vst1_f16(O_out + d,     vcvt_f16_f32(a));
                    vst1_f16(O_out + d + 4, vcvt_f16_f32(b));
                }
                for (; d + 3 < d_v; d += 4)
                {
                    float32x4_t a = vmulq_f32(vld1q_f32(oi_src + d), vinv);
                    vst1_f16(O_out + d, vcvt_f16_f32(a));
                }
                for (; d < d_v; d++)
                    O_out[d] = (__fp16)(oi_src[d] * inv_l);
            }
        }
        #undef PROCESS_NORM_ROW_128
        if (profile) prof.norm += ncnn::get_current_time() - t0;
    }

    if (profile)
        flash_prefill_profile_add(prof);

    if (own_K_pack) free(K_pack_buf);
    if (own_Q_pack) free(Q_pack_buf);
}

#endif // __ARM_FEATURE_FP16_FML

// ============================================================================
// kv_cache=2 fp16 fast path: orchestrates per-head dispatch to the
// flash decode (src_seqlen==1) or flash prefill (src_seqlen>1) kernels.
//
// Returns:
//   0    success
//   -1   cannot handle in this TU (e.g., prefill needs FP16FML but the TU
//        wasn't compiled with +fp16fml). Caller falls back to Gemm.
//   -100 alloc failure
//
// At the top of the function we runtime-dispatch to wrappers compiled with
// the proper -march. Same pattern ncnn uses in innerproduct_fp16s.h /
// gemm_fp16s.h.
// ============================================================================
namespace ncnn {

static int flash_kv_cache2_fp16_path(
    Mat& top_blob, const Mat& query, const Mat& past_key, const Mat& past_value,
    bool has_attn_mask, const Mat& attn_mask_blob,
    int src_seqlen, int dst_seqlen, int embed_dim, int out_embed_dim,
    int num_heads, int num_heads_per_group, float scale, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM82FP16FML && __aarch64__ && !__ARM_FEATURE_FP16_FML
    if (ncnn::cpu_support_arm_asimdfhm())
    {
        return flash_kv_cache2_fp16_path_asimdfhm(
            top_blob, query, past_key, past_value, has_attn_mask, attn_mask_blob,
            src_seqlen, dst_seqlen, embed_dim, out_embed_dim,
            num_heads, num_heads_per_group, scale, opt);
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82 && __aarch64__ && !__ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (ncnn::cpu_support_arm_asimdhp())
    {
        return flash_kv_cache2_fp16_path_asimdhp(
            top_blob, query, past_key, past_value, has_attn_mask, attn_mask_blob,
            src_seqlen, dst_seqlen, embed_dim, out_embed_dim,
            num_heads, num_heads_per_group, scale, opt);
    }
#endif

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    float _scale = scale;
    if (_scale == 0.f) _scale = 1.f / sqrtf((float)embed_dim);

    if (has_attn_mask && attn_mask_blob.dims > 0 && attn_mask_blob.elemsize != 2u)
        return -1;

    if (src_seqlen == 1)
    {
        // Decode path - needs only fp16, no FHM.
        top_blob.create(out_embed_dim, 1, num_heads, 2u, opt.blob_allocator);
        if (top_blob.empty()) return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < num_heads; i++)
        {
            const int gq = i / num_heads_per_group;
            const __fp16* Q_head = (const __fp16*)query.channel(i).data;
            const __fp16* K_head = (const __fp16*)past_key.channel(gq).data;
            const __fp16* V_head = (const __fp16*)past_value.channel(gq).data;
            __fp16* O_head = (__fp16*)top_blob.channel(i).data;

            const __fp16* mask_ptr = NULL;
            if (has_attn_mask && attn_mask_blob.dims > 0)
            {
                const Mat& maskm = attn_mask_blob;
                if (maskm.dims == 3)
                    mask_ptr = (const __fp16*)(maskm.c > 1 ? maskm.channel(i).data : maskm.channel(0).data);
                else
                    mask_ptr = (const __fp16*)maskm.data;
            }
            flash_attn_fp16_decode(Q_head, K_head, V_head, O_head,
                                   dst_seqlen, embed_dim, out_embed_dim, _scale, mask_ptr);
        }
        return 0;
    }
    else
    {
        // Prefill path - needs FP16FML.
#if __ARM_FEATURE_FP16_FML
        if (out_embed_dim > 256)
            return -1;

        top_blob.create(out_embed_dim, src_seqlen, num_heads, 2u, opt.blob_allocator);
        if (top_blob.empty()) return -100;

        bool shared_causal_mask = false;
        if (has_attn_mask && attn_mask_blob.dims == 2 && attn_mask_blob.elemsize == 2u)
            shared_causal_mask = is_causal_mask_fp16((const __fp16*)attn_mask_blob.data, src_seqlen, dst_seqlen);

        if (!shared_causal_mask && !allow_noncausal_flash_prefill())
            return -1;

        const bool profile = flash_prefill_profile_enabled();
        if (profile)
            flash_prefill_profile_reset();

        // Per-thread workspace: K_pack + Q_pack
        const int Bc = 12;
        const int BcMicro = 12;
        const int Br = 8;
        const int Nblocks = (dst_seqlen + Bc - 1) / Bc;
        const int ws_kpack_bytes = Nblocks * Bc * embed_dim * (int)sizeof(__fp16);
        const int ws_qpack_bytes = embed_dim * Br * (int)sizeof(__fp16);
        const int ws_per_thread = ws_kpack_bytes + ws_qpack_bytes;

        Mat workspace(ws_per_thread, 1, opt.num_threads, 1u, opt.workspace_allocator);
        if (workspace.empty()) return -100;

        const int num_group = num_heads / num_heads_per_group;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int gq = 0; gq < num_group; gq++)
        {
            const __fp16* K_head = (const __fp16*)past_key.channel(gq).data;
            const __fp16* V_head = (const __fp16*)past_value.channel(gq).data;

            unsigned char* ws = (unsigned char*)workspace.channel(get_omp_thread_num()).data;
            __fp16* K_pack = (__fp16*)ws;
            __fp16* Q_pack = (__fp16*)(ws + ws_kpack_bytes);

            // K is shared by all heads in the same GQA group. Pack it once,
            // then reuse across the num_heads_per_group Q heads.
            for (int jb_idx = 0; jb_idx < Nblocks; jb_idx++)
            {
                int jb = jb_idx * Bc;
                int bc = (jb + Bc <= dst_seqlen) ? Bc : (dst_seqlen - jb);
                for (int j0 = 0; j0 < bc; j0 += BcMicro)
                {
                    int bc12 = (j0 + BcMicro <= bc) ? BcMicro : (bc - j0);
                    pack_K_12_fp16(K_head + (jb + j0) * embed_dim, bc12, embed_dim, K_pack + jb_idx * Bc * embed_dim + j0 * embed_dim);
                }
            }

            const int head_start = gq * num_heads_per_group;
            const int head_end = head_start + num_heads_per_group;
            for (int i = head_start; i < head_end; i++)
            {
                const __fp16* Q_head = (const __fp16*)query.channel(i).data;
                __fp16* O_head = (__fp16*)top_blob.channel(i).data;

                const __fp16* mask_ptr = NULL;
                bool causal_mask = false;
                if (has_attn_mask && attn_mask_blob.dims > 0)
                {
                    const Mat& maskm = attn_mask_blob;
                    if (maskm.dims == 3)
                    {
                        mask_ptr = (const __fp16*)(maskm.c > 1 ? maskm.channel(i).data : maskm.channel(0).data);
                        causal_mask = is_causal_mask_fp16(mask_ptr, src_seqlen, dst_seqlen);
                    }
                    else
                    {
                        mask_ptr = (const __fp16*)maskm.data;
                        causal_mask = shared_causal_mask;
                    }
                }

                flash_attn_fp16_prefill(Q_head, K_head, V_head, O_head,
                                        src_seqlen, dst_seqlen, embed_dim, out_embed_dim,
                                        _scale, mask_ptr, K_pack, Q_pack, true, causal_mask);
            }
        }
        if (profile)
            flash_prefill_profile_print();
        return 0;
#else
        // FP16FML not available in this TU - caller falls back to Gemm.
        (void)top_blob; (void)query; (void)past_key; (void)past_value;
        (void)has_attn_mask; (void)attn_mask_blob;
        (void)src_seqlen; (void)dst_seqlen; (void)embed_dim; (void)out_embed_dim;
        (void)num_heads; (void)num_heads_per_group; (void)opt;
        return -1;
#endif
    }
#else
    // No fp16 in this TU - should never be called directly here in non-RUNTIME_CPU build;
    // RUNTIME_CPU build dispatches via wrappers above.
    (void)top_blob; (void)query; (void)past_key; (void)past_value;
    (void)has_attn_mask; (void)attn_mask_blob;
    (void)src_seqlen; (void)dst_seqlen; (void)embed_dim; (void)out_embed_dim;
    (void)num_heads; (void)num_heads_per_group; (void)scale; (void)opt;
    return -1;
#endif
}

} // namespace ncnn

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
