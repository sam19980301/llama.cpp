#include "debug.h"

#include <ggml.h>

#include "log.h"

#include <cmath>
#include <string>

static std::string common_ggml_ne_string(const ggml_tensor * t) {
    std::string str;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        str += std::to_string(t->ne[i]);
        if (i + 1 < GGML_MAX_DIMS) {
            str += ", ";
        }
    }
    return str;
}

static std::string common_ggml_nb_string(const ggml_tensor * t) {
    std::string str;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        str += std::to_string(t->nb[i]);
        if (i + 1 < GGML_MAX_DIMS) {
            str += ", ";
        }
    }
    return str;
}

static float common_ggml_get_float_value(const uint8_t * data,
                           ggml_type       type,
                           const size_t *  nb,
                           size_t          i0,
                           size_t          i1,
                           size_t          i2,
                           size_t          i3) {
    size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
    float  v;
    if (type == GGML_TYPE_F16) {
        v = ggml_fp16_to_fp32(*(const ggml_fp16_t *) &data[i]);
    } else if (type == GGML_TYPE_F32) {
        v = *(const float *) &data[i];
    } else if (type == GGML_TYPE_I64) {
        v = (float) *(const int64_t *) &data[i];
    } else if (type == GGML_TYPE_I32) {
        v = (float) *(const int32_t *) &data[i];
    } else if (type == GGML_TYPE_I16) {
        v = (float) *(const int16_t *) &data[i];
    } else if (type == GGML_TYPE_I8) {
        v = (float) *(const int8_t *) &data[i];
    } else if (type == GGML_TYPE_BF16) {
        v = ggml_bf16_to_fp32(*(const ggml_bf16_t *) &data[i]);
    } else {
        GGML_ABORT("fatal error");
    }
    return v;
}

#define INDENT "    "

template <bool abort>
void common_debug_print_tensor(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, int64_t n) {
    GGML_ASSERT(n > 0);
    float sum = 0;
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    const float v = common_ggml_get_float_value(data, type, nb, i0, i1, i2, i3);
                    sum += v;
                }
            }
        }
    }
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        LOG(INDENT "[\n");
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            if (i2 == n && ne[2] > 2 * n) {
                LOG(INDENT INDENT "..., \n");
                i2 = ne[2] - n;
            }
            LOG(INDENT INDENT "[\n");
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                if (i1 == n && ne[1] > 2 * n) {
                    LOG(INDENT INDENT INDENT "..., \n");
                    i1 = ne[1] - n;
                }
                LOG(INDENT INDENT INDENT "[");
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    if (i0 == n && ne[0] > 2 * n) {
                        LOG("   ..., ");
                        i0 = ne[0] - n;
                    }
                    const float v = common_ggml_get_float_value(data, type, nb, i0, i1, i2, i3);
                    LOG("%12.4f", v);
                    if (i0 < ne[0] - 1) {
                        LOG(", ");
                    }
                }
                LOG("  ],\n");
            }
            LOG(INDENT INDENT "],\n");
        }
        LOG(INDENT "]\n");
        LOG(INDENT "sum = %f\n", sum);
    }

    if constexpr (abort) {
        if (std::isnan(sum)) {
            LOG("encountered NaN - aborting\n");
            exit(0);
        }
    }
}

/**
 * GGML operations callback during the graph execution.
 *
 * @param t current tensor
 * @param ask when ask is true, the scheduler wants to know if we are interested in data from this tensor
 *            if we return true, a follow-up call will be made with ask=false in which we can do the actual collection.
 *            see ggml_backend_sched_eval_callback
 * @param user_data user data to pass at each call back
 * @return true to receive data or continue the graph, false otherwise
 */
template <bool abort_on_nan> bool common_debug_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb_data = (base_callback_data *) user_data;

    const struct ggml_tensor * src0 = t->src[0];
    const struct ggml_tensor * src1 = t->src[1];
    const struct ggml_tensor * src2 = t->src[2];

    if (ask) {
        return true;  // Always retrieve data
    }

    bool matches_filter = cb_data->tensor_filters.empty();

    if (!matches_filter) {
        for (const auto & filter : cb_data->tensor_filters) {
            if (std::regex_search(t->name, filter)) {
                matches_filter = true;
                break;
            }
        }
    }

    char t_str[128] = { 0 };
    if (t) {
        snprintf(t_str, sizeof(t_str), "%s{%s}{%s}(%s)", t->name, common_ggml_ne_string(t).c_str(), common_ggml_nb_string(t).c_str(), ggml_type_name(t->type));
    }

    char src0_str[128] = { 0 };
    if (src0) {
        snprintf(src0_str, sizeof(src0_str), "%s{%s}{%s}(%s)", src0->name, common_ggml_ne_string(src0).c_str(), common_ggml_nb_string(src0).c_str(), ggml_type_name(src0->type));
    }

    char src1_str[128] = { 0 };
    if (src1) {
        snprintf(src1_str, sizeof(src1_str), "%s{%s}{%s}(%s)", src1->name, common_ggml_ne_string(src1).c_str(), common_ggml_nb_string(src1).c_str(), ggml_type_name(src1->type));
    }

    char src2_str[128] = { 0 };
    if (src2) {
        snprintf(src2_str, sizeof(src2_str), "%s{%s}{%s}(%s)", src2->name, common_ggml_ne_string(src2).c_str(), common_ggml_nb_string(src2).c_str(), ggml_type_name(src2->type));
    }

    std::string params_str;
    if (
        t->op == GGML_OP_ADD ||
        t->op == GGML_OP_MUL ||
        t->op == GGML_OP_MUL_MAT ||
        t->op == GGML_OP_CONT ||
        t->op == GGML_OP_RESHAPE ||
        t->op == GGML_OP_GET_ROWS ||
        t->op == GGML_OP_SET_ROWS ||
        t->op == GGML_OP_UNARY
    ) {

    }
    else if (t->op == GGML_OP_CONCAT) {
        const int32_t dim = ((int32_t *) t->op_params)[0];
        params_str += "dim: " + std::to_string(dim);
    }
    else if (t->op == GGML_OP_NORM || t->op == GGML_OP_RMS_NORM) {
        float eps;
        memcpy(&eps, t->op_params, sizeof(float));
        params_str += "eps: " + std::to_string(eps);
    }
    else if (t->op == GGML_OP_VIEW) {
        size_t offset;
        memcpy(&offset, t->op_params, sizeof(size_t));
        params_str += "offset: " + std::to_string(offset);
    }
    else if (t->op == GGML_OP_PERMUTE) {
        const int32_t axis0 = ((int32_t *) t->op_params)[0];
        const int32_t axis1 = ((int32_t *) t->op_params)[1];
        const int32_t axis2 = ((int32_t *) t->op_params)[2];
        const int32_t axis3 = ((int32_t *) t->op_params)[3];
        params_str += "axis: [" + \
            std::to_string(axis0) + " " + \
            std::to_string(axis1) + " " + \
            std::to_string(axis2) + " " + \
            std::to_string(axis3) + "]";
    }
    else if (t->op == GGML_OP_SOFT_MAX) {
        float scale    = 1.0f;
        float max_bias = 0.0f;
        memcpy(&scale,    (float *) t->op_params + 0, sizeof(float));
        memcpy(&max_bias, (float *) t->op_params + 1, sizeof(float));
        params_str += "scale: " + std::to_string(scale) + ";";
        params_str += "max_bias: " + std::to_string(max_bias);
    }
    else if (t->op == GGML_OP_ROPE) {
        float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
        int sections[4];
        const int n_dims     = ((int32_t *) t->op_params)[1];
        const int mode       = ((int32_t *) t->op_params)[2];
        const int n_ctx_orig = ((int32_t *) t->op_params)[4];
        memcpy(&freq_base,   (int32_t *) t->op_params +  5, sizeof(float));
        memcpy(&freq_scale,  (int32_t *) t->op_params +  6, sizeof(float));
        memcpy(&ext_factor,  (int32_t *) t->op_params +  7, sizeof(float));
        memcpy(&attn_factor, (int32_t *) t->op_params +  8, sizeof(float));
        memcpy(&beta_fast,   (int32_t *) t->op_params +  9, sizeof(float));
        memcpy(&beta_slow,   (int32_t *) t->op_params + 10, sizeof(float));
        memcpy(&sections,    (int32_t *) t->op_params + 11, sizeof(int)*4);

        params_str += "n_dims: " + std::to_string(n_dims) + ";";
        params_str += "mode: " + std::to_string(mode) + ";";
        params_str += "n_ctx_orig: " + std::to_string(n_ctx_orig) + ";";
        params_str += "freq_base: " + std::to_string(freq_base) + ";";
        params_str += "freq_scale: " + std::to_string(freq_scale) + ";";
        params_str += "ext_factor: " + std::to_string(ext_factor) + ";";
        params_str += "attn_factor: " + std::to_string(attn_factor) + ";";
        params_str += "beta_fast: " + std::to_string(beta_fast) + ";";
        params_str += "beta_slow: " + std::to_string(beta_slow) + ";";
        params_str += "sections: [" + \
            std::to_string(sections[0]) + " " + \
            std::to_string(sections[1]) + " " + \
            std::to_string(sections[2]) + " " + \
            std::to_string(sections[3]) + "]";
    }
    else if (t->op == GGML_OP_IM2COL) {
        const int32_t s0 =    ((int32_t *)(t->op_params))[0];
        const int32_t s1 =    ((int32_t *)(t->op_params))[1];
        const int32_t p0 =    ((int32_t *)(t->op_params))[2];
        const int32_t p1 =    ((int32_t *)(t->op_params))[3];
        const int32_t d0 =    ((int32_t *)(t->op_params))[4];
        const int32_t d1 =    ((int32_t *)(t->op_params))[5];
        const int32_t is_2d = ((int32_t *)(t->op_params))[6];
        params_str += "s0: " +    std::to_string(s0) + ";";
        params_str += "s1: " +    std::to_string(s1) + ";";
        params_str += "p0: " +    std::to_string(p0) + ";";
        params_str += "p1: " +    std::to_string(p1) + ";";
        params_str += "d0: " +    std::to_string(d0) + ";";
        params_str += "d1: " +    std::to_string(d1) + ";";
        params_str += "is_2d: " + std::to_string(is_2d);
    }
    else if (t->op == GGML_OP_UPSCALE) {
        const int32_t mode = ((int32_t *) t->op_params)[0];
        params_str += "mode: " + std::to_string(mode);
    }
    else if (t->op == GGML_OP_PAD) {
        const int32_t lp0 =      ((int32_t *) t->op_params)[0];
        const int32_t rp0 =      ((int32_t *) t->op_params)[1];
        const int32_t lp1 =      ((int32_t *) t->op_params)[2];
        const int32_t rp1 =      ((int32_t *) t->op_params)[3];
        const int32_t lp2 =      ((int32_t *) t->op_params)[4];
        const int32_t rp2 =      ((int32_t *) t->op_params)[5];
        const int32_t lp3 =      ((int32_t *) t->op_params)[6];
        const int32_t rp3 =      ((int32_t *) t->op_params)[7];
        const int32_t circular = ((int32_t *) t->op_params)[8];
        params_str += "lp0: " +      std::to_string(lp0) + ";";
        params_str += "rp0: " +      std::to_string(rp0) + ";";
        params_str += "lp1: " +      std::to_string(lp1) + ";";
        params_str += "rp1: " +      std::to_string(rp1) + ";";
        params_str += "lp2: " +      std::to_string(lp2) + ";";
        params_str += "rp2: " +      std::to_string(rp2) + ";";
        params_str += "lp3: " +      std::to_string(lp3) + ";";
        params_str += "rp3: " +      std::to_string(rp3) + ";";
        params_str += "circular: " + std::to_string(circular);
    }
    else if (t->op == GGML_OP_GLU) {
        const ggml_glu_op gop = ggml_get_glu_op(t);
        if (gop == GGML_GLU_OP_SWIGLU) {
            const int32_t swapped = ((int32_t *) t->op_params)[1];
            params_str += "op: ";
            params_str += ggml_glu_op_name(gop);
            params_str += ";";
            params_str += "swapped: " + std::to_string(swapped);
        }
        else {
            params_str += "skipped...";
        }
    }
    else {
        params_str += "skipped...";
    }

    if (matches_filter) {
        LOG("%s:%110s = %10s(%110s, %110s, %110s, %s)\n", // [%10s]
            __func__,
            // ggml_backend_buffer_name(t->buffer),
            t_str,
            ggml_op_desc(t),
            src0_str,
            src1_str,
            src2_str,
            params_str.c_str()
        );
    }

    const bool is_host = ggml_backend_buffer_is_host(t->buffer);

    if (!is_host) {
        auto n_bytes = ggml_nbytes(t);
        cb_data->data.resize(n_bytes);
        ggml_backend_tensor_get(t, cb_data->data.data(), 0, n_bytes);
    }

    /*
    if (!ggml_is_quantized(t->type) && matches_filter) {
        uint8_t * data = is_host ? (uint8_t *) t->data : cb_data->data.data();
        common_debug_print_tensor<abort_on_nan>(data, t->type, t->ne, t->nb, 3);
    }
    */

    return true;
}

// Explicit template instantiations
template bool common_debug_cb_eval<false>(ggml_tensor *, bool, void *);
template bool common_debug_cb_eval<true>(ggml_tensor *, bool, void *);
template void common_debug_print_tensor<false>(uint8_t *, ggml_type, const int64_t *, const size_t *, int64_t);
template void common_debug_print_tensor<true>(uint8_t *, ggml_type, const int64_t *, const size_t *, int64_t);
