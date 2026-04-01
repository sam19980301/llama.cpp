#include "ggml-ncp-context.h"

#include "ggml-impl.h"

#include <iostream>

// utils

inline static void ggml_ncp_vec_scale_f32(const int n, float * y, const float   v) {
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] *= v;
    }
}

inline static void ggml_ncp_vec_scale_f16(const int n, ggml_fp16_t * y, const float v) {
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_COMPUTE_FP32_TO_FP16(GGML_COMPUTE_FP16_TO_FP32(y[i])*v);
    }
}

// kernels

// ggml_ncp_compute_forward_rms_norm

static void ggml_ncp_compute_forward_rms_norm_f(ggml_ncp & ctx, struct ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    GGML_ASSERT(src0->nb[0] == sizeof(float));

    GGML_TENSOR_UNARY_OP_LOCALS

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    GGML_ASSERT(eps >= 0.0f);

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = 0; i01 < ne01; i01++) {
                if (src0->type == GGML_TYPE_F32) {
                    const float * x = (float *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);

                    double sum = 0.0;
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        sum += (double)(x[i00] * x[i00]);
                    }

                    const float mean = sum/ne00;
                    const float scale = 1.0f/sqrtf(mean + eps);
                    assert(scale > 0.0f);

                    float * y = (float *) ((char *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);
                    memcpy(y, x, ne00 * sizeof(float));
                    ggml_ncp_vec_scale_f32(ne00, y, scale);
                }
                else if (src0->type == GGML_TYPE_F16) {
                    const ggml_fp16_t * x = (ggml_fp16_t *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);

                    double sum = 0.0;
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        sum += (double)(GGML_COMPUTE_FP16_TO_FP32(x[i00]) * GGML_COMPUTE_FP16_TO_FP32(x[i00]));
                    }

                    const float mean = sum/ne00;
                    const float scale = 1.0f/sqrtf(mean + eps);
                    assert(scale > 0.0f);

                    ggml_fp16_t * y = (ggml_fp16_t *) ((char *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);
                    memcpy(y, x, ne00 * sizeof(ggml_fp16_t));
                    ggml_ncp_vec_scale_f16(ne00, y, scale);
                }
                else {
                    GGML_ABORT("fatal error");
                }
            }
        }
    }

    GGML_UNUSED(ctx);
}

static void ggml_ncp_compute_forward_rms_norm(ggml_ncp & ctx, struct ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
            ggml_ncp_compute_forward_rms_norm_f(ctx, dst);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

static bool ggml_ncp_compute_forward(ggml_ncp & ctx, struct ggml_tensor * dst) {

    switch (dst->op) {
        case GGML_OP_RMS_NORM:
            ggml_ncp_compute_forward_rms_norm(ctx, dst);
            break;
        default:
            return false;
    }

    return true;
}

//

ggml_ncp_t ggml_ncp_init(ggml_ncp_device_t dev) {
    GGML_LOG_INFO("%s: allocating\n", __func__);

    // init context
    ggml_ncp_t res = (ggml_ncp_t)calloc(1, sizeof(struct ggml_ncp));

    const struct ggml_ncp_device_props * props_dev = ggml_ncp_device_get_props(dev);

    GGML_LOG_INFO("%s: picking default device: %s\n", __func__, props_dev->name);

    res->dev = dev;

    snprintf(res->name, sizeof(res->name), "%s", props_dev->name);

    return res;
}

void ggml_ncp_free(ggml_ncp_t ctx) {
    GGML_LOG_INFO("%s: deallocating\n", __func__);

    free(ctx);
}

const char * ggml_ncp_get_name(ggml_ncp_t ctx) {
    return ctx->name;
}

enum ggml_status ggml_ncp_graph_compute(ggml_ncp_t ctx, struct ggml_cgraph * gf) {
    // TODO(sam)
    // TODO(sam): consider op fusion

    for (int i = 0; i < gf->n_nodes; i++) {
        ggml_tensor * node = gf->nodes[i];

        if (ggml_is_empty(node) || node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_NONE) {
            continue;
        }

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        bool ok = ggml_ncp_compute_forward(*ctx, node);
        if (!ok) {
            GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
        }
        GGML_ASSERT(ok);
    }

    return GGML_STATUS_SUCCESS;
}