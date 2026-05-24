#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include "ggml-ncp-context.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <iostream>

// utils

template<typename T>
inline static void ggml_ncp_vec_scale_f(const int n, T * y, const float v) {
    // scalar
    for (int i = 0; i < n; ++i) {
        static_assert(std::is_same<T, float>::value || std::is_same<T, ggml_fp16_t>::value);
        if constexpr (std::is_same<T, float>::value) {
            y[i] *= v;
        }
        else if constexpr (std::is_same<T, ggml_fp16_t>::value) {
            y[i] = GGML_COMPUTE_FP32_TO_FP16(GGML_COMPUTE_FP16_TO_FP32(y[i])*v);
        }
    }
}

template<typename T>
inline static void ggml_ncp_vec_cpy_f(const int n, T * y, const T * x) {
    for (int i = 0; i < n; ++i) {
        y[i]  = x[i];
    }
}

template<typename T>
inline static void ggml_ncp_vec_max_f(const int n, float * s, const T * x) {
    float max = -INFINITY;
    for (int i = 0; i < n; ++i) {
        static_assert(std::is_same<T, float>::value || std::is_same<T, ggml_fp16_t>::value);
        if constexpr (std::is_same<T, float>::value) {
            max = MAX(max, x[i]);
        }
        else if constexpr (std::is_same<T, ggml_fp16_t>::value) {
            max = MAX(max, GGML_COMPUTE_FP16_TO_FP32(x[i]));
        }
    }
    *s = max;
}

template<typename T>
inline static void ggml_ncp_vec_mad_f(const int n, T * y, const T * x, const float v) {
    for (int i = 0; i < n; ++i) {
        static_assert(std::is_same<T, float>::value || std::is_same<T, ggml_fp16_t>::value);
        if constexpr (std::is_same<T, float>::value) {
            y[i] += x[i]*v;
        }
        else if constexpr (std::is_same<T, ggml_fp16_t>::value) {
            y[i] = GGML_COMPUTE_FP32_TO_FP16(GGML_COMPUTE_FP16_TO_FP32(y[i]) + GGML_COMPUTE_FP16_TO_FP32(x[i])*v);
        }
    }
}

static inline float op_add(float a, float b) {
    return a + b;
}

static inline float op_mul(float a, float b) {
    return a * b;
}

template <float (*op)(float, float), typename src0_t, typename src1_t, typename dst_t>
static inline void vec_binary_op_non_contiguous(const int64_t n, const int64_t ne10, const int64_t nb10, dst_t * z, const src0_t * x, const src1_t * y) {
    for (int i = 0; i < n; i++) {
        int i10 = i % ne10;
        const src1_t * y_ptr = (const src1_t *)((const char *)y + i10*nb10);
        static_assert(
            (std::is_same<src0_t, float>::value && std::is_same<src1_t, float>::value && std::is_same<dst_t, float>::value) ||
            (std::is_same<src0_t, ggml_fp16_t>::value && std::is_same<src1_t, ggml_fp16_t>::value && std::is_same<dst_t, ggml_fp16_t>::value) ||
            (std::is_same<src0_t, ggml_fp16_t>::value && std::is_same<src1_t, float>::value && std::is_same<dst_t, ggml_fp16_t>::value)
        );
        if constexpr (std::is_same<src0_t, float>::value && std::is_same<src1_t, float>::value && std::is_same<dst_t, float>::value) {
            z[i] = op(x[i], *y_ptr);
        }
        else if constexpr (std::is_same<src0_t, ggml_fp16_t>::value && std::is_same<src1_t, ggml_fp16_t>::value && std::is_same<dst_t, ggml_fp16_t>::value) {
            z[i] = GGML_COMPUTE_FP32_TO_FP16(op(GGML_COMPUTE_FP16_TO_FP32(x[i]), GGML_COMPUTE_FP16_TO_FP32(*y_ptr)));
        }
        else if constexpr (std::is_same<src0_t, ggml_fp16_t>::value && std::is_same<src1_t, float>::value && std::is_same<dst_t, ggml_fp16_t>::value) {
            z[i] = GGML_COMPUTE_FP32_TO_FP16(op(GGML_COMPUTE_FP16_TO_FP32(x[i]), *y_ptr));
        }
    }
}

template <float (*op)(float, float), typename src0_t, typename src1_t, typename dst_t>
static void apply_binary_op(ggml_ncp & ctx, struct ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT( nb0 == sizeof(dst_t));
    GGML_ASSERT(nb00 == sizeof(src0_t));

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = 0; i01 < ne01; i01++) {
                const int64_t i13 = i03 % ne13;
                const int64_t i12 = i02 % ne12;
                const int64_t i11 = i01 % ne11;

                dst_t        * dst_ptr  = (dst_t  *)       ((char *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
                const src0_t * src0_ptr = (const src0_t *) ((const char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);
                const src1_t * src1_ptr = (const src1_t *) ((const char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11);

                vec_binary_op_non_contiguous<op>(ne0, ne10, nb10, dst_ptr, src0_ptr, src1_ptr);
            }
        }
    }

    GGML_UNUSED(ctx);
}

template <float (*op)(float, float)>
static void binary_op(ggml_ncp & ctx, struct ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    /*  */ if (src0->type == GGML_TYPE_F32  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_binary_op<op, float, float, float>(ctx, dst);
    } else if (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_binary_op<op, ggml_fp16_t, ggml_fp16_t, ggml_fp16_t>(ctx, dst);
    } else if (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F16) {
        apply_binary_op<op, ggml_fp16_t, float, ggml_fp16_t>(ctx, dst);
    } else {
        GGML_ABORT("%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type));
    }
}

// kernels

// ggml_ncp_compute_forward_add

static void ggml_ncp_compute_forward_add(ggml_ncp & ctx, struct ggml_tensor * dst) {
    binary_op<op_add>(ctx, dst);
}

// ggml_ncp_compute_forward_mul

static void ggml_ncp_compute_forward_mul(ggml_ncp & ctx, struct ggml_tensor * dst) {
    binary_op<op_mul>(ctx, dst);
}

// ggml_ncp_compute_forward_rms_norm

template<typename T>
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
                const T * x = (T *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);

                double sum = 0.0;
                for (int64_t i00 = 0; i00 < ne00; i00++) {
                    static_assert(std::is_same<T, float>::value || std::is_same<T, ggml_fp16_t>::value);
                    if constexpr (std::is_same<T, float>::value) {
                        sum += (double)(x[i00] * x[i00]);
                    }
                    else if constexpr (std::is_same<T, ggml_fp16_t>::value) {
                        sum += (double)(GGML_COMPUTE_FP16_TO_FP32(x[i00]) * GGML_COMPUTE_FP16_TO_FP32(x[i00]));
                    }
                }

                const float mean = sum/ne00;
                const float scale = 1.0f/sqrtf(mean + eps);
                assert(scale > 0.0f);

                // TODO(sam): support FP32 -> FP16
                T * y = (T *) ((char *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);
                memcpy(y, x, ne00 * sizeof(T));
                ggml_ncp_vec_scale_f<T>(ne00, y, scale);
            }
        }
    }

    GGML_UNUSED(ctx);
}

static void ggml_ncp_compute_forward_rms_norm(ggml_ncp & ctx, struct ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            ggml_ncp_compute_forward_rms_norm_f<float>(ctx, dst);
            break;
        case GGML_TYPE_F16:
            ggml_ncp_compute_forward_rms_norm_f<ggml_fp16_t>(ctx, dst);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

// ggml_ncp_compute_forward_mul_mat

static void ggml_ncp_compute_forward_mul_mat(ggml_ncp & ctx, struct ggml_tensor * dst) {
    /*
        Supported format
        - src0: Q4_0, F16
        - src1: F32, F16
    */
    
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows
    
    GGML_ASSERT(ne00 == ne10);
    const int64_t nc = ne00; // input channel

    GGML_ASSERT(nc % 32 == 0);

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            for (int64_t i1 = 0; i1 < ne1; i1++) {
                for (int64_t i0 = 0; i0 < ne0; i0++) {
                    const int64_t i03 = i3 / (ne13 / ne03);
                    const int64_t i02 = i2 / (ne12 / ne02);
                    const int64_t i01 = i0;

                    const int64_t i13 = i3;
                    const int64_t i12 = i2;
                    const int64_t i11 = i1;
                    
                    double sumf = 0.0;
                    for (int64_t i = 0; i < nc; i += 32) {
                        float src0_f32[32];
                        GGML_ASSERT(src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_F16);
                        const ggml_type_traits * src0_traits = ggml_get_type_traits(src0->type);
                        GGML_ASSERT(32 % src0_traits->blck_size == 0);
                        if (src0->type == GGML_TYPE_Q4_0) {
                            // Q4_0 data is repacked
                            const block_q4_0 * src0_ptr = (const block_q4_0 *)((const char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01 + ((i/src0_traits->blck_size)*nb00));
                            const float d = GGML_FP16_TO_FP32(src0_ptr->d);
                            for (int j = 0; j < QK4_0/2; ++j) {
                                // interpret as signed int4
                                int x0 = (src0_ptr->qs[j] & 0x0F);
                                int x1 = (src0_ptr->qs[j] >>   4);
                                x0 = (x0 & 0x8) ? x0 - 16 : x0;
                                x1 = (x1 & 0x8) ? x1 - 16 : x1;
                                src0_f32[j + 0      ] = x0*d;
                                src0_f32[j + QK4_0/2] = x1*d;
                            }
                        }
                        else {
                            for (int64_t j = 0; j < 32; j += src0_traits->blck_size) {
                                const char * src0_ptr = (const char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01 + ((i+j)/src0_traits->blck_size)*nb00;
                                src0_traits->to_float(src0_ptr, src0_f32 + j, src0_traits->blck_size);
                            }                            
                        }

                        float src1_f32[32];
                        const char * src1_ptr = (const char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + i*nb10;
                        const ggml_type_traits * src1_traits = ggml_get_type_traits(src1->type);
                        if (src1->type == GGML_TYPE_F32) {
                            memcpy(src1_f32, src1_ptr, 32 * src1_traits->type_size);
                        }
                        else if (src1->type == GGML_TYPE_F16) {
                            src1_traits->to_float(src1_ptr, src1_f32, 32);
                        }
                        else {
                            GGML_ABORT("fatal error");
                        }

                        for (int k = 0; k < 32; ++k) {
                            sumf += (double)(src0_f32[k]*src1_f32[k]);
                        }
                    }

                    if (dst->type == GGML_TYPE_F32) {
                        float * dst_ptr = (float *) ((char *) dst->data + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);
                        *dst_ptr = sumf;
                    }
                    else if (dst->type == GGML_TYPE_F16) {
                        ggml_fp16_t * dst_ptr = (ggml_fp16_t *) ((char *) dst->data + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);
                        *dst_ptr = GGML_COMPUTE_FP32_TO_FP16((float)sumf);
                    }
                    else {
                        GGML_ABORT("fatal error");
                    }
                }
            }
        }
    }
    GGML_UNUSED(ctx);
}

// ggml_ncp_compute_forward_cpy

template <typename T>
void ggml_ncp_compute_forward_cpy_f(ggml_ncp & ctx, struct ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(src0) == ggml_nelements(dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    ggml_fp16_t * dst_ptr = (ggml_fp16_t *) dst->data;
    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = 0; i01 < ne01; i01++) {
                for (int64_t i00 = 0; i00 < ne00; i00++) {
                    T * src_ptr = (T *) ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                    static_assert(std::is_same<T, float>::value || std::is_same<T, ggml_fp16_t>::value);
                    if constexpr (std::is_same<T, float>::value) {
                        *dst_ptr++ = GGML_COMPUTE_FP32_TO_FP16(*src_ptr);
                    }
                    else if constexpr (std::is_same<T, ggml_fp16_t>::value) {
                        *dst_ptr++ = *src_ptr;
                    }
                }
            }
        }
    }

    GGML_UNUSED(ctx);
}

static void ggml_ncp_compute_forward_cpy(ggml_ncp & ctx, struct ggml_tensor * dst) {

    GGML_ASSERT(ggml_is_contiguous(dst) && dst->type == GGML_TYPE_F16);

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            ggml_ncp_compute_forward_cpy_f<float>(ctx, dst);
            break;
        case GGML_TYPE_F16:
            ggml_ncp_compute_forward_cpy_f<ggml_fp16_t>(ctx, dst);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

// ggml_ncp_compute_forward_cont

template <typename T>
void ggml_ncp_compute_forward_cont_f(ggml_ncp & ctx, struct ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(src0) == ggml_nelements(dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    T * dst_ptr = (T *) dst->data;
    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = 0; i01 < ne01; i01++) {
                for (int64_t i00 = 0; i00 < ne00; i00++) {
                    T * src_ptr = (T *) ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                    *dst_ptr++ = *src_ptr;
                }
            }
        }
    }
    
    GGML_UNUSED(ctx);
}

static void ggml_ncp_compute_forward_cont(ggml_ncp & ctx, struct ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            ggml_ncp_compute_forward_cont_f<float>(ctx, dst);
            break;
        case GGML_TYPE_F16:
            ggml_ncp_compute_forward_cont_f<ggml_fp16_t>(ctx, dst);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

// ggml_ncp_compute_forward_get_rows

template<typename T>
static void ggml_ncp_compute_forward_get_rows_f(ggml_ncp & ctx, struct ggml_tensor * dst) {
    // TODO(sam): currently support in a general form, while it may be possible to support only when nr == 1
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t nc = ne00;
    const int64_t nr = ggml_nelements(src1);

    assert(ne0  == nc);
    assert(ne02 == ne11);
    assert(ne13 == 1);
    assert(nb00 == sizeof(T));
    assert(ggml_nrows(dst) == nr);

    for (int64_t i10 = 0; i10 < ne10; i10++) {
        for (int64_t i11 = 0; i11 < ne11; i11++) {
            for (int64_t i12 = 0; i12 < ne12; i12++) {
                const int64_t i01 = *(int32_t *) ((char *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

                GGML_ASSERT(i01 >= 0 && i01 < ne01);

                ggml_ncp_vec_cpy_f<T>(nc,
                        (T *) ((char *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3),
                        (T *) ((char *) src0->data + i01*nb01 + i11*nb02 + i12*nb03));
            }
        }
    }
    
    GGML_UNUSED(ctx);
}

static void ggml_ncp_compute_forward_get_rows(ggml_ncp & ctx, struct ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            ggml_ncp_compute_forward_get_rows_f<float>(ctx, dst);
            break;
        case GGML_TYPE_F16:
            ggml_ncp_compute_forward_get_rows_f<ggml_fp16_t>(ctx, dst);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

// ggml_ncp_compute_forward_set_rows

template <typename src0_t, typename idx_t>
static void ggml_ncp_compute_forward_set_rows_f(ggml_ncp & ctx, struct ggml_tensor * dst) {
    // f32, i64, f16
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t nc = ne00;

    assert(ne0  == nc);
    assert(ne2  == ne02);
    assert(ne3  == ne03);
    assert(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    assert(ne02 % ne11 == 0);
    assert(ne03 % ne12 == 0);

    GGML_ASSERT(dst->type == GGML_TYPE_F16);

    for (int64_t i03 = 0; i03 < ne03; ++i03) {
        for (int64_t i02 = 0; i02 < ne02; ++i02) {
            for (int64_t i = 0; i < ne01; ++i) {
                const int64_t i12 = i03%ne12;
                const int64_t i11 = i02%ne11;
                const int64_t i10 = i;

                const int64_t i1 = *(idx_t *) ((char *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

                GGML_ASSERT(i1 >= 0 && i1 < ne1);
                
                src0_t      * src0_ptr = (src0_t      *) ((char *) src0->data +  i*nb01 + i02*nb02 + i03*nb03);
                ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((char *) dst->data  + i1*nb1  + i02*nb2  + i03*nb3);
                for (int64_t j = 0; j < nc; j++) {
                    static_assert(std::is_same<src0_t, float>::value || std::is_same<src0_t, ggml_fp16_t>::value);
                    if constexpr (std::is_same<src0_t, float>::value) {
                        dst_ptr[j] = GGML_COMPUTE_FP32_TO_FP16(src0_ptr[j]);
                    }
                    else if constexpr (std::is_same<src0_t, ggml_fp16_t>::value) {
                        dst_ptr[j] = src0_ptr[j];
                    }
                }
            }
        }
    }

    GGML_UNUSED(ctx);
}

static void ggml_ncp_compute_forward_set_rows(ggml_ncp & ctx, struct ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    switch (src0->type) {
        case GGML_TYPE_F32:
            if (src1->type == GGML_TYPE_I64) {
                ggml_ncp_compute_forward_set_rows_f<float, int64_t>(ctx, dst);
            }
            else {
                GGML_ABORT("src1->type = %d (%s) not supported", src1->type, ggml_type_name(src1->type));
            }
            break;
        case GGML_TYPE_F16:
            if (src1->type == GGML_TYPE_I64) {
                ggml_ncp_compute_forward_set_rows_f<ggml_fp16_t, int64_t>(ctx, dst);
            }
            else {
                GGML_ABORT("src1->type = %d (%s) not supported", src1->type, ggml_type_name(src1->type));
            }
            break;
        default:
            GGML_ABORT("src0->type = %d (%s) not supported", src0->type, ggml_type_name(src0->type));
    }
}

// ggml_ncp_compute_forward_soft_max

template <typename T>
static double ggml_vec_soft_max_f(const int n, T * y, const T * x, float max) {
    double sum = 0;
    for (int i = 0; i < n; ++i) {
        static_assert(std::is_same<T, float>::value || std::is_same<T, ggml_fp16_t>::value);
        if constexpr (std::is_same<T, float>::value) {
            float val = expf(x[i] - max);
            sum += (double)val;
            y[i] = val;
        }
        else if constexpr (std::is_same<T, ggml_fp16_t>::value) {
            float val = expf(GGML_COMPUTE_FP16_TO_FP32(x[i]) - max);
            sum += (double)val;
            y[i] = GGML_COMPUTE_FP32_TO_FP16(val);
        }
    }
    return sum;
}

template <typename T>
static void ggml_ncp_compute_forward_soft_max_f(ggml_ncp & ctx, struct ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    assert(ggml_is_contiguous(dst));
    assert(ggml_are_same_shape(src0, dst));

    float scale    = 1.0f;
    float max_bias = 0.0f;

    memcpy(&scale,    (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (float *) dst->op_params + 1, sizeof(float));

    GGML_ASSERT(max_bias == 0.0f);

    GGML_TENSOR_UNARY_OP_LOCALS

    const int64_t nb11 = src1 ? src1->nb[1] : 1;
    const int64_t nb12 = src1 ? src1->nb[2] : 1;
    const int64_t nb13 = src1 ? src1->nb[3] : 1;

    const int64_t ne12 = src1 ? src1->ne[2] : 1;
    const int64_t ne13 = src1 ? src1->ne[3] : 1;

    GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F32);

    // sinks
    GGML_ASSERT(!src2);

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = 0; i01 < ne01; i01++) {
                const int64_t i11 = i01;
                const int64_t i12 = i02%ne12;
                const int64_t i13 = i03%ne13;

                T * sp = (T *)((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                T * dp = (T *)((char *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3);

                // broadcast the mask across rows
                float       * mp_f32 = src1 ? (float       *)((char *) src1->data + i11*nb11 + i12*nb12 + i13*nb13) : NULL;

                GGML_ASSERT(ne00 <= 8192);
                T wp[8192] = {};
                ggml_ncp_vec_cpy_f<T>  (ne00, wp, sp);
                ggml_ncp_vec_scale_f<T>(ne00, wp, scale);
                if (mp_f32) {
                    for (int i = 0; i < ne00; ++i) {
                        static_assert(std::is_same<T, float>::value || std::is_same<T, ggml_fp16_t>::value);
                        if constexpr (std::is_same<T, float>::value) {
                            wp[i] += mp_f32[i];
                        }
                        else if constexpr (std::is_same<T, ggml_fp16_t>::value) {
                            wp[i] = GGML_COMPUTE_FP32_TO_FP16(GGML_COMPUTE_FP16_TO_FP32(wp[i]) + mp_f32[i]);
                        }
                    }
                }

                float max = -INFINITY;
                ggml_ncp_vec_max_f<T>(ne00, &max, wp);

                double sum = ggml_vec_soft_max_f<T>(ne00, dp, wp, max);
                assert(sum > 0.0);

                sum = 1.0/sum;
                ggml_ncp_vec_scale_f<T>(ne00, dp, sum);
            }
        }
    }

    GGML_UNUSED(ctx);
}

static void ggml_ncp_compute_forward_soft_max(ggml_ncp & ctx, struct ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F16:
            ggml_ncp_compute_forward_soft_max_f<ggml_fp16_t>(ctx, dst);
            break;
        case GGML_TYPE_F32:
            ggml_ncp_compute_forward_soft_max_f<float>(ctx, dst);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

// ggml_ncp_compute_forward_rope

static void ggml_rope_cache_init(float theta_base, int64_t ne0, float * cache, float theta_scale) {
    float theta = theta_base;
    for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        cache[i0 + 0] = cosf(theta);
        cache[i0 + 1] = sinf(theta);
        theta *= theta_scale;
    }
}

static void ggml_mrope_cache_init(
     float theta_base_t, float theta_base_h, float theta_base_w, float theta_base_e, int sections[4], bool is_imrope, bool indep_sects,
     int64_t ne0, float * cache, float theta_scale) {
    float theta_t = theta_base_t;
    float theta_h = theta_base_h;
    float theta_w = theta_base_w;
    float theta_e = theta_base_e;  // extra position id for vision encoder
    int sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    int sec_w = sections[1] + sections[0];
    int sec_e = sections[2] + sec_w;
    GGML_ASSERT(sect_dims <= ne0);

    for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        int sector = (i0 / 2) % sect_dims;
        if (indep_sects) {
            // compute theta independently for each dim sections
            // (i.e. reset corresponding theta when `i0` go from one section to another)
            if (sector == 0) {
                theta_t = theta_base_t;
            }
            else if (sector == sections[0]) {
                theta_h = theta_base_h;;
            }
            else if (sector == sec_w) {
                theta_w = theta_base_w;
            }
            else if (sector == sec_e) {
                theta_e = theta_base_e;
            }
        }

        float theta = theta_t;
        if (is_imrope) { // qwen3vl apply interleaved mrope
            if (sector % 3 == 1 && sector < 3 * sections[1]) {
                theta = theta_h;
            } else if (sector % 3 == 2 && sector < 3 * sections[2]) {
                theta = theta_w;
            } else if (sector % 3 == 0 && sector < 3 * sections[0]) {
                theta = theta_t;
            } else {
                theta = theta_e;
            }
        } else {
            if (sector >= sections[0] && sector < sec_w) {
                theta = theta_h;
            }
            else if (sector >= sec_w && sector < sec_w + sections[2]) {
                theta = theta_w;
            }
            else if (sector >= sec_w + sections[2]) {
                theta = theta_e;
            }
        }

        cache[i0 + 0] = cosf(theta);
        cache[i0 + 1] = sinf(theta);

        theta_t *= theta_scale;
        theta_w *= theta_scale;
        theta_h *= theta_scale;
        theta_e *= theta_scale;
    }
}

template<typename T>
static void rotate_pairs(const int64_t n, const int64_t n_offset, const float * cache, const T * src_data, T * dst_data, const int scale = 2) {
  for (int64_t i0 = 0; i0 < n; i0 += 2) {
    const int64_t ic = i0/scale; // hack for GGML_ROPE_TYPE_NORMAL, where we need ic = i0; for all other cases, ic = i0/2

    const float cos_theta = cache[i0 + 0];
    const float sin_theta = cache[i0 + 1];

    const T * const src = src_data + ic;
    T * dst             = dst_data + ic;

    static_assert(std::is_same<T, float>::value || std::is_same<T, ggml_fp16_t>::value);

    float x0 = 0.0f;
    if constexpr (std::is_same<T, float>::value) {
        x0 = src[0];
    }
    else if constexpr (std::is_same<T, ggml_fp16_t>::value) {
        x0 = GGML_COMPUTE_FP16_TO_FP32(src[0]);
    }

    float x1 = 0.0f;
    if constexpr (std::is_same<T, float>::value) {
        x1 = src[n_offset];
    }
    else if constexpr (std::is_same<T, ggml_fp16_t>::value) {
        x1 = GGML_COMPUTE_FP16_TO_FP32(src[n_offset]);
    }

    if constexpr (std::is_same<T, float>::value) {
        dst[0]        = x0*cos_theta - x1*sin_theta;
        dst[n_offset] = x0*sin_theta + x1*cos_theta;
    }
    else if constexpr (std::is_same<T, ggml_fp16_t>::value) {
        dst[0]        = GGML_COMPUTE_FP32_TO_FP16(x0*cos_theta - x1*sin_theta);
        dst[n_offset] = GGML_COMPUTE_FP32_TO_FP16(x0*sin_theta + x1*cos_theta);
    }
  }
}

template<typename T>
static void ggml_ncp_compute_forward_rope_flt(ggml_ncp & ctx, ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(!src2);

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);

    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    int sections[4];

    //const int n_past     = ((int32_t *) dst->op_params)[0];
    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    //const int n_ctx      = ((int32_t *) dst->op_params)[3];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];

    GGML_UNUSED(n_ctx_orig);

    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));
    memcpy(&sections,    (int32_t *) dst->op_params + 11, sizeof(int)*4);

    GGML_ASSERT(freq_scale == 1.0f);
    GGML_ASSERT(ext_factor == 0.0f);
    GGML_ASSERT(attn_factor == 1.0f);
    GGML_ASSERT(beta_fast == 32.0f);
    GGML_ASSERT(beta_slow == 1.0f);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(nb0 == nb00);
    GGML_ASSERT(nb0 == sizeof(T));

    GGML_ASSERT(n_dims <= ne0);
    GGML_ASSERT(n_dims % 2 == 0);

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    const bool is_imrope = mode == GGML_ROPE_TYPE_IMROPE; // qwen3vl apply interleaved mrope
    const bool mrope_used = mode & GGML_ROPE_TYPE_MROPE;  // ggml_rope_multi, note: also true for vision (24 & 8 == true) and for imrope
    const bool is_vision = mode == GGML_ROPE_TYPE_VISION;

    if (mrope_used) {
        GGML_ASSERT(sections[0] > 0 || sections[1] > 0 || sections[2] > 0);
    }

    if (is_vision) {
        GGML_ASSERT(n_dims == ne0/2);
    }

    const int32_t * pos = (const int32_t *) src1->data;

    int64_t last_i2 = -1;

    GGML_ASSERT(ne0 <= 1024);

    float cache[1024]{ };
    for (int64_t i3 = 0; i3 < ne3; i3++) { // batch
        for (int64_t i2 = 0; i2 < ne2; i2++) { // seq-len
            for (int64_t i1 = 0; i1 < ne1; i1++) { // attn-heads
                if (last_i2 != i2) {
                    if (!mrope_used) {
                        const int64_t p = pos[i2];
                        ggml_rope_cache_init(p, ne0, cache, theta_scale);
                    }
                    else {
                        const int64_t p_t = pos[i2];
                        const int64_t p_h = pos[i2 + ne2];
                        const int64_t p_w = pos[i2 + ne2 * 2];
                        const int64_t p_e = pos[i2 + ne2 * 3];
                        ggml_mrope_cache_init(
                            p_t, p_h, p_w, p_e, sections, is_imrope, is_vision,
                            ne0, cache, theta_scale);
                    }

                    last_i2 = i2;
                }

                T * src = (T *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
                T * dst_data  = (T *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1);

                switch (mode) {
                    /*
                    case GGML_ROPE_TYPE_NORMAL:
                        rotate_pairs<T>(n_dims, 1, cache, src, dst_data, 1);
                        break;
                    */
                    case GGML_ROPE_TYPE_NEOX:
                    /*
                    case GGML_ROPE_TYPE_MROPE:
                    */
                    case GGML_ROPE_TYPE_IMROPE:
                        rotate_pairs<T>(n_dims, n_dims/2, cache, src, dst_data);
                        break;
                    case GGML_ROPE_TYPE_VISION:
                        rotate_pairs<T>(ne0, n_dims, cache, src, dst_data);
                        break;
                    default:
                        GGML_ABORT("rope type not supported");
                }

                if (!is_vision) {
                    bool reached = false;
                    // fill the remain channels with data from src tensor
                    for (int64_t i0 = n_dims; i0 < ne0; i0 += 2) {
                        const T * const src = (T *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                        T * dst_data  = (T *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                        dst_data[0] = src[0];
                        dst_data[1] = src[1];
                        reached = true;
                    }
                    if (reached) {
                        GGML_ABORT("fatal error");
                    }
                }
            } //attn-heads
        }
    }

    GGML_UNUSED(ctx);
}

static void ggml_ncp_compute_forward_rope(ggml_ncp & ctx, struct ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F16:
            ggml_ncp_compute_forward_rope_flt<ggml_fp16_t>(ctx, dst);
            break;
        case GGML_TYPE_F32:
            ggml_ncp_compute_forward_rope_flt<float>(ctx, dst);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

// ggml_ncp_compute_forward_flash_attn_ext

static void ggml_compute_forward_flash_attn_ext_f16_one_chunk(ggml_tensor * dst, int ir0, int ir1, int64_t ic_start, int64_t ic_end) {
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    const int64_t DK = nek0;
    const int64_t DV = nev0;
    const int64_t N  = neq1;

    GGML_ASSERT(ne0 == DV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_size(q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(k->type));
    GGML_ASSERT(nbv0 == ggml_type_size(v->type));

    GGML_ASSERT(neq0 == DK);
    GGML_ASSERT(nek0 == DK);
    GGML_ASSERT(nev0 == DV);

    GGML_ASSERT(neq1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    const int64_t rk2 = neq2/nek2;
    const int64_t rk3 = neq3/nek3;

    const int64_t rv2 = neq2/nev2;
    const int64_t rv3 = neq3/nev3;

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (float *) dst->op_params + 2, sizeof(float));

    GGML_ASSERT(max_bias == 0.0f);
    GGML_ASSERT(logit_softcap == 0.0f);

    GGML_ASSERT(!sinks);
    GGML_ASSERT(k->type == GGML_TYPE_F16);
    GGML_ASSERT(v->type == GGML_TYPE_F16);

    GGML_ASSERT(DK < 1024);
    GGML_ASSERT(DV < 1024);
    GGML_ASSERT(nb1 < 1024 * sizeof(float));
    float       VKQ32[1024] = { 0 }; // FP32 VKQ accumulator
    ggml_fp16_t VKQ16[1024] = { 0 }; // (temporary) FP16 VKQ accumulator
    ggml_fp16_t Q_q  [1024] = { 0 }; // (temporary) buffer for Q converted to quantized/FP16

    for (int ir = ir0; ir < ir1; ++ir) {
        // q indices
        const int iq3 = ir/(neq2*neq1);
        const int iq2 = (ir - iq3*neq2*neq1)/neq1;
        const int iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        float S = 0.0f;      // sum
        float M = -INFINITY; // maximum KQ value

        memset(VKQ16, 0, DV*sizeof(ggml_fp16_t));

        const ggml_fp16_t * mp = mask ? (ggml_fp16_t *)((char *) mask->data + iq1*mask->nb[1] + (iq2%mask->ne[2])*mask->nb[2] + (iq3%mask->ne[3])*mask->nb[3]) : NULL;

        // k indices
        const int ik3 = iq3 / rk3;
        const int ik2 = iq2 / rk2;

        // v indices
        const int iv3 = iq3 / rv3;
        const int iv2 = iq2 / rv2;

        const float * pq = (const float *) ((char *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3));
        for (int64_t i = 0; i < DK; ++i) {
            Q_q[i] = GGML_COMPUTE_FP32_TO_FP16(pq[i]);
        }

        // online softmax / attention
        // loop over n_kv and n_head_kv
        // ref: https://arxiv.org/pdf/2112.05682.pdf

        for (int64_t ic = ic_start; ic < ic_end; ++ic) {
            const float mv = mp ? GGML_COMPUTE_FP16_TO_FP32(mp[ic]) : 0.0f;
            if (mv == -INFINITY) {
                continue;
            }

            const char * k_data = (const char *) k->data + ( ic*nbk1 + ik2*nbk2 + ik3*nbk3);
            double sumf = 0.0;

            for (int i = 0; i < DK; ++i) {
                sumf += (double)(GGML_COMPUTE_FP16_TO_FP32(*((const ggml_fp16_t * )k_data + i))*GGML_COMPUTE_FP16_TO_FP32(Q_q[i]));
            }

            float s = sumf; // KQ value

            s = s*scale; // scale KQ value

            s += mv; // apply mask

            const float Mold = M;

            float ms = 1.0f; // upon new higher max val, scale VKQ and KQ sum with this value
            float vs = 1.0f; // post-softmax KQ value, expf(s - M)

            const char * v_data = ((const char *) v->data + (ic*nbv1 + iv2*nbv2 + iv3*nbv3));

            if (s > M) {
                // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                M = s;
                ms = expf(Mold - M);

                // V = V*expf(Mold - M)
                ggml_ncp_vec_scale_f<ggml_fp16_t>(DV, VKQ16, ms);
            } else {
                // no new maximum, ms == 1.0f, vs != 1.0f
                vs = expf(s - M);
            }

            // V += v*expf(s - M)
            ggml_ncp_vec_mad_f<ggml_fp16_t>(DV, VKQ16, (const ggml_fp16_t *) v_data, vs);

            S = S*ms + vs; // scale and increment sum with partial sum
        }

        for (int64_t d = 0; d < DV; ++d) {
            VKQ32[d] = GGML_COMPUTE_FP16_TO_FP32(VKQ16[d]);
        }

        // V /= S
        const float S_inv = S == 0.0f ? 0.0f : 1.0f/S;
        ggml_ncp_vec_scale_f<float>(DV, VKQ32, S_inv);

        // dst indices
        const int i1 = iq1;
        const int i2 = iq2;
        const int i3 = iq3;

        // permute(0, 2, 1, 3)
        memcpy((char *) dst->data + (i3*ne2*ne1 + i2 + i1*ne1)*nb1, VKQ32, nb1);
    }
}

static void ggml_ncp_compute_forward_flash_attn_ext_f16(ggml_ncp & ctx, ggml_tensor * dst) {

    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    const int64_t DK = nek0;
    const int64_t DV = nev0;
    const int64_t N  = neq1;


    GGML_ASSERT(ne0 == DV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_size(q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(k->type));
    GGML_ASSERT(nbv0 == ggml_type_size(v->type));

    GGML_ASSERT(neq0 == DK);
    GGML_ASSERT(nek0 == DK);
    GGML_ASSERT(nev0 == DV);

    GGML_ASSERT(neq1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // total rows in q
    const int64_t nr = neq1*neq2*neq3;

    ggml_compute_forward_flash_attn_ext_f16_one_chunk(dst, 0, nr, 0, nek1);

    GGML_UNUSED(ctx);
}

static void ggml_ncp_compute_forward_flash_attn_ext(ggml_ncp & ctx, ggml_tensor * dst) {
    switch (dst->op_params[3]) {
        case GGML_PREC_DEFAULT:
        case GGML_PREC_F32:
            // uses F32 accumulators
            ggml_ncp_compute_forward_flash_attn_ext_f16(ctx, dst);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

// ggml_ncp_compute_forward_glu

inline static float ggml_ncp_silu_f32(float x) {
    return x/(1.0f + expf(-x));
}

inline static float ggml_ncp_swiglu_f32(float a, float b) {
    return ggml_ncp_silu_f32(a) * b;
}

template <typename T>
static void ggml_ncp_compute_forward_swiglu_f(ggml_ncp & ctx, struct ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    GGML_ASSERT(src1);

    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(src0->type == src1->type);

    const int nc = src0->ne[0];
    const int nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    const int32_t swapped = ggml_get_op_params_i32(dst, 1);
    GGML_ASSERT(!swapped);

    GGML_TENSOR_BINARY_OP_LOCALS

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = 0; i01 < ne01; i01++) {
                for (int i00 = 0; i00 < ne00; i00++) {
                    T       * dst_ptr  = (T  *)      ((char *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1  + i00*nb0 );
                    const T * src0_ptr = (const T *) ((const char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);
                    const T * src1_ptr = (const T *) ((const char *) src1->data + i03*nb13 + i02*nb12 + i01*nb11 + i00*nb10);
                    static_assert(std::is_same<T, float>::value || std::is_same<T, ggml_fp16_t>::value);
                    if constexpr (std::is_same<T, float>::value) {
                        *dst_ptr = ggml_ncp_swiglu_f32(*src0_ptr, *src1_ptr);
                    }
                    else if constexpr (std::is_same<T, ggml_fp16_t>::value) {
                        *dst_ptr = GGML_COMPUTE_FP32_TO_FP16(ggml_ncp_swiglu_f32(GGML_COMPUTE_FP16_TO_FP32(*src0_ptr), GGML_COMPUTE_FP16_TO_FP32(*src1_ptr)));
                    }
                }
            }
        }
    }

    GGML_UNUSED(ctx);
}

static void ggml_ncp_compute_forward_swiglu(ggml_ncp & ctx, struct ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            ggml_ncp_compute_forward_swiglu_f<float>(ctx, dst);
            break;
        case GGML_TYPE_F16:
            ggml_ncp_compute_forward_swiglu_f<ggml_fp16_t>(ctx, dst);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

static void ggml_ncp_compute_forward_glu(ggml_ncp & ctx, struct ggml_tensor * dst) {

    const ggml_glu_op op = ggml_get_glu_op(dst);

    switch (op) {
        case GGML_GLU_OP_SWIGLU:
            ggml_ncp_compute_forward_swiglu(ctx, dst);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

static bool ggml_ncp_compute_forward(ggml_ncp & ctx, struct ggml_tensor * dst) {

    switch (dst->op) {
        case GGML_OP_ADD:
            ggml_ncp_compute_forward_add(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_ncp_compute_forward_mul(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_ncp_compute_forward_rms_norm(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            ggml_ncp_compute_forward_mul_mat(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggml_ncp_compute_forward_cpy(ctx, dst);
            break;
        case GGML_OP_CONT:
            ggml_ncp_compute_forward_cont(ctx, dst);
            break;
        case GGML_OP_RESHAPE:
            // nop
            break;
        case GGML_OP_VIEW:
            // nop
            break;
        case GGML_OP_PERMUTE:
            // nop
            break;
        case GGML_OP_GET_ROWS:
            ggml_ncp_compute_forward_get_rows(ctx, dst);
            break;
        case GGML_OP_SET_ROWS:
            ggml_ncp_compute_forward_set_rows(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_ncp_compute_forward_soft_max(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_ncp_compute_forward_rope(ctx, dst);
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_ncp_compute_forward_flash_attn_ext(ctx, dst);
            break;
        case GGML_OP_GLU:
            ggml_ncp_compute_forward_glu(ctx, dst);
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

static ggml_ncp_buffer_layout * ggml_ncp_get_layout(ggml_tensor * node) {
    if (!node || !node->extra) {
        return nullptr;
    }
    return &((ggml_ncp_tensor_extra *)node->extra)->layout;
}

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

static const char * ggml_backend_ncp_tensor_get_layout_name(const struct ggml_tensor * tensor) {
    if (!tensor || !tensor->buffer || !tensor->extra) {
        return NULL;
    }
    
    // TODO(sam): fix
    // if (tensor->buffer->buft->iface.get_name != ggml_backend_ncp_buffer_type_get_name) {
    //     return NULL;
    // }
    
    const ggml_ncp_tensor_extra * extra = (const ggml_ncp_tensor_extra *) tensor->extra;
    switch (extra->layout) {
        case GGML_NCP_LAYOUT_UNINITIALIZED: return "UNINIT";
        case GGML_NCP_LAYOUT_0123:          return "0123";
        case GGML_NCP_LAYOUT_0123_C4N4:     return "0123_C4N4";
        case GGML_NCP_LAYOUT_0213:          return "0213";
        case GGML_NCP_LAYOUT_1023_C16:      return "1023_C16";
        default:                            return "UNKNOWN";
    }
}

static bool common_debug_cb_eval(struct ggml_tensor * t) {    const struct ggml_tensor * src0 = t->src[0];
    const struct ggml_tensor * src1 = t->src[1];
    const struct ggml_tensor * src2 = t->src[2];
    const struct ggml_tensor * src3 = t->src[3];
    const struct ggml_tensor * src4 = t->src[4];

    const bool matches_filter = true;

    auto get_layout_str = [](const ggml_tensor * tensor) {
        std::string layout_str = ggml_backend_buffer_name(tensor->buffer);
        const char * ncp_layout = ggml_backend_ncp_tensor_get_layout_name(tensor);
        layout_str += std::string(" layout ") + (ncp_layout ? ncp_layout : "null");
        return layout_str;
    };

    char t_str[256] = { 0 };
    if (t) {
        snprintf(t_str, sizeof(t_str), "%s{%s}{%s}(%s)(%s)(view_src=%d)", t->name, common_ggml_ne_string(t).c_str(), common_ggml_nb_string(t).c_str(), ggml_type_name(t->type), get_layout_str(t).c_str(), (bool)t->view_src);
    }

    char src0_str[256] = { 0 };
    if (src0) {
        snprintf(src0_str, sizeof(src0_str), "%s{%s}{%s}(%s)(%s)(view_src=%d)", src0->name, common_ggml_ne_string(src0).c_str(), common_ggml_nb_string(src0).c_str(), ggml_type_name(src0->type), get_layout_str(src0).c_str(), (bool)src0->view_src);
    }

    char src1_str[256] = { 0 };
    if (src1) {
        snprintf(src1_str, sizeof(src1_str), "%s{%s}{%s}(%s)(%s)(view_src=%d)", src1->name, common_ggml_ne_string(src1).c_str(), common_ggml_nb_string(src1).c_str(), ggml_type_name(src1->type), get_layout_str(src1).c_str(), (bool)src1->view_src);
    }

    char src2_str[256] = { 0 };
    if (src2) {
        snprintf(src2_str, sizeof(src2_str), "%s{%s}{%s}(%s)(%s)(view_src=%d)", src2->name, common_ggml_ne_string(src2).c_str(), common_ggml_nb_string(src2).c_str(), ggml_type_name(src2->type), get_layout_str(src2).c_str(), (bool)src2->view_src);
    }

    char src3_str[256] = { 0 };
    if (src3) {
        snprintf(src3_str, sizeof(src3_str), "%s{%s}{%s}(%s)(%s)(view_src=%d)", src3->name, common_ggml_ne_string(src3).c_str(), common_ggml_nb_string(src3).c_str(), ggml_type_name(src3->type), get_layout_str(src3).c_str(), (bool)src3->view_src);
    }

    char src4_str[256] = { 0 };
    if (src4) {
        snprintf(src4_str, sizeof(src4_str), "%s{%s}{%s}(%s)(%s)(view_src=%d)", src4->name, common_ggml_ne_string(src4).c_str(), common_ggml_nb_string(src4).c_str(), ggml_type_name(src4->type), get_layout_str(src4).c_str(), (bool)src4->view_src);
    }

    std::string params_str;
    if (
        t->op == GGML_OP_ADD ||
        t->op == GGML_OP_MUL ||
        t->op == GGML_OP_MUL_MAT ||
        t-> op == GGML_OP_CPY ||
        t->op == GGML_OP_CONT ||
        t->op == GGML_OP_RESHAPE ||
        t->op == GGML_OP_TRANSPOSE ||
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
    else if (t->op == GGML_OP_FLASH_ATTN_EXT) {
        float scale         = 1.0f;
        float max_bias      = 0.0f;
        float logit_softcap = 0.0f;
        memcpy(&scale,         (float *) t->op_params + 0, sizeof(float));
        memcpy(&max_bias,      (float *) t->op_params + 1, sizeof(float));
        memcpy(&logit_softcap, (float *) t->op_params + 2, sizeof(float));
        const int32_t acc_precision = ((int32_t *) t->op_params)[3];
        params_str += "scale: "         + std::to_string(scale)         + ";";
        params_str += "max_bias: "      + std::to_string(max_bias)      + ";";
        params_str += "logit_softcap: " + std::to_string(logit_softcap) + ";";
        params_str += "acc_precision: " + std::to_string(acc_precision);
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
        GGML_LOG("%s:%125s = %15s(%125s, %125s, %125s, %125s, %125s, %s)\n",
            __func__,
            t_str,
            ggml_op_desc(t),
            src0_str,
            src1_str,
            src2_str,
            src3_str,
            src4_str,
            params_str.c_str()
        );
    }
    return true;
}

static void ggml_ncp_graph_infer_layout(ggml_ncp_t ctx, struct ggml_cgraph * gf) {
    // TODO(sam): may skip inferring if previous layout is set
    const int n = gf->n_nodes;
    for (int i = 0; i < n; i++) {
        ggml_tensor * node = gf->nodes[i];

        ggml_tensor * src0 = node->src[0];
        ggml_tensor * src1 = node->src[1];
        ggml_tensor * src2 = node->src[2];
        ggml_tensor * src3 = node->src[3];
        ggml_tensor * src4 = node->src[4];

        ggml_ncp_buffer_layout * dst_layout  = ggml_ncp_get_layout(node);
        ggml_ncp_buffer_layout * src0_layout = ggml_ncp_get_layout(src0);
        ggml_ncp_buffer_layout * src1_layout = ggml_ncp_get_layout(src1);
        ggml_ncp_buffer_layout * src2_layout = ggml_ncp_get_layout(src2);
        ggml_ncp_buffer_layout * src3_layout = ggml_ncp_get_layout(src3);
        ggml_ncp_buffer_layout * src4_layout = ggml_ncp_get_layout(src4);

        switch (node->op)
        {
            case GGML_OP_ADD:
            case GGML_OP_MUL:
                {
                    GGML_ASSERT(*src0_layout == GGML_NCP_LAYOUT_0123 || *src0_layout == GGML_NCP_LAYOUT_1023_C16);
                    if (*src1_layout == GGML_NCP_LAYOUT_UNINITIALIZED) {
                        // TODO(sam) check weight layout ?
                        *src1_layout = GGML_NCP_LAYOUT_0123;
                    }
                    GGML_ASSERT(*src1_layout == GGML_NCP_LAYOUT_0123 /* weight */ || *src1_layout == GGML_NCP_LAYOUT_1023_C16) /* feature */;
                    GGML_ASSERT(*src0_layout == *src1_layout);
                    *dst_layout = *src0_layout;
                }
                break;
            case GGML_OP_RMS_NORM:
                {
                    if (*src0_layout == GGML_NCP_LAYOUT_UNINITIALIZED) { // first layer
                        // TODO(sam): input from CPU backend (input feature), should be tranformed to specified layout during transfer
                        *src0_layout = GGML_NCP_LAYOUT_1023_C16;
                    }
                    GGML_ASSERT(*src0_layout == GGML_NCP_LAYOUT_0123 || *src0_layout == GGML_NCP_LAYOUT_1023_C16);
                    *dst_layout = GGML_NCP_LAYOUT_0123;
                }
                break;
            case GGML_OP_MUL_MAT:
                {
                    if (*src0_layout == GGML_NCP_LAYOUT_UNINITIALIZED) {
                        *src0_layout = GGML_NCP_LAYOUT_0123_C4N4;
                    }
                    GGML_ASSERT(*src0_layout == GGML_NCP_LAYOUT_0123_C4N4);
                    
                    // TODO(sam): MUL_MAT + RESHAPE -> QKV project and will specify different layout (to be compatible with rms_norm and kv cache set rows)
                    bool is_followed_by_reshape = false;
                    if (i + 1 < n) {
                        ggml_tensor * next_node = gf->nodes[i + 1];
                        if (next_node->op == GGML_OP_RESHAPE && node == next_node->src[0]) {
                            is_followed_by_reshape = true;
                        }
                    }

                    if (is_followed_by_reshape) {
                        GGML_ASSERT(*src1_layout == GGML_NCP_LAYOUT_0123);
                        *dst_layout = GGML_NCP_LAYOUT_0123;
                    }
                    else {
                        GGML_ASSERT(*src1_layout == GGML_NCP_LAYOUT_0123 || *src1_layout == GGML_NCP_LAYOUT_1023_C16);
                        *dst_layout = GGML_NCP_LAYOUT_1023_C16;
                    }
                }
                break;
            case GGML_OP_CPY:
                {
                    // TODO(sam): input from CPU backend (attention mask)
                    if (*src0_layout == GGML_NCP_LAYOUT_UNINITIALIZED) {
                        *src0_layout = GGML_NCP_LAYOUT_0123;
                    }
                    GGML_ASSERT(*src0_layout == GGML_NCP_LAYOUT_0123);
                    GGML_ASSERT(*dst_layout == *src1_layout);
                    *dst_layout = *src0_layout;
                    GGML_ASSERT(*dst_layout == *src1_layout);
                }
                break;
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
                {
                    GGML_ASSERT(*src0_layout == GGML_NCP_LAYOUT_0123);
                    *dst_layout = *src0_layout;
                }
                break;
            case GGML_OP_PERMUTE:
                {
                    const int32_t axis0 = node->op_params[0];
                    const int32_t axis1 = node->op_params[1];
                    const int32_t axis2 = node->op_params[2];
                    const int32_t axis3 = node->op_params[3];
                    GGML_ASSERT(axis0 == 0 && axis1 == 2 && axis2 == 1 && axis3 == 3);
                    GGML_ASSERT(*src0_layout == GGML_NCP_LAYOUT_0123);
                    *dst_layout = GGML_NCP_LAYOUT_0213;
                }
                break;
            case GGML_OP_GET_ROWS:
                {
                    GGML_ASSERT(ggml_is_vector(node));
                    GGML_ASSERT(*src0_layout == GGML_NCP_LAYOUT_1023_C16);
                    GGML_ASSERT(ggml_is_scalar(src1));
                    if (*src1_layout == GGML_NCP_LAYOUT_UNINITIALIZED) {
                        *src1_layout = GGML_NCP_LAYOUT_0123;
                    }
                    GGML_ASSERT(*src1_layout == GGML_NCP_LAYOUT_0123);
                    *dst_layout = GGML_NCP_LAYOUT_1023_C16;
                }
                break;
            case GGML_OP_SET_ROWS:
                {
                    GGML_ASSERT(*src0_layout == GGML_NCP_LAYOUT_0123);
                    if (*src1_layout == GGML_NCP_LAYOUT_UNINITIALIZED) {
                        *src1_layout = GGML_NCP_LAYOUT_0123;
                    }
                    GGML_ASSERT(*src1_layout == GGML_NCP_LAYOUT_0123);
                    if (*src2_layout == GGML_NCP_LAYOUT_UNINITIALIZED) {
                        *src2_layout = GGML_NCP_LAYOUT_0123;
                    }
                    GGML_ASSERT(*src2_layout == GGML_NCP_LAYOUT_0123);
                    *dst_layout = *src0_layout;
                }
                break;
            case GGML_OP_ROPE:
                {
                    GGML_ASSERT(*src0_layout == GGML_NCP_LAYOUT_0123);
                    if (*src1_layout == GGML_NCP_LAYOUT_UNINITIALIZED) {
                        *src1_layout = GGML_NCP_LAYOUT_0123;
                    }
                    GGML_ASSERT(*src1_layout == GGML_NCP_LAYOUT_0123);
                    *dst_layout = *src0_layout;
                }
                break;
            case GGML_OP_FLASH_ATTN_EXT:
                {
                    GGML_ASSERT(*src0_layout == GGML_NCP_LAYOUT_0213);
                    GGML_ASSERT(*src1_layout == GGML_NCP_LAYOUT_0213);
                    GGML_ASSERT(*src2_layout == GGML_NCP_LAYOUT_0213);
                    if (src3_layout) {
                        GGML_ASSERT(*src3_layout == GGML_NCP_LAYOUT_0123);
                    }
                    GGML_ASSERT(!src4_layout);
                    *dst_layout = GGML_NCP_LAYOUT_0123;
                }
                break;
            case GGML_OP_GLU:
                {
                    GGML_ASSERT(*src0_layout == GGML_NCP_LAYOUT_1023_C16);
                    GGML_ASSERT(*src0_layout == *src1_layout);
                    *dst_layout = *src0_layout;
                }
                break;
            default:
                GGML_ABORT("fatal error: unsupported op in layout inference: %s", ggml_op_name(node->op));
        }

        // TODO(sam) add flag to enable this
        // common_debug_cb_eval(node);
    }

    GGML_UNUSED(ctx);
}

static bool ggml_ncp_can_fuse(const struct ggml_cgraph * gf, int node_idx, std::initializer_list<enum ggml_op> ops) {
    if (!ggml_can_fuse(gf, node_idx, ops)) {
        return false;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        return true;
    }

    return false;
}

enum ggml_status ggml_ncp_graph_compute(ggml_ncp_t ctx, struct ggml_cgraph * gf) {
    // TODO(sam): somewhat similar to graph plan, may change to be compatible with it
    ggml_ncp_graph_infer_layout(ctx, gf);

    for (int i = 0; i < gf->n_nodes; i++) {
        ggml_tensor * node = gf->nodes[i];

        if (ggml_is_empty(node) || ggml_op_is_empty(node->op)) {
            continue;
        }

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        
        // make sure all rms norm is fusible
        if (node->op == GGML_OP_RMS_NORM) {
            GGML_ASSERT(ggml_ncp_can_fuse(gf, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }));
        }
        if (ggml_ncp_can_fuse(gf, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
            // TODO(sam): replace with actual fused kernel
            for (int j = i; j < i+2; j++) {
                ggml_tensor * fused_node = gf->nodes[j];
                bool ok = ggml_ncp_compute_forward(*ctx, fused_node);
                if (!ok) {
                    GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, fused_node->name, ggml_op_name(fused_node->op));
                }
                GGML_ASSERT(ok);
            }
            i++;
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