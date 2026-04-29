#include "ggml-ncp-context.h"

#include "ggml-impl.h"

#include <iostream>

// utils

template<typename T>
inline static void ggml_ncp_vec_scale_f(const int n, T * y, const float v) {
    // scalar
    for (int i = 0; i < n; ++i) {
        if constexpr (std::is_same<T, float>::value) {
            y[i] *= v;
        }
        else if constexpr (std::is_same<T, ggml_fp16_t>::value) {
            y[i] = GGML_COMPUTE_FP32_TO_FP16(GGML_COMPUTE_FP16_TO_FP32(y[i])*v);
        }
        else {
            static_assert(0);
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
        if constexpr (std::is_same<T, float>::value) {
            max = MAX(max, x[i]);
        }
        else if constexpr (std::is_same<T, ggml_fp16_t>::value) {
            max = MAX(max, GGML_COMPUTE_FP16_TO_FP32(x[i]));
        }
        else {
            static_assert(0);
        }    
    }
    *s = max;
}

template<typename T>
inline static void ggml_ncp_vec_mad_f(const int n, T * y, const T * x, const float v) {
    for (int i = 0; i < n; ++i) {
        if constexpr (std::is_same<T, float>::value) {
            y[i] += x[i]*v;
        }
        else if constexpr (std::is_same<T, ggml_fp16_t>::value) {
            y[i] = GGML_COMPUTE_FP32_TO_FP16(GGML_COMPUTE_FP16_TO_FP32(y[i]) + GGML_COMPUTE_FP16_TO_FP32(x[i])*v);
        }
        else {
            static_assert(0);
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
        if constexpr (std::is_same<src0_t, float>::value && std::is_same<src1_t, float>::value && std::is_same<dst_t, float>::value) {
            z[i] = op(x[i], *y_ptr);
        }
        else if constexpr (std::is_same<src0_t, ggml_fp16_t>::value && std::is_same<src1_t, ggml_fp16_t>::value && std::is_same<dst_t, ggml_fp16_t>::value) {
            z[i] = GGML_COMPUTE_FP32_TO_FP16(op(GGML_COMPUTE_FP16_TO_FP32(x[i]), GGML_COMPUTE_FP16_TO_FP32(*y_ptr)));
        }
        else if constexpr (std::is_same<src0_t, ggml_fp16_t>::value && std::is_same<src1_t, float>::value && std::is_same<dst_t, ggml_fp16_t>::value) {
            z[i] = GGML_COMPUTE_FP32_TO_FP16(op(GGML_COMPUTE_FP16_TO_FP32(x[i]), *y_ptr));
        }
        else {
            static_assert(0);
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
                    if constexpr (std::is_same<T, float>::value) {
                        sum += (double)(x[i00] * x[i00]);
                    }
                    else if constexpr (std::is_same<T, ggml_fp16_t>::value) {
                        sum += (double)(GGML_COMPUTE_FP16_TO_FP32(x[i00]) * GGML_COMPUTE_FP16_TO_FP32(x[i00]));
                    }
                    else {
                        static_assert(0);
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
                        for (int64_t j = 0; j < 32; j += src0_traits->blck_size) {
                            const char * src0_ptr = (const char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01 + ((i+j)/src0_traits->blck_size)*nb00;
                            src0_traits->to_float(src0_ptr, src0_f32 + j, src0_traits->blck_size);
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
                    if constexpr (std::is_same<src0_t, float>::value) {
                        dst_ptr[j] = GGML_COMPUTE_FP32_TO_FP16(src0_ptr[j]);
                    }
                    else if constexpr (std::is_same<src0_t, ggml_fp16_t>::value) {
                        dst_ptr[j] = src0_ptr[j];
                    }
                    else {
                        static_assert(0);
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
        else {
            static_assert(0);
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
                        if constexpr (std::is_same<T, float>::value) {
                            wp[i] += mp_f32[i];
                        }
                        else if constexpr (std::is_same<T, ggml_fp16_t>::value) {
                            wp[i] = GGML_COMPUTE_FP32_TO_FP16(GGML_COMPUTE_FP16_TO_FP32(wp[i]) + mp_f32[i]);
                        }
                        else {
                            static_assert(0);
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

    float x0 = 0.0f;
    if constexpr (std::is_same<T, float>::value) {
        x0 = src[0];
    }
    else if constexpr (std::is_same<T, ggml_fp16_t>::value) {
        x0 = GGML_COMPUTE_FP16_TO_FP32(src[0]);
    }
    else {
        static_assert(0);
    }

    float x1 = 0.0f;
    if constexpr (std::is_same<T, float>::value) {
        x1 = src[n_offset];
    }
    else if constexpr (std::is_same<T, ggml_fp16_t>::value) {
        x1 = GGML_COMPUTE_FP16_TO_FP32(src[n_offset]);
    }
    else {
        static_assert(0);
    }

    if constexpr (std::is_same<T, float>::value) {
        dst[0]        = x0*cos_theta - x1*sin_theta;
        dst[n_offset] = x0*sin_theta + x1*cos_theta;
    }
    else if constexpr (std::is_same<T, ggml_fp16_t>::value) {
        dst[0]        = GGML_COMPUTE_FP32_TO_FP16(x0*cos_theta - x1*sin_theta);
        dst[n_offset] = GGML_COMPUTE_FP32_TO_FP16(x0*sin_theta + x1*cos_theta);
    }
    else {
        static_assert(0);
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
                    if constexpr (std::is_same<T, float>::value) {
                        *dst_ptr = ggml_ncp_swiglu_f32(*src0_ptr, *src1_ptr);
                    }
                    else if constexpr (std::is_same<T, ggml_fp16_t>::value) {
                        *dst_ptr = GGML_COMPUTE_FP32_TO_FP16(ggml_ncp_swiglu_f32(GGML_COMPUTE_FP16_TO_FP32(*src0_ptr), GGML_COMPUTE_FP16_TO_FP32(*src1_ptr)));
                    }
                    else {
                        static_assert(0);
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

enum ggml_status ggml_ncp_graph_compute(ggml_ncp_t ctx, struct ggml_cgraph * gf) {
    // TODO(sam): consider op fusion

    for (int i = 0; i < gf->n_nodes; i++) {
        ggml_tensor * node = gf->nodes[i];

        if (ggml_is_empty(node) || ggml_op_is_empty(node->op)) {
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