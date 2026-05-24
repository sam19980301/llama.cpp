#include "ggml-ncp.h"

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include "ggml-ncp-device.h"

#include "ggml-impl.h"

#include <memory>
#include <vector>

#include <cassert>
#include <cstdlib>

static ggml_ncp_device_t ggml_ncp_device_init(int device) {
    ggml_ncp_device_t dev = (ggml_ncp_device_t) calloc(1, sizeof(struct ggml_ncp_device));

    assert(dev != NULL);

    dev->props.device = device;

    dev->props.max_buffer_size = 1 * 1024 * 1024 * 1024; // TODO(sam)

    snprintf(dev->props.name, sizeof(dev->props.name), "%s%d", GGML_NCP_NAME, device);
    snprintf(dev->props.desc, sizeof(dev->props.desc), "%s", GGML_NCP_NAME);

    GGML_LOG_INFO("%s: GPU name:   %s\n", __func__, dev->props.name);
    return dev;
}

static void ggml_ncp_device_free(ggml_ncp_device_t dev) {
    free(dev);
}

struct ggml_ncp_device_deleter {
    void operator()(ggml_ncp_device_t ctx) {
        ggml_ncp_device_free(ctx);
    }
};

typedef std::unique_ptr<ggml_ncp_device, ggml_ncp_device_deleter> ggml_ncp_device_ptr;

ggml_ncp_device_t ggml_ncp_device_get(int device) {
    static std::vector<ggml_ncp_device_ptr> devs;

    devs.emplace_back(ggml_ncp_device_init(device));

    return devs.back().get();
}

void ggml_ncp_device_get_memory(ggml_ncp_device_t dev, size_t * free, size_t * total) {
    *total = dev->props.max_buffer_size;
    *free = *total;
}

bool ggml_ncp_device_supports_op(ggml_ncp_device_t dev, const struct ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_RMS_NORM:
        case GGML_OP_MUL_MAT:
        case GGML_OP_CPY:
        case GGML_OP_CONT:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_GET_ROWS:
        case GGML_OP_SET_ROWS:
        case GGML_OP_ROPE:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_GLU:
            return true;
        default:
            return false;
    }

    GGML_UNUSED(dev);
}

const struct ggml_ncp_device_props * ggml_ncp_device_get_props(ggml_ncp_device_t dev) {
    return &dev->props;
}

// device buffers

ggml_ncp_buffer_t ggml_ncp_buffer_init(ggml_ncp_device_t dev, size_t size) {
    ggml_ncp_buffer_t res = (ggml_ncp_buffer_t)calloc(1, sizeof(struct ggml_ncp_buffer));
    res->data = ggml_aligned_malloc(size);
    res->size = size;
    return res;

    GGML_UNUSED(dev);
}

void ggml_ncp_buffer_free(ggml_ncp_buffer_t buf) {
    ggml_aligned_free(buf->data, buf->size);
    free(buf);
}

void * ggml_ncp_buffer_get_base(ggml_ncp_buffer_t buf) {
    return buf->data;
}

void ggml_ncp_buffer_memset_tensor(ggml_ncp_buffer_t buf, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    memset((char *) tensor->data + offset, value, size);

    GGML_UNUSED(buf);
}

static void ggml_ncp_buffer_repack_q4_0(ggml_tensor * tensor, const void * data, size_t size) {
    GGML_ASSERT(tensor->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(size == ggml_nbytes(tensor));
    GGML_ASSERT(ggml_is_contiguous(tensor));

    memcpy((char *) tensor->data, data, size);

    const int64_t n_blocks = size / sizeof(block_q4_0);
    const block_q4_0 * src = (const block_q4_0 *) data;
          block_q4_0 * dst = (      block_q4_0 *) tensor->data;

    for (int64_t b = 0; b < n_blocks; b++) {
        for (int i = 0; i < QK4_0 / 2; i++) {
            dst[b].qs[i] = src[b].qs[i] ^ 0x88; // excess-8 to signed int4
        }
    }
}

void ggml_ncp_buffer_set_tensor(ggml_ncp_buffer_t buf, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    if (tensor->type == GGML_TYPE_Q4_0) {
        GGML_ASSERT(offset == 0);
        GGML_ASSERT(offset + size == ggml_nbytes(tensor));
        ggml_ncp_buffer_repack_q4_0(tensor, data, size);
        return;
    }

    memcpy((char *) tensor->data + offset, data, size);

    GGML_UNUSED(buf);
}

void ggml_ncp_buffer_get_tensor(ggml_ncp_buffer_t buf, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    if (tensor->type == GGML_TYPE_Q4_0) {
        GGML_ASSERT(0); // TODO(sam): support un-repack for q4_0 data?
    }
    memcpy(data, (const char *) tensor->data + offset, size);

    GGML_UNUSED(buf);
}

void ggml_ncp_buffer_clear(ggml_ncp_buffer_t buf, uint8_t value) {
    memset(buf->data, value, buf->size);
}