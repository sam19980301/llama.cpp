#pragma once

#include "ggml.h"

#include <vector>

// device

struct ggml_ncp_device;
typedef struct ggml_ncp_device * ggml_ncp_device_t;

struct ggml_ncp_device_props {
    int device;
    char name[128];
    char desc[128];

    size_t max_buffer_size;
};

struct ggml_ncp_device {
    struct ggml_ncp_device_props props;
};

ggml_ncp_device_t ggml_ncp_device_get(int device);

void ggml_ncp_device_get_memory(ggml_ncp_device_t dev, size_t * free, size_t * total);
bool ggml_ncp_device_supports_op(ggml_ncp_device_t dev, const struct ggml_tensor * op);

const struct ggml_ncp_device_props * ggml_ncp_device_get_props(ggml_ncp_device_t dev);

// device buffer

enum ggml_ncp_buffer_layout {
    GGML_NCP_LAYOUT_UNINITIALIZED,
    GGML_NCP_LAYOUT_0123,           // contiguous layout
    GGML_NCP_LAYOUT_0123_C4N4,      // optimal weight layout
    GGML_NCP_LAYOUT_0213,
    GGML_NCP_LAYOUT_1023_C16,       // (mostly) optimal feature layout
};

struct ggml_ncp_tensor_extra {
    ggml_ncp_buffer_layout layout;
};

struct ggml_ncp_buffer;
typedef struct ggml_ncp_buffer * ggml_ncp_buffer_t;

struct ggml_ncp_buffer {
    void * data;
    size_t size;
    std::vector<ggml_ncp_tensor_extra *> tensor_extras;
};

ggml_ncp_buffer_t ggml_ncp_buffer_init(ggml_ncp_device_t dev, size_t size);

void   ggml_ncp_buffer_free    (ggml_ncp_buffer_t buf);
void * ggml_ncp_buffer_get_base(ggml_ncp_buffer_t buf);

void   ggml_ncp_buffer_memset_tensor(ggml_ncp_buffer_t buf, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size);
void   ggml_ncp_buffer_set_tensor   (ggml_ncp_buffer_t buf, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
void   ggml_ncp_buffer_get_tensor   (ggml_ncp_buffer_t buf, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
void   ggml_ncp_buffer_clear        (ggml_ncp_buffer_t buf, uint8_t value);
