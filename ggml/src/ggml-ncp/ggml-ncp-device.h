#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

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

struct ggml_ncp_buffer;
typedef struct ggml_ncp_buffer * ggml_ncp_buffer_t;

struct ggml_ncp_buffer {
    void * data;
    size_t size;
};

ggml_ncp_buffer_t ggml_ncp_buffer_init(ggml_ncp_device_t dev, size_t size);

void   ggml_ncp_buffer_free    (ggml_ncp_buffer_t buf);
void * ggml_ncp_buffer_get_base(ggml_ncp_buffer_t buf);

void   ggml_ncp_buffer_memset_tensor(ggml_ncp_buffer_t buf, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size);
void   ggml_ncp_buffer_set_tensor   (ggml_ncp_buffer_t buf, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
void   ggml_ncp_buffer_get_tensor   (ggml_ncp_buffer_t buf, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
void   ggml_ncp_buffer_clear        (ggml_ncp_buffer_t buf, uint8_t value);

#ifdef __cplusplus
}
#endif
