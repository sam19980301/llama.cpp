#pragma once

#include "ggml-ncp-device.h"

// backend context

struct ggml_ncp;
typedef struct ggml_ncp * ggml_ncp_t;

struct ggml_ncp {
    char name[128];

    ggml_ncp_device_t dev;

    // TODO(sam)
    ggml_abort_callback abort_callback;
    void *              abort_callback_data;
};

ggml_ncp_t ggml_ncp_init(ggml_ncp_device_t dev);
void ggml_ncp_free(ggml_ncp_t ctx);

const char * ggml_ncp_get_name(ggml_ncp_t ctx);

enum ggml_status ggml_ncp_graph_compute(ggml_ncp_t ctx, struct ggml_cgraph * gf);
