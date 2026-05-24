#include "ggml-ncp.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-ncp-device.h"
#include "ggml-ncp-context.h"

#include <memory>
#include <mutex>
#include <vector>

// number of NCP devices
static int g_devices = 1;

///////////////////////
// backend interface //
///////////////////////

// buffer

static void ggml_backend_ncp_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_ncp_buffer_t ctx = (ggml_ncp_buffer_t)buffer->context;

    for (ggml_ncp_tensor_extra * tensor_extra : ctx->tensor_extras) {
        delete tensor_extra;
    }

    ggml_ncp_buffer_free(ctx);
}

static void * ggml_backend_ncp_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_ncp_buffer_t ctx = (ggml_ncp_buffer_t)buffer->context;

    return ggml_ncp_buffer_get_base(ctx);
}

static enum ggml_status ggml_backend_ncp_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_ncp_buffer_t ctx = (ggml_ncp_buffer_t)buffer->context;

    ggml_ncp_tensor_extra * extra = new ggml_ncp_tensor_extra{
        /* .ggml_ncp_buffer_layout  = */ GGML_NCP_LAYOUT_UNINITIALIZED,
    };
    ctx->tensor_extras.push_back(extra);
    tensor->extra = extra;

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_ncp_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_ncp_buffer_t ctx = (ggml_ncp_buffer_t)buffer->context;

    ggml_ncp_buffer_memset_tensor(ctx, tensor, value, offset, size);
}

static void ggml_backend_ncp_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_ncp_buffer_t ctx = (ggml_ncp_buffer_t)buffer->context;

    ggml_ncp_buffer_set_tensor(ctx, tensor, data, offset, size);
}

static void ggml_backend_ncp_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_ncp_buffer_t ctx = (ggml_ncp_buffer_t)buffer->context;

    ggml_ncp_buffer_get_tensor(ctx, tensor, data, offset, size);
}

static bool ggml_backend_ncp_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    // TODO(sam)
    ggml_ncp_buffer_t ctx = (ggml_ncp_buffer_t)buffer->context;

    GGML_UNUSED(ctx);

    GGML_UNUSED(buffer);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);

    return false;
}

static void ggml_backend_ncp_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_ncp_buffer_t ctx = (ggml_ncp_buffer_t)buffer->context;

    ggml_ncp_buffer_clear(ctx, value);
}

static ggml_backend_buffer_i ggml_backend_ncp_buffer_i = {
    /* .free_buffer     = */ ggml_backend_ncp_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_ncp_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_ncp_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_ncp_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_ncp_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_ncp_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_ncp_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_ncp_buffer_clear,
    /* .reset           = */ NULL, // TODO(sam)
};

// buffer type

struct ggml_backend_ncp_buffer_type_ctx {
    int device;
    std::string name;
};

struct ggml_backend_ncp_buffer_type_ctx_deleter {
    void operator()(ggml_backend_ncp_buffer_type_ctx * ctx) const {
        delete ctx;
    }
};

typedef std::unique_ptr<ggml_backend_ncp_buffer_type_ctx, ggml_backend_ncp_buffer_type_ctx_deleter> ggml_backend_ncp_buffer_type_ctx_ptr;

static const char * ggml_backend_ncp_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_ncp_buffer_type_ctx * ctx = (ggml_backend_ncp_buffer_type_ctx *)buft->context;

    return ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_ncp_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_ncp_device_t ctx_dev = (ggml_ncp_device_t)buft->device->context;
    ggml_ncp_buffer_t res = ggml_ncp_buffer_init(ctx_dev, size);
    ggml_backend_buffer_i buf_i = ggml_backend_ncp_buffer_i;
    return ggml_backend_buffer_init(buft, buf_i, res, size);
}

static size_t ggml_backend_ncp_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 32; // TODO(sam)

    GGML_UNUSED(buft);
}

static size_t ggml_backend_ncp_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_ncp_device_t ctx_dev = (ggml_ncp_device_t)buft->device->context;

    return ggml_ncp_device_get_props(ctx_dev)->max_buffer_size;
}

static size_t ggml_backend_ncp_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    size_t res = ggml_nbytes(tensor);

    // TODO(sam)
    // some operations require additional memory for fleeting data:

    return res;

    GGML_UNUSED(buft);
}

static bool ggml_backend_ncp_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    // TODO(sam) may support cpy tensor if buffer type is not host (?)
    return true;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_ncp_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static std::vector<ggml_backend_buffer_type> bufts;
    static std::vector<ggml_backend_ncp_buffer_type_ctx_ptr> ctxs;

    static bool initialized = false;
    if (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        for (int i = 0; i < g_devices; ++i) {
            ggml_backend_ncp_buffer_type_ctx * raw_ctx =
                new ggml_backend_ncp_buffer_type_ctx {
                    /* .device = */ i,
                    /* .name   = */ GGML_NCP_NAME + std::to_string(i),
                };
            ctxs.emplace_back(raw_ctx);

            ggml_backend_buffer_type buft = {
                /* .iface = */ {
                    /* .get_name         = */ ggml_backend_ncp_buffer_type_get_name,
                    /* .alloc_buffer     = */ ggml_backend_ncp_buffer_type_alloc_buffer,
                    /* .get_alignment    = */ ggml_backend_ncp_buffer_type_get_alignment,
                    /* .get_max_size     = */ ggml_backend_ncp_buffer_type_get_max_size,
                    /* .get_alloc_size   = */ ggml_backend_ncp_buffer_type_get_alloc_size,
                    /* .is_host          = */ ggml_backend_ncp_buffer_type_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_ncp_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = true;
    }

    return &bufts[device];
}

// backend

static const char * ggml_backend_ncp_name(ggml_backend_t backend) {
    ggml_ncp_t ctx = (ggml_ncp_t)backend->context;

    return ggml_ncp_get_name(ctx);
}

static void ggml_backend_ncp_free(ggml_backend_t backend) {
    ggml_ncp_t ctx = (ggml_ncp_t)backend->context;

    // TODO(sam): necessary?
    // wait for any ongoing async operations to finish

    ggml_ncp_free(ctx);

    free(backend);
}

static enum ggml_status ggml_backend_ncp_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    // TODO(sam)
    // TODO(sam): consider graph mode
    // Reference: CPU, CANN, CUDA, SYCL counterpart

    ggml_ncp_t ctx = (ggml_ncp_t)backend->context;

    return ggml_ncp_graph_compute(ctx, cgraph);
}

static ggml_backend_i ggml_backend_ncp_i = {
    /* .get_name                = */ ggml_backend_ncp_name,
    /* .free                    = */ ggml_backend_ncp_free,
    /* .set_tensor_async        = */ NULL, // TODO(sam)
    /* .get_tensor_async        = */ NULL, // TODO(sam)
    /* .cpy_tensor_async        = */ NULL, // only needed for multi-GPU setups // TODO(sam)
    /* .synchronize             = */ NULL, // TODO(sam)
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_ncp_graph_compute,
    /* .event_record            = */ NULL, // TODO(sam)
    /* .event_wait              = */ NULL, // TODO(sam)
    /* .graph_optimize          = */ NULL, // TODO(sam)
};

static ggml_guid_t ggml_backend_ncp_guid(void) {
    static ggml_guid guid = { 0x2c, 0x83, 0x97, 0xf3, 0xd9, 0x0d, 0x4d, 0xd4, 0xb2, 0x6d, 0x8c, 0x8a, 0x38, 0xaf, 0xa0, 0x98 };
    return &guid;
}

// backend device

static const char * ggml_backend_ncp_device_get_name(ggml_backend_dev_t dev) {
    ggml_ncp_device_t ctx_dev = (ggml_ncp_device_t)dev->context;

    const ggml_ncp_device_props * props_dev = ggml_ncp_device_get_props(ctx_dev);

    return props_dev->name;
}

static const char * ggml_backend_ncp_device_get_description(ggml_backend_dev_t dev) {
    ggml_ncp_device_t ctx_dev = (ggml_ncp_device_t)dev->context;

    return ggml_ncp_device_get_props(ctx_dev)->desc;
}

static void ggml_backend_ncp_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_ncp_device_t ctx_dev = (ggml_ncp_device_t)dev->context;

    ggml_ncp_device_get_memory(ctx_dev, free, total);
}

static enum ggml_backend_dev_type ggml_backend_ncp_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_ncp_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_ncp_device_get_name(dev);
    props->description = ggml_backend_ncp_device_get_description(dev);
    props->type        = ggml_backend_ncp_device_get_type(dev);

    ggml_backend_ncp_device_get_memory(dev, &props->memory_free, &props->memory_total);

    // TODO(sam)
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ false,
    };
}

static ggml_backend_t ggml_backend_ncp_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_ncp_device_t ctx_dev = (ggml_ncp_device_t)dev->context;

    ggml_ncp_t ctx = ggml_ncp_init(ctx_dev);
    if (ctx == NULL) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        return NULL;
    }

    ggml_backend_t backend = (ggml_backend_t) malloc(sizeof(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_ncp_guid(),
        /* .interface = */ ggml_backend_ncp_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    return backend;

    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_ncp_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_ncp_device_t ctx_dev = (ggml_ncp_device_t)dev->context;

    const ggml_ncp_device_props * props_dev = ggml_ncp_device_get_props(ctx_dev);

    return ggml_backend_ncp_buffer_type(props_dev->device);
}

static bool ggml_backend_ncp_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_ncp_device_t ctx_dev = (ggml_ncp_device_t)dev->context;

    return ggml_ncp_device_supports_op(ctx_dev, op);
}

static bool ggml_backend_ncp_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft->device == dev && buft->iface.get_name == ggml_backend_ncp_buffer_type_get_name;

    GGML_UNUSED(dev);
}

static ggml_backend_device_i ggml_backend_ncp_device_i = {
    /* .get_name             = */ ggml_backend_ncp_device_get_name,
    /* .get_description      = */ ggml_backend_ncp_device_get_description,
    /* .get_memory           = */ ggml_backend_ncp_device_get_memory,
    /* .get_type             = */ ggml_backend_ncp_device_get_type,
    /* .get_props            = */ ggml_backend_ncp_device_get_props,
    /* .init_backend         = */ ggml_backend_ncp_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_ncp_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL, // TODO(sam)
    /* .supports_op          = */ ggml_backend_ncp_device_supports_op,
    /* .supports_buft        = */ ggml_backend_ncp_device_supports_buft,
    /* .offload_op           = */ NULL, // TODO(sam)
    /* .event_new            = */ NULL, // TODO(sam)
    /* .event_free           = */ NULL, // TODO(sam)
    /* .event_synchronize    = */ NULL, // TODO(sam)
};

// backend registry

struct ggml_backend_ncp_reg {
    std::vector<ggml_backend_dev_t> devices;
};

typedef struct ggml_backend_ncp_reg * ggml_backend_ncp_reg_t;

static ggml_backend_ncp_reg_t ggml_backend_ncp_reg_init(void) {
    ggml_backend_ncp_reg_t ctx = new struct ggml_backend_ncp_reg;

    return ctx;
}

static void ggml_backend_ncp_reg_free(ggml_backend_ncp_reg_t ctx) {
    delete ctx;
}

struct ggml_backend_ncp_reg_deleter {
    void operator()(ggml_backend_ncp_reg_t ctx) {
        ggml_backend_ncp_reg_free(ctx);
    }
};

typedef std::unique_ptr<struct ggml_backend_ncp_reg, ggml_backend_ncp_reg_deleter> ggml_backend_ncp_reg_ptr;

static const char * ggml_backend_ncp_reg_get_name(ggml_backend_reg_t reg) {
    return GGML_NCP_NAME;

    GGML_UNUSED(reg);
}

static size_t ggml_backend_ncp_reg_device_count(ggml_backend_reg_t reg) {
    ggml_backend_ncp_reg_t ctx = (ggml_backend_ncp_reg_t)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_ncp_reg_device_get(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_ncp_reg_t ctx = (ggml_backend_ncp_reg_t)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static ggml_backend_feature g_ggml_backend_ncp_features[] = {
    { NULL, NULL },
};

static ggml_backend_feature * ggml_backend_ncp_get_features(ggml_backend_reg_t reg) {
    return g_ggml_backend_ncp_features;

    GGML_UNUSED(reg);
}

static void * ggml_backend_ncp_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    // TODO(sam)
    /*
    if (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        ggml_backend_dev_get_extra_bufts_t fct = ggml_backend_ncp_device_get_extra_buffers_type;
        return (void *)fct;
    }
    */
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_ncp_get_features;
    }
    // TODO(sam)
    /*
    if (strcmp(name, "ggml_backend_set_abort_callback") == 0) {
        return (void *)ggml_backend_ncp_set_abort_callback;
    }
    */

    return NULL;

    GGML_UNUSED(reg);
}

static ggml_backend_reg_i ggml_backend_ncp_reg_i = {
    /* .get_name         = */ ggml_backend_ncp_reg_get_name,
    /* .get_device_count = */ ggml_backend_ncp_reg_device_count,
    /* .get_device       = */ ggml_backend_ncp_reg_device_get,
    /* .get_proc_address = */ ggml_backend_ncp_get_proc_address,
};

static ggml_backend_dev_t ggml_backend_ncp_device_init(ggml_backend_reg_t reg, int device) {
    return new ggml_backend_device {
        /* .iface   = */ ggml_backend_ncp_device_i,
        /* .reg     = */ reg,
        /* .context = */ ggml_ncp_device_get(device),
    };
}

static void ggml_backend_ncp_device_free(ggml_backend_dev_t dev) {
    delete dev;
}

struct ggml_backend_device_deleter {
    void operator()(ggml_backend_dev_t ctx) {
        ggml_backend_ncp_device_free(ctx);
    }
};

typedef std::unique_ptr<ggml_backend_device, ggml_backend_device_deleter> ggml_backend_device_ptr;

ggml_backend_reg_t ggml_backend_ncp_reg(void) {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);

        const char * env = getenv("GGML_NCP_DEVICES");
        if (env) {
            g_devices = atoi(env);
        }

        static std::vector<ggml_backend_device_ptr> devs;

        if (!initialized) {
            static ggml_backend_ncp_reg_ptr reg_ctx(ggml_backend_ncp_reg_init());

            for (int i = 0; i < g_devices; ++i) {
                auto * dev = ggml_backend_ncp_device_init(&reg, i);
                devs.emplace_back(dev);

                reg_ctx->devices.push_back(dev);
            }

            reg = {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_ncp_reg_i,
                /* .context     = */ reg_ctx.get(),
            };
        }

        initialized = true;
    }

    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_ncp_reg)
