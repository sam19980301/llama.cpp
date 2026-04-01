#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_NCP_NAME "NCP"

// backend API

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_ncp_reg(void);

#ifdef __cplusplus
}
#endif