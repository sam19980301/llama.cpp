message(STATUS "NCP found")
set(TARGET_NAME ggml-ncp)

ggml_add_backend_library(${TARGET_NAME}
                         ggml-ncp.cpp
                         ../../include/ggml-ncp.h)

file(GLOB   GGML_HEADERS_NCP "*.h")
file(GLOB   GGML_SOURCES_NCP "*.cpp")
target_sources(${TARGET_NAME} PRIVATE ${GGML_HEADERS_NCP} ${GGML_SOURCES_NCP})