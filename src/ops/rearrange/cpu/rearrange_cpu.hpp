#pragma once
#include "llaisys.h"
#include <cstddef>
namespace llaisys::ops::cpu {
void rearrange(std::byte *out, const std::byte *in, llaisysDataType_t type, const size_t *shape, const ptrdiff_t *strides, size_t ndim, size_t numel);
}