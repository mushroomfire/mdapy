// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Sorting and selection on the device: CUB with CUDA; with HIP, hipCUB on
// rocPRIM, which has the names and the arguments of CUB, and the stable
// order of its radix sort and of its selection.

#pragma once

#include "cuda/portable.cuh"

#include <thrust/iterator/counting_iterator.h>

#if defined(__HIP__) || defined(__HIPCC__)
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#else
#include <cub/cub.cuh>
#endif
