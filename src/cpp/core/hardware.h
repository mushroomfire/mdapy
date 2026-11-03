// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <cstdint>
#include <string>

namespace mdapy {

// The name of the processor, or empty where the system does not tell it.
std::string cpu_name();

// The memory of the machine in bytes, or 0 where the system does not tell it.
std::uint64_t memory_bytes();

}  // namespace mdapy
