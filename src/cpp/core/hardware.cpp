// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "core/hardware.h"

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <winreg.h>
#else
#include <unistd.h>

#include <fstream>
#endif

namespace mdapy {

std::string cpu_name() {
#if defined(__APPLE__)
    char name[256] = {0};
    size_t size = sizeof(name) - 1;
    if (sysctlbyname("machdep.cpu.brand_string", name, &size, nullptr, 0) == 0) {
        return name;
    }
#elif defined(_WIN32)
    char name[256] = {0};
    DWORD size = sizeof(name);
    if (RegGetValueA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                     "ProcessorNameString", RRF_RT_REG_SZ, nullptr, name,
                     &size) == ERROR_SUCCESS) {
        return name;
    }
#else
    // x86 processors write "model name"; many ARM ones write nothing of the kind
    std::ifstream info("/proc/cpuinfo");
    std::string line;
    while (std::getline(info, line)) {
        const size_t colon = line.find(':');
        if (line.rfind("model name", 0) == 0 && colon != std::string::npos) {
            const size_t start = line.find_first_not_of(" \t", colon + 1);
            return start == std::string::npos ? std::string() : line.substr(start);
        }
    }
#endif
    return "";
}

std::uint64_t memory_bytes() {
#if defined(__APPLE__)
    std::uint64_t bytes = 0;
    size_t size = sizeof(bytes);
    return sysctlbyname("hw.memsize", &bytes, &size, nullptr, 0) == 0 ? bytes : 0;
#elif defined(_WIN32)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    return GlobalMemoryStatusEx(&status) != 0 ? status.ullTotalPhys : 0;
#else
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page = sysconf(_SC_PAGE_SIZE);
    return pages > 0 && page > 0
               ? static_cast<std::uint64_t>(pages) * static_cast<std::uint64_t>(page)
               : 0;
#endif
}

}  // namespace mdapy
