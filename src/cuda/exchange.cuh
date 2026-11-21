// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// What the Exchange keeps (cuda.h): the messages of a build and the memory
// each sends and receives through, shared by exchange.cu (each step) and
// domain.cu (the builds).

#pragma once

#include <memory>
#include <stdexcept>
#include <vector>

#include "cuda/atoms.cuh"
#include "cuda/cuda.h"

namespace mdapy::cuda {

// Memory of the host that the device copies to and from at full speed,
// kept for the messages of the next steps.
struct Pinned {
    void* data = nullptr;
    std::size_t size = 0;
    Pinned() = default;
    Pinned(const Pinned&) = delete;
    Pinned& operator=(const Pinned&) = delete;
    ~Pinned() {
        if (data != nullptr) cudaFreeHost(data);
    }
    void* reserve(std::size_t n) {
        n = n > 0 ? n : 1;
        if (n > size) {
            if (data != nullptr) cudaFreeHost(data);
            data = nullptr;
            size = 0;
            MDAPY_CUDA(cudaHostAlloc(&data, n, cudaHostAllocDefault));
            size = n;
        }
        return data;
    }
};

struct Exchange::Impl {
    bool device = false;
    // A message of the borders, or the atoms leaving to one side.
    struct Slot {
        Buffer<std::int32_t> chosen;          // the rows it sends, the caller's at the build
        std::int64_t chosen_count = 0;
        Buffer<std::int32_t> sent, received;  // the places of those rows, and of its ghosts
        std::int64_t start = 0, count = 0;    // the first row of what it receives, their number
        Buffer<unsigned char> out, in;        // what it sends and receives, on the device
        Pinned host_out, host_in;             // their copies on the host
        View out_view{nullptr, 0, false};
    };
    std::vector<std::unique_ptr<Slot>> slots;
    int used = 0;
    Slot sides[2];
    Buffer<std::int32_t> stay;  // the rows that stay in the brick
    std::int64_t stayed = 0;
    Buffer<std::int8_t> step;
    Buffer<int> found;  // what the selections found, and the largest step
    Buffer<unsigned char> temp;
    Buffer<std::int32_t> place;

    Slot& at(int k) {
        if (k < 0 || k >= used) throw std::invalid_argument("no such message");
        return *slots[static_cast<std::size_t>(k)];
    }
    const Slot& at(int k) const {
        if (k < 0 || k >= used) throw std::invalid_argument("no such message");
        return *slots[static_cast<std::size_t>(k)];
    }
    // What a kernel wrote into out, bytes of it, as MPI is given it: on the
    // device once the kernels are done, else copied to the host.
    View staged(Slot& s, std::size_t bytes) {
        if (device) {
            MDAPY_CUDA(cudaStreamSynchronize(nullptr));
            return View{s.out.data(), bytes, true};
        }
        void* host = s.host_out.reserve(bytes);
        if (bytes > 0) {
            MDAPY_CUDA(cudaMemcpy(host, s.out.data(), bytes, cudaMemcpyDeviceToHost));
        }
        return View{host, bytes, false};
    }
    // The memory MPI receives bytes into.
    View receive(Slot& s, std::size_t bytes) {
        if (device) {
            s.in.resize(bytes);
            return View{s.in.data(), bytes, true};
        }
        return View{s.host_in.reserve(bytes), bytes, false};
    }
    // What came, bytes of it, on the device.
    const unsigned char* arrived(Slot& s, std::size_t bytes) {
        if (!device) {
            s.in.resize(bytes);
            if (bytes > 0) {
                MDAPY_CUDA(cudaMemcpy(s.in.data(), s.host_in.data, bytes, cudaMemcpyHostToDevice));
            }
        } else if (s.in.size() != bytes) {
            throw std::logic_error("nothing was received for the message");
        }
        return s.in.data();
    }
};

}  // namespace mdapy::cuda
