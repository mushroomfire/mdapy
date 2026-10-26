// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Vectors whose new elements of a trivial type are left uninitialized. A
// large result is then not cleared first, and its memory is first touched by
// the threads that write it.

#pragma once

#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

namespace mdapy {

template <class T>
class UninitializedAllocator : public std::allocator<T> {
public:
    using value_type = T;
    template <class U>
    struct rebind {
        using other = UninitializedAllocator<U>;
    };

    UninitializedAllocator() noexcept = default;
    template <class U>
    UninitializedAllocator(const UninitializedAllocator<U>&) noexcept {}

    template <class U, class... Args>
    void construct(U* p, Args&&... args) {
        ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
    }
    template <class U>
    void construct(U* p) noexcept(std::is_nothrow_default_constructible_v<U>) {
        ::new (static_cast<void*>(p)) U;
    }
};

template <class T>
using Buffer = std::vector<T, UninitializedAllocator<T>>;

}  // namespace mdapy
