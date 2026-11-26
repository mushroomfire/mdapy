// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Where the time of a calculation goes. Built with -DMDAPY_DFT_PROFILE, a
// MDAPY_SPAN("name") waits for the GPU where it stands and where its scope
// ends, and adds the seconds between to the row of its name under the names
// of the spans open around it on this thread ("davidson/apply/fft"). The
// process prints the rows at its end, and mdapy_dft_profile(label) prints and
// clears them before. MDAPY_SPANS(parts) declares spans one after the other in
// a scope, MDAPY_SPAN_NEXT(parts, "name") ends the one open and opens the next,
// and the last ends with the scope. Without the macro a span is nothing.

#pragma once

#ifdef MDAPY_DFT_PROFILE

#include <optional>

namespace mdapy::dft {

class Span {
public:
    explicit Span(const char* name);
    ~Span();
    Span(const Span&) = delete;
    Span& operator=(const Span&) = delete;

private:
    unsigned long length_;  // of the path before this span
    long long start_;
};

}  // namespace mdapy::dft

#define MDAPY_SPAN_JOIN2(a, b) a##b
#define MDAPY_SPAN_JOIN(a, b) MDAPY_SPAN_JOIN2(a, b)
#define MDAPY_SPAN(name) const ::mdapy::dft::Span MDAPY_SPAN_JOIN(mdapy_span_, __LINE__)(name)
#define MDAPY_SPANS(parts) std::optional<::mdapy::dft::Span> parts
#define MDAPY_SPAN_NEXT(parts, name) (parts.reset(), parts.emplace(name))

#else

#define MDAPY_SPAN(name)
#define MDAPY_SPANS(parts)
#define MDAPY_SPAN_NEXT(parts, name)

#endif
