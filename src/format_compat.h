#pragma once

// Compatibility shim: use std::format/print/println when available (C++23),
// otherwise fall back to the {fmt} library.
//
// HAVE_STD_FORMAT is set by CMake (CheckSourceCompiles) and passed via
// target_compile_definitions, so this header just reads what CMake detected.

#ifdef HAVE_STD_FORMAT
#  include <format>
#  include <print>
namespace compat {
    using std::format;
    using std::print;
    using std::println;
}
#else
#  include <fmt/format.h>
#  include <fmt/ostream.h>   // fmt::print/println with std::ostream& (e.g. ofstream)
namespace compat {
    using fmt::format;
    using fmt::print;
    using fmt::println;
}
#endif
