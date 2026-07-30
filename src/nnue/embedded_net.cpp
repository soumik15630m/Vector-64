#include "embedded_net.h"

// Platform accessors for the embedded default net. On MSVC the net is a Windows
// RCDATA resource (embedded_net.rc); elsewhere it is bracketed by two labels in
// embedded_net.S (.incbin). This file is compiled only into ChessEngine-nnue.

#if defined(_MSC_VER)

#include <windows.h>

namespace NNUE {
namespace {
HRSRC find_res() {
  // MAKEINTRESOURCEW(10) == the wide RT_RCDATA (RT_RCDATA itself is the ANSI
  // macro, which FindResourceW rejects).
  return FindResourceW(nullptr, L"STK_EMBEDDED_NET", MAKEINTRESOURCEW(10));
}
} // namespace

const unsigned char *embedded_net_data() {
  return static_cast<const unsigned char *>(
      LockResource(LoadResource(nullptr, find_res())));
}

std::size_t embedded_net_size() {
  return static_cast<std::size_t>(SizeofResource(nullptr, find_res()));
}
} // namespace NNUE

#else

extern "C" {
extern const unsigned char stk_embedded_net_data[];
extern const unsigned char stk_embedded_net_end[];
}

namespace NNUE {

const unsigned char *embedded_net_data() { return stk_embedded_net_data; }

std::size_t embedded_net_size() {
  return static_cast<std::size_t>(stk_embedded_net_end - stk_embedded_net_data);
}

} // namespace NNUE

#endif
